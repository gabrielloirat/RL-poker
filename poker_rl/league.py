"""Per-run self-play league with immutable, hashed opponent checkpoints."""
import copy
import hashlib
from pathlib import Path
import shutil

import numpy as np

from poker_rl.evaluation import _preserve_random_state
from poker_rl.game import CARD_INDEX, PokerGame
from poker_rl.opponents import MixedOpponent
from poker_rl.records import write_json

HEURISTICS = ('random', 'check_call', 'tight', 'aggressive')


def frozen_spec(path, *, name='ppo', agents_dir='agents', label=None, kwargs=None):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'name': name, 'agents_dir': str(Path(agents_dir).resolve()),
            'label': label or path.stem, 'kwargs': kwargs or {}}


def load_frozen(spec):
    from poker_rl.plugins import AgentRegistry
    path = Path(spec['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError(f"Checkpoint de ligue modifié : {spec['label']}")
    with _preserve_random_state():
        registry = AgentRegistry(spec['agents_dir'])
        agent = registry.load(spec['name'], {**spec.get('kwargs', {}), 'autoload': False})
        agent.load_checkpoint(path)
        model = getattr(agent, 'model', None)
        if model is not None:
            model.policy.set_training_mode(False)
            for parameter in model.policy.parameters():
                parameter.requires_grad_(False)
    return agent


def freeze_anchors(registry, root, *, num_players=2, candidate_root=None, candidate_seed=None):
    """Copy compatible catalog policies once; every seed gets the same anchors."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    anchors, skipped, hashes = {}, [], set()
    for model_id, model in registry.saved_models.items():
        if model['num_players'] != num_players or not registry.native_training(model_id):
            continue
        digest = model['checkpoint_sha256']
        if digest in hashes:
            continue
        try:
            # Fail before launching a campaign if an old architecture is incompatible.
            with _preserve_random_state():
                registry.load(model_id)
            target = root / f'{digest[:16]}.checkpoint'
            shutil.copyfile(model['path'], target)
            key = f'anchor_{digest[:12]}'
            anchors[key] = frozen_spec(target, name=model['name'], agents_dir=registry.directory,
                label=registry.label(model_id), kwargs=model.get('kwargs'))
            anchors[key].update(source_model_id=model_id, catalog_kind=model.get('catalog_kind'))
            hashes.add(digest)
        except Exception as exc:
            skipped.append({'model': model_id, 'error': str(exc)})
    if candidate_root is not None and candidate_seed is not None:
        import json
        for manifest_path in sorted(Path(candidate_root).glob('*/campaign.json'), reverse=True):
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('status') != 'completed' or manifest['config']['num_players'] != num_players:
                continue
            candidate = next((r for r in manifest['runs'] if r['seed'] == candidate_seed and r['status'] == 'completed'), None)
            if not candidate:
                continue
            try:
                source = Path(candidate['path']) / 'selected.checkpoint'
                spec = frozen_spec(source, agents_dir=registry.directory,
                    label=f"Campagne {manifest['id']} · seed {candidate_seed}")
                if candidate.get('selected_sha256') and candidate['selected_sha256'] != spec['sha256']:
                    raise ValueError('Checkpoint du candidat modifié depuis sa validation.')
                load_frozen(spec)
                if spec['sha256'] not in hashes:
                    target = root / f"{spec['sha256'][:16]}.checkpoint"
                    shutil.copyfile(source, target)
                    spec.update(path=str(target.resolve()), source_campaign=manifest['id'], source_seed=candidate_seed)
                    anchors[f"anchor_{spec['sha256'][:12]}"] = spec
                    hashes.add(spec['sha256'])
            except Exception as exc:
                skipped.append({'model': f"{manifest['id']}:{candidate_seed}", 'error': str(exc)})
            break
    return anchors, skipped


class League:
    def __init__(self, root, *, seed, anchors=None, agents_dir='agents', max_history=8,
                 diversity_threshold=.02, num_players=2):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.seed, self.agents_dir, self.max_history = seed, agents_dir, max_history
        self.anchors = dict(anchors or {})
        self.history = {}
        self.retired = []
        self.exposure = {}
        self.diversity_threshold, self.num_players = diversity_threshold, num_players
        self.signatures = {}
        self.save()

    @property
    def members(self):
        return {**self.anchors, **self.history}

    def snapshot(self, agent, steps):
        key = f'seed_{self.seed}_step_{steps}'
        if key in self.history or any(item['id'] == key for item in self.retired):
            return
        target = self.root / f'{key}.checkpoint'
        agent.save_checkpoint(target)
        spec = frozen_spec(target, agents_dir=self.agents_dir, label=f'Seed {self.seed} · {steps} décisions')
        signature = behaviour_signature(agent, num_players=self.num_players)
        self.signatures[key] = signature
        similar = next((member for member in self.history
                        if signature_distance(signature, self.signatures[member]) < self.diversity_threshold), None)
        if similar is not None:
            self.retired.append({'id': key, **spec, 'reason': 'similar_behaviour', 'similar_to': similar})
            self.save()
            return
        self.history[key] = spec
        while len(self.history) > self.max_history:
            old = next(iter(self.history))
            self.retired.append({'id': old, **self.history.pop(old), 'reason': 'capacity'})
        self.save()

    def record(self, opponent, depth, reward, terminated):
        key = f'{depth:g}bb:{opponent}'
        row = self.exposure.setdefault(key, {'opponent': opponent, 'depth_bb': depth,
                                            'hands': 0, 'truncated': 0, 'net_bb': 0.})
        row['hands'] += int(terminated)
        row['truncated'] += int(not terminated)
        if terminated:
            row['net_bb'] += reward

    def state(self):
        return {'seed': self.seed, 'heuristics': list(HEURISTICS), 'heuristic_probability': .5,
                'anchors': self.anchors, 'history': self.history, 'retired': self.retired,
                'diversity_threshold': self.diversity_threshold,
                'behaviour': {key: {k: v for k, v in signature.items() if k != 'probabilities'}
                              for key, signature in self.signatures.items()},
                'exposure': list(self.exposure.values()), 'data_role': 'training_diagnostic'}

    def save(self):
        write_json(self.root / 'league.json', self.state())


class LeagueOpponent:
    """Only act/on_episode_start; no observe or learning reaches frozen policies."""
    def __init__(self, league, seed):
        self.league, self.seed = league, seed
        self.current_id = None
        self.current = None
        self.cache = {}

    def on_episode_start(self, *, seed=None, training=False):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, seed or 0]))
        members = self.league.members
        if not members or rng.random() < .5:
            self.current_id = str(rng.choice(HEURISTICS))
            self.current = MixedOpponent(seed=int(rng.integers(2**31)), styles=(self.current_id,))
        else:
            self.current_id = str(rng.choice(list(members)))
            if self.current_id not in self.cache:
                self.cache[self.current_id] = load_frozen(members[self.current_id])
            self.current = self.cache[self.current_id]
            self.cache = {k: v for k, v in self.cache.items() if k in members}
        start = getattr(self.current, 'on_episode_start', None)
        if callable(start):
            start(seed=seed, training=False)

    def act(self, observation, action_mask, *, training=False):
        return self.current.act(observation, action_mask, training=False)


def behaviour_signature(agent, *, num_players=2):  # Compare des situations fixes
    with _preserve_random_state():
        clone = agent.evaluation_copy() if callable(getattr(agent, 'evaluation_copy', None)) else copy.deepcopy(agent)
        holdings = (('AH', 'AD'), ('AC', 'KC'), ('QH', 'JD'), ('9H', '9D'),
                    ('8C', '7C'), ('7C', '2D'), ('TC', '4D'), ('5H', '3H'))
        vectors = []
        for depth in (20, 100):
            for raised in (False, True):
                game = PokerGame(num_players=num_players, initial_stacks=depth * 2)
                try:
                    game.reset(seed=8765, options={'dealer_seat': 0})
                    if raised and game.action_masks()[3]:
                        game.step(3)
                    seat = game.current_seat
                    obs, mask = game.observe(seat), game.action_masks(seat)
                    for cards in holdings:
                        probe = obs.copy()
                        probe[:52] = 0
                        for card in cards:
                            probe[CARD_INDEX[card]] = 1
                        clone.on_episode_start(seed=1, training=False)
                        vectors.append(clone.action_probabilities(probe, mask).tolist())
                finally:
                    game.close()
    values = np.asarray(vectors)
    return {'version': 1, 'probabilities': values.tolist(), 'contexts': len(values),
            'mean_all_in_probability': float(values[:, 7].mean()),
            'greedy_all_in_rate': float((values.argmax(axis=1) == 7).mean())}


def signature_distance(a, b):  # Mesure l'écart entre deux politiques
    return float(np.abs(np.asarray(a['probabilities']) - np.asarray(b['probabilities'])).sum(axis=1).mean() / 2)
