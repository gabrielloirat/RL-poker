"""Fixed-seed curricula and held-out comparisons of PPO, DQN and MCTS.

Run with experiments/train_benchmark.py. Artifacts remain under ignored runs/.
Independent hands reset stacks; profit is measured in big blinds, not chips.
"""
import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import random
from statistics import NormalDist
import time

import numpy as np
import torch

from poker_rl.evaluation import _preserve_random_state
from poker_rl.game import ACTION_NAMES, PokerGame
from poker_rl.plugins import AgentRegistry
from poker_rl.records import write_json
from poker_rl.sb3_training import train_sb3_session
from poker_rl.table_session import TableSession
from poker_rl.benchmark_observation import UniversalAgent

ALGORITHMS = ('ppo', 'dqn', 'mcts')


@dataclass
class BenchmarkConfig:
    seed: int = 42
    players: tuple = (2, 4)
    ppo_steps: int = 8192
    dqn_hands: int = 2000
    mcts_hands: int = 500
    max_random_blocks: int = 8
    selfplay_blocks: int = 2
    specialist_blocks: int = 1
    gate_hands: int = 800
    confirmation_hands: int = 4800
    second_confirmation_hands: int = 9600
    final_hands: int = 1200
    gate_bb100: float = 100.
    mcts_simulations: int = 80
    evaluation_workers: int = 2

    def validate(self):
        if not self.players or len(set(self.players)) != len(self.players):
            raise ValueError('Formats de table vides ou dupliqués.')
        if any(type(p) is not int or p not in (2, 4) for p in self.players):
            raise ValueError('Les formats pris en charge sont 2 et 4 joueurs.')
        for key, value in asdict(self).items():
            if key not in ('players', 'gate_bb100', 'seed') and (type(value) is not int or value < 1):
                raise ValueError(f'{key} doit être un entier positif.')
        if min(self.ppo_steps, self.gate_hands, self.confirmation_hands, self.second_confirmation_hands, self.final_hands) < 2:
            raise ValueError('Au moins deux décisions / mains sont nécessaires.')
        if self.seed < 0 or not np.isfinite(self.gate_bb100):
            raise ValueError('Seed ou seuil invalide.')


def fixed_seed(base, purpose, players, algorithm=0, block=0):
    """Separate, reproducible RNG streams; purpose=1 train, 2 gate, 3 final."""
    return int(np.random.SeedSequence([base, purpose, players, algorithm, block]).generate_state(1)[0])


def reward_summary(rewards, *, critical=1.96):
    values = np.asarray(rewards, dtype=float)
    n = len(values)
    if not n:
        return {'hands': 0, 'net_bb': 0., 'bb100': None, 'ci_low': None, 'ci_high': None,
                'win_rate': None, 'tie_rate': None, 'standard_error_bb': None}
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    margin = critical * se * 100 if se is not None else None
    return {'hands': n, 'net_bb': float(values.sum()), 'bb100': mean * 100,
            'ci_low': mean * 100 - margin if margin is not None else None,
            'ci_high': mean * 100 + margin if margin is not None else None,
            'standard_error_bb': se, 'win_rate': float(np.mean(values > 1e-9)),
            'tie_rate': float(np.mean(np.abs(values) <= 1e-9))}


def passes_gate(summary, threshold):
    return (summary['hands'] >= 2 and summary['ci_low'] is not None
            and summary['ci_low'] > 0 and summary['bb100'] >= threshold)


def agent_options(kind, seed, config, *, frozen=False):
    if kind == 'ppo':
        return dict(seed=seed, autoload=False, deterministic=False, n_steps=512,
                    batch_size=128, learning_rate=3e-4, n_epochs=10, ent_coef=.01)
    if kind == 'dqn':
        return dict(seed=seed, buffer_size=1 if frozen else 50000,
                    epsilon_decay_steps=15000, checkpoint_every=10**9)
    if kind == 'mcts':
        return dict(seed=seed, num_simulations=config.mcts_simulations)
    if kind == 'random':
        return dict(seed=seed)
    raise ValueError(f'Agent non pris en charge : {kind}')


def save_policy(agent, kind, path, options):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save_checkpoint(path)
    spec = {'kind': kind, 'path': str(path), 'options': options,
            'observation_schema': 'universal_5slots_v1' if isinstance(agent, UniversalAgent) else 'native',
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(path.with_suffix('.json'), spec)
    return spec


def load_policy(registry, spec, config, seed):
    if hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('Checkpoint modifié depuis son enregistrement.')
    # Checkpoint loading must never reset the learner's process-wide generators.
    with _preserve_random_state():
        agent = registry.load(spec['kind'], agent_options(spec['kind'], seed, config, frozen=True))
        agent.load_checkpoint(spec['path'])
        if spec.get('observation_schema') == 'universal_5slots_v1':
            agent = UniversalAgent(agent)
    return agent


def evaluate_lineup(registry, candidate, rivals, config, *, seed, hands, output=None, critical=1.96):
    """Frozen policies only; rotate hero and dealer over all N*N combinations.

    The same deal stream is used across models/scenarios within a format. No
    observe or training callbacks are invoked, including for adaptive MCTS.
    """
    with _preserve_random_state():
        clone = getattr(candidate, 'evaluation_copy', None)
        policy = clone() if callable(clone) else copy.deepcopy(candidate)
        players_count = len(rivals) + 1
        opponents = [registry.load('random', {'seed': seed + i}) if rival == 'random'
                     else load_policy(registry, rival, config, seed + i)
                     for i, rival in enumerate(rivals)]
        game = PokerGame(num_players=players_count)
        rng = np.random.default_rng(seed)
        records, rewards, truncated = [], [], 0
        actions = np.zeros(len(ACTION_NAMES), dtype=int)
        try:
            for hand in range(hands):
                hero = hand % players_count
                dealer = (hand // players_count) % players_count
                hand_seed = int(rng.integers(0, 2**31 - 1))
                lineup = list(opponents)
                lineup.insert(hero, policy)
                # Rotate heterogeneous opponents relative to each other too.
                if len(opponents) > 1:
                    shift = (hand // (players_count ** 2)) % len(opponents)
                    ordered = opponents[shift:] + opponents[:shift]
                    lineup = list(ordered)
                    lineup.insert(hero, policy)
                for seat, player in enumerate(lineup):
                    callback = getattr(player, 'on_episode_start', None)
                    if callable(callback):
                        callback(seed=hand_seed + seat, training=False)
                game.reset(seed=hand_seed, options={'dealer_seat': dealer})
                while not game.finished:
                    seat = game.current_seat
                    mask = game.action_masks(seat)
                    action = int(lineup[seat].act(game.observe(seat), mask, training=False))
                    if action < 0 or action >= len(mask) or not mask[action]:
                        raise ValueError(f'Action illégale pendant le test : {action}')
                    if seat == hero:
                        actions[action] += 1
                    game.step(action)
                reward = float(game.rewards()[hero])
                if game.truncated:
                    truncated += 1
                else:
                    rewards.append(reward)
                records.append({'hand': hand, 'seed': hand_seed, 'hero': hero, 'dealer': dealer,
                                'reward_bb': reward, 'truncated': game.truncated})
        finally:
            game.close()
        if output:
            write_csv(output, records)
        result = reward_summary(rewards, critical=critical)
        result.update(seed=seed, requested_hands=hands, truncated=truncated,
                      actions=dict(zip(ACTION_NAMES, actions.tolist())),
                      position_counts=[sum(r['hero'] == i for r in records) for i in range(players_count)])
        if truncated:
            raise RuntimeError(f'{truncated} mains tronquées : comparaison refusée.')
        return result


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_block(registry, learner, kind, rivals, config, *, seed, folder):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    opponents = [registry.load('random', {'seed': seed + i + 1}) if rival == 'random'
                 else load_policy(registry, rival, config, seed + i + 1)
                 for i, rival in enumerate(rivals)]
    agents = [learner, *opponents]
    selections = [(kind, {}), *[(r if r == 'random' else r['kind'], {}) for r in rivals]]
    hands, curve = [], []
    started = time.monotonic()
    session = None

    def on_hand(table):
        if table.game.truncated:
            raise RuntimeError('Main tronquée pendant l’entraînement.')
        hands.append({'hand': len(hands) + 1, 'reward_bb': float(table.game.rewards()[0]),
                      'seed': table.seed + table.hand - 1})

    def attach(table):
        table.hand_observers.append(on_hand)

    def point(table, decisions):
        recent = [h['reward_bb'] for h in hands[-500:]]
        curve.append({'decisions': decisions, 'hands': len(hands),
                      'rolling_bb100': float(np.mean(recent) * 100) if recent else None})

    try:
        if kind == 'ppo':
            session = train_sb3_session(registry, selections, agents=agents, training_seats=(0,),
                total_timesteps=config.ppo_steps, seed=seed, record=False, on_session=attach,
                on_progress=lambda done, total, stats: curve.append({
                    'decisions': done, 'hands': len(hands),
                    'rolling_bb100': float(np.mean([h['reward_bb'] for h in hands[-500:]]) * 100) if hands else None})
                    if not curve or done - curve[-1]['decisions'] >= 512 else None,
                perform_validation=False, normalize_rewards=True, independent_hands=True)
            decisions = session.sb3_timesteps
            updates = learner.training_stats
        else:
            target = config.dqn_hands if kind == 'dqn' else config.mcts_hands
            session = TableSession(registry, selections, agents=agents, mode='train', training_seats=(0,),
                                   seed=seed, record=False, continuous=False)
            attach(session)
            while session.stats.hands < target:
                session.run_batch(target, max_actions=1024)
                if not curve or len(hands) - curve[-1]['hands'] >= 100:
                    point(session, session.stats.actions[0])
            decisions = session.stats.actions[0]
            updates = {'learn_steps': learner.learn_steps, 'epsilon': learner.epsilon} if kind == 'dqn' else {
                'policy_weight_l2': float(np.linalg.norm(learner.policy.weights))}
        point(session, decisions)
    finally:
        if session is not None:
            session.close()
    write_csv(folder / 'hands.csv', hands)
    write_csv(folder / 'curve.csv', curve)
    result = {'seed': seed, 'decisions': decisions, 'seconds': time.monotonic() - started,
              'training': reward_summary([h['reward_bb'] for h in hands]), 'updates': updates,
              'curve': curve, 'opponents': [r if r == 'random' else r['sha256'] for r in rivals]}
    write_json(folder / 'training.json', result)
    return result


def run_benchmark(config, root, *, resume=False):
    """Six curricula, twelve opponent-specific branches, then a locked test suite."""
    config.validate()
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    previous = None
    if (root / 'benchmark.json').exists() and resume:
        previous = json.loads((root / 'benchmark.json').read_text(encoding='utf-8'))
        old_config = previous['config'].copy()
        old_config.setdefault('evaluation_workers', config.evaluation_workers)
        old_config['evaluation_workers'] = config.evaluation_workers  # scheduling does not change the experiment
        old_config.setdefault('confirmation_hands', config.confirmation_hands)
        old_config.setdefault('second_confirmation_hands', config.second_confirmation_hands)
        if old_config != json.loads(json.dumps(asdict(config))):
            raise ValueError('Une reprise doit conserver exactement la configuration de l’expérience.')
    elif (root / 'benchmark.json').exists():
        raise ValueError('Utilisez un nouveau dossier pour conserver les résultats existants.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from poker_rl.benchmark_fast_inference import install_fast_features
    install_fast_features()
    registry = AgentRegistry()
    state = {'status': 'running', 'config': asdict(config), 'runtime': {
        'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__},
        'protocol': {'independent_hands': True, 'stack_bb': 50, 'policy_mode': 'ppo_stochastic_dqn_greedy_mcts_search',
            'gate': 'mean >= threshold and familywise one-sided normal lower bound > 0',
            'gate_familywise_alpha': .05, 'final_ci': 'pointwise normal 95%; not simultaneous',
            'specialists': 'fine-tune copies of completed curriculum; opponents are frozen curricula',
            'selfplay': 'independent frozen copy per opponent; refresh between blocks',
            'test': 'no updates, no selection on final data; shared deals and position rotation',
            'cross_format': 'same weights tested at 2 and 4 players; neural observations padded to 5 slots',
            'environment_rng': 'TableTrainingEnv initialized from session.seed, including resumed PPO',
            'fixed_features': 'CPU batch-one NumPy equivalent; bitwise verified against torch; training gradients use torch',
            'limitations': 'one training seed; no between-seed uncertainty or GTO claim'},
        'models': [], 'tests': [], 'active': None}
    if previous:
        # Completed checkpoints are immutable. A partially trained model is
        # rerun from its fixed initialization (DQN does not serialize replay/RNG).
        finished_curricula = {r['id'] for r in previous['models']
                              if r['regime'] == 'curriculum' and r['status'] in ('completed', 'random_gate_not_reached')}
        state['models'] = [r for r in previous['models'] if r['status'] in ('completed', 'random_gate_not_reached')
                           and (r['regime'] != 'random_only' or f"{r['kind']}_{r['players']}p_curriculum" in finished_curricula)]
        state['tests'] = previous['tests']

    def publish(message):
        state['active'] = message
        write_json(root / 'benchmark.json', state)
        print(message, flush=True)

    try:
        curricula = {}
        # Reserve two independent, increasingly precise confirmations per model.
        critical = NormalDist().inv_cdf(1 - .05 / (len(ALGORITHMS) * len(config.players) * (config.max_random_blocks + 2)))
        state['protocol']['gate_critical_value'] = critical
        state['protocol']['gate_max_checks_per_model'] = config.max_random_blocks + 2
        for completed in state['models']:
            for block in completed.get('blocks', []):
                gate = block.get('gate')
                if gate is not None:
                    margin = critical * gate['standard_error_bb'] * 100
                    gate.update(ci_low=gate['bb100'] - margin, ci_high=gate['bb100'] + margin)
            for key in ('confirmation_gate', 'second_confirmation_gate'):
                gate = completed.get(key)
                if gate:
                    margin = critical * gate['standard_error_bb'] * 100
                    gate.update(ci_low=gate['bb100'] - margin, ci_high=gate['bb100'] + margin)
            if completed.get('gate_passed'):
                successful = [b['gate'] for b in completed['blocks'] if b.get('gate') and passes_gate(b['gate'], config.gate_bb100)]
                if not successful and not completed.get('confirmation_gate'):
                    raise ValueError('Le seuil antérieur ne satisfait plus la correction statistique ; nouvelle expérience nécessaire.')
        for players in config.players:
            for ai, kind in enumerate(ALGORITHMS):
                model_id = f'{kind}_{players}p_curriculum'
                completed = next((r for r in state['models'] if r['id'] == model_id), None)
                if completed:
                    curricula[(players, kind)] = completed['checkpoint']
                    publish(f'{model_id}: checkpoint terminé réutilisé')
                    continue
                path = root / model_id
                options = agent_options(kind, fixed_seed(config.seed, 0, players, ai), config)
                learner = registry.load(kind, options)
                state['tests'] = []  # Any new training invalidates prior suite results.
                if kind in ('ppo', 'dqn'):
                    learner = UniversalAgent(learner)
                row = {'id': model_id, 'kind': kind, 'players': players, 'regime': 'curriculum',
                       'blocks': [], 'gate_passed': False, 'status': 'training'}
                state['models'].append(row)
                for block in range(config.max_random_blocks):
                    publish(f'{model_id}: random {block + 1}/{config.max_random_blocks}')
                    result = train_block(registry, learner, kind, ['random'] * (players - 1), config,
                        seed=fixed_seed(config.seed, 1, players, ai, block), folder=path / f'random_{block}')
                    result['phase'] = 'random'
                    row['blocks'].append(result)
                    spec = save_policy(learner, kind, path / f'random_{block}.checkpoint', options)
                    publish(f'{model_id}: contrôle random ({config.gate_hands} mains)')
                    gate = evaluate_lineup(registry, learner, ['random'] * (players - 1), config,
                        seed=fixed_seed(config.seed, 2, players, ai, block), hands=config.gate_hands,
                        critical=critical, output=path / f'gate_{block}.csv')
                    result['gate'] = gate
                    publish(f"{model_id}: random {gate['bb100']:.1f} bb/100; borne corrigée {gate['ci_low']:.1f}")
                    if passes_gate(gate, config.gate_bb100):
                        row['gate_passed'] = True
                        row['random_checkpoint'] = spec
                        break
                if row['gate_passed']:
                    # Keep the pre-self-play policy as a separate test row: this
                    # measures transfer gained/lost specifically through self-play.
                    state['models'].append({'id': f'{kind}_{players}p_random_only', 'kind': kind,
                        'players': players, 'regime': 'random_only', 'blocks': [],
                        'checkpoint': spec, 'status': 'completed'})
                    for block in range(config.selfplay_blocks):
                        publish(f'{model_id}: self-play {block + 1}/{config.selfplay_blocks}')
                        result = train_block(registry, learner, kind, [spec] * (players - 1), config,
                            seed=fixed_seed(config.seed, 1, players, ai, 100 + block), folder=path / f'selfplay_{block}')
                        result['phase'] = 'selfplay'
                        row['blocks'].append(result)
                        spec = save_policy(learner, kind, path / f'selfplay_{block}.checkpoint', options)
                row.update(checkpoint=spec, status='completed' if row['gate_passed'] else 'random_gate_not_reached')
                curricula[(players, kind)] = spec
                publish(f'{model_id}: {row["status"]}')
                del learner

        for players in config.players:
            for ai, kind in enumerate(ALGORITHMS):
                row = next(r for r in state['models'] if r['id'] == f'{kind}_{players}p_curriculum')
                if row['gate_passed'] or row.get('second_confirmation_gate'):
                    continue
                path = root / row['id']
                learner = load_policy(registry, row['checkpoint'], config, fixed_seed(config.seed, 0, players, ai))
                if kind == 'dqn':
                    from agents.dqn import ReplayBuffer
                    learner.agent.buffer = ReplayBuffer(50000, learner.input_dim)
                for key, purpose, budget in (('confirmation_gate', 4, config.confirmation_hands),
                                             ('second_confirmation_gate', 5, config.second_confirmation_hands)):
                    if row.get(key):
                        gate = row[key]
                    else:
                        publish(f"{row['id']}: {key}, {budget} mains indépendantes")
                        gate = evaluate_lineup(registry, learner, ['random'] * (players - 1), config,
                            seed=fixed_seed(config.seed, purpose, players, ai), hands=budget,
                            critical=critical, output=path / f'{key}.csv')
                        row[key] = gate
                        publish(f"{row['id']}: {key} {gate['bb100']:.1f}, borne {gate['ci_low']:.1f} bb/100")
                    if passes_gate(gate, config.gate_bb100):
                        break
                if passes_gate(gate, config.gate_bb100):
                    state['tests'] = []
                    row['status'] = 'training'
                    row['gate_passed'] = True
                    row['random_checkpoint'] = spec = row['checkpoint']
                    state['models'].append({'id': f'{kind}_{players}p_random_only', 'kind': kind,
                        'players': players, 'regime': 'random_only', 'blocks': [],
                        'checkpoint': spec, 'status': 'completed'})
                    for block in range(config.selfplay_blocks):
                        publish(f"{row['id']}: self-play confirmé {block + 1}/{config.selfplay_blocks}")
                        result = train_block(registry, learner, kind, [spec] * (players - 1), config,
                            seed=fixed_seed(config.seed, 1, players, ai, 100 + block), folder=path / f'selfplay_{block}')
                        result['phase'] = 'selfplay'
                        row['blocks'].append(result)
                        spec = save_policy(learner, kind, path / f'selfplay_{block}.checkpoint', agent_options(kind, config.seed, config))
                    row.update(checkpoint=spec, status='completed')
                    curricula[(players, kind)] = spec
                publish(f"{row['id']}: confirmation {gate['bb100']:.1f} bb/100, borne {gate['ci_low']:.1f}; {row['status']}")
                del learner

        # Lock every reference before any opponent-specific branch is trained.
        state['references'] = {f'{p}p_{k}': v for (p, k), v in curricula.items()}
        for players in config.players:
            for ai, kind in enumerate(ALGORITHMS):
                for oi, opponent in enumerate(ALGORITHMS):
                    if opponent == kind:
                        continue
                    model_id = f'{kind}_{players}p_vs_{opponent}'
                    if any(r['id'] == model_id for r in state['models']):
                        publish(f'{model_id}: checkpoint terminé réutilisé')
                        continue
                    path = root / model_id
                    state['tests'] = []
                    learner = load_policy(registry, curricula[(players, kind)], config,
                                          fixed_seed(config.seed, 0, players, ai, oi + 1))
                    # DQN resumes with a fresh replay buffer, keeping optimizer and weights.
                    if kind == 'dqn':
                        from agents.dqn import ReplayBuffer
                        learner.agent.buffer = ReplayBuffer(50000, learner.input_dim)
                    row = {'id': model_id, 'kind': kind, 'players': players, 'regime': f'vs_{opponent}',
                           'parent': curricula[(players, kind)], 'blocks': [], 'status': 'training'}
                    state['models'].append(row)
                    for block in range(config.specialist_blocks):
                        publish(f'{model_id}: entraînement {block + 1}/{config.specialist_blocks}')
                        result = train_block(registry, learner, kind, [curricula[(players, opponent)]] * (players - 1), config,
                            seed=fixed_seed(config.seed, 1, players, ai, 200 + oi * 10 + block),
                            folder=path / f'block_{block}')
                        result['phase'] = f'vs_{opponent}'
                        row['blocks'].append(result)
                    row.update(checkpoint=save_policy(learner, kind, path / 'final.checkpoint',
                                                      agent_options(kind, config.seed, config)), status='completed')
                    publish(f'{model_id}: terminé')
                    del learner

        state['phase'] = 'final_test'
        from poker_rl.benchmark_workers import evaluate_suite
        evaluate_suite(config, root, state, curricula, publish)
        state['status'] = 'completed'
        state['phase'] = 'completed'
        publish('Entraînements et évaluations terminés.')
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        publish(f'Arrêt : {error}')
        raise
    finally:
        write_csv(root / 'results.csv', [{k: v for k, v in row.items() if k not in ('actions', 'position_counts')}
                                        for row in state['tests']])
    return state
