import copy
"""Reproducible multi-seed experiments, separate from the bot catalog."""
import json
import hashlib
from pathlib import Path

import numpy as np

from poker_rl.evaluation import attach_validation
from poker_rl.records import SessionRecorder, save_model_version, write_json

CAMPAIGNS_DIR = Path('runs/campaigns')


class CampaignConfig:
    def __init__(self,
                 seeds=(101, 202, 303), total_timesteps=100000, checkpoint_interval=50000,
                 selection_hands=500, final_hands=1000, stack_depths=(20, 50, 100),
                 num_players=2, selection_seed=None, final_seed=None,
                 deterministic=True, n_steps=1024, learning_rates=(0.0003, 0.0001),
                 compare_modes=True, league_snapshot_interval=5000, diversity_threshold=0.02,
                 include_seed202=True, curriculum_probability=0.2, curriculum_warmup=1024,
                 postflop_target=0.35, champion_hands=5000, champion_seed=None):  # Enregistre les données
        self.seeds = seeds
        self.total_timesteps = total_timesteps
        self.checkpoint_interval = checkpoint_interval
        self.selection_hands = selection_hands
        self.final_hands = final_hands
        self.stack_depths = stack_depths
        self.num_players = num_players
        self.selection_seed = selection_seed
        self.final_seed = final_seed
        self.deterministic = deterministic
        self.n_steps = n_steps
        self.learning_rates = learning_rates
        self.compare_modes = compare_modes
        self.league_snapshot_interval = league_snapshot_interval
        self.diversity_threshold = diversity_threshold
        self.include_seed202 = include_seed202
        self.curriculum_probability = curriculum_probability
        self.curriculum_warmup = curriculum_warmup
        self.postflop_target = postflop_target
        self.champion_hands = champion_hands
        self.champion_seed = champion_seed

    def to_dict(self):  # Prépare les données à sauvegarder
        return {
            "seeds": self.seeds,
            "total_timesteps": self.total_timesteps,
            "checkpoint_interval": self.checkpoint_interval,
            "selection_hands": self.selection_hands,
            "final_hands": self.final_hands,
            "stack_depths": self.stack_depths,
            "num_players": self.num_players,
            "selection_seed": self.selection_seed,
            "final_seed": self.final_seed,
            "deterministic": self.deterministic,
            "n_steps": self.n_steps,
            "learning_rates": self.learning_rates,
            "compare_modes": self.compare_modes,
            "league_snapshot_interval": self.league_snapshot_interval,
            "diversity_threshold": self.diversity_threshold,
            "include_seed202": self.include_seed202,
            "curriculum_probability": self.curriculum_probability,
            "curriculum_warmup": self.curriculum_warmup,
            "postflop_target": self.postflop_target,
            "champion_hands": self.champion_hands,
            "champion_seed": self.champion_seed,
        }

    def copy(self):  # Copie les données avant modification
        return copy.copy(self)

    def __eq__(self, other):  # Compare les valeurs
        if not isinstance(other, CampaignConfig):
            return False
        return self.__dict__ == other.__dict__

    def validate(self):
        if not 3 <= len(self.seeds) <= 5 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('Choisissez 3 à 5 seeds distinctes.')
        if any(type(s) is not int or s < 0 or s >= 2**32 for s in self.seeds):
            raise ValueError('Les seeds doivent être des entiers entre 0 et 2^32-1.')
        if (self.selection_seed is not None and self.selection_seed == self.final_seed) or any(s in (self.selection_seed, self.final_seed) for s in self.seeds):
            raise ValueError('Seeds de train, de sélection et de test final distinctes requises.')
        evaluation_seeds = [s for s in (self.selection_seed, self.final_seed, self.champion_seed) if s is not None]
        if any(type(s) is not int or not 0 <= s < 2**63 for s in evaluation_seeds):
            raise ValueError('Seeds d’évaluation entières entre 0 et 2^63-1 requises.')
        if len(set(evaluation_seeds)) != len(evaluation_seeds) or any(s in self.seeds for s in evaluation_seeds):
            raise ValueError('Les trois seeds d’évaluation doivent être distinctes et séparées du train.')
        if not self.stack_depths or len(set(self.stack_depths)) != len(self.stack_depths) or any(
                type(d) is not int or d <= 1 for d in self.stack_depths):
            raise ValueError('Profondeurs entières, distinctes et supérieures à 1 BB requises.')
        if type(self.num_players) is not int or not 2 <= self.num_players <= 5:
            raise ValueError('2 à 5 joueurs requis.')
        if any(type(v) is not int or v < 2 for v in (self.total_timesteps, self.checkpoint_interval, self.n_steps)):
            raise ValueError('Budget, intervalle et taille de rollout >= 2 requis.')
        if any(type(n) is not int or n < 1 for n in (self.selection_hands, self.final_hands)):
            raise ValueError('Au moins une main par scénario est requise.')
        if not 1 <= len(self.learning_rates) <= 3 or len(set(self.learning_rates)) != len(self.learning_rates) or any(
                not np.isfinite(rate) or rate <= 0 for rate in self.learning_rates):
            raise ValueError('Choisissez un à trois taux d’apprentissage positifs, finis et distincts.')
        if not 0 <= self.curriculum_probability <= 1 or not 0 <= self.postflop_target <= 1 or self.curriculum_warmup < 0:
            raise ValueError('Paramètres de curriculum invalides.')
        if self.league_snapshot_interval < 2 or self.diversity_threshold < 0 or self.champion_hands < 1:
            raise ValueError('Paramètres de ligue ou de test du champion invalides.')


def scenario_scores(report):
    if not report or report.get('status') != 'completed' or not report.get('agents'):
        return {}
    candidate = report['agents'][0]
    if candidate.get('status') != 'completed':
        return {}
    return {b.get('scenario', b['opponent']): b['bb_per_100'] for b in candidate['benchmarks']
            if b.get('bb_per_100') is not None}


def distribution(values):
    return {'n': len(values), 'mean': float(np.mean(values)) if values else None,
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def aggregate_seeds(runs, expected):
    complete = [r for r in runs if r.get('status') == 'completed']
    scores = [scenario_scores(r.get('final_validation')) for r in complete]
    # Only compare identical scenarios; a missing benchmark never becomes zero.
    mismatch = bool(scores and any(set(s) != set(scores[0]) for s in scores))
    scenarios = sorted(scores[0]) if scores and not mismatch else []
    means = [float(np.mean([s[k] for k in scenarios])) for s in scores] if scenarios else []
    return {'unit': 'BB/100', 'weighting': 'equal_scenarios_then_equal_training_seeds',
            'dispersion': 'sample_standard_deviation_between_independent_training_seeds',
            'expected_seeds': expected, 'completed_seeds': len(complete),
            'scenario_mismatch': mismatch,
            'overall': distribution(means),
            'scenarios': {k: distribution([s[k] for s in scores]) for k in scenarios}}


def campaign_paths(root=CAMPAIGNS_DIR):
    return sorted(Path(root).glob('*/campaign.json'), reverse=True)


def read_campaign(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run_campaign(registry, config=None, *, root=CAMPAIGNS_DIR, stop_event=None, on_update=None,
                 agent_kwargs=None, use_catalog_anchors=True):
    from poker_rl.campaign_runner import run_campaign as execute
    return execute(registry, config, root=root, stop_event=stop_event, on_update=on_update,
                   agent_kwargs=agent_kwargs, use_catalog_anchors=use_catalog_anchors)


def save_campaign_version(registry, campaign_path, seed, *, trial_id=None):
    """Explicit export only. Experiment checkpoints never become current bots."""
    path = Path(campaign_path)
    manifest = read_campaign(path)
    if manifest.get('status') == 'running':
        raise ValueError('Attendez la fin ou l’arrêt de la campagne avant d’enregistrer une version.')
    trial_id = trial_id or manifest.get('selected_trial')
    run = next(r for r in manifest['runs'] if r['seed'] == seed and
               (trial_id is None or r.get('trial_id') == trial_id))
    if run.get('saved_version'):
        return run['saved_version']
    if run['status'] != 'completed':
        raise ValueError('Seule une seed avec test final terminé peut être enregistrée.')
    agent = registry.load('ppo', {'autoload': False})
    checkpoint = Path(run['path']) / 'selected.checkpoint'
    if run.get('selected_sha256') and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != run['selected_sha256']:
        raise ValueError('Checkpoint sélectionné modifié : sa validation ne correspond plus à ses poids.')
    agent.load_checkpoint(checkpoint)
    config = manifest['config']
    recorder = SessionRecorder([('ppo', {}), *[('mixed', {}) for _ in range(config['num_players'] - 1)]],
        kind='experiment', training_seats=(0,), seed=seed, game_kwargs={'initial_stacks': 100},
        root=registry.records_dir, registry=registry)
    recorder.manifest.update(campaign=manifest['id'], training_history_recorded=False,
                            training_protocol=manifest['protocol'])
    report = read_campaign(Path(run['path']) / 'final_validation.json')
    attach_validation(recorder, report)
    if run.get('champion_validation'):
        champion = read_campaign(Path(run['path']) / 'champion_validation.json')
        write_json(recorder.path / 'champion_validation.json', champion)
        from poker_rl.hand_analysis import compact_candidate
        recorder.manifest['agents'][0]['champion_validation'] = {
            **compact_candidate(champion['agents'][0], hands_file='champion_validation.json'),
            'protocol': champion['protocol'], 'created_at': champion['created_at']}
    recorder.manifest['agents'][0]['promotion_review'] = run.get('promotion_review')
    recorder.finish({'completed': sum(r['hands'] for r in run.get('training_by_depth', {}).values()),
                     'training_history_recorded': False, 'sb3_timesteps': run['steps']},
                    agents=[agent, *[None] * (config['num_players'] - 1)])
    model_id = recorder.manifest['agents'][0]['model_id']
    version = save_model_version(registry.records_dir, model_id, label=f"Campagne {manifest['id']} · seed {seed}")
    run['saved_version'] = version
    write_json(path, manifest)
    return version
