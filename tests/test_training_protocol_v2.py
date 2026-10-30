import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

import numpy as np

from poker_rl.campaigns import CampaignConfig, run_campaign
from poker_rl.campaign_runner import promotion_review
from poker_rl.evaluation import evaluate_depths
from poker_rl.league import League, load_frozen
from poker_rl.plugins import AgentRegistry
from poker_rl.policy_diagnostics import decision_diagnostics, summarize_decisions
from poker_rl.sb3_training import TableTrainingEnv, train_sb3_session
from poker_rl.table_session import TableSession


class ProtocolV2Tests(unittest.TestCase):
    def test_diffuse_policy_can_have_one_greedy_action_without_low_entropy(self):
        probabilities = np.array([.17, 0, .14, .14, 0, .21, .11, .23])
        detail = decision_diagnostics(probabilities, 7, deterministic=True)
        self.assertGreater(detail['normalized_entropy'], .95)
        self.assertEqual(detail['greedy_action'], 7)
        self.assertAlmostEqual(detail['probabilities'][7], .23)
        summary = summarize_decisions([{'stage': 'PREFLOP', 'action': 'ALL_IN', 'policy': detail}] * 10)
        self.assertAlmostEqual(summary['overall']['mean_probabilities']['ALL_IN'], .23)
        self.assertEqual(summary['overall']['action_counts']['ALL_IN'], 10)

    def test_evaluation_seeds_are_explicitly_replayable_and_disjoint(self):
        CampaignConfig(selection_seed=4001, final_seed=5001, champion_seed=6001).validate()
        for kwargs in ({'selection_seed': -1}, {'champion_seed': 101},
                       {'final_seed': 4001, 'champion_seed': 4001}):
            with self.assertRaises(ValueError):
                CampaignConfig(**kwargs).validate()

    def test_curriculum_reaches_late_streets_through_legal_prefixes(self):
        with TemporaryDirectory() as root:
            registry = AgentRegistry('agents', records_dir=root)
            session = TableSession(registry, [('random', {}), ('random', {})], mode='infer',
                                   continuous=False, defer_callbacks=True, record=True)
            env = TableTrainingEnv(session, 0, curriculum_probability=1, curriculum_warmup=0, postflop_target=1)
            seen = set()
            for i in range(30):
                env.reset(seed=42 if i == 0 else None)
                seen.add(session.game.table.stage.name)
                self.assertNotEqual(session.game.table.stage.name, 'PREFLOP')
                while not session.game.finished:
                    env.step(next(a for a in (1, 2, 0) if env.action_masks()[a]))
            coverage = env.coverage()
            self.assertTrue({'FLOP', 'TURN', 'RIVER'}.issubset(seen))
            self.assertEqual(coverage['natural_decisions_by_street'], {})
            self.assertTrue(coverage['prefix_actions_by_street']['PREFLOP'] > 0)
            self.assertAlmostEqual(coverage['postflop_fraction'], 1.)
            self.assertFalse(session.curriculum_prefix)
            history = session.recorder.path / 'history.jsonl'
            session.close()
            events = [json.loads(line) for line in history.read_text(encoding='utf-8').splitlines()]
            origins = {event['training_origin'] for event in events if event['event'] == 'action'}
            self.assertEqual(origins, {'natural', 'curriculum_prefix'})

    def test_natural_training_coverage_counts_exactly_the_collected_decisions(self):
        with TemporaryDirectory() as root:
            registry = AgentRegistry('agents', records_dir=root)
            session = train_sb3_session(registry, [('ppo', {'autoload': False, 'n_steps': 16, 'batch_size': 16,
                'n_epochs': 1}), ('mixed', {})], training_seats=(0,), total_timesteps=32,
                record=False, perform_validation=False)
            coverage = session.training_coverage
            self.assertEqual(sum(coverage['decisions_by_street'].values()), 32)
            self.assertEqual(coverage['curriculum_decisions_by_street'], {})
            self.assertEqual(coverage['prefix_actions_by_street'], {})
            session.close()

    def test_learning_rate_trials_start_from_identical_weights_for_a_shared_seed(self):
        with TemporaryDirectory() as root:
            registry = AgentRegistry('agents', records_dir=root)
            initial = []
            for rate in (3e-4, 1e-4):
                def capture(session, steps):
                    if steps == 0:
                        initial.append({key: tensor.detach().cpu().numpy().copy()
                            for key, tensor in session.agents[0].model.policy.state_dict().items()})
                session = train_sb3_session(registry, [('ppo', {'autoload': False, 'seed': 25, 'learning_rate': rate,
                    'n_steps': 8, 'batch_size': 8, 'n_epochs': 1}), ('mixed', {})],
                    training_seats=(0,), seed=25, total_timesteps=8, on_rollout=capture,
                    record=False, perform_validation=False)
                session.close()
            self.assertEqual(set(initial[0]), set(initial[1]))
            self.assertTrue(all(np.array_equal(initial[0][key], initial[1][key]) for key in initial[0]))

    def test_modes_have_matching_deals_and_probability_diagnostics(self):
        with TemporaryDirectory() as root:
            registry = AgentRegistry('agents', records_dir=root)
            agent = registry.load('ppo', {'autoload': False, 'n_steps': 8, 'batch_size': 8,
                'n_epochs': 1})
            session = train_sb3_session(registry, [('ppo', {}), ('mixed', {})], agents=[agent, registry.load('mixed')],
                training_seats=(0,), total_timesteps=8, record=False, perform_validation=False)
            reports = [evaluate_depths([agent, None], (0,), stack_depths=(20,), hands_per_opponent=6,
                opponents=('check_call',), deterministic=mode, seed=444) for mode in (True, False)]
            a, b = [report['agents'][0]['benchmarks'][0] for report in reports]
            self.assertEqual([h['seed'] for h in a['hand_analysis']['hands']], [h['seed'] for h in b['hand_analysis']['hands']])
            self.assertIsNotNone(a['policy_diagnostics'])
            self.assertTrue(all(d['policy']['deterministic'] for h in a['hand_analysis']['hands'] for d in h['actions'] if d['seat'] == h['bot_seat']))
            self.assertTrue(all(not d['policy']['deterministic'] for h in b['hand_analysis']['hands'] for d in h['actions'] if d['seat'] == h['bot_seat']))
            session.close()

    def test_behaviourally_identical_snapshots_do_not_fill_the_active_league(self):
        with TemporaryDirectory() as root:
            registry = AgentRegistry('agents', records_dir=Path(root) / 'sessions')
            session = train_sb3_session(registry, [('ppo', {'autoload': False, 'n_steps': 8, 'batch_size': 8,
                'n_epochs': 1}), ('mixed', {})], training_seats=(0,),
                total_timesteps=8, record=False, perform_validation=False)
            learner = session.agents[0]
            rng_before = copy.deepcopy(learner.rng.bit_generator.state)
            league = League(Path(root) / 'league', seed=11, agents_dir=registry.directory)
            league.snapshot(learner, 8)
            league.snapshot(learner, 16)
            self.assertEqual(len(league.history), 1)
            self.assertEqual(league.retired[-1]['reason'], 'similar_behaviour')
            self.assertEqual(learner.rng.bit_generator.state, rng_before)
            self.assertTrue(Path(league.retired[-1]['path']).exists())
            frozen = load_frozen(league.retired[-1])
            self.assertFalse(any(p.requires_grad for p in frozen.model.policy.parameters()))
            session.close()

    def test_trial_choice_is_locked_before_final_and_uses_same_budget(self):
        with TemporaryDirectory() as root:
            registry = AgentRegistry('agents', records_dir=Path(root) / 'sessions')
            states = []
            cfg = CampaignConfig(total_timesteps=32, n_steps=16, checkpoint_interval=32, selection_hands=2,
                final_hands=2, champion_hands=2, learning_rates=(3e-4, 1e-4), curriculum_warmup=0)
            result = run_campaign(registry, cfg, root=Path(root) / 'campaigns',
                use_catalog_anchors=False, agent_kwargs={'n_epochs': 1},
                on_update=lambda state: states.append((state['selected_trial'], [r['phase'] for r in state['runs']])))
            self.assertEqual(result['status'], 'completed', result.get('error'))
            self.assertEqual(len(result['runs']), 6)
            self.assertEqual(result['aggregate']['completed_seeds'], 3)
            self.assertEqual(len(registry.saved_models), 0)
            for run in result['runs']:
                self.assertEqual(run['steps'], 32)
                self.assertEqual(set(run['selection'][0]['modes']), {'deterministic', 'stochastic'})
                self.assertEqual(run['agent_options']['learning_rate'], run['learning_rate'])
                selected = run['trial_id'] == result['selected_trial']
                self.assertEqual('final_validation' in run, selected)
                if selected:
                    mode = run['final_validation']['agents'][0]['policy']['deterministic']
                    self.assertEqual(mode, run['selected_mode'] == 'deterministic')
            for trial, phases in states:
                if 'final_test' in phases:
                    self.assertIsNotNone(trial)
            self.assertNotEqual(result['config']['selection_seed'], result['config']['final_seed'])
            self.assertNotEqual(result['champion_seed'], result['config']['final_seed'])

    def test_promotion_review_requires_positive_simultaneous_bounds(self):
        benchmarks = [{'depth_bb': d, 'bb_per_100': 10, 'standard_error_bb': .2, 'completed': 1000} for d in (20, 50, 100)]
        report = {'status': 'completed', 'agents': [{'benchmarks': benchmarks}]}
        review = promotion_review(report, expected_depths=(20, 50, 100))
        self.assertEqual(review['status'], 'inconclusive')
        self.assertFalse(review['automatic_promotion'])
        for b in benchmarks:
            b['bb_per_100'] = 200
        self.assertEqual(promotion_review(report, expected_depths=(20, 50, 100))['status'], 'supported_on_tested_depths')


if __name__ == '__main__':
    unittest.main()
