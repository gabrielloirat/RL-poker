import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest

import numpy as np

from poker_rl.campaigns import CampaignConfig, aggregate_seeds, run_campaign, save_campaign_version
from poker_rl.evaluation import evaluate_depths, ReferenceAgent
from poker_rl.game import PokerGame
from poker_rl.hand_analysis import decision_snapshot, describe_hand, analyze_hands, filter_hands
from poker_rl.league import League, LeagueOpponent, load_frozen
from poker_rl.plugins import AgentRegistry
from poker_rl.records import write_json
from poker_rl.sb3_training import TableTrainingEnv
from poker_rl.strategy_metrics import starting_hand, matrix_key, analyze_strategy, position_name
from poker_rl.table_session import TableSession
from gym_env.enums import Stage


class StrategyMetricsTests(unittest.TestCase):
    def hand(self, actions, **kwargs):
        return {'terminated': True, 'bot_seat': 0, 'dealer': 0, 'player_cards': [['AH', 'KD'], ['2C', '3C']],
                'hole_cards': ['AH', 'KD'], 'actions': actions, 'net_bb': 3., **kwargs}

    def action(self, action='CHECK', **kwargs):
        return {'seat': 0, 'stage': 'PREFLOP', 'action': action, 'pot': 4., 'stack': 20., 'to_call': 0., **kwargs}

    def test_matrix_has_169_unique_cells_and_card_order(self):
        self.assertEqual(len({matrix_key(r, c) for r in range(13) for c in range(13)}), 169)
        self.assertEqual(starting_hand(['AH', 'KH']), 'AKs')
        self.assertEqual(starting_hand(['KC', 'AD']), 'AKo')
        self.assertEqual(starting_hand(['AH', 'AD']), 'AA')
        self.assertEqual(matrix_key(0, 1), 'AKs')
        self.assertEqual(matrix_key(1, 0), 'AKo')
        legacy = self.hand([], hand=1, outcome='win')
        self.assertEqual(filter_hands([legacy], dimension='starting_hand', category='AKo', outcome='all'), [legacy])

    def test_vpip_walks_checks_calls_and_all_in_call(self):
        hands = [self.hand([]), self.hand([self.action()]),
                 self.hand([self.action('CALL', to_call=2.)]),
                 self.hand([self.action('ALL_IN', to_call=20., stack=20.)]),
                 self.hand([self.action('RAISE_POT')])]
        stats = analyze_strategy(hands)['overall']
        self.assertEqual(stats['vpip']['denominator'], 4)
        self.assertEqual(stats['vpip']['numerator'], 3)
        self.assertEqual(stats['pfr']['numerator'], 1)
        self.assertEqual(stats['hands'], 5)

    def test_3bet_and_large_bet_denominators_and_truncation(self):
        decision = self.action('RAISE_POT', preflop_raises=1, can_raise=True, to_call=2)
        hands = [self.hand([decision]), self.hand([self.action('FOLD', to_call=2)]),
                 self.hand([self.action('FOLD', to_call=1.99)]),
                 self.hand([decision], terminated=False, net_bb=1000)]
        stats = analyze_strategy(hands)['overall']
        self.assertEqual(stats['three_bet'], {'numerator': 1, 'denominator': 1, 'rate': 1.})
        self.assertEqual(stats['fold_large_bet']['denominator'], 2)
        self.assertEqual(stats['fold_large_bet']['rate'], .5)
        self.assertEqual(stats['net_bb'], 9.)
        empty = analyze_strategy([])['starting_hands']['AA']
        self.assertIsNone(empty['bb_per_100'])
        self.assertIsNone(empty['vpip']['rate'])

    def test_real_trajectory_depth_and_position(self):
        game = PokerGame(initial_stacks=40)
        game.reset(seed=19, options={'dealer_seat': 0})
        actions = []
        while not game.finished:
            seat = game.current_seat
            mask = game.action_masks(seat)
            action = next(a for a in (1, 2, 0) if mask[a])
            actions.append(decision_snapshot(game, seat, action))
            game.step(action)
        hand = describe_hand(game, 0, hand=1, seed=19, actions=actions)
        self.assertEqual(hand['depth_bb'], 20)
        self.assertEqual(hand['position'], 'BTN/SB')
        self.assertEqual(analyze_hands([hand])['strategy']['overall']['hands'], 1)
        game.close()

    def test_short_all_in_call_is_voluntary_but_not_a_raise(self):
        table = SimpleNamespace(stage=Stage.PREFLOP, min_call=10, player_pots=[7],
            big_blind=2, players=[SimpleNamespace(stack=3)], community_pot=40,
            current_round_pot=10, table_cards=[])
        game = SimpleNamespace(table=table, action_masks=lambda seat: np.ones(8, dtype=bool))
        action = decision_snapshot(game, 0, 7)
        self.assertTrue(action['voluntary'])
        self.assertFalse(action['is_raise'])
        self.assertFalse(action['can_raise'])
        self.assertEqual(action['to_call'], 3)


class CampaignTests(unittest.TestCase):
    def test_config_rejects_confounded_or_duplicate_seeds(self):
        for kwargs in ({'seeds': (1, 1, 2)}, {'seeds': (1, 2)}, {'seeds': (1, 2, 3), 'final_seed': 1},
                       {'stack_depths': (20, 20)}, {'total_timesteps': 0}):
            with self.assertRaises(ValueError):
                CampaignConfig(**kwargs).validate()

    def test_dispersion_is_between_seeds_not_pooled_hands(self):
        def run(value):
            return {'status': 'completed', 'final_validation': {'status': 'completed', 'agents': [
                {'status': 'completed', 'benchmarks': [{'opponent': 'random', 'scenario': '20bb:random', 'bb_per_100': value}]}]}}
        stats = aggregate_seeds([run(10), run(20), run(30), {'status': 'interrupted'}], 4)
        self.assertEqual(stats['overall']['mean'], 20)
        self.assertEqual(stats['overall']['std'], 10)
        self.assertEqual(stats['completed_seeds'], 3)
        self.assertEqual(stats['expected_seeds'], 4)
        incomplete = run(30)
        incomplete['final_validation']['agents'][0]['benchmarks'] = []
        mismatch = aggregate_seeds([run(10), incomplete], 2)
        self.assertTrue(mismatch['scenario_mismatch'])
        self.assertIsNone(mismatch['overall']['mean'])

    def test_depth_suite_keeps_scenarios_and_traces_separate(self):
        report = evaluate_depths([ReferenceAgent('check_call', 1), None], (0,),
            stack_depths=(20, 50, 100), hands_per_opponent=4, opponents=('random',))
        self.assertEqual(report['status'], 'completed')
        self.assertEqual([b['depth_bb'] for b in report['agents'][0]['benchmarks']], [20, 50, 100])
        for benchmark in report['agents'][0]['benchmarks']:
            self.assertEqual(benchmark['hand_analysis']['hands'][0]['depth_bb'], benchmark['depth_bb'])

    def test_variable_depth_rewards_and_sampling(self):
        with TemporaryDirectory() as root:
            registry = AgentRegistry('agents', records_dir=root)
            session = TableSession(registry, [('random', {}), ('random', {})], mode='infer',
                defer_callbacks=True, continuous=False)
            env = TableTrainingEnv(session, 0, stack_depths=(20, 50, 100))
            obs, _ = env.reset(seed=7)
            depths = set()
            for _ in range(100):
                depth = session.game.hand_start_stacks[0] / 2
                depths.add(depth)
                scale = env.reward_scale
                obs, reward, terminated, truncated, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
                self.assertEqual(scale, env.reward_scale)
                if terminated:
                    self.assertAlmostEqual(reward, session.game.rewards()[0] * scale)
                if terminated or truncated:
                    env.reset()
            self.assertTrue({20, 50, 100}.issubset(depths))
            session.close()

    def test_small_real_campaign_and_explicit_export(self):
        with TemporaryDirectory() as root:
            registry = AgentRegistry('agents', records_dir=Path(root) / 'sessions')
            config = CampaignConfig(total_timesteps=64, n_steps=32, checkpoint_interval=32,
                selection_hands=2, final_hands=3, learning_rates=(3e-4,))
            result = run_campaign(registry, config, root=Path(root) / 'campaigns',
                agent_kwargs={'n_epochs': 1}, use_catalog_anchors=False)
            self.assertEqual(result['status'], 'completed', result.get('error'))
            self.assertEqual(result['aggregate']['overall']['n'], 3)
            self.assertEqual(len(registry.saved_models), 0)
            own_hashes = []
            for run in result['runs']:
                self.assertEqual(run['steps'], 64)
                self.assertEqual(run['curves'][-1]['steps'], run['steps'])
                self.assertEqual(len(run['final_validation']['agents'][0]['benchmarks']), 12)
                self.assertTrue(all(key.startswith(f"seed_{run['seed']}_") for key in run['league']['history']))
                self.assertNotEqual(config.selection_seed, run['final_validation']['protocol']['seed'])
                self.assertTrue(set(run['training_by_depth']).issubset({'20', '50', '100'}))
                spec = list(run['league']['history'].values())[-1]
                frozen = load_frozen(spec)
                self.assertFalse(any(p.requires_grad for p in frozen.model.policy.parameters()))
                self.assertIn('selected_sha256', run)
                self.assertEqual(run['agent_options']['seed'], run['seed'])
                own_hashes.append(spec['sha256'])
            self.assertEqual(len(set(own_hashes)), 3)
            campaign_path = Path(result['path']) / 'campaign.json'
            version = save_campaign_version(registry, campaign_path, 101)
            self.assertEqual(version, save_campaign_version(registry, campaign_path, 101))
            updated = AgentRegistry('agents', records_dir=registry.records_dir)
            self.assertEqual(list(updated.saved_models), [version])
            second = result['runs'][1]
            Path(second['path'], 'selected.checkpoint').write_bytes(b'changed after validation')
            with self.assertRaises(ValueError):
                save_campaign_version(registry, campaign_path, second['seed'])
            # Hash verification rejects external mutation of a league checkpoint.
            Path(spec['path']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                load_frozen(spec)

    def test_stop_before_launch_persists_partial_protocol(self):
        with TemporaryDirectory() as root:
            stop = Event()
            stop.set()
            result = run_campaign(AgentRegistry('agents', records_dir=Path(root) / 'sessions'),
                root=Path(root) / 'campaigns', stop_event=stop, use_catalog_anchors=False)
            self.assertEqual(result['status'], 'interrupted')
            self.assertEqual(result['aggregate']['overall']['n'], 0)
            self.assertTrue((Path(result['path']) / 'campaign.json').exists())

    def test_export_does_not_race_a_running_campaign(self):
        with TemporaryDirectory() as root:
            path = Path(root) / 'campaign.json'
            write_json(path, {'status': 'running', 'runs': []})
            with self.assertRaisesRegex(ValueError, 'campagne'):
                save_campaign_version(None, path, 101)


if __name__ == '__main__':
    unittest.main()
