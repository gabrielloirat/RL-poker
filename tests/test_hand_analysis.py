import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from gym_env.enums import Stage
from poker_rl.evaluation import evaluate_agents, validate_model
from poker_rl.hand_analysis import analyze_hands, describe_hand, filter_hands, compact_candidate
from poker_rl.plugins import AgentRegistry
from poker_rl.table_session import TableSession
from test_records_arena import SAVING_AGENT


class CheckCall:
    def act(self, observation, action_mask, *, training=False):
        return 1 if action_mask[1] else 2


class HandAnalysisTests(unittest.TestCase):
    def describe(self, cards, board, *, reward=-2, stage=Stage.SHOWDOWN, folded=False,
                 terminated=True, actions=None):
        table = SimpleNamespace(stage=stage, dealer_pos=1, table_cards=board, big_blind=2,
            players=[SimpleNamespace(cards=cards), SimpleNamespace(cards=['2D', '3D'])],
            player_cycle=SimpleNamespace(folder=[folded, False]))
        game = SimpleNamespace(table=table, terminated=terminated, truncated=not terminated, hand_start_stacks=(100, 100),
                               rewards=lambda: [reward, -reward])
        return describe_hand(game, 0, hand=1, seed=42, actions=actions or [])

    def test_actual_showdown_combinations_are_classified(self):
        cases = [(['AS', 'KD'], ['2C', '4D', '6S', '8H', 'TC'], 'HighCard'),
                 (['AS', 'AD'], ['2C', '4D', '6S', '8H', 'TC'], 'Pair'),
                 (['AS', 'AD'], ['2C', '2D', '6S', '8H', 'TC'], 'TwoPair'),
                 (['AS', 'AD'], ['AC', '4D', '6S', '8H', 'TC'], 'ThreeOfAKind'),
                 (['AS', 'KD'], ['QC', 'JD', 'TS', '8H', '2C'], 'Straight'),
                 (['AS', 'KS'], ['2S', '4S', '6S', '8H', 'TC'], 'Flush'),
                 (['AS', 'AD'], ['AC', '4D', '4S', '8H', 'TC'], 'FullHouse'),
                 (['AS', 'AD'], ['AC', 'AH', '6S', '8H', 'TC'], 'FourOfAKind'),
                 (['AS', 'KS'], ['QS', 'JS', 'TS', '8H', '2C'], 'StraightFlush')]
        for hole, board, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(self.describe(hole, board)['category'], expected)

    def test_folds_and_wins_without_showdown_are_not_fake_combinations(self):
        hand = self.describe(['AS', 'AD'], ['AC', 'AH', '6S', '8H', 'TC'], folded=True)
        self.assertEqual(hand['category'], 'Fold')
        self.assertFalse(hand['showdown'])
        hand = self.describe(['AS', 'AD'], [], stage=Stage.END_HIDDEN, reward=1)
        self.assertEqual(hand['category'], 'NoShowdown')
        self.assertEqual(hand['outcome'], 'win')

    def test_financial_outcomes_ties_and_truncations_have_correct_denominators(self):
        hands = [self.describe(['AS', 'AD'], ['2C', '4D', '6S', '8H', 'TC'], reward=reward)
                 for reward in (-3., 2., 0., 1e-12)]
        hands.append(self.describe(['AS', 'AD'], [], terminated=False, reward=0))
        analysis = analyze_hands(hands)
        for dimension in analysis['breakdown'].values():
            self.assertEqual(sum(row['hands'] for row in dimension), 4)
            self.assertEqual(sum(row['wins'] for row in dimension), 1)
            self.assertEqual(sum(row['losses'] for row in dimension), 1)
            self.assertEqual(sum(row['neutral'] for row in dimension), 2)
            self.assertAlmostEqual(sum(row['net_bb'] for row in dimension), -1.)
            self.assertAlmostEqual(sum(row['won_bb'] for row in dimension), 2.)
            self.assertEqual(sum(row['lost_bb'] for row in dimension), 3.)

    def test_last_action_is_counted_once_and_filters_select_largest_losses(self):
        hands = []
        for hand, reward in enumerate((-2, -5, 3, 0), start=1):
            record = self.describe(['AS', 'AD'], ['2C', '4D', '6S', '8H', 'TC'], reward=reward,
                actions=[{'seat': 0, 'action': 'CALL', 'stage': 'PREFLOP'},
                         {'seat': 1, 'action': 'CHECK', 'stage': 'FLOP'},
                         {'seat': 0, 'action': 'ALL_IN', 'stage': 'FLOP'}])
            record['hand'] = hand
            hands.append(record)
        analysis = analyze_hands(hands)
        all_in = next(row for row in analysis['breakdown']['action'] if row['category'] == 'ALL_IN')
        call = next(row for row in analysis['breakdown']['action'] if row['category'] == 'CALL')
        self.assertEqual(all_in['hands'], 4)
        self.assertEqual(call['hands'], 0)
        losses = filter_hands(hands, dimension='action', category='ALL_IN', outcome='loss')
        self.assertEqual([hand['net_bb'] for hand in losses], [-5, -2])
        self.assertEqual(len(filter_hands(hands, dimension='street', category='FLOP', outcome='all')), 4)
        self.assertEqual(len(filter_hands(hands, outcome='neutral')), 1)

    def test_validation_collects_private_post_hand_traces_and_preserves_agent(self):
        agent = CheckCall()
        original = copy.deepcopy(agent.__dict__)
        report = evaluate_agents([agent] * 3, (0,), hands_per_opponent=12)
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(agent.__dict__, original)
        for benchmark in report['agents'][0]['benchmarks']:
            analysis = benchmark['hand_analysis']
            self.assertEqual(len(analysis['hands']), 12)
            for dimension in analysis['breakdown'].values():
                self.assertEqual(sum(row['hands'] for row in dimension), benchmark['completed'])
                self.assertAlmostEqual(sum(row['net_bb'] for row in dimension), benchmark['net_bb'])
            for hand in analysis['hands']:
                self.assertEqual(len(hand['hole_cards']), 2)
                self.assertEqual(len(hand['player_cards']), 3)
                for decision in hand['actions']:
                    self.assertIn('board', decision)
                    self.assertGreaterEqual(decision['to_call'], 0)
                    self.assertGreaterEqual(decision['pot'], 0)

    def test_manifests_store_small_summaries_and_trace_reports_remain_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plugins = root / 'agents'
            plugins.mkdir()
            (plugins / 'learner.py').write_text(SAVING_AGENT, encoding='utf-8')
            registry = AgentRegistry(plugins)
            session = TableSession(registry, [('learner', {})] * 2,
                                   mode='train', training_seats=(0,), record=True)
            try:
                session.run_batch(2)
                session.complete_training(validation_hands=12)
                manifest = json.loads((session.last_archive / 'manifest.json').read_text(encoding='utf-8'))
                compact = manifest['agents'][0]['validation']['benchmarks'][0]['hand_analysis']
                self.assertNotIn('hands', compact)
                self.assertEqual(compact['hands_file'], 'validation.json')
                full = json.loads((session.last_archive / compact['hands_file']).read_text(encoding='utf-8'))
                self.assertEqual(len(full['agents'][0]['benchmarks'][0]['hand_analysis']['hands']), 12)
                # Compacting must not strip the in-memory hand history.
                candidate = session.validation_report['agents'][0]
                compact_candidate(candidate)
                self.assertEqual(len(candidate['benchmarks'][0]['hand_analysis']['hands']), 12)
                registry = AgentRegistry(plugins)
                name = next(iter(registry.saved_models))
                validation = validate_model(registry, name, hands_per_opponent=8)
                self.assertEqual(validation['benchmarks'][0]['hand_analysis']['hands_file'], 'validation_seat_0.json')
            finally:
                session.close()

    def test_chart_clicks_filter_winning_losing_and_neutral_hands_and_back_preserves_context(self):
        from poker_rl.gui import PokerApp
        app = PokerApp.__new__(PokerApp)
        app.metrics_open, app.metrics_tab = True, 'hands'
        app.analysis_parent = 'chart'
        for position, outcome in (((100, 322), 'all'), ((400, 320), 'win'),
                                  ((400, 332), 'loss'), ((950, 320), 'neutral')):
            app.mouse_position = position
            app.select_analysis_bar('Pair', 317)
            self.assertEqual(app.analysis_category, 'Pair')
            self.assertEqual(app.analysis_outcome, outcome)
            self.assertEqual(app.analysis_view, 'list')
            self.assertEqual(app.analysis_page, 0)
        app.inspect_analysis_hand({'hand': 1})
        self.assertEqual(app.analysis_view, 'detail')
        app.back()
        self.assertEqual(app.analysis_view, 'list')
        app.back()
        self.assertEqual(app.analysis_view, 'chart')
        app.back()
        self.assertEqual(app.metrics_tab, 'summary')
        self.assertTrue(app.metrics_open)


if __name__ == '__main__':
    unittest.main()
