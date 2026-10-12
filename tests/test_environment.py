import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gym_env.env import HoldemTable
from gym_env.enums import Action, Stage


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.equity_patch = patch("gym_env.env.MONTEACRLO_RUNS", 10)
        self.equity_patch.start()
        self.addCleanup(self.equity_patch.stop)
        self.env = HoldemTable()
        self.addCleanup(self.env.close)
        for name in ("Player 1", "Player 2"):
            self.env.add_player(SimpleNamespace(name=name))
        self.env.reset()

    def assert_chips_conserved(self):
        env = self.env
        pot = 0 if env.done else env.community_pot + env.current_round_pot
        self.assertAlmostEqual(sum(player.stack for player in env.players) + pot, 200)
        self.assertTrue(all(player.stack >= 0 for player in env.players))

    def test_call_and_check_reach_flop_with_one_pot(self):
        self.env.step(Action.CALL.value)
        self.env.step(Action.CHECK.value)
        self.assertEqual(self.env.stage, Stage.FLOP)
        self.assertEqual(len(self.env.table_cards), 3)
        self.assertEqual(self.env.community_pot, 4)
        self.assertEqual(self.env.current_round_pot, 0)
        self.assert_chips_conserved()

    def test_check_call_completes_several_hands(self):
        for _ in range(32):
            action = Action.CHECK if Action.CHECK in self.env.legal_moves else Action.CALL
            self.env.step(action.value)
            self.assert_chips_conserved()
        self.assertGreater(len(self.env.funds_history), 3)

    def test_fold_starts_another_hand(self):
        self.env.step(Action.FOLD.value)
        self.assertEqual(self.env.stage, Stage.PREFLOP)
        self.assertEqual(len(self.env.funds_history), 2)
        self.assert_chips_conserved()

    def test_all_in_finishes_and_reset_allows_play_again(self):

        self.env.current_player.cards = ["AS", "AH"]
        self.env.players[1 - self.env.current_player.seat].cards = ["KS", "KH"]
        board = iter(["2C", "3D", "7H", "8S", "9C"])
        with patch.object(self.env, "_distribute_cards_to_table", side_effect=lambda n: self.env.table_cards.extend(next(board) for _ in range(n))):
            self.env.step(Action.ALL_IN.value)
            _, _, done, info = self.env.step(Action.CALL.value)
        self.assertTrue(done)
        self.assertEqual(info["legal_moves"], [])
        self.assert_chips_conserved()
        with self.assertRaises(RuntimeError):
            self.env.step(Action.CHECK.value)
        self.env.reset()
        self.assertFalse(self.env.done)
        self.assertTrue(self.env.legal_moves)
        self.assert_chips_conserved()

    def test_raise_tracks_total_bet_and_resets_next_hand(self):
        raiser = self.env.current_player
        self.env.step(Action.RAISE_3BB.value)
        self.assertEqual(self.env.min_call, 6)
        self.assertNotIn(Action.RAISE_3BB, self.env.legal_moves)
        self.env.step(Action.FOLD.value)
        self.assertEqual(raiser.num_raises_in_street[Stage.PREFLOP], 0)
        self.assertIn(Action.RAISE_3BB, self.env.legal_moves)
        self.assert_chips_conserved()

    def test_illegal_action_keeps_state(self):
        before = [player.stack for player in self.env.players]
        _, reward, done, _ = self.env.step(Action.CHECK.value)
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.assertEqual([player.stack for player in self.env.players], before)

    def test_reset_requires_two_players(self):
        empty = HoldemTable()
        self.addCleanup(empty.close)
        with self.assertRaises(ValueError):
            empty.reset()


if __name__ == "__main__":
    unittest.main()
