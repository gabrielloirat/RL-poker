import random
import unittest
from unittest.mock import patch

import numpy as np

from poker_rl import PokerGame
from poker_rl.display_equity import EquitySnapshot, estimate_equities


class DisplayEquityTests(unittest.TestCase):
    def test_snapshot_stays_unchanged_when_the_game_advances(self):
        game = PokerGame(num_players=5)
        self.addCleanup(game.close)
        game.reset(seed=42)
        snapshot = EquitySnapshot.from_game(game)
        folded = game.current_seat
        game.step(0)
        updated = EquitySnapshot.from_game(game)
        self.assertNotEqual(snapshot, updated)
        self.assertTrue(snapshot.eligible[folded])
        self.assertFalse(updated.eligible[folded])

    def test_folded_players_are_zero_and_all_in_players_remain_eligible(self):
        game = PokerGame(num_players=3)
        self.addCleanup(game.close)
        game.reset(seed=42)
        game.step(7)
        all_in = next(i for i, player in enumerate(game.table.players) if player.stack == 0)
        game.step(0)
        snapshot = EquitySnapshot.from_game(game)
        self.assertTrue(snapshot.eligible[all_in])
        with patch("poker_rl.display_equity.get_equity", return_value=0.4) as calculate:
            result = estimate_equities(snapshot)
        self.assertEqual(calculate.call_count, sum(snapshot.eligible))
        self.assertEqual(result[all_in], 0.4)
        for seat, eligible in enumerate(snapshot.eligible):
            self.assertEqual(result[seat], 0.4 if eligible else 0.0)
        self.assertTrue(all(call.args[2] == 2 for call in calculate.call_args_list))

    def test_royal_flush_on_board_splits_equity_at_every_table_size(self):
        for count in range(2, 6):
            hands = (("2C", "3D"), ("4C", "5D"), ("6C", "7D"), ("8C", "9D"), ("TC", "JD"))[:count]
            snapshot = EquitySnapshot(hands, ("TH", "JH", "QH", "KH", "AH"), (True,) * count, False)
            np.testing.assert_allclose(estimate_equities(snapshot, runs=10), [1 / count] * count)
            np.testing.assert_allclose(estimate_equities(EquitySnapshot(hands, snapshot.board, snapshot.eligible, True)),
                                       [1 / count] * count)

    def test_finished_hand_has_known_winner_without_sampling(self):
        snapshot = EquitySnapshot((("AS", "AH"), ("KS", "KH"), ("QS", "QH")),
                                  ("2C", "3D", "7H", "8S", "9C"), (True, True, True), True)
        with patch("poker_rl.display_equity.get_equity", side_effect=AssertionError("sampling at showdown")):
            self.assertEqual(estimate_equities(snapshot), (1.0, 0.0, 0.0))

    def test_last_player_has_full_equity_without_sampling(self):
        snapshot = EquitySnapshot((("AS", "AH"), ("KS", "KH")), (), (False, True), True)
        with patch("poker_rl.display_equity.get_equity", side_effect=AssertionError("sampling with one player")):
            self.assertEqual(estimate_equities(snapshot), (0.0, 1.0))

    def test_simulations_do_not_consume_agent_random_state_or_change_game(self):
        game = PokerGame(num_players=5)
        self.addCleanup(game.close)
        game.reset(seed=42)
        before = [game.observe(seat) for seat in range(5)]
        random_state = random.getstate()
        result = estimate_equities(EquitySnapshot.from_game(game), runs=20)
        self.assertEqual(random.getstate(), random_state)
        self.assertTrue(all(0 <= value <= 1 for value in result))
        for seat in range(5):
            np.testing.assert_array_equal(before[seat], game.observe(seat))
        self.assertEqual(game.action_count, 0)


if __name__ == "__main__":
    unittest.main()
