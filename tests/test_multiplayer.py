from pathlib import Path
import tempfile
import unittest

import numpy as np

from poker_rl import AgentRegistry, PokerEnv, PokerGame, run_session
from poker_rl.table_session import TableSession


class RecordingAgent:
    def __init__(self):
        self.transitions = []

    def act(self, observation, action_mask, *, training=False):
        return 1 if action_mask[1] else 2

    def observe(self, transition):
        self.transitions.append(transition)


class MultiplayerTests(unittest.TestCase):
    def test_all_table_sizes_observations_turns_and_chip_conservation(self):
        rng = np.random.default_rng(19)
        for count in range(2, 6):
            game = PokerGame(num_players=count)
            self.addCleanup(game.close)
            for seed in range(50):
                game.reset(seed=seed)
                self.assertEqual(game.observation_version, 1 if count == 2 else 2)
                while not game.finished:
                    for seat in range(count):
                        self.assertTrue(game.observation_space.contains(game.observe(seat)))
                    legal = np.flatnonzero(game.action_masks())
                    self.assertGreater(len(legal), 0)
                    game.step(int(rng.choice(legal)))
                    self.assertTrue(all(p.stack >= 0 for p in game.table.players))
                    pot = 0 if game.terminated else game.table.community_pot + game.table.current_round_pot
                    self.assertAlmostEqual(sum(p.stack for p in game.table.players) + pot, 100 * count)
                self.assertTrue(game.terminated)
                self.assertAlmostEqual(sum(game.rewards()), 0)

    def test_positions_before_and_after_flop(self):
        for count in range(3, 6):
            game = PokerGame(num_players=count)
            self.addCleanup(game.close)
            game.reset(seed=2, options={"dealer_seat": 0})
            self.assertEqual(game.current_seat, 3 % count)
            self.assertEqual(game.table.player_pots[1:3], [1, 2])
            seats = []
            while game.table.stage.name == "PREFLOP":
                seats.append(game.current_seat)
                game.step(1 if game.action_masks()[1] else 2)
            self.assertEqual(len(seats), count)
            self.assertEqual(set(seats), set(range(count)))
            self.assertEqual(game.current_seat, 1)

    def test_observations_hide_opponent_cards_and_track_folds(self):
        game = PokerGame(num_players=5)
        self.addCleanup(game.close)
        game.reset(seed=3)
        before = game.observe(0)
        game.table.players[1].cards = ["AS", "KS"]
        np.testing.assert_array_equal(before, game.observe(0))
        folded = game.current_seat
        game.step(0)
        observation = game.observe(0)
        fields = {}
        offset = 0
        for name, size in game.observation_fields:
            fields[name] = observation[offset:offset + size]
            offset += size
        self.assertFalse(fields["active_players"][folded])
        self.assertTrue(all(game.info(s)["num_players"] == 5 for s in range(5)))

    def test_side_pots_have_different_winners_and_refund_uncalled_chips(self):
        game = PokerGame(num_players=3)
        self.addCleanup(game.close)
        game.reset(seed=1)
        table = game.table
        table.table_cards = ["2C", "3D", "7H", "8S", "9C"]
        for player, cards, amount in zip(table.players, (["AS", "AH"], ["KS", "KH"], ["QS", "QH"]), (30, 70, 100)):
            player.cards = cards
            player.stack = 100 - amount
        table.player_max_win = [30, 70, 100]
        table.community_pot = 200
        table.current_round_pot = 0
        table._end_hand()
        self.assertEqual([p.stack for p in table.players], [160, 110, 30])
        self.assertEqual(table.winner_indices, [0, 1])

    def test_tied_multiplayer_pots_are_split_and_folded_player_cannot_win(self):
        game = PokerGame(num_players=4)
        self.addCleanup(game.close)
        game.reset(seed=1)
        table = game.table
        table.table_cards = ["TH", "JH", "QH", "KH", "AH"]
        table.player_max_win = [20, 40, 40, 40]
        table.community_pot = 140
        table.current_round_pot = 0
        table.player_cycle.folder[3] = True
        for player, amount in zip(table.players, table.player_max_win):
            player.stack = 100 - amount
        table._end_hand()
        self.assertAlmostEqual(table.players[0].stack, 80 + 80 / 3)
        self.assertAlmostEqual(table.players[1].stack, 60 + 80 / 3 + 30)
        self.assertAlmostEqual(table.players[2].stack, 60 + 80 / 3 + 30)
        self.assertEqual(table.players[3].stack, 60)
        self.assertAlmostEqual(sum(p.stack for p in table.players), 400)

    def test_training_session_routes_each_selected_multiplayer_seat(self):
        agents = [RecordingAgent() for _ in range(5)]
        results = run_session(agents, episodes=3, training_seats=(0, 4))
        self.assertEqual(len(results[0].rewards_bb), 5)
        for seat in range(5):
            terminal = [t for t in agents[seat].transitions if t.terminated]
            self.assertEqual(len(terminal), 3 if seat in (0, 4) else 0)
            if terminal:
                self.assertEqual([t.reward for t in terminal], [r.rewards_bb[seat] for r in results])

    def test_gui_session_humans_and_automatic_plugin_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            human = Path(__file__).resolve().parents[1] / "agents" / "human.py"
            (root / "human.py").write_text(human.read_text(encoding="utf-8"), encoding="utf-8")
            (root / "auto.py").write_text("class Agent:\n    def act(self, observation, action_mask, *, training=False):\n        return 1 if action_mask[1] else 2\n", encoding="utf-8")
            registry = AgentRegistry(root)
            for count in range(2, 6):
                selections = [("human" if i % 2 == 0 else "auto", {}) for i in range(count)]
                session = TableSession(registry, selections)
                try:
                    self.assertEqual(len({id(a) for a in session.agents}), count)
                    while not session.game.finished:
                        if session.human_turn:
                            session.submit_human_action(1 if session.game.action_masks()[1] else 2)
                        session.play(session.choose_action())
                    self.assertTrue(session.game.terminated)
                    session.next_hand()
                    self.assertEqual(session.game.table.dealer_pos, 1)
                finally:
                    session.close()

    def test_invalid_player_counts_and_fixed_perspective_env(self):
        for count in (1, 6, True):
            with self.assertRaises(ValueError):
                PokerGame(num_players=count)
        with self.assertRaisesRegex(ValueError, "two players"):
            PokerEnv(opponent=RecordingAgent(), num_players=3)


if __name__ == "__main__":
    unittest.main()
