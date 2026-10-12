from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import numpy as np

from poker_rl import PokerGame, load_agent
from run_agents import main


class HumanAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = load_agent("human", {"name": "Alice"})
        self.game = PokerGame()
        self.addCleanup(self.game.close)
        self.game.reset(seed=42, options={"dealer_seat": 0})

    def test_invalid_and_illegal_inputs_are_retried_without_advancing_game(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["texte", "e", "-1", "9", " C "]) as read, redirect_stdout(output):
            action = self.agent.act(self.game.observe(0), self.game.action_masks(0))
        self.assertEqual(action, 2)
        self.assertEqual(read.call_count, 5)
        self.assertEqual(self.game.action_count, 0)
        self.assertIn("Saisie invalide", output.getvalue())
        self.assertIn("Action indisponible", output.getvalue())

    def test_numeric_choice_and_private_observation_display(self):
        self.game.table.players[0].cards = ["2C", "3D"]
        self.game.table.players[1].cards = ["AS", "KH"]
        output = io.StringIO()
        with patch("builtins.input", return_value="0"), redirect_stdout(output):
            action = self.agent.act(self.game.observe(0), self.game.action_masks(0))
        self.assertEqual(action, 0)
        self.assertIn("Alice — siège 0 — Préflop", output.getvalue())
        self.assertIn("Vos cartes : 2C 3D", output.getvalue())
        self.assertNotIn("AS", output.getvalue())
        self.assertNotIn("KH", output.getvalue())
        self.assertNotIn("1 [e]", output.getvalue())

    def test_quit_and_closed_input_interrupt_session(self):
        for response in ("q", EOFError()):
            with self.subTest(response=response), patch("builtins.input", side_effect=[response]), redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    self.agent.act(self.game.observe(0), self.game.action_masks(0))
        with patch("builtins.input") as read:
            with self.assertRaises(ValueError):
                self.agent.act(self.game.observe(0), np.zeros(8, dtype=bool))
            read.assert_not_called()

    def test_cli_two_humans_complete_hand_with_user_choices(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["c"] + ["e"] * 7), redirect_stdout(output):
            report = main(["infer", "--agent0", "human", "--agent1", "human", "--episodes", "1"])
        self.assertEqual(report["completed"], 1)
        self.assertEqual(report["illegal_actions"], 0)
        self.assertIn("siège 0", output.getvalue())
        self.assertIn("siège 1", output.getvalue())
        self.assertIn("gain net", output.getvalue())


if __name__ == "__main__":
    unittest.main()
