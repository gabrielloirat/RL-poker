import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from poker_rl import AgentRegistry, PokerGame, load_agent, run_session, Agent, TrainableAgent
from agents.mcts import MCTSAgent, MCTSPolicy, evaluate_hand_strength, _observation_to_sim_state_factory


class TestMCTSAgent(unittest.TestCase):
    def test_registry_discovery(self):
        registry = AgentRegistry()
        self.assertIn("mcts", registry.agents)
        self.assertTrue(registry.agents["mcts"].trainable)
        agent = registry.load("mcts")
        self.assertEqual(type(agent).__name__, "MCTSAgent")
        self.assertIsInstance(agent, Agent)
        self.assertIsInstance(agent, TrainableAgent)

    def test_load_agent_spec(self):
        agent = load_agent("mcts")
        self.assertEqual(type(agent).__name__, "MCTSAgent")
        self.assertIsInstance(agent, Agent)
        self.assertIsInstance(agent, TrainableAgent)
        self.assertTrue(callable(getattr(agent, "act", None)))
        self.assertTrue(callable(getattr(agent, "observe", None)))

    def test_hand_strength_evaluation(self):
        # AA should be stronger than 72o preflop
        strength_aa = evaluate_hand_strength(["AS", "AH"], [])
        strength_72 = evaluate_hand_strength(["7S", "2D"], [])
        self.assertGreater(strength_aa, strength_72)
        self.assertGreater(strength_aa, 0.9)
        self.assertLess(strength_72, 0.25)

        # Flop evaluations
        board = ["KS", "QD", "2C"]
        strength_set = evaluate_hand_strength(["KD", "KH"], board)
        strength_air = evaluate_hand_strength(["7H", "3S"], board)
        self.assertGreater(strength_set, strength_air)
        self.assertGreater(strength_set, 0.7)

    def test_act_returns_legal_action(self):
        agent = MCTSAgent(num_simulations=20)
        game = PokerGame(num_players=2)
        obs, info = game.reset(seed=42)

        mask = game.action_masks(game.current_seat)
        action = agent.act(obs, mask)
        self.assertIsInstance(action, (int, np.integer))
        self.assertTrue(mask[action])

    def test_single_legal_action_fast_path(self):
        agent = MCTSAgent(num_simulations=50)
        obs = np.zeros(147, dtype=np.float32)
        # Only CHECK (1) is legal
        mask = np.zeros(8, dtype=bool)
        mask[1] = True
        action = agent.act(obs, mask)
        self.assertEqual(action, 1)

    def test_empty_action_mask_raises(self):
        agent = MCTSAgent()
        obs = np.zeros(147, dtype=np.float32)
        mask = np.zeros(8, dtype=bool)
        with self.assertRaises(ValueError):
            agent.act(obs, mask)

    def test_checkpoint_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "mcts.json"
            agent = MCTSAgent()
            agent.policy.weights[0, 1] = 2.5
            agent.save_checkpoint(ckpt_path)
            self.assertTrue(ckpt_path.is_file())

            agent2 = MCTSAgent(checkpoint=ckpt_path)
            self.assertAlmostEqual(agent2.policy.weights[0, 1], 2.5)

    def test_run_session_infer_and_train(self):
        mcts0 = MCTSAgent(num_simulations=30)
        mcts1 = load_agent("random")
        results = run_session([mcts0, mcts1], episodes=5, seed=123)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertEqual(r.illegal_actions, 0)
            self.assertTrue(r.terminated)

        # Test in training mode
        results_train = run_session([mcts0, mcts1], episodes=5, training_seats=[0], seed=456)
        self.assertEqual(len(results_train), 5)
        for r in results_train:
            self.assertEqual(r.illegal_actions, 0)

    def test_multiplayer_session(self):
        mcts = MCTSAgent(num_simulations=20)
        r1 = load_agent("random")
        r2 = load_agent("random")
        results = run_session([mcts, r1, r2], episodes=3, game_kwargs={"num_players": 3}, seed=789)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.illegal_actions, 0)


if __name__ == "__main__":
    unittest.main()
