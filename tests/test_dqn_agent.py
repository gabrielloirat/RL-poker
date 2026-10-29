from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from poker_rl import AgentRegistry, PokerGame, load_agent, run_session, Agent, TrainableAgent
from agents.dqn import DQNAgent, QNetwork, ReplayBuffer

SMALL = dict(hidden_sizes=(32, 32), batch_size=16, learning_starts=16,
             train_every=1, target_update=5, buffer_size=500, seed=0)


def make_agent(**overrides):
    return DQNAgent(**{**SMALL, **overrides})


class TestDQNAgent(unittest.TestCase):
    def test_registry_discovery(self):
        registry = AgentRegistry()
        self.assertIn("dqn", registry.agents)
        self.assertTrue(registry.agents["dqn"].trainable)
        agent = registry.load("dqn", dict(SMALL))
        self.assertEqual(type(agent).__name__, "DQNAgent")
        self.assertIsInstance(agent, Agent)
        self.assertIsInstance(agent, TrainableAgent)

    def test_load_agent_spec_with_json_like_kwargs(self):
        agent = load_agent("dqn", {"hidden_sizes": [16, 16], "seed": 1})
        self.assertEqual(agent.hidden_sizes, (16, 16))

    def test_act_returns_legal_action_in_both_modes(self):
        agent = make_agent()
        game = PokerGame(num_players=2)
        self.addCleanup(game.close)
        obs, _ = game.reset(seed=42)
        mask = game.action_masks(game.current_seat)
        for training in (False, True):
            action = agent.act(obs, mask, training=training)
            self.assertIsInstance(action, (int, np.integer))
            self.assertTrue(mask[action])

    def test_never_plays_illegal_action(self):
        agent = make_agent(epsilon_start=0.5)
        obs = np.random.default_rng(0).random(147).astype(np.float32)
        mask = np.zeros(8, dtype=bool)
        mask[[2, 5]] = True
        for training in (False, True):
            for _ in range(200):
                self.assertIn(agent.act(obs, mask, training=training), (2, 5))

    def test_empty_action_mask_raises(self):
        agent = make_agent()
        with self.assertRaises(ValueError):
            agent.act(np.zeros(147, dtype=np.float32), np.zeros(8, dtype=bool))

    def test_epsilon_decays_only_with_training_decisions(self):
        agent = make_agent(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=10)
        obs = np.zeros(147, dtype=np.float32)
        mask = np.ones(8, dtype=bool)
        self.assertAlmostEqual(agent.epsilon, 1.0)
        agent.act(obs, mask, training=False)
        self.assertEqual(agent.act_steps, 0)
        for _ in range(10):
            agent.act(obs, mask, training=True)
        self.assertAlmostEqual(agent.epsilon, 0.1)
        for _ in range(10):
            agent.act(obs, mask, training=True)
        self.assertAlmostEqual(agent.epsilon, 0.1)

    def test_replay_buffer_is_circular(self):
        buffer = ReplayBuffer(3, 4)
        mask = np.ones(8, dtype=bool)
        for i in range(5):
            buffer.push(np.full(4, i, np.float32), mask, 1, float(i), np.full(4, i + 1, np.float32), mask, i == 4)
        self.assertEqual(len(buffer), 3)
        self.assertEqual(sorted(buffer.reward.tolist()), [2.0, 3.0, 4.0])
        batch = buffer.sample(6, np.random.default_rng(0))
        self.assertEqual(batch[0].shape, (6, 4))
        self.assertEqual(batch[1].dtype, torch.bool)

    def test_training_updates_network_and_inference_does_not(self):
        dqn = make_agent()
        opponent = load_agent("random", {"seed": 3})
        run_session([dqn, opponent], episodes=1, training_seats=[0], seed=5)
        self.assertIsNotNone(dqn.online)
        before = [p.detach().clone() for p in dqn.online.parameters()]

        results = run_session([dqn, opponent], episodes=30, training_seats=[0], seed=6)
        self.assertEqual(len(results), 30)
        self.assertTrue(all(r.illegal_actions == 0 and r.terminated for r in results))
        self.assertGreater(len(dqn.buffer), 16)
        self.assertGreater(dqn.learn_steps, 0)
        self.assertTrue(np.isfinite(dqn.last_loss))
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(before, dqn.online.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for p in dqn.online.parameters()))

        steps, size = dqn.learn_steps, len(dqn.buffer)
        run_session([dqn, opponent], episodes=10, seed=7)
        self.assertEqual(dqn.learn_steps, steps)
        self.assertEqual(len(dqn.buffer), size)

    def test_terminal_transitions_are_stored_with_done_flag(self):
        dqn = make_agent(learning_starts=10_000)
        run_session([dqn, load_agent("random", {"seed": 1})], episodes=5, training_seats=[0], seed=9)
        self.assertEqual(int(dqn.buffer.done[:len(dqn.buffer)].sum()), 5)
        self.assertEqual(len(dqn.buffer), dqn.act_steps)

    def test_checkpoint_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dqn.pt"
            agent = make_agent()
            run_session([agent, load_agent("random", {"seed": 2})], episodes=10, training_seats=[0], seed=11)
            agent.save_checkpoint(path)
            self.assertTrue(path.is_file())

            restored = DQNAgent(checkpoint=path, **{**SMALL, "hidden_sizes": (8,)})
            self.assertEqual(restored.hidden_sizes, (32, 32))
            self.assertEqual(restored.input_dim, agent.input_dim)
            self.assertEqual(restored.learn_steps, agent.learn_steps)
            x = torch.rand(4, agent.input_dim)
            with torch.no_grad():
                torch.testing.assert_close(agent.online(x), restored.online(x))
            obs = np.random.default_rng(1).random(agent.input_dim).astype(np.float32)
            mask = np.ones(8, dtype=bool)
            self.assertEqual(agent.act(obs, mask), restored.act(obs, mask))

    def test_automatic_checkpoint_only_in_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "auto.pt"
            agent = make_agent(checkpoint=path, checkpoint_every=2)
            opponent = load_agent("random", {"seed": 4})
            run_session([agent, opponent], episodes=4, seed=1)
            self.assertFalse(path.exists())
            run_session([agent, opponent], episodes=4, training_seats=[0], seed=1)
            self.assertTrue(path.is_file())

    def test_network_input_adapts_to_table_size(self):
        for count in (2, 3, 5):
            agent = make_agent()
            players = [agent] + [load_agent("random", {"seed": i}) for i in range(count - 1)]
            results = run_session(players, episodes=3, training_seats=[0],
                                  game_kwargs={"num_players": count}, seed=21)
            self.assertEqual(agent.input_dim, {2: 147, 3: 164, 5: 194}[count])
            self.assertTrue(all(r.illegal_actions == 0 for r in results))

    def test_self_play_trains_two_independent_networks(self):
        a, b = make_agent(seed=1), make_agent(seed=2)
        run_session([a, b], episodes=15, training_seats=[0, 1], seed=31)
        self.assertGreater(a.learn_steps, 0)
        self.assertGreater(b.learn_steps, 0)
        self.assertIsNot(a.online, b.online)

    def test_qnetwork_shape(self):
        net = QNetwork(147, (16, 16))
        self.assertEqual(tuple(net(torch.zeros(3, 147)).shape), (3, 8))


if __name__ == "__main__":
    unittest.main()