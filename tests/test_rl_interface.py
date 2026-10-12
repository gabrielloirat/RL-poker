import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from gym_env.enums import Action, Stage
from poker_rl import PokerEnv, PokerGame, OBSERVATION_FIELDS, load_agent
from poker_rl.session import run_session


class _RecordingAgent:


    def __init__(self, action=None):
        self.action = action
        self.transitions = []
        self.training_flags = []
        self.starts = []

    def act(self, observation, action_mask, *, training=False):
        self.training_flags.append(training)
        if self.action is not None:
            return self.action
        return 1 if action_mask[1] else 2

    def observe(self, transition):
        self.transitions.append(transition)

    def on_episode_start(self, *, seed, training):
        self.starts.append((seed, training))


class RLInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.game = PokerGame()
        self.addCleanup(self.game.close)

    def test_observations_are_private_finite_and_seeded(self):
        with patch.object(self.game.table, "get_equity", side_effect=AssertionError("Monte Carlo called")):
            first, first_info = self.game.reset(seed=42)
            second, second_info = self.game.reset(seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_info["hole_cards"], second_info["hole_cards"])
        self.assertTrue(self.game.observation_space.contains(second))
        own = self.game.observe(0)
        before_info = self.game.info(0)
        self.game.table.players[1].cards = ["AS", "KS"]
        np.testing.assert_array_equal(own, self.game.observe(0))
        self.assertEqual(before_info["hole_cards"], self.game.info(0)["hole_cards"])
        self.assertNotIn("opponent_cards", before_info)
        self.assertEqual(sum(size for _, size in OBSERVATION_FIELDS), len(own))

    def test_heads_up_positions_and_fold_rewards(self):
        for dealer in (0, 1):
            self.game.reset(seed=7, options={"dealer_seat": dealer})
            self.assertEqual(self.game.current_seat, dealer)
            self.assertEqual(self.game.table.player_pots[dealer], 1)
            self.assertEqual(self.game.table.player_pots[1 - dealer], 2)
            _, rewards, terminated, truncated, _ = self.game.step(Action.FOLD.value)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertAlmostEqual(rewards[dealer], -0.5)
            self.assertAlmostEqual(rewards[1 - dealer], 0.5)
            self.assertFalse(self.game.action_masks(dealer).any())
            with self.assertRaises(RuntimeError):
                self.game.step(1)

    def test_postflop_non_dealer_acts_first(self):
        self.game.reset(seed=7, options={"dealer_seat": 0})
        self.game.step(Action.CALL.value)
        self.game.step(Action.CHECK.value)
        self.assertEqual(self.game.table.stage, Stage.FLOP)
        self.assertEqual(self.game.current_seat, 1)

    def test_raise_limit_never_skips_the_last_response(self):
        self.game.reset(seed=7, options={"dealer_seat": 0})
        for action in (Action.RAISE_3BB, Action.RAISE_POT,
                       Action.RAISE_POT, Action.RAISE_POT):
            self.assertIn(action, self.game.table.legal_moves)
            self.game.step(action.value)
            self.assertEqual(self.game.table.stage, Stage.PREFLOP)
        self.assertEqual(self.game.current_seat, 0)
        self.assertEqual(set(self.game.table.legal_moves), {Action.CALL, Action.FOLD})
        self.game.step(Action.CALL.value)
        self.assertEqual(self.game.table.stage, Stage.FLOP)
        self.assertEqual(self.game.table.community_pot, 84)

    def test_showdown_tie_returns_chips_and_zero_reward(self):
        self.game.reset(seed=7)
        self.game.table.players[0].cards = ["2C", "3C"]
        self.game.table.players[1].cards = ["4D", "5D"]
        board = iter(["TH", "JH", "QH", "KH", "AH"])
        with patch.object(self.game.table, "_distribute_cards_to_table",
                          side_effect=lambda n: self.game.table.table_cards.extend(next(board) for _ in range(n))):
            self.game.step(Action.ALL_IN.value)
            _, rewards, terminated, _, info = self.game.step(Action.CALL.value)
        self.assertTrue(terminated)
        np.testing.assert_array_equal(rewards, [0, 0])
        self.assertEqual(set(info["winners"]), {0, 1})
        self.assertEqual([p.stack for p in self.game.table.players], [100, 100])

    def test_invalid_action_does_not_move_player_and_can_truncate(self):
        game = PokerGame(max_actions=2)
        self.addCleanup(game.close)
        observation, _ = game.reset(seed=11)
        seat = game.current_seat
        after, rewards, terminated, truncated, info = game.step(Action.CHECK.value)
        np.testing.assert_array_equal(observation, after)
        self.assertEqual(game.current_seat, seat)
        self.assertTrue(info["illegal_action"])
        self.assertEqual(rewards[seat], -1)
        self.assertFalse(terminated or truncated)
        _, _, terminated, truncated, _ = game.step(Action.CHECK.value)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_gymnasium_checker_and_registration(self):
        env = PokerEnv(_RecordingAgent())
        self.addCleanup(env.close)
        check_env(env, skip_render_check=True)
        wrapped = gym.make("poker-rl-v0", opponent=_RecordingAgent())
        self.addCleanup(wrapped.close)
        observation, info = wrapped.reset(seed=42)
        self.assertTrue(wrapped.observation_space.contains(observation))
        self.assertTrue(info["action_mask"].any())

    def test_single_agent_returns_its_loss_after_opponent_call(self):
        env = PokerEnv(_RecordingAgent(), agent_seat=0)
        self.addCleanup(env.close)
        env.reset(seed=10)
        env.game.table.players[0].cards = ["KS", "KH"]
        env.game.table.players[1].cards = ["AS", "AH"]
        board = iter(["2C", "3D", "7H", "8S", "9C"])
        with patch.object(env.game.table, "_distribute_cards_to_table",
                          side_effect=lambda n: env.game.table.table_cards.extend(next(board) for _ in range(n))):
            observation, reward, terminated, truncated, info = env.step(Action.ALL_IN.value)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -50)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["seat"], 0)
        self.assertFalse(env.action_masks().any())

    def test_training_routes_same_player_transitions_and_final_rewards(self):
        agents = [_RecordingAgent(), _RecordingAgent()]
        results = run_session(agents, episodes=3, training_seats=(0, 1), seed=8)
        for seat, agent in enumerate(agents):
            terminal = [t for t in agent.transitions if t.terminated]
            self.assertEqual(len(terminal), 3)
            self.assertEqual([t.reward for t in terminal], [r.rewards_bb[seat] for r in results])
            for transition in agent.transitions:
                self.assertEqual(transition.info["seat"], seat)
                self.assertTrue(transition.action_mask[transition.action])
                if not transition.terminated:
                    self.assertEqual(transition.info["current_player"], seat)
                    self.assertEqual(transition.reward, 0)
                    self.assertTrue(transition.next_action_mask.any())
                else:
                    self.assertFalse(transition.next_action_mask.any())
            self.assertTrue(all(agent.training_flags))

    def test_inference_never_sends_training_transitions(self):
        agents = [_RecordingAgent(), _RecordingAgent()]
        run_session(agents, episodes=2, seed=3)
        for agent in agents:
            self.assertEqual(agent.transitions, [])
            self.assertFalse(any(agent.training_flags))
        run_session(agents, episodes=1, training_seats=(1,), seed=3)
        self.assertEqual(agents[0].transitions, [])
        self.assertGreater(len(agents[1].transitions), 0)

    def test_terminal_opponent_reward_is_delivered_to_pending_decision(self):
        agents = [_RecordingAgent(Action.ALL_IN.value), _RecordingAgent()]
        result = run_session(agents, episodes=1, training_seats=(0, 1), seed=2)[0]
        for seat, agent in enumerate(agents):
            self.assertEqual(len(agent.transitions), 1)
            self.assertTrue(agent.transitions[0].terminated)
            self.assertEqual(agent.transitions[0].reward, result.rewards_bb[seat])

    def test_custom_loader_instantiates_user_factory(self):
        agent = load_agent(f"{__name__}:_RecordingAgent", {"action": 0})
        self.assertEqual(agent.action, 0)
        with self.assertRaises(ValueError):
            load_agent("MissingSeparator")

    def test_random_legal_trajectories_conserve_chips(self):
        rng = np.random.default_rng(19)
        for raise_limit in (1, 2, 4):
            game = PokerGame(max_raises_per_player_round=raise_limit)
            self.addCleanup(game.close)
            for seed in range(100):
                game.reset(seed=seed)
                while not game.finished:
                    legal = np.flatnonzero(game.action_masks())
                    self.assertGreater(len(legal), 0)
                    _, _, _, _, info = game.step(int(rng.choice(legal)))
                    self.assertFalse(info["illegal_action"])
                    self.assertTrue(all(p.stack >= 0 for p in game.table.players))
                    for seat in (0, 1):
                        self.assertTrue(game.observation_space.contains(game.observe(seat)))
                    pot = 0 if game.terminated else game.table.community_pot + game.table.current_round_pot
                    self.assertAlmostEqual(sum(p.stack for p in game.table.players) + pot, 200)
                self.assertTrue(game.terminated)
                self.assertAlmostEqual(sum(game.rewards()), 0)


if __name__ == "__main__":
    unittest.main()
