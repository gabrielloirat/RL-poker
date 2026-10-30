import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from poker_rl.benchmark import (BenchmarkConfig, evaluate_lineup, fixed_seed, passes_gate,
                                reward_summary, train_block, save_policy, load_policy)
from poker_rl.plugins import AgentRegistry
from poker_rl.benchmark_observation import (UniversalAgent, universal_observation, TARGET_FIELDS,
                                            TARGET_SIZE, SOURCE_FIELDS)
from poker_rl.game import PokerGame


class BenchmarkTests(unittest.TestCase):
    def test_seed_streams_reproduce_and_separate_roles(self):
        self.assertEqual(fixed_seed(42, 1, 2), fixed_seed(42, 1, 2))
        self.assertEqual(len({fixed_seed(42, role, n, alg, block)
                             for role in (0, 1, 2, 3) for n in (2, 4)
                             for alg in range(3) for block in range(8)}), 192)

    def test_gate_requires_positive_bound_and_material_profit(self):
        self.assertFalse(passes_gate(reward_summary([50, -49]), 100))
        self.assertFalse(passes_gate(reward_summary([.5] * 20), 100))
        self.assertTrue(passes_gate(reward_summary([2.] * 20), 100))
        summary = reward_summary([1., 2., 3.])
        self.assertAlmostEqual(summary['bb100'], 200.)
        self.assertAlmostEqual(summary['standard_error_bb'], 1 / np.sqrt(3))
        split_pots = reward_summary([1e-14, -1e-14, 1., -1.])
        self.assertEqual(split_pots['tie_rate'], .5)
        self.assertEqual(split_pots['win_rate'], .25)

    def test_validation_repeats_rotates_and_does_not_mutate_policy(self):
        registry = AgentRegistry()
        policy = registry.load('random', {'seed': 43})
        before = copy.deepcopy(policy.rng.bit_generator.state)
        config = BenchmarkConfig()
        a = evaluate_lineup(registry, policy, ['random'] * 3, config, seed=123, hands=32)
        b = evaluate_lineup(registry, policy, ['random'] * 3, config, seed=123, hands=32)
        self.assertEqual(a, b)
        self.assertEqual(a['position_counts'], [8] * 4)
        self.assertEqual(before, policy.rng.bit_generator.state)

    def test_non_native_selfplay_opponent_checkpoint_stays_immutable(self):
        registry = AgentRegistry()
        config = BenchmarkConfig(dqn_hands=8)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            learner = registry.load('dqn', {'seed': 42, 'buffer_size': 32, 'learning_starts': 2, 'batch_size': 2})
            train_block(registry, learner, 'dqn', ['random'], config, seed=42, folder=root / 'initial')
            spec = save_policy(learner, 'dqn', root / 'frozen.checkpoint', {})
            before = (root / 'frozen.checkpoint').read_bytes()
            rival = load_policy(registry, spec, config, 43)
            weights = [p.detach().clone() for p in rival.online.parameters()]
            result = train_block(registry, learner, 'dqn', [spec], config, seed=44, folder=root / 'self')
            self.assertEqual(result['training']['hands'], 8)
            self.assertEqual(before, (root / 'frozen.checkpoint').read_bytes())
            self.assertTrue(all((x == y).all() for x, y in zip(weights, rival.online.parameters())))

    def test_ppo_checkpoint_validation_and_selfplay_resume(self):
        import torch
        registry = AgentRegistry()
        config = BenchmarkConfig(ppo_steps=16)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            learner = registry.load('ppo', {'seed': 42, 'autoload': False,
                                           'n_steps': 16, 'batch_size': 8, 'n_epochs': 1})
            train_block(registry, learner, 'ppo', ['random'], config, seed=42, folder=root / 'initial')
            before = [p.detach().clone() for p in learner.model.policy.parameters()]
            result = evaluate_lineup(registry, learner, ['random'], config, seed=123, hands=16)
            repeated = evaluate_lineup(registry, learner, ['random'], config, seed=123, hands=16)
            self.assertEqual(result, repeated)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(before, learner.model.policy.parameters())))
            spec = save_policy(learner, 'ppo', root / 'frozen.checkpoint', {})
            result = train_block(registry, learner, 'ppo', [spec], config, seed=44, folder=root / 'self')
            self.assertEqual(result['decisions'], 16)
            self.assertEqual(learner.model.num_timesteps, 32)

    def test_common_observation_keeps_all_seats_and_consistent_money_scale(self):
        for players in (2, 4):
            game = PokerGame(num_players=players)
            self.addCleanup(game.close)
            game.reset(seed=42)
            original = game.observe(game.current_seat)
            padded = universal_observation(original)
            self.assertEqual(padded.shape, (TARGET_SIZE,))
            np.testing.assert_array_equal(original[:104], padded[:104])
            original_fields = SOURCE_FIELDS[players][0]
            for name in ('stacks', 'street_bets', 'hand_contributions'):
                np.testing.assert_allclose(padded[TARGET_FIELDS[name]][:players],
                                           original[original_fields[name]] * (players / 5))
                np.testing.assert_array_equal(padded[TARGET_FIELDS[name]][players:], 0)

    def test_same_neural_weights_act_at_two_and_four_player_tables(self):
        import torch
        torch.set_num_threads(1)
        registry = AgentRegistry()
        with TemporaryDirectory() as directory:
            config = BenchmarkConfig(ppo_steps=16)
            learner = UniversalAgent(registry.load('ppo', {'seed': 42, 'autoload': False,
                                                         'n_steps': 16, 'batch_size': 8, 'n_epochs': 1}))
            train_block(registry, learner, 'ppo', ['random'], config, seed=42, folder=Path(directory) / 'train')
            weights = [p.detach().clone() for p in learner.model.policy.parameters()]
            for players in (2, 4):
                result = evaluate_lineup(registry, learner, ['random'] * (players - 1), config, seed=123, hands=16)
                self.assertEqual(result['hands'], 16)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(weights, learner.model.policy.parameters())))

    def test_fast_cpu_inference_features_are_bitwise_equal(self):
        import torch
        from gymnasium import spaces
        from poker_rl.features import PokerFeaturesExtractor
        from poker_rl.benchmark_fast_inference import cpu_features
        extractor = PokerFeaturesExtractor(spaces.Box(0, 1, shape=(TARGET_SIZE,), dtype=np.float32))
        for players in (2, 4):
            game = PokerGame(num_players=players)
            self.addCleanup(game.close)
            rng = np.random.default_rng(42)
            for seed in range(32):
                game.reset(seed=seed)
                while not game.finished:
                    obs = universal_observation(game.observe(game.current_seat))
                    tensor = torch.from_numpy(obs[None])
                    with torch.no_grad():
                        expected = extractor(tensor)
                        actual = cpu_features(extractor, tensor)
                    self.assertTrue(torch.equal(expected, actual),
                                    f'{players}p seed {seed}: {torch.max(torch.abs(expected - actual))}')
                    game.step(int(rng.choice(np.flatnonzero(game.action_masks()))))

    def test_ppo_repeated_training_sessions_reproduce_weights(self):
        import torch
        torch.set_num_threads(1)
        registry = AgentRegistry()
        results = []
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = BenchmarkConfig(ppo_steps=32)
            for repeat in range(2):
                learner = UniversalAgent(registry.load('ppo', {'seed': 42, 'autoload': False,
                    'n_steps': 16, 'batch_size': 8, 'n_epochs': 2}))
                for block in range(2):
                    train_block(registry, learner, 'ppo', ['random'] * 3, config,
                                seed=100 + block, folder=root / str(repeat) / str(block))
                results.append({k: v.clone() for k, v in learner.model.policy.state_dict().items()})
        self.assertTrue(all(torch.equal(v, results[1][k]) for k, v in results[0].items()))

    def test_mcts_strength_cache_keeps_actions_and_random_stream_identical(self):
        from poker_rl.benchmark_fast_inference import install_strength_cache
        registry = AgentRegistry()
        agent = registry.load('mcts', {'seed': 42, 'num_simulations': 8})
        game = PokerGame(num_players=4)
        self.addCleanup(game.close)
        rng = np.random.default_rng(42)
        observations = []
        for seed in range(8):
            game.reset(seed=seed)
            while not game.finished:
                mask = game.action_masks()
                observations.append((game.observe(game.current_seat), mask))
                game.step(int(rng.choice(np.flatnonzero(mask))))

        def decisions():
            result = []
            for index, (observation, mask) in enumerate(observations):
                agent.on_episode_start(seed=1000 + index)
                result.append((agent.act(observation, mask), agent.rng.getstate()))
            return result

        expected = decisions()
        cached = install_strength_cache(registry)
        actual = decisions()
        self.assertEqual(expected, actual)
        self.assertGreater(cached.cache_info().hits, 0)

    def test_isolated_case_commits_result_and_reuses_it_without_new_process(self):
        from unittest.mock import patch
        from poker_rl.benchmark_workers import evaluate_subprocess
        registry = AgentRegistry()
        config = BenchmarkConfig(dqn_hands=8, final_hands=16)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            learner = registry.load('dqn', {'seed': 42, 'buffer_size': 32})
            train_block(registry, learner, 'dqn', ['random'], config, seed=42, folder=root / 'train')
            spec = save_policy(learner, 'dqn', root / 'model.checkpoint', {})
            row = {'id': 'probe_dqn', 'kind': 'dqn', 'players': 2, 'regime': 'curriculum', 'checkpoint': spec}
            task = (config, root, row, 2, 'random', ['random'])
            result = evaluate_subprocess(task)
            self.assertEqual(result['hands'], 16)
            config.evaluation_workers = 1
            with patch('poker_rl.benchmark_workers.subprocess.run', side_effect=AssertionError('Cache non utilisé')):
                repeated = evaluate_subprocess(task)
            self.assertEqual(result, repeated)

    def test_cached_mcts_weight_factors_match_scalar_calculation_and_invalidate(self):
        from poker_rl.benchmark_fast_inference import policy_weight_factors
        from types import SimpleNamespace
        rng = np.random.default_rng(42)
        policy = SimpleNamespace(weights=rng.normal(0, 5, (32, 8)).astype(np.float32))
        for bucket in range(32):
            expected = tuple(np.exp(np.clip(policy.weights[bucket, a], -3., 3.)) for a in range(8))
            self.assertEqual(expected, policy_weight_factors(policy, bucket))
        old = policy_weight_factors(policy, 0)
        policy.weights[0] = 0
        self.assertNotEqual(old, policy_weight_factors(policy, 0))
        self.assertEqual(policy_weight_factors(policy, 0), (1.,) * 8)


if __name__ == '__main__':
    unittest.main()
