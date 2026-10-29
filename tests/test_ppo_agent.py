"""Integration checks for the real Stable-Baselines3 learner (no PPO implementation here)."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback

from agents.ppo import PPOAgent
from agents.random import RandomAgent
from poker_rl import AgentRegistry, PokerEnv, run_session
from poker_rl.arena import run_arena
from poker_rl.sb3_training import train_sb3_session


class CaptureStart(BaseCallback):
    def _on_training_start(self):
        self.before = {key: value.detach().clone() for key, value in self.model.policy.state_dict().items()}

    def _on_step(self):
        assert not any(info.get('illegal_action', False) for info in self.locals['infos'])
        return True


class PPOAgentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def agent(self):
        return PPOAgent(seed=42, autoload=False, n_steps=16, batch_size=8, n_epochs=2)

    def env(self):
        env = PokerEnv(opponent=RandomAgent(seed=7))
        self.addCleanup(env.close)
        return env

    def test_native_learn_updates_actor_and_critic_and_masks_actions(self):
        agent = self.agent()
        capture = CaptureStart()
        agent.learn(self.env(), total_timesteps=64, callback=capture)
        after = agent.model.policy.state_dict()
        self.assertFalse(torch.equal(after['action_net.weight'], capture.before['action_net.weight']))
        self.assertFalse(torch.equal(after['value_net.weight'], capture.before['value_net.weight']))
        self.assertIsInstance(agent.model, MaskablePPO)
        self.assertFalse(any(isinstance(module, torch.nn.RNNBase) for module in agent.model.policy.modules()))
        self.assertGreater(agent.model._n_updates, 0)
        self.assertEqual(agent.model.num_timesteps, 64)
        self.assertTrue(all(torch.isfinite(value).all() for value in after.values()))

    def test_checkpoint_restores_sb3_weights_optimizer_and_resumes_learning(self):
        agent = self.agent()
        agent.learn(self.env(), total_timesteps=32)
        target = self.root / 'model.checkpoint'
        agent.save_checkpoint(target)
        self.assertTrue(target.is_file())
        self.assertFalse(Path(str(target) + '.zip').exists())
        loaded = PPOAgent(checkpoint=target, autoload=False)
        for key, value in agent.model.policy.state_dict().items():
            torch.testing.assert_close(value, loaded.model.policy.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(loaded.model._n_updates, agent.model._n_updates)
        for parameter, state in agent.model.policy.optimizer.state_dict()['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(value, loaded.model.policy.optimizer.state_dict()['state'][parameter][key])
        loaded.learn(self.env(), total_timesteps=16)
        self.assertEqual(loaded.model.num_timesteps, 48)
        self.assertGreater(loaded.model._n_updates, agent.model._n_updates)

    def test_small_budget_caps_the_rollout_and_keeps_library_gae(self):
        agent = PPOAgent(seed=42, autoload=False, n_steps=1024, batch_size=128, n_epochs=1)
        agent.learn(self.env(), total_timesteps=8)
        self.assertEqual(agent.model.num_timesteps, 8)
        self.assertEqual(agent.model.n_steps, 8)
        self.assertEqual(agent.model.batch_size, 8)
        self.assertEqual(agent.model.gae_lambda, .95)
        self.assertEqual(agent.model.gamma, .99)
        self.assertEqual(agent.model.ent_coef, 0)
        self.assertIsNone(agent.model.target_kl)
        buffer = agent.model.rollout_buffer
        np.testing.assert_allclose(buffer.returns, buffer.values + buffer.advantages, atol=1e-6)
        self.assertTrue(np.isfinite(buffer.advantages).all())

    def test_probabilities_do_not_depend_on_previous_decisions(self):
        agent = self.agent()
        env = self.env()
        agent.learn(env, total_timesteps=16)
        obs, _ = env.reset(seed=123)
        mask = env.action_masks()
        expected = agent.action_probabilities(obs, mask)
        other = obs.copy()
        other[:104] = 0
        agent.action_probabilities(other, mask)
        np.testing.assert_array_equal(agent.action_probabilities(obs, mask), expected)
        agent.on_episode_start(seed=123)
        np.testing.assert_array_equal(agent.action_probabilities(obs, mask), expected)
        self.assertFalse(hasattr(agent, '_state'))

    def test_training_and_collection_use_the_same_masks_and_log_probs(self):
        agent = self.agent()
        env = self.env()
        agent.learn(env, total_timesteps=16)
        obs, _ = env.reset(seed=7)
        batch = torch.tensor(np.stack([obs] * 6))
        masks = np.zeros((6, 8), dtype=bool)
        for index in range(6):
            masks[index, index] = True
            masks[index, (index + 1) % 8] = True
        policy = agent.model.policy
        actions, values, log_probs = policy(batch, action_masks=masks)
        new_values, new_log_probs, entropy = policy.evaluate_actions(batch, actions, action_masks=masks)
        torch.testing.assert_close(log_probs, new_log_probs)
        torch.testing.assert_close(values, new_values)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(masks[np.arange(6), actions.detach().numpy()].all())

    def test_existing_feedforward_checkpoints_remain_loadable(self):
        model = MaskablePPO('MlpPolicy', self.env(), seed=42, n_steps=16, batch_size=8, n_epochs=1)
        model.learn(total_timesteps=16)
        target = self.root / 'old_mlp.checkpoint'
        with target.open('wb') as stream:
            model.save(stream)
        loaded = PPOAgent(checkpoint=target, autoload=False)
        self.assertIsInstance(loaded.model, MaskablePPO)
        loaded.learn(self.env(), total_timesteps=16)
        self.assertEqual(loaded.model.num_timesteps, 32)

    def test_recurrent_checkpoints_require_a_new_model(self):
        model = RecurrentPPO('MlpLstmPolicy', self.env(), n_steps=8, batch_size=8,
                             policy_kwargs={'lstm_hidden_size': 8})
        target = self.root / 'recurrent.checkpoint'
        with target.open('wb') as stream:
            model.save(stream)
        with self.assertRaisesRegex(ValueError, 'nouveau modèle'):
            PPOAgent(checkpoint=target, autoload=False)

    def test_invalid_masks_are_rejected_without_modifying_observations(self):
        agent = self.agent()
        env = self.env()
        agent.learn(env, total_timesteps=16)
        obs, _ = env.reset(seed=3)
        original = obs.copy()
        for mask in (np.zeros(8, dtype=bool), np.ones(7, dtype=bool)):
            with self.assertRaises(ValueError):
                agent.act(obs, mask)
        agent.act(obs, env.action_masks())
        np.testing.assert_array_equal(obs, original)

    def test_supplied_agents_keep_the_real_registry_and_archive_metadata(self):
        from unittest.mock import patch
        registry = AgentRegistry(records_dir=self.root / 'records')
        agents = [self.agent(), RandomAgent(seed=7)]
        with patch.object(registry, 'load', side_effect=AssertionError('Unexpected agent load')):
            session = train_sb3_session(registry, [('ppo', {}), ('random', {})], agents=agents,
                training_seats=(0,), total_timesteps=16, validation_hands=4)
        self.addCleanup(session.close)
        self.assertIs(session.registry, registry)
        self.assertIs(session.agents[0], agents[0])
        self.assertIs(session.agents[1], agents[1])
        self.assertTrue((session.last_archive / 'manifest.json').is_file())

    def test_original_train_classmethod_saves_a_loadable_default_model(self):
        from unittest.mock import patch
        with patch.object(PPOAgent, 'MODEL_PATH', self.root / 'default_ppo'):
            model = PPOAgent.train(self.env(), total_timesteps=16, seed=42)
            self.assertTrue((self.root / 'default_ppo.zip').is_file())
            agent = PPOAgent()
            self.assertEqual(agent.model.num_timesteps, model.num_timesteps)

    def test_table_adapter_supports_two_to_five_players(self):
        registry = AgentRegistry(records_dir=self.root / 'records')
        for count in range(2, 6):
            selections = [('ppo', {'seed':42, 'autoload':False, 'n_steps':16, 'batch_size':8, 'n_epochs':1})]
            selections += [('random', {'seed':seat + 8}) for seat in range(1, count)]
            session = train_sb3_session(registry, selections, training_seats=(0,),
                total_timesteps=32, record=False, max_hands_per_match=20, validation_hands=12)
            try:
                self.assertGreater(session.agents[0].model._n_updates, 0)
                self.assertEqual(session.sb3_timesteps, 32)
                self.assertGreater(session.stats.hands, 0)
            finally:
                session.close()

    def test_archive_reload_and_arena_preserve_weights_and_do_not_learn(self):
        registry = AgentRegistry(records_dir=self.root / 'records')
        session = train_sb3_session(registry,
            [('ppo', {'seed':42, 'autoload':False, 'n_steps':16, 'batch_size':8, 'n_epochs':1}),
             ('random', {'seed':8})], training_seats=(0,), total_timesteps=32, max_hands_per_match=20, validation_hands=12)
        self.addCleanup(session.close)
        import json
        manifest = json.loads((session.last_archive / 'manifest.json').read_text(encoding='utf-8'))
        self.assertGreater(manifest['agents'][0]['training_metrics']['ppo_epochs'], 0)
        catalog = AgentRegistry(records_dir=self.root / 'records')
        names = [entry['model_id'] for entry in manifest['agents']]
        loaded = catalog.load(names[0])
        epochs = loaded.model._n_updates
        frozen = {key: value.clone() for key, value in loaded.model.policy.state_dict().items()}
        run_session([loaded, RandomAgent(seed=9)], episodes=5)
        self.assertEqual(loaded.model._n_updates, epochs)
        self.assertTrue(all(torch.equal(value, frozen[key]) for key, value in loaded.model.policy.state_dict().items()))
        report = run_arena(catalog, [(name,{}) for name in names], matches=3, max_hands_per_match=20)
        self.assertEqual(report['matches_completed'] + report['matches_truncated'], 3)

    def test_generic_observe_loop_does_not_silently_pretend_to_train_sb3(self):
        with self.assertRaisesRegex(ValueError, 'model.learn'):
            run_session([self.agent(), RandomAgent()], episodes=5, training_seats=(0,))
        with self.assertRaisesRegex(ValueError, 'un seul'):
            train_sb3_session(AgentRegistry(), [('ppo', {}), ('ppo', {})], training_seats=(0,1), total_timesteps=16)

    def test_held_out_validation_preserves_weights_optimizer_rng_and_training_counters(self):
        import copy
        from poker_rl.evaluation import evaluate_agents
        agent = self.agent()
        env = self.env()
        agent.learn(env, total_timesteps=32)
        obs, info = env.reset(seed=11)
        agent.act(obs, info['action_mask'])
        weights = {key: value.clone() for key, value in agent.model.policy.state_dict().items()}
        optimizer = copy.deepcopy(agent.model.policy.optimizer.state_dict())
        rng_state = copy.deepcopy(agent.rng.bit_generator.state)
        decision = copy.deepcopy(agent.last_decision)
        torch_state = torch.random.get_rng_state().clone()
        stats = dict(agent.training_stats)
        report = evaluate_agents([agent, RandomAgent(seed=3)], (0,), hands_per_opponent=12)
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(agent.training_stats, stats)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        for key, value in weights.items():
            torch.testing.assert_close(value, agent.model.policy.state_dict()[key], rtol=0, atol=0)
        for parameter, values in optimizer['state'].items():
            for key, value in values.items():
                torch.testing.assert_close(value, agent.model.policy.optimizer.state_dict()['state'][parameter][key])
        self.assertEqual(agent.rng.bit_generator.state, rng_state)
        self.assertEqual(agent.last_decision, decision)
        self.assertFalse(report['agents'][0]['policy']['recurrent'])

    def test_cli_uses_native_sb3_training_and_archives_it(self):
        import json
        import io
        from contextlib import redirect_stdout
        from run_agents import main
        path = self.root / 'config.json'
        path.write_text(json.dumps({'agents':[
            {'name':'ppo','kwargs':{'seed':42,'autoload':False,'n_steps':16,'batch_size':8,'n_epochs':1}},
            {'name':'random','kwargs':{'seed':8}}]}), encoding='utf-8')
        with redirect_stdout(io.StringIO()):
            report = main(['train','--config',str(path),'--records-dir',str(self.root / 'records'),'--timesteps','32','--validation-hands','12'])
        self.assertEqual(report['timesteps'], 32)
        self.assertGreater(report['training'][0]['ppo_epochs'], 0)
        self.assertTrue(Path(report['archive']).is_dir())
        self.assertEqual(report['validation']['status'], 'completed')
        self.assertEqual(report['validation']['protocol']['hands_per_opponent'], 12)

    def gui_config(self):
        from poker_rl.gui import PokerApp
        # Exercise the actual launch handlers without requiring a desktop window.
        app = PokerApp.__new__(PokerApp)
        app.registry = AgentRegistry(records_dir=self.root / 'records')
        app.count = 2
        app.mode = 'train'
        app.selected = ['human', 'ppo']
        app.training = [False, True]
        app.kwargs = ['{"name":"Moi"}', '{"autoload":false,"seed":42,"n_steps":16,"batch_size":8,"n_epochs":1}']
        app.episodes_text = '32'
        app.max_hands_text = '20'
        app.record_sessions = True
        app.validation_text = '12'
        app.loading = False
        app.pending = None
        app.error = ''
        return app

    def test_gui_rejects_human_opponent_before_starting_or_creating_an_archive(self):
        app = self.gui_config()
        app.start()
        self.assertIn('Joueur(s) 1 : human', app.error)
        self.assertIn('Entraîner contre random', app.error)
        self.assertFalse(app.loading)
        self.assertIsNone(app.pending)
        self.assertEqual(list(app.registry.records_dir.glob('*/manifest.json')), [])

    def test_gui_random_shortcut_trains_ppo_in_the_original_learner_seat(self):
        app = self.gui_config()
        app.start()
        app.train_against_random()
        self.assertEqual(app.selected, ['random', 'ppo'])
        self.assertEqual(app.kwargs[0], '{}')
        self.assertEqual(app.training, [False, True])
        self.assertEqual(app.pending_kind, 'ppo_train')
        self.assertEqual(app.error, '')
        session = app.pending.result(timeout=30)
        self.addCleanup(session.close)
        self.assertIsInstance(session.agents[1].model, MaskablePPO)
        self.assertEqual(session.sb3_timesteps, 32)
        self.assertGreater(session.agents[1].model._n_updates, 0)
        self.assertTrue((session.last_archive / 'manifest.json').is_file())


if __name__ == '__main__':
    unittest.main()
