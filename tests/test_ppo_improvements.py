import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from agents.ppo import PPOAgent
from poker_rl import AgentRegistry, PokerEnv
from poker_rl.evaluation import evaluate_agents
from poker_rl.features import PokerFeaturesExtractor
from poker_rl.game import CARD_INDEX, PokerGame, observation_fields
from poker_rl.opponents import MixedOpponent
from poker_rl.sb3_training import TableTrainingEnv, train_sb3_session
from poker_rl.table_session import TableSession


class PPOImprovementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def agent(self, *, deterministic=False):
        agent = PPOAgent(seed=42, autoload=False, deterministic=deterministic,
                         n_steps=16, batch_size=8, n_epochs=1)
        env = PokerEnv(MixedOpponent(seed=43))
        self.addCleanup(env.close)
        agent.learn(env, total_timesteps=16)
        return agent, env

    def test_stochastic_sampling_is_local_seeded_masked_and_checkpointed(self):
        agent, env = self.agent()
        obs, info = env.reset(seed=5)
        agent.on_episode_start(seed=1234)
        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
        path = self.root / 'model.checkpoint'
        agent.save_checkpoint(path)
        expected = [agent.act(obs,info['action_mask']) for _ in range(24)]
        self.assertGreater(len(set(expected)), 1)
        loaded = PPOAgent(checkpoint=path)
        actual = [loaded.act(obs,info['action_mask']) for _ in range(24)]
        self.assertEqual(expected, actual)
        self.assertTrue(all(info['action_mask'][action] for action in actual))
        # Checkpoint loading may seed global RNGs; sampling must not.
        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
        loaded.act(obs,info['action_mask'])
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        np.testing.assert_equal(numpy_state,np.random.get_state())

    def test_explicit_policy_mode_overrides_archive_and_validation_respects_it(self):
        agent, _ = self.agent(deterministic=True)
        path = self.root / 'legacy.checkpoint'
        agent.save_checkpoint(path)
        loaded = PPOAgent(checkpoint=path, deterministic=False)
        self.assertFalse(loaded.deterministic)
        original_rng = copy.deepcopy(loaded.rng.bit_generator.state)
        report = evaluate_agents([loaded,None], (0,), hands_per_opponent=12)
        self.assertFalse(report['agents'][0]['policy']['deterministic'])
        self.assertEqual(original_rng,loaded.rng.bit_generator.state)
        second = evaluate_agents([loaded,None], (0,), hands_per_opponent=12)
        self.assertEqual(report['agents'],second['agents'])
        greedy = evaluate_agents([loaded,None], (0,), hands_per_opponent=12,deterministic=True)
        self.assertTrue(greedy['agents'][0]['policy']['deterministic'])
        self.assertFalse(loaded.deterministic)

    def test_features_expose_hand_structure_and_remove_absolute_seat(self):
        for count in range(2,6):
            game=PokerGame(num_players=count)
            self.addCleanup(game.close)
            game.reset(seed=6)
            feature=PokerFeaturesExtractor(game.observation_space)
            obs=game.observe(0)
            weak=obs.copy()
            strong=obs.copy()
            weak[:52]=strong[:52]=0
            weak[[CARD_INDEX['7C'],CARD_INDEX['2D']]]=1
            strong[[CARD_INDEX['AC'],CARD_INDEX['AD']]]=1
            with torch.no_grad():
                result=feature(torch.tensor(np.stack([weak,strong])))
            self.assertEqual(result.shape,(2,len(obs)+feature.EXTRA_SIZE))
            self.assertTrue(torch.isfinite(result).all())
            self.assertFalse(torch.equal(result[0,len(obs):],result[1,len(obs):]))
            fields=dict(observation_fields(count))
            seat_start=52+52+4
            self.assertTrue((result[:,seat_start:seat_start+count]==0).all())
            # Suit-invariant derived features; raw card identity remains available.
            permuted=strong.copy()
            permuted[:104]=strong[:104].reshape(2,13,4)[...,[2,0,3,1]].reshape(-1)
            with torch.no_grad():
                remapped=feature(torch.tensor(permuted[None,:]))
            torch.testing.assert_close(result[1,len(obs):],remapped[0,len(obs):])

    def test_custom_policy_options_keep_default_poker_features(self):
        agent=PPOAgent(seed=42,autoload=False,n_steps=8,batch_size=4,n_epochs=1,
                       policy_kwargs={'net_arch':{'pi':[16],'vf':[16]}})
        env=PokerEnv(MixedOpponent(seed=43))
        self.addCleanup(env.close)
        agent.learn(env,total_timesteps=8)
        self.assertIsInstance(agent.model.policy.features_extractor,PokerFeaturesExtractor)

    def test_training_reward_scaling_suit_augmentation_and_raw_stats(self):
        registry=AgentRegistry(records_dir=self.root/'records')
        session=TableSession(registry,[('mixed',{'seed':1}),('mixed',{'seed':2})],continuous=False)
        self.addCleanup(session.close)
        env=TableTrainingEnv(session,0)
        obs,_=env.reset(seed=8)
        suits=env._suits.copy()
        np.testing.assert_equal(env._observation(),obs)
        done=False
        while not done:
            action=int(np.flatnonzero(env.action_masks())[0])
            obs,reward,terminated,truncated,_=env.step(action)
            done=terminated or truncated
        self.assertAlmostEqual(reward,float(session.game.rewards()[0])/50)
        np.testing.assert_equal(env._suits,suits)
        self.assertEqual(session.stats.net_bb[0],float(session.game.rewards()[0]))

    def test_mixed_opponents_are_legal_reproducible_and_not_trainable(self):
        registry=AgentRegistry(records_dir=self.root/'records')
        self.assertFalse(registry.agents['mixed'].trainable)
        left,right=MixedOpponent(seed=3),MixedOpponent(seed=3)
        styles=set()
        for seed in range(30):
            left.on_episode_start(seed=seed)
            right.on_episode_start(seed=seed)
            self.assertEqual(left.style,right.style)
            styles.add(left.style)
        self.assertEqual(styles,{'random','check_call','tight','aggressive'})
        self.assertRaises(ValueError,MixedOpponent,styles=('typo',))

    def test_native_training_preserves_old_reward_units_and_records_new_scaling(self):
        registry=AgentRegistry(records_dir=self.root/'records')
        for existing in (False,True):
            agents=None
            if existing:
                old,_=self.agent()
                self.assertEqual(old.model.poker_reward_scale,1.)
                agents=[old,MixedOpponent(seed=43)]
            session=train_sb3_session(registry,
                [('ppo',{'autoload':False,'seed':42,'n_steps':16,'batch_size':8,'n_epochs':1}),('mixed',{'seed':43})],
                training_seats=(0,),total_timesteps=16,validation_hands=4,record=False,agents=agents)
            try:
                self.assertEqual(session.agents[0].model.poker_reward_scale,1. if existing else .02)
            finally:
                session.close()


if __name__=='__main__':
    unittest.main()
