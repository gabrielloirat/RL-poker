"""Opponent-preserving observation padding for neural transfer between table sizes.

No opponents are dropped: all relative-seat fields occupy five slots. Money
fields use the same scale at every table size. MCTS keeps native observations
because its simulator already understands the real number of players.
Absolute hero seat is replaced by a canonical seat; relative position is kept.
"""
import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from poker_rl.game import observation_fields


def slices(players):
    result, offset = {}, 0
    for name, width in observation_fields(players):
        result[name] = slice(offset, offset + width)
        offset += width
    return result, offset


TARGET_FIELDS, TARGET_SIZE = slices(5)
SOURCE_FIELDS = {n: slices(n) for n in range(2, 6)}
MONEY_FIELDS = {'stacks', 'street_bets', 'hand_contributions', 'pots', 'to_call'}


def universal_observation(observation):
    observation = np.asarray(observation, dtype=np.float32)
    players = next((n for n, (_, size) in SOURCE_FIELDS.items() if size == observation.size), None)
    if players is None:
        raise ValueError(f'Observation poker inconnue : {observation.size}')
    source = SOURCE_FIELDS[players][0]
    result = np.zeros(TARGET_SIZE, dtype=np.float32)
    for name, destination in TARGET_FIELDS.items():
        if name == 'seat':
            # Absolute table seat carries no poker information. Relative
            # dealer/opponent slots already encode position; do not expose an
            # unseen one-hot when the same hero is rotated during validation.
            result[destination][0] = 1
        elif name == 'active_players' and players == 2:
            last = observation[source['last_actions']].reshape(players, 8)
            result[destination][:players] = last[:, 0] != 1
        else:
            values = observation[source[name]]
            if name in MONEY_FIELDS:
                values = values * (players / 5.)
            result[destination][:len(values)] = values
    return result


class UniversalEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=1, shape=(TARGET_SIZE,), dtype=np.float32)

    def observation(self, observation):
        return universal_observation(observation)

    def action_masks(self):
        return self.env.action_masks()

    @property
    def reward_scale(self):
        return self.env.reward_scale


class UniversalAgent:
    def __init__(self, agent):
        self.agent = agent

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def act(self, observation, action_mask, *, training=False):
        return self.agent.act(universal_observation(observation), action_mask, training=training)

    def observe(self, transition):
        transition = copy.copy(transition)
        transition.observation = universal_observation(transition.observation)
        transition.next_observation = universal_observation(transition.next_observation)
        self.agent.observe(transition)

    def learn(self, env, **kwargs):
        return self.agent.learn(UniversalEnv(env), **kwargs)

    def evaluation_copy(self):
        method = getattr(self.agent, 'evaluation_copy', None)
        return UniversalAgent(method() if callable(method) else copy.deepcopy(self.agent))
