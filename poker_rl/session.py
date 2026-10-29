
import time

import numpy as np

from poker_rl.game import PokerGame
from poker_rl.interface import Transition
from poker_rl.records import snapshot


class EpisodeResult:
    def __init__(self,
                 episode, rewards_bb, winners,
                 terminated, truncated, actions,
                 illegal_actions):  # Enregistre les données
        self.episode = episode
        self.rewards_bb = rewards_bb
        self.winners = winners
        self.terminated = terminated
        self.truncated = truncated
        self.actions = actions
        self.illegal_actions = illegal_actions

    def to_dict(self):  # Prépare les données à sauvegarder
        return {
            "episode": self.episode,
            "rewards_bb": self.rewards_bb,
            "winners": self.winners,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "actions": self.actions,
            "illegal_actions": self.illegal_actions,
        }

    def __eq__(self, other):  # Compare les valeurs
        if not isinstance(other, EpisodeResult):
            return False
        return self.__dict__ == other.__dict__


class _Pending:
    def __init__(self,
                 observation, action_mask, action,
                 reward=0.0):  # Enregistre les données
        self.observation = observation
        self.action_mask = action_mask
        self.action = action
        self.reward = reward


def run_session(agents, *, episodes=100, training_seats=(),
                seed=0, game_kwargs=None, render=False, delay=0.0,
                on_episode_end = None, recorder=None):


    if not 2 <= len(agents) <= 5 or not all(callable(getattr(a, "act", None)) for a in agents):
        raise ValueError("Supply 2..5 agents with an act() method.")
    training_seats = set(training_seats)
    if not training_seats.issubset(range(len(agents))):
        raise ValueError("training_seats must identify supplied agents.")
    for seat in training_seats:
        if getattr(agents[seat], "native_training", False):
            raise ValueError("PPO utilise model.learn() : lancez train_sb3_session ou run_agents.py train --timesteps N.")
        if not callable(getattr(agents[seat], "observe", None)):
            raise TypeError(f"Training agent at seat {seat} must provide observe(transition).")
    if not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer.")
    if not np.isfinite(delay) or not 0 <= delay <= 60:
        raise ValueError("delay must be between 0 and 60 seconds.")
    rng = np.random.default_rng(seed)
    settings = dict(game_kwargs or {})
    if settings.get("num_players", len(agents)) != len(agents):
        raise ValueError("num_players must match the number of agents.")
    settings["num_players"] = len(agents)
    game = PokerGame(**settings)
    results = []
    completed_run = False

    def deliver(seat, pending):
        if seat in training_seats:
            agents[seat].observe(Transition(
                observation=pending.observation.copy(), action_mask=pending.action_mask.copy(),
                action=pending.action, reward=pending.reward,
                next_observation=game.observe(seat), next_action_mask=game.action_masks(seat),
                terminated=game.terminated, truncated=game.truncated, info=game.info(seat),
            ))

    try:
        for episode in range(episodes):
            hand_seed = int(rng.integers(2**32))
            for seat, agent in enumerate(agents):
                callback = getattr(agent, "on_episode_start", None)
                if callable(callback):
                    callback(seed=int(rng.integers(2**32)), training=seat in training_seats)
            game.reset(seed=hand_seed)
            if recorder:
                recorder.hand_start(game, hand=episode + 1, match=episode + 1, seed=hand_seed)
            pending = [None] * len(agents)
            illegal_actions = 0
            if render:
                game.render()
            while not game.finished:
                seat = game.current_seat
                if pending[seat] is not None:
                    deliver(seat, pending[seat])
                    pending[seat] = None
                observation = game.observe(seat)
                mask = game.action_masks(seat)
                action = agents[seat].act(observation.copy(), mask.copy(), training=seat in training_seats)

                before = snapshot(game) if recorder else None
                _, rewards, _, _, info = game.step(action)
                action_value = int(getattr(action, "value", action))
                if recorder:
                    recorder.action(before, game, action_value, illegal=info["illegal_action"])
                pending[seat] = _Pending(observation, mask, action_value)
                illegal_actions += int(info["illegal_action"])
                for player, decision in enumerate(pending):
                    if decision is not None:
                        decision.reward += float(rewards[player])
                if render:
                    game.render()
                    if game.table.viewer and not game.table.viewer.active:
                        return results
                    if delay:
                        time.sleep(delay)
            for seat, decision in enumerate(pending):
                if decision is not None:
                    deliver(seat, decision)
            rewards = tuple(float(r) for r in game.rewards())
            result = EpisodeResult(
                episode=episode, rewards_bb=rewards, winners=tuple(game.table.winner_indices)
                if game.terminated else (), terminated=game.terminated, truncated=game.truncated,
                actions=game.action_count, illegal_actions=illegal_actions,
            )
            results.append(result)
            for seat, agent in enumerate(agents):
                callback = getattr(agent, "on_episode_end", None)
                if callable(callback):
                    callback(reward=rewards[seat], terminated=game.terminated,
                             truncated=game.truncated, training=seat in training_seats)
            if on_episode_end:
                on_episode_end(result)
            if recorder:
                recorder.hand_end(game)
        completed_run = True
        return results
    finally:
        try:
            if recorder:
                recorder.finish({"hands": len(results), "completed": sum(r.terminated for r in results),
                    "truncated": sum(r.truncated for r in results),
                    "illegal_actions": sum(r.illegal_actions for r in results),
                    "net_bb": [sum(r.rewards_bb[seat] for r in results) for seat in range(len(agents))]},
                    agents=agents, status="completed" if completed_run else "interrupted")
        finally:
            game.close()
