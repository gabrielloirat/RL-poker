
import gymnasium as gym

from poker_rl.game import PokerGame
from poker_rl.interface import load_agent


class PokerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, opponent, *, agent_seat=0, render_mode=None,
                 agents_dir=None, opponent_kwargs=None, **game_kwargs):
        super().__init__()
        if game_kwargs.get("num_players", 2) != 2:
            raise ValueError("PokerEnv supports two players; use PokerGame for 2..5 players.")
        if agent_seat not in (0, 1):
            raise ValueError("agent_seat must be 0 or 1.")
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'.")
        if isinstance(opponent, str):
            opponent = load_agent(opponent, opponent_kwargs, agents_dir=agents_dir)
        elif opponent_kwargs is not None:
            raise ValueError("opponent_kwargs requires an opponent name or module:factory.")
        if not callable(getattr(opponent, "act", None)):
            raise TypeError("Supply an opponent with act(observation, action_mask, *, training=False).")
        self.opponent = opponent
        self.agent_seat = agent_seat
        self.render_mode = render_mode
        self.game = PokerGame(**game_kwargs)
        self.observation_space = self.game.observation_space
        self.action_space = self.game.action_space
        self._reset_required = True

    def action_masks(self):
        return self.game.action_masks(self.agent_seat)

    def _play_opponent(self):
        while not self.game.finished and self.game.current_seat != self.agent_seat:
            seat = self.game.current_seat
            mask = self.game.action_masks(seat)
            action = self.opponent.act(self.game.observe(seat), mask.copy(), training=False)
            if not self.action_space.contains(action) or not mask[int(action)]:
                raise ValueError("The supplied opponent returned an illegal action.")
            self.game.step(action)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(2**32))
        callback = getattr(self.opponent, "on_episode_start", None)
        if callable(callback):
            callback(seed=game_seed, training=False)


        game_options = {"dealer_seat": self.agent_seat, **(options or {})}
        self.game.reset(seed=game_seed, options=game_options)
        self._play_opponent()
        if self.game.finished:
            raise ValueError("Opponent ended the hand before the learner's first decision; use dealer_seat=agent_seat.")
        self._reset_required = False
        if self.render_mode == "human":
            self.render()
        return self.game.observe(self.agent_seat), self.game.info(self.agent_seat)

    def step(self, action):
        if self._reset_required:
            raise RuntimeError("Call reset() before stepping a new episode.")
        _, rewards, terminated, truncated, step_info = self.game.step(action)
        penalty = 0.0
        if step_info["illegal_action"]:
            penalty = float(rewards[self.agent_seat])
        self._play_opponent()
        terminated, truncated = self.game.terminated, self.game.truncated
        reward = float(self.game.rewards()[self.agent_seat]) + penalty
        info = self.game.info(self.agent_seat)
        info["illegal_action"] = step_info["illegal_action"]
        self._reset_required = terminated or truncated
        if self.render_mode == "human":
            self.render()
        return self.game.observe(self.agent_seat), reward, terminated, truncated, info

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()
