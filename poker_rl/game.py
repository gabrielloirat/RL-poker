from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from gymnasium import spaces

from gym_env.env import HoldemTable
from gym_env.enums import Action, Stage


CARD_ORDER = tuple(rank + suit for rank in "23456789TJQKA" for suit in "CDHS")
CARD_INDEX = {card: i for i, card in enumerate(CARD_ORDER)}
ACTION_NAMES = tuple(Action(i).name for i in range(8))
OBSERVATION_FIELDS = (
    ("hole_cards", 52), ("board", 52), ("stage", 4),
    ("seat", 2), ("dealer", 2), ("stacks", 2), ("street_bets", 2),
    ("hand_contributions", 2), ("pots", 2), ("to_call", 1),
    ("raises_remaining", 2), ("last_actions", 16), ("action_mask", 8),
)
OBSERVATION_SIZE = sum(size for _, size in OBSERVATION_FIELDS)
OBSERVATION_VERSION = 1


def observation_fields(num_players=2):

    if num_players == 2:
        return OBSERVATION_FIELDS
    sizes = {"seat": num_players, "dealer": num_players, "stacks": num_players,
             "street_bets": num_players, "hand_contributions": num_players,
             "raises_remaining": num_players, "last_actions": num_players * 8}
    return tuple((name, sizes.get(name, size)) for name, size in OBSERVATION_FIELDS) + (("active_players", num_players),)


class PokerGame:


    def __init__(self, *, num_players=2, initial_stacks=100, small_blind=1, big_blind=2,
                 max_raises_per_player_round=2, max_actions=200):
        if type(num_players) is not int or not 2 <= num_players <= 5:
            raise ValueError("num_players must be an integer from 2 to 5.")
        self.num_players = num_players
        self.observation_fields = observation_fields(num_players)
        self.observation_version = 1 if num_players == 2 else 2
        if not all(np.isfinite(v) for v in (initial_stacks, small_blind, big_blind)):
            raise ValueError("Stacks and blinds must be finite.")
        if not 0 < small_blind <= big_blind < initial_stacks:
            raise ValueError("Require 0 < small_blind <= big_blind < initial_stacks.")
        if (not isinstance(max_actions, int) or max_actions < 1
                or not isinstance(max_raises_per_player_round, int)
                or max_raises_per_player_round < 1):
            raise ValueError("max_actions and max_raises_per_player_round must be positive integers.")
        self.table = HoldemTable(
            initial_stacks=initial_stacks, small_blind=small_blind, big_blind=big_blind,
            max_raises_per_player_round=max_raises_per_player_round,
            single_hand=True, compute_equity=False,
        )
        for seat in range(num_players):
            self.table.add_player(SimpleNamespace(name=f"Seat {seat}"))
        self.action_space = spaces.Discrete(8)
        size = sum(size for _, size in self.observation_fields)
        self.observation_space = spaces.Box(0.0, 1.0, (size,), dtype=np.float32)
        self.max_actions = max_actions
        self._rng = np.random.default_rng()
        self._started = False
        self.terminated = False
        self.truncated = False
        self.action_count = 0
        self.hand_start_stacks = (float(initial_stacks),) * num_players

    @property
    def finished(self):
        return self.terminated or self.truncated

    @property
    def current_seat(self):
        if not self._started or self.finished:
            return None
        return self.table.current_player.seat

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        options = options or {}
        dealer = options.get("dealer_seat", int(self._rng.integers(self.num_players)))
        if dealer not in range(self.num_players):
            raise ValueError("dealer_seat must identify an active seat.")
        self.table.reset(seed=int(self._rng.integers(2**32)), dealer_start=dealer)
        self.hand_start_stacks = (float(self.table.initial_stacks),) * self.num_players
        self._started = True
        self.terminated = self.truncated = False
        self.action_count = 0
        return self.observe(self.current_seat), self.info(self.current_seat)

    def start_next_hand(self, *, seed=None):

        if not self.terminated or self.truncated:
            raise RuntimeError("Finish and settle the current hand before continuing it.")
        if sum(player.stack > 0 for player in self.table.players) < 2:
            raise RuntimeError("The match is over; reset() starts a new match.")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        self.hand_start_stacks = tuple(float(player.stack) for player in self.table.players)
        self.table._card_rng = np.random.default_rng(int(self._rng.integers(2**32)))
        self.table.done = False
        self.table.current_player = None
        self.table.winner_ix = None
        self.table.winner_indices = []
        self.table._start_new_hand()

        if self.table.stage in (Stage.END_HIDDEN, Stage.SHOWDOWN):
            self.table._end_hand()
            self.table.done = True
        self.table._get_environment()
        self.terminated = bool(self.table.done)
        self.truncated = False
        self.action_count = 0
        seat = self.table.winner_ix if self.finished else self.current_seat
        return self.observe(seat), self.info(seat)

    def action_masks(self, seat=None):
        mask = np.zeros(8, dtype=np.bool_)
        seat = self.current_seat if seat is None else seat
        if seat == self.current_seat and self.current_seat is not None:
            for action in self.table.legal_moves:
                mask[action.value] = True
        return mask

    def observe(self, seat):
        if not self._started:
            raise RuntimeError("Call reset() before requesting an observation.")
        if seat not in range(self.num_players):
            raise ValueError("seat must identify a player.")
        table = self.table
        order = tuple((seat + offset) % self.num_players for offset in range(self.num_players))
        scale = float(table.initial_stacks * self.num_players)
        cards = np.zeros(52, dtype=np.float32)
        board = np.zeros(52, dtype=np.float32)
        for card in table.players[seat].cards:
            cards[CARD_INDEX[card]] = 1
        for card in table.table_cards:
            board[CARD_INDEX[card]] = 1
        stage = np.zeros(4, dtype=np.float32)
        stage[min(table.stage.value, 3)] = 1
        position = np.eye(self.num_players, dtype=np.float32)[seat]
        dealer = np.array([table.dealer_pos == i for i in order], dtype=np.float32)
        stacks = [table.players[i].stack / scale for i in order]
        bets = [table.player_pots[i] / scale for i in order]
        contributions = [table.player_max_win[i] / scale for i in order]

        pots = [0, 0] if self.terminated else [table.community_pot / scale, table.current_round_pot / scale]
        to_call = 0 if self.finished else max(0, min(
            table.min_call - table.player_pots[seat], table.players[seat].stack
        )) / scale
        raises = [max(0, table.max_raises_per_player_round - table.players[i].num_raises_in_street[
            Stage(min(table.stage.value, 3))
        ]) / table.max_raises_per_player_round for i in order]
        last_actions = np.zeros((self.num_players, 8), dtype=np.float32)
        for relative_seat, i in enumerate(order):
            name = table.players[i].last_action_in_stage
            if name in ACTION_NAMES:
                last_actions[relative_seat, ACTION_NAMES.index(name)] = 1
        parts = (cards, board, stage, position, dealer, stacks, bets,
                               contributions, pots, [to_call], raises,
                               last_actions.ravel(), self.action_masks(seat))
        if self.num_players > 2:
            eligible = table.player_cycle.get_potential_winners()
            parts += ([eligible[i] for i in order],)
        return np.concatenate(parts).astype(np.float32)

    def rewards(self):
        if not self.terminated:
            return np.zeros(self.num_players, dtype=np.float64)
        return np.array([(p.stack - self.hand_start_stacks[seat]) / self.table.big_blind
                         for seat, p in enumerate(self.table.players)], dtype=np.float64)

    def info(self, seat):
        return {
            "seat": seat, "current_player": self.current_seat,
            "dealer_seat": self.table.dealer_pos,
            "action_mask": self.action_masks(seat),
            "hole_cards": tuple(self.table.players[seat].cards),
            "board": tuple(self.table.table_cards),
            "action_count": self.action_count,
            "net_reward_bb": float(self.rewards()[seat]),
            "winners": tuple(self.table.winner_indices) if self.terminated else (),
            "observation_version": self.observation_version,
            "num_players": self.num_players,
            "observation_fields": self.observation_fields,
        }

    def step(self, action):
        if not self._started or self.finished:
            raise RuntimeError("Call reset() before stepping a new hand.")
        if isinstance(action, Action):
            action = action.value
        if not self.action_space.contains(action):
            raise ValueError("Action must be an integer from 0 to 7.")
        actor = self.current_seat
        valid = bool(self.action_masks()[int(action)])
        self.action_count += 1
        if valid:
            self.table.step(int(action))
        self.terminated = bool(self.table.done)
        self.truncated = self.action_count >= self.max_actions and not self.terminated
        rewards = self.rewards()
        if not valid:
            rewards[actor] = -1.0
        perspective = actor if self.finished else self.current_seat
        info = self.info(perspective)
        info.update(acting_player=actor, illegal_action=not valid)
        return self.observe(perspective), rewards, self.terminated, self.truncated, info

    def render(self):
        if self._started:
            self.table.render()

    def close(self):
        self.table.close()
