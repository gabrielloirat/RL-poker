
from types import SimpleNamespace

import numpy as np
from gymnasium import spaces

from gym_env.env import HoldemTable
from gym_env.enums import Action, Stage


cards = []
for rank in "23456789TJQKA":
    for suit in "CDHS":
        cards.append(rank + suit)
CARD_ORDER = tuple(cards)
CARD_INDEX = {}
for index, card in enumerate(CARD_ORDER):
    CARD_INDEX[card] = index
actions = []
for index in range(8):
    actions.append(Action(index).name)
ACTION_NAMES = tuple(actions)
OBSERVATION_FIELDS = (
    ("hole_cards", 52), ("board", 52), ("stage", 4),
    ("seat", 2), ("dealer", 2), ("stacks", 2), ("street_bets", 2),
    ("hand_contributions", 2), ("pots", 2), ("to_call", 1),
    ("raises_remaining", 2), ("last_actions", 16), ("action_mask", 8),
)
OBSERVATION_SIZE = 0
for name, size in OBSERVATION_FIELDS:
    OBSERVATION_SIZE += size
OBSERVATION_VERSION = 1


def observation_fields(num_players=2):  # Décrit les champs du réseau

    if num_players == 2:
        return OBSERVATION_FIELDS
    sizes = {"seat": num_players, "dealer": num_players, "stacks": num_players,
             "street_bets": num_players, "hand_contributions": num_players,
             "raises_remaining": num_players, "last_actions": num_players * 8}
    fields = []
    for name, size in OBSERVATION_FIELDS:
        fields.append((name, sizes.get(name, size)))
    fields.append(("active_players", num_players))
    return tuple(fields)


class PokerGame:


    def __init__(self, *, num_players=2, initial_stacks=100, small_blind=1, big_blind=2,
                 max_raises_per_player_round=2, max_actions=200):  # Crée une table de poker
        if type(num_players) is not int or not 2 <= num_players <= 5:
            raise ValueError("num_players must be an integer from 2 to 5.")
        self.num_players = num_players
        self.observation_fields = observation_fields(num_players)
        self.observation_version = 1
        if num_players > 2:
            self.observation_version = 2
        for value in (initial_stacks, small_blind, big_blind):
            if not np.isfinite(value):
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
        size = 0
        for name, length in self.observation_fields:
            size += length
        self.observation_space = spaces.Box(0.0, 1.0, (size,), dtype=np.float32)
        self.max_actions = max_actions
        self._rng = np.random.default_rng()
        self._started = False
        self.terminated = False
        self.truncated = False
        self.action_count = 0
        self.hand_start_stacks = (float(initial_stacks),) * num_players

    @property
    def finished(self):  # Indique si la main est finie
        return self.terminated or self.truncated

    @property
    def current_seat(self):  # Indique le joueur actif
        if not self._started or self.finished:
            return None
        return self.table.current_player.seat

    def reset(self, *, seed=None, options=None):  # Commence une nouvelle partie
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

    def start_next_hand(self, *, seed=None):  # Commence la main suivante

        if not self.terminated or self.truncated:
            raise RuntimeError("Finish and settle the current hand before continuing it.")
        active_players = 0
        for player in self.table.players:
            if player.stack > 0:
                active_players += 1
        if active_players < 2:
            raise RuntimeError("The match is over; reset() starts a new match.")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        stacks = []
        for player in self.table.players:
            stacks.append(float(player.stack))
        self.hand_start_stacks = tuple(stacks)
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
        seat = self.current_seat
        if self.finished:
            seat = self.table.winner_ix
        return self.observe(seat), self.info(seat)

    def action_masks(self, seat=None):  # Repère les actions légales
        mask = np.zeros(8, dtype=np.bool_)
        if seat is None:
            seat = self.current_seat
        if seat == self.current_seat and self.current_seat is not None:
            for action in self.table.legal_moves:
                mask[action.value] = True
        return mask

    def observe(self, seat):  # Construit les données du joueur
        if not self._started:
            raise RuntimeError("Call reset() before requesting an observation.")
        if seat not in range(self.num_players):
            raise ValueError("seat must identify a player.")
        table = self.table
        order = []
        for offset in range(self.num_players):
            order.append((seat + offset) % self.num_players)
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
        dealer = []
        stacks = []
        bets = []
        contributions = []
        raises = []
        street = Stage(min(table.stage.value, 3))
        for player in order:
            dealer.append(table.dealer_pos == player)
            stacks.append(table.players[player].stack / scale)
            bets.append(table.player_pots[player] / scale)
            contributions.append(table.player_max_win[player] / scale)
            used = table.players[player].num_raises_in_street[street]
            raises.append(max(0, table.max_raises_per_player_round - used) / table.max_raises_per_player_round)
        dealer = np.array(dealer, dtype=np.float32)
        pots = [0, 0]
        if not self.terminated:
            pots = [table.community_pot / scale, table.current_round_pot / scale]
        to_call = 0
        if not self.finished:
            missing = table.min_call - table.player_pots[seat]
            to_call = max(0, min(missing, table.players[seat].stack)) / scale
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
            active = []
            for player in order:
                active.append(eligible[player])
            parts += (active,)
        return np.concatenate(parts).astype(np.float32)

    def rewards(self):  # Calcule les gains en blindes
        if not self.terminated:
            return np.zeros(self.num_players, dtype=np.float64)
        rewards = []
        for seat, player in enumerate(self.table.players):
            gain = player.stack - self.hand_start_stacks[seat]
            rewards.append(gain / self.table.big_blind)
        return np.array(rewards, dtype=np.float64)

    def info(self, seat):  # Regroupe les informations de jeu
        winners = ()
        if self.terminated:
            winners = tuple(self.table.winner_indices)
        return {
            "seat": seat, "current_player": self.current_seat,
            "dealer_seat": self.table.dealer_pos,
            "action_mask": self.action_masks(seat),
            "hole_cards": tuple(self.table.players[seat].cards),
            "board": tuple(self.table.table_cards),
            "action_count": self.action_count,
            "net_reward_bb": float(self.rewards()[seat]),
            "winners": winners,
            "observation_version": self.observation_version,
            "num_players": self.num_players,
            "observation_fields": self.observation_fields,
        }

    def step(self, action):  # Joue une action
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
        perspective = self.current_seat
        if self.finished:
            perspective = actor
        info = self.info(perspective)
        info.update(acting_player=actor, illegal_action=not valid)
        return self.observe(perspective), rewards, self.terminated, self.truncated, info

    def render(self):  # Affiche la table
        if self._started:
            self.table.render()

    def close(self):  # Ferme la table
        self.table.close()
