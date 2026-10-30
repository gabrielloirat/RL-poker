import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Box, Discrete

from gym_env.cycle import PlayerCycle
from gym_env.enums import Action, Stage
from gym_env.rendering import PygletWindow, WHITE, RED, GREEN, BLUE
from tools.hand_evaluator import get_winners
from tools.helper import flatten

log = logging.getLogger(__name__)

MONTEACRLO_RUNS = 1000


class CommunityData:
    def __init__(self, num_players):
        self.current_player_position = [False] * num_players
        self.stage = [False] * 4
        self.community_pot = None
        self.current_round_pot = None
        self.active_players = [False] * num_players
        self.big_blind = 0
        self.small_blind = 0
        self.legal_moves = [0 for action in Action]


class StageData:
    def __init__(self, num_players):
        self.calls = [False] * num_players
        self.raises = [False] * num_players
        self.min_call_at_action = [0] * num_players
        self.contribution = [0] * num_players
        self.stack_at_action = [0] * num_players
        self.community_pot_at_action = [0] * num_players


class PlayerData:
    def __init__(self):
        self.position = None
        self.equity_to_river_alive = 0
        self.equity_to_river_2plr = 0
        self.equity_to_river_3plr = 0
        self.stack = None


class HoldemTable(Env):


    def __init__(self, initial_stacks=100, small_blind=1, big_blind=2, render=False, funds_plot=False,
                 max_raises_per_player_round=2, raise_illegal_moves=False,
                 single_hand=False, compute_equity=True):
        from tools.montecarlo_python import get_equity
        self.get_equity = get_equity
        self._equity_cache = {}
        self.single_hand = single_hand
        self.compute_equity = compute_equity
        self._card_rng = np.random.default_rng()
        self.winner_indices = []
        self.num_of_players = 0
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.render_switch = render
        self.players = []
        self.table_cards = None
        self.dealer_pos = None
        self.player_status = []
        self.current_player = None
        self.player_cycle = None
        self.stage = None
        self.last_player_pot = None
        self.viewer = None
        self.player_max_win = None
        self.round_number_in_street = 0
        self.last_caller = None
        self.last_raiser = None
        self.raisers = []
        self.callers = []
        self.played_in_round = None
        self.min_call = None
        self.community_data = None
        self.player_data = None
        self.stage_data = None
        self.deck = None
        self.action = None
        self.winner_ix = None
        self.initial_stacks = initial_stacks
        self.acting_agent = None
        self.funds_plot = funds_plot
        self.max_raises_per_player_round = max_raises_per_player_round
        self.community_pot = 0
        self.current_round_pot = 0
        self.player_pots = None
        self.observation = None
        self.reward = None
        self.info = None
        self.done = False
        self.funds_history = None
        self.array_everything = None
        self.legal_moves = None
        self.illegal_move_reward = -1
        self.action_space = Discrete(len(Action) - 2)
        self.first_action_for_hand = None
        self.raise_illegal_moves = raise_illegal_moves

    def reset(self, *, seed=None, dealer_start=0):
        if seed is not None:
            self._card_rng = np.random.default_rng(seed)
        self.observation = None
        self.reward = None
        self.info = None
        self.done = False
        self.funds_history = pd.DataFrame()
        self.first_action_for_hand = [True] * len(self.players)
        if len(self.players) < 2:
            raise ValueError("Add at least two players before resetting the environment.")
        if not 0 <= dealer_start < len(self.players):
            raise ValueError("dealer_start must identify a player seat.")
        for player in self.players:
            player.stack = self.initial_stacks
            player.actions.clear()
            player.temp_stack.clear()
            player.equity_alive = 0
        self.current_player = None
        self.winner_ix = None
        self.winner_indices = []
        self.dealer_pos = 0
        self.player_cycle = PlayerCycle(self.players, dealer_idx=dealer_start - 1, max_steps_after_raiser=None,
                                        max_steps_after_big_blind=len(self.players),
                                        max_raises_per_player_round=self.max_raises_per_player_round)
        self._start_new_hand()
        self._get_environment()
        return self.array_everything

    def step(self, action):
        if self.player_cycle is None:
            raise RuntimeError("Reset the environment before taking an action.")
        if self.done:
            raise RuntimeError("The game is over. Reset the environment to play again.")
        self.reward = 0
        self.acting_agent = self.player_cycle.idx
        self._get_legal_moves()
        if Action(action) not in self.legal_moves:
            self._illegal_move(action)
        else:
            self._execute_step(Action(action))
            if self.first_action_for_hand[self.acting_agent] or self.done:
                self.first_action_for_hand[self.acting_agent] = False
                self._calculate_reward(action)
        return self.array_everything, self.reward, self.done, self.info

    def _execute_step(self, action):
        self._process_decision(action)
        self._next_player()
        if self.stage in [Stage.END_HIDDEN, Stage.SHOWDOWN]:
            self._end_hand()
            if self.single_hand:
                self.done = True
            else:
                self._start_new_hand()
        self._get_environment()

    def _illegal_move(self, action):
        if self.raise_illegal_moves:
            raise ValueError(f"{action} is illegal. Allowed: {self.legal_moves}")
        self.reward = self.illegal_move_reward

    def _get_environment(self):
        if not self.current_player:
            self.current_player = self.players[self.winner_ix]
        if not self.done:
            self._get_legal_moves()
        else:
            self.legal_moves = []
        self.observation = None
        self.reward = 0
        self.info = None
        self.community_data = CommunityData(len(self.players))
        self.community_data.community_pot = self.community_pot / (self.big_blind * 100)
        self.community_data.current_round_pot = self.current_round_pot / (self.big_blind * 100)
        self.community_data.small_blind = self.small_blind
        self.community_data.big_blind = self.big_blind
        self.community_data.stage[np.minimum(self.stage.value, 3)] = 1
        self.community_data.legal_moves = [action in self.legal_moves for action in Action]
        self.community_data.current_player_position[self.current_player.seat] = True
        self.community_data.active_players = list(self.player_cycle.alive)
        self.player_data = PlayerData()
        self.player_data.stack = [player.stack / (self.big_blind * 100) for player in self.players]
        self.player_data.position = self.current_player.seat


        alive_count = int(sum(self.player_cycle.alive))
        for player, alive in zip(self.players, self.player_cycle.alive):
            if not self.compute_equity or not alive:
                player.equity_alive = 0.0
            elif alive_count == 1:
                player.equity_alive = 1.0
            else:
                cache_key = (tuple(player.cards), tuple(self.table_cards), alive_count)
                if cache_key not in self._equity_cache:
                    self._equity_cache[cache_key] = self.get_equity(
                        player.cards, self.table_cards, alive_count, MONTEACRLO_RUNS
                    )
                player.equity_alive = self._equity_cache[cache_key]
        self.player_data.equity_to_river_alive = self.current_player.equity_alive
        self.player_data.equity_to_river_2plr = 0.0
        self.player_data.equity_to_river_3plr = 0.0

        arr1 = np.array(list(flatten(self.player_data.__dict__.values())))
        arr2 = np.array(list(flatten(self.community_data.__dict__.values())))
        arr3 = np.array([list(flatten(sd.__dict__.values())) for sd in self.stage_data]).flatten()
        self.array_everything = np.concatenate([arr1, arr2, arr3]).flatten().astype(np.float32)
        self.observation = self.array_everything
        self.info = {
            'player_data': self.player_data.__dict__,
            'community_data': self.community_data.__dict__,
            'stage_data': [stage.__dict__ for stage in self.stage_data],
            'legal_moves': self.legal_moves,
        }
        self.observation_space = Box(-np.inf, np.inf, shape=self.array_everything.shape, dtype=np.float32)
        if self.render_switch:
            self.render()

    def _calculate_reward(self, last_action):
        _ = last_action
        if self.done:
            self.reward = self.initial_stacks * len(self.players)
        elif len(self.funds_history) > 1:
            self.reward = self.funds_history.iloc[-1, self.acting_agent] - self.funds_history.iloc[-2, self.acting_agent]

    def _process_decision(self, action):
        if action not in [Action.SMALL_BLIND, Action.BIG_BLIND]:
            assert action in set(self.legal_moves), "Illegal decision"
        if action == Action.FOLD:
            self.player_cycle.deactivate_current()
            self.player_cycle.mark_folder()
        else:
            if action == Action.CALL:
                contribution = min(self.min_call - self.player_pots[self.current_player.seat], self.current_player.stack)
                self.callers.append(self.current_player.seat)
                self.last_caller = self.current_player.seat
            elif action == Action.CHECK:
                contribution = 0
                self.player_cycle.mark_checker()
            elif action == Action.RAISE_3BB:
                contribution = 3 * self.big_blind - self.player_pots[self.current_player.seat]
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
            elif action == Action.RAISE_HALF_POT:
                contribution = (self.community_pot + self.current_round_pot) / 2
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
            elif action == Action.RAISE_POT:
                contribution = self.community_pot + self.current_round_pot
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
            elif action == Action.RAISE_2POT:
                contribution = (self.community_pot + self.current_round_pot) * 2
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
            elif action == Action.ALL_IN:
                contribution = self.current_player.stack
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
            elif action == Action.SMALL_BLIND:
                contribution = np.minimum(self.small_blind, self.current_player.stack)
            elif action == Action.BIG_BLIND:
                contribution = np.minimum(self.big_blind, self.current_player.stack)
                self.player_cycle.mark_bb()
            else:
                raise RuntimeError("Illegal action")
            total_contribution = self.player_pots[self.current_player.seat] + contribution
            if total_contribution > self.min_call and action not in [Action.BIG_BLIND, Action.SMALL_BLIND]:
                self.player_cycle.mark_raiser()
            self.current_player.stack -= contribution
            self.player_pots[self.current_player.seat] += contribution
            self.current_round_pot += contribution
            self.last_player_pot = self.player_pots[self.current_player.seat]
            if self.current_player.stack == 0 and contribution > 0:
                self.player_cycle.mark_out_of_cash_but_contributed()
            self.min_call = max(self.min_call, total_contribution)
            self.current_player.actions.append(action)
            self.current_player.last_action_in_stage = action.name
            self.current_player.temp_stack.append(self.current_player.stack)
            self.player_max_win[self.current_player.seat] += contribution
            pos = self.player_cycle.idx
            rnd = self.stage.value + self.round_number_in_street
            self.stage_data[rnd].calls[pos] = action == Action.CALL
            self.stage_data[rnd].raises[pos] = action in [Action.RAISE_2POT, Action.RAISE_HALF_POT, Action.RAISE_POT]
            self.stage_data[rnd].min_call_at_action[pos] = self.min_call / (self.big_blind * 100)
            self.stage_data[rnd].community_pot_at_action[pos] = self.community_pot / (self.big_blind * 100)
            self.stage_data[rnd].contribution[pos] += contribution / (self.big_blind * 100)
            self.stage_data[rnd].stack_at_action[pos] = self.current_player.stack / (self.big_blind * 100)
        self.player_cycle.update_alive()

    def _start_new_hand(self):
        self._equity_cache.clear()
        self._save_funds_history()
        if self._check_game_over():
            return
        self.table_cards = []
        self._create_card_deck()
        self.stage = Stage.PREFLOP
        self.stage_data = [StageData(len(self.players)) for _ in range(8)]
        self.community_pot = 0
        self.current_round_pot = 0
        self.player_pots = [0] * len(self.players)
        self.player_max_win = [0] * len(self.players)
        self.last_player_pot = 0
        self.played_in_round = 0
        self.first_action_for_hand = [True] * len(self.players)
        for player in self.players:
            player.cards = []
            player.num_raises_in_street = {
                Stage.PREFLOP: 0, Stage.FLOP: 0, Stage.TURN: 0, Stage.RIVER: 0
            }
        self._next_dealer()
        self._distribute_cards()
        self._initiate_round()

    def _save_funds_history(self):
        funds_dict = {i: player.stack for i, player in enumerate(self.players)}
        self.funds_history = pd.concat([self.funds_history, pd.DataFrame(funds_dict, index=[0])])

    def _check_game_over(self):
        player_alive = []
        self.player_cycle.new_hand_reset()
        for idx, player in enumerate(self.players):
            if player.stack > 0:
                player_alive.append(True)
            else:
                self.player_cycle.deactivate_player(idx)
        self.player_cycle.update_alive()
        if sum(player_alive) < 2:
            self._game_over()
            return True
        return False

    def _game_over(self):
        self.done = True
        player_names = [f"{i} - {player.name}" for i, player in enumerate(self.players)]
        self.funds_history.columns = player_names
        if self.funds_plot:
            self.funds_history.reset_index(drop=True).plot()
            plt.show()

    def _initiate_round(self):
        self.last_caller = None
        self.last_raiser = None
        self.raisers = []
        self.callers = []
        self.min_call = 0
        for player in self.players:
            player.last_action_in_stage = ''
        self.player_cycle.new_street_reset()
        if self.stage == Stage.PREFLOP and sum(self.player_cycle.alive) == 2:

            self.player_cycle.idx -= 1
        if self.stage == Stage.PREFLOP:
            self.player_cycle.max_steps_total = len(self.players) * (
                len(self.players) * self.max_raises_per_player_round + 1
            ) + 2
            self._next_player()
            self._process_decision(Action.SMALL_BLIND)
            self._next_player()
            self._process_decision(Action.BIG_BLIND)
            self._next_player()
        elif self.stage in [Stage.FLOP, Stage.TURN, Stage.RIVER]:
            self.player_cycle.max_steps_total = len(self.players) * (
                len(self.players) * self.max_raises_per_player_round + 1
            )
            self._next_player()
        elif self.stage != Stage.SHOWDOWN:
            raise RuntimeError()

    def add_player(self, player_obj):

        self.num_of_players += 1
        player = PlayerShell(stack_size=self.initial_stacks, name=player_obj.name)
        player.seat = len(self.players)
        player.stack = self.initial_stacks
        self.players.append(player)
        self.player_status = [True] * len(self.players)
        self.player_pots = [0] * len(self.players)

    def _end_round(self):
        self._close_round()
        if self.stage == Stage.PREFLOP:
            self.stage = Stage.FLOP
            self._distribute_cards_to_table(3)
        elif self.stage == Stage.FLOP:
            self.stage = Stage.TURN
            self._distribute_cards_to_table(1)
        elif self.stage == Stage.TURN:
            self.stage = Stage.RIVER
            self._distribute_cards_to_table(1)
        elif self.stage == Stage.RIVER:
            self.stage = Stage.SHOWDOWN
        self._clean_up_pots()

    def _clean_up_pots(self):
        self.community_pot += self.current_round_pot
        self.current_round_pot = 0
        self.player_pots = [0] * len(self.players)

    def _end_hand(self):
        self._clean_up_pots()
        potential = self.player_cycle.get_potential_winners()
        seats = [i for i, eligible in enumerate(potential) if eligible]
        if len(seats) == 1:
            self.winner_indices = seats
        else:
            indices, _ = get_winners([self.players[i].cards for i in seats], self.table_cards)
            self.winner_indices = [seats[i] for i in indices]
        self.winner_ix = self.winner_indices[0]


        previous = 0
        winners = set()
        for level in sorted(set(self.player_max_win) - {0}):
            contributors = [i for i, amount in enumerate(self.player_max_win) if amount >= level]
            delta = level - previous
            previous = level
            eligible = [i for i in contributors if potential[i]]
            if len(contributors) == 1 or not eligible:

                for i in contributors:
                    self.players[i].stack += delta
                continue
            if len(eligible) == 1:
                pot_winners = eligible
            else:
                indices, _ = get_winners([self.players[i].cards for i in eligible], self.table_cards)
                pot_winners = [eligible[i] for i in indices]
            share = delta * len(contributors) / len(pot_winners)
            for i in pot_winners:
                self.players[i].stack += share
            winners.update(pot_winners)
        if winners:
            self.winner_indices = sorted(winners)
            self.winner_ix = self.winner_indices[0]

    def _next_dealer(self):
        self.dealer_pos = self.player_cycle.next_dealer().seat

    def _next_player(self):
        self.current_player = self.player_cycle.next_player()
        if not self.current_player:
            if sum(self.player_cycle.alive) < 2:
                self.stage = Stage.END_HIDDEN
            else:
                self._end_round()
                self._initiate_round()
        elif self.current_player in ['max_steps_total', 'max_steps_after_raiser']:
            self._end_round()

    def _get_legal_moves(self):
        self.legal_moves = []
        if self.done or self.stage in (Stage.END_HIDDEN, Stage.SHOWDOWN):
            return
        if self.player_pots[self.current_player.seat] == max(self.player_pots):
            self.legal_moves.append(Action.CHECK)
        else:
            self.legal_moves.extend([Action.CALL, Action.FOLD])
        if self.current_player.num_raises_in_street[self.stage] < self.max_raises_per_player_round:
            if 3 * self.big_blind > self.min_call and self.current_player.stack >= 3 * self.big_blind - self.player_pots[self.current_player.seat]:
                self.legal_moves.append(Action.RAISE_3BB)
            if self.current_player.stack >= ((self.community_pot + self.current_round_pot) / 2) >= self.min_call:
                self.legal_moves.append(Action.RAISE_HALF_POT)
            if self.current_player.stack >= (self.community_pot + self.current_round_pot) >= self.min_call:
                self.legal_moves.append(Action.RAISE_POT)
            if self.current_player.stack >= ((self.community_pot + self.current_round_pot) * 2) >= self.min_call:
                self.legal_moves.append(Action.RAISE_2POT)
            if self.current_player.stack > 0:
                self.legal_moves.append(Action.ALL_IN)

    def _close_round(self):
        self.played_in_round = 0

    def _create_card_deck(self):
        self.deck = [value + suit for value in "23456789TJQKA" for suit in "CDHS"]

    def _distribute_cards(self):
        for player in self.players:
            player.cards = []
            if player.stack <= 0:
                continue
            for _ in range(2):
                card = self._card_rng.integers(0, len(self.deck))
                player.cards.append(self.deck.pop(card))

    def _distribute_cards_to_table(self, amount_of_cards):
        for _ in range(amount_of_cards):
            card = self._card_rng.integers(0, len(self.deck))
            self.table_cards.append(self.deck.pop(card))

    def render(self, mode='human'):
        if mode != "human":
            return
        screen_width, screen_height, table_radius, face_radius = 600, 400, 200, 10
        if self.viewer is None:
            self.viewer = PygletWindow(screen_width + 50, screen_height + 50)
        self.viewer.reset()
        self.viewer.circle(screen_width / 2, screen_height / 2, table_radius, color=BLUE, thickness=0)
        for i in range(len(self.players)):
            degrees = i * (360 / len(self.players))
            radian = degrees * (np.pi / 180)
            x = (face_radius + table_radius) * np.cos(radian) + screen_width / 2
            y = (face_radius + table_radius) * np.sin(radian) + screen_height / 2
            color = GREEN if self.player_cycle.alive[i] else RED
            self.viewer.circle(x, y, face_radius, color=color, thickness=2)
            if self.current_player and i == self.current_player.seat:
                self.viewer.rectangle(x - 60, y, 150, -50, (255, 0, 0, 10))
            self.viewer.text(self.players[i].name, x - 60, y - 15, font_size=10, color=WHITE)
            self.viewer.text(f"Player {self.players[i].seat}: {self.players[i].cards}", x - 60, y,
                             font_size=10, color=WHITE)
            stack_label = f"${self.players[i].stack}"
            if self.compute_equity:
                equity_alive = int(round(float(self.players[i].equity_alive) * 100))
                stack_label += f" (EQ: {equity_alive}%)"
            self.viewer.text(stack_label, x - 60, y + 15,
                             font_size=10, color=WHITE)
            self.viewer.text(self.players[i].last_action_in_stage, x - 60, y + 30, font_size=10, color=WHITE)
            x_inner = (-face_radius + table_radius - 60) * np.cos(radian) + screen_width / 2
            y_inner = (-face_radius + table_radius - 60) * np.sin(radian) + screen_height / 2
            self.viewer.text(f"${self.player_pots[i]}", x_inner, y_inner + 60, font_size=10, color=WHITE)
            self.viewer.text(f"{self.table_cards}", screen_width / 2 - 40, screen_height / 2,
                             font_size=10, color=WHITE)
            self.viewer.text(f"${self.community_pot}", screen_width / 2 - 15, screen_height / 2 + 30,
                             font_size=10, color=WHITE)
            self.viewer.text(f"${self.current_round_pot}", screen_width / 2 - 15, screen_height / 2 + 50,
                             font_size=10, color=WHITE)
            x_button = (-face_radius + table_radius - 20) * np.cos(radian) + screen_width / 2
            y_button = (-face_radius + table_radius - 20) * np.sin(radian) + screen_height / 2
            if i == self.player_cycle.dealer_idx:
                self.viewer.circle(x_button, y_button, 5, color=BLUE, thickness=2)
        self.viewer.update()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class PlayerShell:
    def __init__(self, stack_size, name):
        self.stack = stack_size
        self.seat = None
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.cards = None
        self.num_raises_in_street = {
            Stage.PREFLOP: 0, Stage.FLOP: 0, Stage.TURN: 0, Stage.RIVER: 0
        }

    def __repr__(self):
        return f"Player {self.name} at seat {self.seat} with stack of {self.stack} and cards {self.cards}"
