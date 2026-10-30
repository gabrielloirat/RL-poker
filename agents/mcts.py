from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np

from poker_rl import CARD_ORDER, observation_fields
from tools.hand_evaluator import _calc_score, get_winners

CARD_SET = set(CARD_ORDER)
ACTION_NAMES = (
    "FOLD",
    "CHECK",
    "CALL",
    "RAISE_3BB",
    "RAISE_HALF_POT",
    "RAISE_POT",
    "RAISE_2POT",
    "ALL_IN",
)

HAND_TYPE_SCORES = {
    "HighCard": 0.12,
    "Pair": 0.38,
    "TwoPair": 0.62,
    "ThreeOfAKind": 0.76,
    "Straight": 0.86,
    "Flush": 0.90,
    "FullHouse": 0.94,
    "FourOfAKind": 0.98,
    "StraightFlush": 1.0,
}


def evaluate_hand_strength(hole_cards: Sequence[str], board_cards: Sequence[str]) -> float:
    if not hole_cards or len(hole_cards) < 2:
        return 0.20

    if len(board_cards) < 3:
        c1, c2 = hole_cards[0], hole_cards[1]
        ranks = "23456789TJQKA"
        r1, s1 = ranks.index(c1[0]), c1[1]
        r2, s2 = ranks.index(c2[0]), c2[1]
        high_r, low_r = max(r1, r2), min(r1, r2)
        if r1 == r2:
            score = 0.50 + 0.50 * (high_r / 12.0)
        else:
            score = 0.22 * (high_r / 12.0) + 0.16 * (low_r / 12.0)
            if s1 == s2:
                score += 0.08
            if high_r - low_r == 1:
                score += 0.05
        return float(np.clip(score, 0.05, 1.0))

    score_tuple, rank_tuple, hand_type = _calc_score(list(hole_cards) + list(board_cards))
    base = HAND_TYPE_SCORES.get(hand_type, 0.20)
    rank_bonus = 0.0
    if rank_tuple and len(rank_tuple) > 0 and isinstance(rank_tuple[0], (int, float)):
        rank_bonus = (max(0, rank_tuple[0]) / 12.0) * 0.14
    return float(np.clip(base + rank_bonus, 0.05, 1.0))


class SimState:

    def __init__(
        self,
        num_players: int,
        hero_seat: int,
        current_player: int,
        dealer_pos: int,
        stage: int,
        stacks: list[float],
        player_pots: list[float],
        player_max_win: list[float],
        community_pot: float,
        current_round_pot: float,
        min_call: float,
        raises_remaining: list[int],
        hole_cards: list[list[str]],
        board: list[str],
        deck: list[str],
        active: list[bool],
        start_stacks: list[float],
        acted_in_street: list[bool] | None = None,
        big_blind: float = 2.0,
        small_blind: float = 1.0,
        max_raises_per_round: int = 2,
    ):
        self.num_players = num_players
        self.hero_seat = hero_seat
        self.current_player = current_player
        self.dealer_pos = dealer_pos
        self.stage = stage
        self.stacks = [float(s) for s in stacks]
        self.player_pots = [float(p) for p in player_pots]
        self.player_max_win = [float(m) for m in player_max_win]
        self.community_pot = float(community_pot)
        self.current_round_pot = float(current_round_pot)
        self.min_call = float(min_call)
        self.raises_remaining = list(raises_remaining)
        self.hole_cards = [list(c) for c in hole_cards]
        self.board = list(board)
        self.deck = list(deck)
        self.active = list(active)
        self.start_stacks = [float(s) for s in start_stacks]
        self.acted_in_street = list(acted_in_street) if acted_in_street else [False] * num_players
        self.big_blind = big_blind
        self.small_blind = small_blind
        self.max_raises_per_round = max_raises_per_round
        self.terminal = False

    def is_terminal(self) -> bool:
        if self.terminal:
            return True
        if sum(self.active) <= 1:
            self.terminal = True
            return True
        if self.stage >= 4:
            self.terminal = True
            return True
        return False

    def get_legal_actions(self, seat: int | None = None) -> list[int]:
        if self.is_terminal():
            return []
        p = self.current_player if seat is None else seat
        if not self.active[p] or self.stacks[p] <= 0:
            return []

        legal = []
        max_pot = max(self.player_pots)
        if self.player_pots[p] == max_pot:
            legal.append(1)
        else:
            legal.extend([0, 2])

        if self.raises_remaining[p] > 0:
            pot_now = self.community_pot + self.current_round_pot
            needed_3bb = 3 * self.big_blind - self.player_pots[p]
            if 3 * self.big_blind > self.min_call and self.stacks[p] >= needed_3bb:
                legal.append(3)
            if self.stacks[p] >= (pot_now / 2.0) >= self.min_call:
                legal.append(4)  # RAISE_HALF_POT
            if self.stacks[p] >= pot_now >= self.min_call:
                legal.append(5)  # RAISE_POT
            if self.stacks[p] >= (pot_now * 2.0) >= self.min_call:
                legal.append(6)  # RAISE_2POT
            if self.stacks[p] > 0:
                legal.append(7)  # ALL_IN

        return legal

    def apply_action(self, action: int, rng: random.Random) -> None:
        p = self.current_player
        if action == 0:  # FOLD
            self.active[p] = False
            self.acted_in_street[p] = True
            if sum(self.active) <= 1:
                self.terminal = True
                self._settle_hand()
                return
        elif action == 1:  # CHECK
            self.acted_in_street[p] = True
        elif action == 2:  # CALL
            needed = max(0.0, self.min_call - self.player_pots[p])
            contrib = min(needed, self.stacks[p])
            self.stacks[p] -= contrib
            self.player_pots[p] += contrib
            self.player_max_win[p] += contrib
            self.current_round_pot += contrib
            self.acted_in_street[p] = True
        elif action in (3, 4, 5, 6, 7):  # RELANCES
            pot_now = self.community_pot + self.current_round_pot
            if action == 3:
                contrib = 3 * self.big_blind - self.player_pots[p]
            elif action == 4:
                contrib = pot_now / 2.0
            elif action == 5:
                contrib = pot_now
            elif action == 6:
                contrib = pot_now * 2.0
            else:  # ALL_IN
                contrib = self.stacks[p]

            contrib = min(contrib, self.stacks[p])
            self.stacks[p] -= contrib
            self.player_pots[p] += contrib
            self.player_max_win[p] += contrib
            self.current_round_pot += contrib

            total_contrib = self.player_pots[p]
            if total_contrib > self.min_call:
                self.min_call = total_contrib
                # Réinitialiser acted pour les autres joueurs actifs avec jetons
                for other in range(self.num_players):
                    if other != p and self.active[other] and self.stacks[other] > 0:
                        self.acted_in_street[other] = False

            self.raises_remaining[p] = max(0, self.raises_remaining[p] - 1)
            self.acted_in_street[p] = True

        self._advance_turn(rng)

    def _advance_turn(self, rng: random.Random) -> None:
        if self.is_terminal():
            self._settle_hand()
            return

        can_act = [i for i in range(self.num_players) if self.active[i] and self.stacks[i] > 0]
        if len(can_act) == 0:
            # Tous les joueurs restants sont à tapis
            self._runout_board(rng)
            self.terminal = True
            self._settle_hand()
            return

        if len(can_act) == 1:
            solo = can_act[0]
            max_p = max(self.player_pots)
            if self.acted_in_street[solo] and self.player_pots[solo] == max_p:
                self._runout_board(rng)
                self.terminal = True
                self._settle_hand()
                return

        max_p = max(self.player_pots)
        street_complete = all(self.acted_in_street[i] and self.player_pots[i] == max_p for i in can_act)
        if street_complete:
            self._advance_street(rng)
        else:
            curr = self.current_player
            for offset in range(1, self.num_players + 1):
                nxt = (curr + offset) % self.num_players
                if nxt in can_act and (not self.acted_in_street[nxt] or self.player_pots[nxt] < max_p):
                    self.current_player = nxt
                    return
            self._advance_street(rng)

    def _advance_street(self, rng: random.Random) -> None:
        self.community_pot += self.current_round_pot
        self.current_round_pot = 0.0
        self.player_pots = [0.0] * self.num_players
        self.min_call = 0.0
        self.raises_remaining = [self.max_raises_per_round] * self.num_players
        self.acted_in_street = [False] * self.num_players

        if self.stage == 0:  # PREFLOP -> FLOP
            self.stage = 1
            for _ in range(3):
                if self.deck:
                    self.board.append(self.deck.pop())
        elif self.stage == 1:  # FLOP -> TURN
            self.stage = 2
            if self.deck:
                self.board.append(self.deck.pop())
        elif self.stage == 2:  # TURN -> RIVER
            self.stage = 3
            if self.deck:
                self.board.append(self.deck.pop())
        elif self.stage >= 3:  # RIVER -> SHOWDOWN
            self.stage = 4
            self.terminal = True
            self._settle_hand()
            return

        can_act = [i for i in range(self.num_players) if self.active[i] and self.stacks[i] > 0]
        if len(can_act) <= 1:
            self._runout_board(rng)
            self.terminal = True
            self._settle_hand()
            return

        # Le premier joueur à parler postflop est le premier joueur actif à gauche du dealer
        for offset in range(1, self.num_players + 1):
            cand = (self.dealer_pos + offset) % self.num_players
            if cand in can_act:
                self.current_player = cand
                return
        self.current_player = can_act[0]

    def _runout_board(self, rng: random.Random) -> None:
        needed = 5 - len(self.board)
        for _ in range(needed):
            if self.deck:
                self.board.append(self.deck.pop())
        self.stage = 4

    def _settle_hand(self) -> None:
        potential = list(self.active)
        if sum(potential) <= 1:
            winner = next((i for i, alive in enumerate(potential) if alive), None)
            if winner is not None:
                self.stacks[winner] += self.community_pot + self.current_round_pot
            return

        self.community_pot += self.current_round_pot
        self.current_round_pot = 0.0

        previous = 0.0
        for level in sorted(set(self.player_max_win) - {0.0}):
            contributors = [i for i, amount in enumerate(self.player_max_win) if amount >= level]
            delta = level - previous
            previous = level
            eligible = [i for i in contributors if potential[i]]
            if len(contributors) == 1 or not eligible:
                for i in contributors:
                    self.stacks[i] += delta
                continue
            if len(eligible) == 1:
                pot_winners = eligible
            else:
                indices, _ = get_winners([self.hole_cards[i] for i in eligible], self.board)
                pot_winners = [eligible[i] for i in indices]
            share = (delta * len(contributors)) / float(len(pot_winners))
            for i in pot_winners:
                self.stacks[i] += share

    def get_terminal_rewards_bb(self) -> list[float]:
        return [(self.stacks[i] - self.start_stacks[i]) / self.big_blind for i in range(self.num_players)]


class MCTSPolicy:
    """Politique de poker pour l'estimation des priors d'actions et les rollouts."""

    def __init__(self, learning_rate: float = 0.05, discount_factor: float = 0.99):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.weights = np.zeros((32, 8), dtype=np.float32)

    def get_bucket(self, stage: int, strength: float, facing_bet: bool) -> int:
        stage_idx = min(3, max(0, stage))
        if strength < 0.35:
            bucket_str = 0
        elif strength < 0.60:
            bucket_str = 1
        elif strength < 0.80:
            bucket_str = 2
        else:
            bucket_str = 3
        facing_idx = 1 if facing_bet else 0
        return stage_idx * 8 + bucket_str * 2 + facing_idx

    def get_action_priors(self, state: SimState, seat: int, legal_actions: list[int]) -> dict[int, float]:
        if not legal_actions:
            return {}
        if len(legal_actions) == 1:
            return {legal_actions[0]: 1.0}

        cards = state.hole_cards[seat]
        strength = evaluate_hand_strength(cards, state.board)
        to_call = max(0.0, state.min_call - state.player_pots[seat])
        facing_bet = to_call > 0.0
        pot = max(1.0, state.community_pot + state.current_round_pot)
        pot_odds = to_call / (pot + to_call) if facing_bet else 0.0
        stack = max(1.0, state.stacks[seat])
        stack_commitment = to_call / (stack + to_call)

        priors: dict[int, float] = {}
        for a in legal_actions:
            if a == 0:  # FOLD
                if facing_bet:
                    if stack_commitment > 0.40 or to_call > 2.0 * pot:
                        priors[a] = 5.0 if strength < 0.65 else 0.10
                    elif strength < pot_odds:
                        priors[a] = 2.5
                    else:
                        priors[a] = max(0.05, 1.0 - strength)
                else:
                    priors[a] = 0.001
            elif a == 1:  # CHECK
                priors[a] = 2.0 if strength < 0.65 else 0.50
            elif a == 2:  # CALL
                if facing_bet:
                    if stack_commitment > 0.40 or to_call > 2.0 * pot:
                        priors[a] = 3.0 if strength >= 0.65 else 0.05
                    elif strength >= pot_odds:
                        priors[a] = 2.0
                    else:
                        priors[a] = 0.30
                else:
                    priors[a] = 0.10
            elif a in (3, 4, 5, 6):
                if strength >= 0.70:
                    priors[a] = 2.5
                elif strength >= 0.55:
                    priors[a] = 1.0
                else:
                    priors[a] = 0.10
            elif a == 7:  # ALL_IN
                if strength >= 0.85:
                    priors[a] = 2.0
                elif strength >= 0.70 and stack < pot * 0.5:
                    priors[a] = 1.5
                else:
                    priors[a] = 0.05

        bucket = self.get_bucket(state.stage, strength, facing_bet)
        factor_provider = getattr(self, '_benchmark_weight_factors', None)
        factors = factor_provider(bucket) if callable(factor_provider) else None
        logits = np.array(
            [priors.get(a, 0.1) * (factors[a] if factors is not None else
                np.exp(np.clip(self.weights[bucket, a], -3.0, 3.0))) for a in legal_actions],
            dtype=np.float64,
        )
        total = np.sum(logits)
        if total <= 0 or not np.isfinite(total):
            prob = 1.0 / len(legal_actions)
            return {a: prob for a in legal_actions}
        probs = logits / total
        return {a: float(probs[i]) for i, a in enumerate(legal_actions)}

    def get_rollout_action(self, state: SimState, rng: random.Random) -> int:
        legal = state.get_legal_actions()
        if not legal:
            return 0
        if len(legal) == 1:
            return legal[0]

        p = state.current_player
        cards = state.hole_cards[p]
        strength = evaluate_hand_strength(cards, state.board)
        to_call = max(0.0, state.min_call - state.player_pots[p])
        facing_bet = to_call > 0.0
        pot = max(1.0, state.community_pot + state.current_round_pot)
        pot_odds = to_call / (pot + to_call) if facing_bet else 0.0
        stack = max(1.0, state.stacks[p])
        stack_commitment = to_call / (stack + to_call)

        r = rng.random()
        if facing_bet:
            if stack_commitment > 0.40 or to_call > 2.0 * pot:
                if strength < 0.65:
                    return 0 if 0 in legal else legal[0]
                return 2 if 2 in legal else legal[0]

            if strength < pot_odds * 0.75:
                if r < 0.85 and 0 in legal:
                    return 0
                if 2 in legal:
                    return 2
            elif strength < pot_odds + 0.15:
                if r < 0.80 and 2 in legal:
                    return 2
                if 0 in legal:
                    return 0
            else:
                raises = [a for a in legal if a in (3, 4, 5, 6, 7)]
                if raises and r < 0.60:
                    return rng.choice(raises)
                if 2 in legal:
                    return 2
        else:
            if strength >= 0.75:
                raises = [a for a in legal if a in (3, 4, 5, 6, 7)]
                if raises and r < 0.70:
                    return rng.choice(raises)
            elif strength >= 0.55:
                small_raises = [a for a in legal if a in (3, 4)]
                if small_raises and r < 0.35:
                    return rng.choice(small_raises)
            if 1 in legal:
                return 1

        return rng.choice(legal)

    def update(self, bucket: int, action: int, td_target: float) -> None:
        """Met à jour les poids pour le bucket d'état et l'action observée."""
        if 0 <= bucket < 32 and 0 <= action < 8:
            error = td_target - self.weights[bucket, action]
            self.weights[bucket, action] += self.lr * float(np.clip(error, -5.0, 5.0))


class MCTSNode:
    """Nœud d'ensemble d'information dans l'arbre ISMCTS."""

    def __init__(self, player: int):
        self.player = player
        self.visits = 0
        self.total_reward = 0.0
        self.children: dict[int, MCTSNode] = {}
        self.action_visits: dict[int, int] = {}
        self.action_rewards: dict[int, float] = {}


class ISMCTS:
    """Moteur de recherche par arbre Monte Carlo pour information imparfaite."""

    def __init__(
        self,
        policy: MCTSPolicy,
        num_simulations: int = 80,
        exploration_c: float = 1.414,
        rng: random.Random | None = None,
    ):
        self.policy = policy
        self.num_simulations = num_simulations
        self.c = exploration_c
        self.rng = rng or random.Random()

    def search(self, root_state_factory, legal_root_actions: list[int]) -> tuple[dict[int, int], int]:
        if not legal_root_actions:
            raise ValueError("Aucune action légale disponible pour la recherche MCTS.")
        if len(legal_root_actions) == 1:
            return {legal_root_actions[0]: 1}, legal_root_actions[0]

        root = MCTSNode(player=root_state_factory(self.rng).current_player)

        for _ in range(self.num_simulations):
            sim_state = root_state_factory(self.rng)
            node = root
            path: list[tuple[MCTSNode, int]] = []

            while not sim_state.is_terminal():
                legal = sim_state.get_legal_actions()
                if not legal:
                    break

                unexpanded = [a for a in legal if a not in node.children]
                if unexpanded:
                    priors = self.policy.get_action_priors(sim_state, sim_state.current_player, unexpanded)
                    chosen_action = max(unexpanded, key=lambda a: priors.get(a, 0.0) + self.rng.random() * 0.01)
                    sim_state.apply_action(chosen_action, self.rng)
                    new_node = MCTSNode(player=sim_state.current_player)
                    node.children[chosen_action] = new_node
                    path.append((node, chosen_action))
                    node = new_node
                    break
                else:
                    priors = self.policy.get_action_priors(sim_state, sim_state.current_player, legal)
                    total_n = sum(node.action_visits.get(a, 0) for a in legal)
                    best_score = -float("inf")
                    best_action = legal[0]

                    q_vals = [
                        (node.action_rewards[a] / node.action_visits[a])
                        if node.action_visits.get(a, 0) > 0
                        else 0.0
                        for a in legal
                    ]
                    min_q, max_q = min(q_vals), max(q_vals)
                    range_q = max_q - min_q if max_q > min_q else 1.0

                    for a in legal:
                        n_a = node.action_visits.get(a, 0)
                        q_raw = (node.action_rewards[a] / n_a) if n_a > 0 else 0.0
                        q_norm = (q_raw - min_q) / range_q
                        prior = priors.get(a, 1.0 / len(legal))
                        u = self.c * prior * (np.sqrt(max(1, total_n)) / (1 + n_a))
                        score = q_norm + u
                        if score > best_score:
                            best_score = score
                            best_action = a

                    sim_state.apply_action(best_action, self.rng)
                    path.append((node, best_action))
                    node = node.children[best_action]

            while not sim_state.is_terminal():
                action = self.policy.get_rollout_action(sim_state, self.rng)
                sim_state.apply_action(action, self.rng)

            terminal_rewards = sim_state.get_terminal_rewards_bb()
            for parent_node, action in path:
                r = terminal_rewards[parent_node.player]
                parent_node.visits += 1
                parent_node.total_reward += r
                parent_node.action_visits[action] = parent_node.action_visits.get(action, 0) + 1
                parent_node.action_rewards[action] = parent_node.action_rewards.get(action, 0.0) + r

        visits = {a: root.action_visits.get(a, 0) for a in legal_root_actions}
        best_act = max(
            legal_root_actions, key=lambda a: (visits.get(a, 0), root.action_rewards.get(a, -float("inf")))
        )
        return visits, best_act


def _observation_to_sim_state_factory(observation: np.ndarray):
    num_players = 2 if len(observation) == 147 else (len(observation) - 119) // 15
    fields = {}
    offset = 0
    for field, size in observation_fields(num_players):
        fields[field] = observation[offset : offset + size]
        offset += size

    scale = 100.0 * num_players
    hero_seat = int(np.argmax(fields["seat"]))
    dealer_rel = int(np.argmax(fields["dealer"]))
    dealer_pos = (hero_seat + dealer_rel) % num_players
    stage = int(np.argmax(fields["stage"]))

    hero_cards = [CARD_ORDER[i] for i in np.flatnonzero(fields["hole_cards"])]
    board_cards = [CARD_ORDER[i] for i in np.flatnonzero(fields["board"])]
    known = set(hero_cards + board_cards)
    base_deck = [c for c in CARD_ORDER if c not in known]

    stacks = [float(fields["stacks"][(s - hero_seat) % num_players] * scale) for s in range(num_players)]
    street_bets = [float(fields["street_bets"][(s - hero_seat) % num_players] * scale) for s in range(num_players)]
    hand_contrib = [float(fields["hand_contributions"][(s - hero_seat) % num_players] * scale) for s in range(num_players)]
    raises_rem = [int(round(fields["raises_remaining"][(s - hero_seat) % num_players] * 2)) for s in range(num_players)]

    if num_players > 2:
        active = [bool(fields["active_players"][(s - hero_seat) % num_players] > 0.5) for s in range(num_players)]
    else:
        active = [True, True]

    community_pot = float(fields["pots"][0] * scale)
    current_round_pot = float(fields["pots"][1] * scale)
    min_call = max(street_bets) if any(street_bets) else 0.0
    start_stacks = [stacks[s] + hand_contrib[s] for s in range(num_players)]

    acted_in_street = [False] * num_players
    for s in range(num_players):
        rel = (s - hero_seat) % num_players
        acted_in_street[s] = bool(np.any(fields["last_actions"][rel * 8 : (rel + 1) * 8] > 0.5))
    acted_in_street[hero_seat] = False

    def make_determinized_state(sample_rng: random.Random) -> SimState:
        deck = base_deck.copy()
        sample_rng.shuffle(deck)
        hole_cards = [[] for _ in range(num_players)]
        hole_cards[hero_seat] = list(hero_cards)

        for s in range(num_players):
            if s != hero_seat:
                if active[s]:
                    c1 = deck.pop() if deck else "2C"
                    c2 = deck.pop() if deck else "3D"
                    hole_cards[s] = [c1, c2]
                else:
                    hole_cards[s] = []

        return SimState(
            num_players=num_players,
            hero_seat=hero_seat,
            current_player=hero_seat,
            dealer_pos=dealer_pos,
            stage=stage,
            stacks=stacks,
            player_pots=street_bets,
            player_max_win=hand_contrib,
            community_pot=community_pot,
            current_round_pot=current_round_pot,
            min_call=min_call,
            raises_remaining=raises_rem,
            hole_cards=hole_cards,
            board=list(board_cards),
            deck=deck,
            active=active,
            start_stacks=start_stacks,
            acted_in_street=list(acted_in_street),
        )

    return make_determinized_state


class MCTSAgent:
    """Agent MCTS pour Texas Hold'em avec politique heuristique et apprenante."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        num_simulations: int = 80,
        exploration_c: float = 1.414,
        temperature: float = 0.0,
        learning_rate: float = 0.05,
        name: str = "MCTS",
        seed: int | None = None,
        **kwargs: Any,
    ):
        self.name = name
        self.num_simulations = int(num_simulations)
        self.exploration_c = float(exploration_c)
        self.temperature = float(temperature)
        self.checkpoint_path = Path(checkpoint) if checkpoint else None

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.policy = MCTSPolicy(learning_rate=learning_rate)
        self.mcts = ISMCTS(
            policy=self.policy,
            num_simulations=self.num_simulations,
            exploration_c=self.exploration_c,
            rng=self.rng,
        )

        self._pending_decisions: list[tuple[int, int]] = []

        if self.checkpoint_path and self.checkpoint_path.is_file():
            self.load_checkpoint(self.checkpoint_path)

    def act(self, observation: np.ndarray, action_mask: np.ndarray, *, training: bool = False) -> int:
        legal = [int(a) for a in np.flatnonzero(action_mask)]
        if not legal:
            raise ValueError("L'agent MCTS doit avoir au moins une action légale.")
        if len(legal) == 1:
            return legal[0]

        factory = _observation_to_sim_state_factory(observation)
        visits, best_action = self.mcts.search(factory, legal)

        if training:
            root_state = factory(self.rng)
            strength = evaluate_hand_strength(root_state.hole_cards[root_state.hero_seat], root_state.board)
            to_call = max(0.0, root_state.min_call - root_state.player_pots[root_state.hero_seat])
            bucket = self.policy.get_bucket(root_state.stage, strength, to_call > 0.0)
            self._pending_decisions.append((bucket, best_action))

        if training and self.temperature > 0.0:
            total_v = sum(visits.get(a, 0) for a in legal)
            if total_v > 0:
                probs = [float((visits.get(a, 0) / total_v) ** (1.0 / self.temperature)) for a in legal]
                sum_p = sum(probs)
                if sum_p > 0:
                    probs = [p / sum_p for p in probs]
                    return int(self.np_rng.choice(legal, p=probs))

        return best_action

    def observe(self, transition: Any) -> None:
        """Met à jour les statistiques de la politique à partir de la transition reçue."""
        reward = float(getattr(transition, "reward", 0.0))
        if self._pending_decisions:
            bucket, action = self._pending_decisions.pop()
            self.policy.update(bucket, action, reward)

    def on_episode_start(self, *, seed: int | None = None, training: bool = False) -> None:
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self._pending_decisions.clear()

    def on_episode_end(
        self,
        *,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        training: bool = False,
    ) -> None:
        self._pending_decisions.clear()
        if training and self.checkpoint_path:
            self.save_checkpoint(self.checkpoint_path)

    def save_checkpoint(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "name": self.name,
            "weights": self.policy.weights.tolist(),
        }
        file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_checkpoint(self, path: str | Path) -> None:
        file_path = Path(path)
        if not file_path.is_file():
            return
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if "weights" in data:
                self.policy.weights = np.array(data["weights"], dtype=np.float32)
        except Exception:
            pass
