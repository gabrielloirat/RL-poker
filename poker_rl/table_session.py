from dataclasses import dataclass, replace
from threading import Event

from poker_rl.game import PokerGame
from poker_rl.interface import Transition


@dataclass(frozen=True)
class SessionStats:
    completed: int
    truncated: int
    net_bb: tuple[float, ...]
    transitions: tuple[int, ...]
    matches_completed: int = 0
    matches_truncated: int = 0
    wins: tuple[int, ...] = ()

    @property
    def hands(self):
        return self.completed + self.truncated

    @property
    def mean_bb(self):
        return tuple(value / self.completed if self.completed else 0.0 for value in self.net_bb)

    @property
    def matches(self):
        return self.matches_completed + self.matches_truncated


@dataclass
class PendingDecision:
    observation: object
    action_mask: object
    action: int
    reward: float = 0.0


class TableSession:
    def __init__(self, registry, selections, *, mode="infer", training_seats=(),
                 seed=42, game_kwargs=None, defer_callbacks=False, continuous=False,
                 max_hands_per_match=1000):
        if not 2 <= len(selections) <= 5:
            raise ValueError("Choisissez entre 2 et 5 joueurs.")
        if mode not in ("train", "infer"):
            raise ValueError("Mode attendu : train ou infer.")
        self.mode = mode
        if type(max_hands_per_match) is not int or max_hands_per_match < 1:
            raise ValueError("La limite de mains par partie doit être un entier positif.")
        self.continuous = continuous
        self.max_hands_per_match = max_hands_per_match
        self.training_seats = frozenset(training_seats) if mode == "train" else frozenset()
        if not self.training_seats.issubset(range(len(selections))):
            raise ValueError("Les sièges à entraîner doivent participer à la partie.")
        if mode == "train" and not self.training_seats:
            raise ValueError("Sélectionnez au moins un agent à entraîner.")
        self.agents = [registry.load(name, kwargs) for name, kwargs in selections]
        self.names = [name for name, _ in selections]
        for seat in self.training_seats:
            if not callable(getattr(self.agents[seat], "observe", None)):
                raise TypeError(f"Joueur {seat + 1} ({self.names[seat]}) : observe(transition) est requis pour entraîner cet agent.")
        for agent in self.agents:
            enable = getattr(agent, "enable_window_input", None)
            if callable(enable):
                enable()
        self.game = PokerGame(**{**(game_kwargs or {}), "num_players": len(selections)})
        self.seed = seed
        self.hand = 0
        self.match = 0
        self.hand_in_match = 0
        self.match_finished = False
        self.match_truncated = False
        self.match_winner = None
        self.defer_callbacks = defer_callbacks
        self.stop_event = Event()
        self.pause_event = Event()
        self.stats = SessionStats(0, 0, (0.0,) * len(selections), (0,) * len(selections), wins=(0,) * len(selections))
        self.decisions = [None] * len(selections)
        self._hand_recorded = True
        try:
            self.next_hand()
        except BaseException:
            self.close()
            raise

    def next_hand(self):
        if self.stop_event.is_set():
            raise RuntimeError("La session est arrêtée.")
        if self.hand and (not self.game.finished or not self._hand_recorded):
            raise RuntimeError("Terminez la main et ses callbacks avant la suivante.")
        if not self.continuous or self.match == 0:
            self._begin_match()
        elif self.match_finished:
            raise RuntimeError("La partie est terminée ; advance() démarre la suivante.")
        self.hand += 1
        self.hand_in_match += 1
        for seat, agent in enumerate(self.agents):
            callback = getattr(agent, "on_episode_start", None)
            if callable(callback):
                callback(seed=self.seed + self.hand - 1, training=seat in self.training_seats)
        if self.hand_in_match == 1:
            self.game.reset(seed=self.seed + self.hand - 1,
                            options={"dealer_seat": (self.match - 1) % len(self.agents)})
        else:
            self.game.start_next_hand(seed=self.seed + self.hand - 1)
        self.decisions = [None] * len(self.agents)
        self._hand_recorded = False
        for seat, name in enumerate(self.names):
            self.game.table.players[seat].name = f"{seat + 1}. {name}"
        if self.game.finished and not self.defer_callbacks:
            self.finish_hand()

    def _begin_match(self):
        self.match += 1
        self.hand_in_match = 0
        self.match_finished = False
        self.match_truncated = False
        self.match_winner = None

    def advance(self):

        if self.stop_event.is_set() or not self.game.finished or not self._hand_recorded:
            raise RuntimeError("Terminez la main et ses callbacks avant de continuer.")
        if self.continuous and self.match_finished:
            self._begin_match()
        self.next_hand()

    @property
    def human_turn(self):
        seat = self.game.current_seat
        return seat is not None and getattr(self.agents[seat], "input_mode", None) == "window"

    @property
    def has_human(self):
        return any(getattr(agent, "input_mode", None) == "window" for agent in self.agents)

    def _deliver(self, seat):
        decision = self.decisions[seat]
        if decision is None:
            return
        if seat in self.training_seats:
            info = self.game.info(seat)
            info.update(match=self.match, hand_in_match=self.hand_in_match,
                        match_finished=self.match_finished, match_winner=self.match_winner)
            self.agents[seat].observe(Transition(
                observation=decision.observation.copy(), action_mask=decision.action_mask.copy(),
                action=decision.action, reward=decision.reward,
                next_observation=self.game.observe(seat), next_action_mask=self.game.action_masks(seat),
                terminated=self.game.terminated, truncated=self.game.truncated, info=info))
            counts = list(self.stats.transitions)
            counts[seat] += 1
            self.stats = replace(self.stats, transitions=tuple(counts))
        self.decisions[seat] = None

    def choose_action(self):
        seat = self.game.current_seat
        if seat is None:
            raise RuntimeError("La main est terminée.")


        self._deliver(seat)
        return self.agents[seat].act(self.game.observe(seat), self.game.action_masks(seat),
                                     training=seat in self.training_seats)

    def submit_human_action(self, action):
        if not self.human_turn or not self.game.action_masks()[action]:
            raise ValueError("Action humaine indisponible.")
        self.agents[self.game.current_seat].submit_action(action)

    def play(self, action):
        game = self.game
        if game.finished:
            raise RuntimeError("La main est terminée ; lancez la suivante.")
        seat = game.current_seat
        action = getattr(action, "value", action)
        if not game.action_space.contains(action) or not game.action_masks()[int(action)]:
            raise ValueError(f"{self.names[seat]} a retourné une action illégale : {action}.")
        if self.decisions[seat] is not None:
            raise RuntimeError("Appelez choose_action() avant play() pour livrer la transition précédente.")
        self.decisions[seat] = PendingDecision(game.observe(seat), game.action_masks(seat), int(action))
        _, rewards, _, _, _ = game.step(action)
        for player, decision in enumerate(self.decisions):
            if decision is not None:
                decision.reward += float(rewards[player])
        if game.finished and not self.defer_callbacks:
            self.finish_hand()

    def finish_hand(self):
        if not self.game.finished or self._hand_recorded:
            return
        survivors = [seat for seat, player in enumerate(self.game.table.players) if player.stack > 0]
        self.match_finished = (not self.continuous or self.game.truncated or len(survivors) == 1
                               or self.hand_in_match >= self.max_hands_per_match)
        self.match_truncated = self.game.truncated or (self.continuous and len(survivors) > 1
                                                      and self.hand_in_match >= self.max_hands_per_match)
        self.match_winner = survivors[0] if self.continuous and len(survivors) == 1 and self.game.terminated else None
        for seat in range(len(self.agents)):
            self._deliver(seat)
        rewards = self.game.rewards()
        for seat, agent in enumerate(self.agents):
            callback = getattr(agent, "on_episode_end", None)
            if callable(callback):
                callback(reward=float(rewards[seat]), terminated=self.game.terminated,
                         truncated=self.game.truncated, training=seat in self.training_seats)
        totals = tuple(total + float(rewards[seat]) for seat, total in enumerate(self.stats.net_bb))
        wins = list(self.stats.wins)
        if self.match_finished and self.match_winner is not None:
            wins[self.match_winner] += 1
        self.stats = replace(self.stats, completed=self.stats.completed + int(self.game.terminated),
                             truncated=self.stats.truncated + int(self.game.truncated), net_bb=totals,
                             matches_completed=self.stats.matches_completed + int(self.match_finished and not self.match_truncated),
                             matches_truncated=self.stats.matches_truncated + int(self.match_finished and self.match_truncated),
                             wins=tuple(wins))
        self._hand_recorded = True

    def run_batch(self, target, *, max_actions=128):

        if self.has_human:
            raise ValueError("Le mode rapide nécessite uniquement des agents automatiques.")
        for _ in range(max_actions):
            done = self.stats.matches if self.continuous else self.stats.hands
            if self.stop_event.is_set() or self.pause_event.is_set() or done >= target:
                return
            if self.game.finished:
                self.finish_hand()
                done = self.stats.matches if self.continuous else self.stats.hands
                if done >= target:
                    return
                self.advance()
                if self.game.finished:
                    self.finish_hand()
                    continue
            self.play(self.choose_action())
            if self.game.finished:
                self.finish_hand()

    def use_inference(self):
        if not self.game.finished or not self._hand_recorded:
            raise RuntimeError("Terminez la main avant de passer en observation.")
        self.mode = "infer"
        self.training_seats = frozenset()
        self.advance()

    def close(self):
        self.stop_event.set()
        for agent in self.agents:
            if getattr(agent, "input_mode", None) == "window":
                agent.submit_action(None)
        self.game.close()
