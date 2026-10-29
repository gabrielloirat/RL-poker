import copy
from threading import Event
import json

from poker_rl.game import PokerGame
from poker_rl.interface import Transition
from poker_rl.records import DEFAULT_RECORDS_DIR, SessionRecorder, snapshot


class SessionStats:
    def __init__(self,
                 completed, truncated, net_bb,
                 transitions, matches_completed=0, matches_truncated=0,
                 wins=(), hand_wins=(), hand_ties=(),
                 actions=()):  # Enregistre les données
        self.completed = completed
        self.truncated = truncated
        self.net_bb = net_bb
        self.transitions = transitions
        self.matches_completed = matches_completed
        self.matches_truncated = matches_truncated
        self.wins = wins
        self.hand_wins = hand_wins
        self.hand_ties = hand_ties
        self.actions = actions

    def to_dict(self):  # Prépare les données à sauvegarder
        return {
            "completed": self.completed,
            "truncated": self.truncated,
            "net_bb": self.net_bb,
            "transitions": self.transitions,
            "matches_completed": self.matches_completed,
            "matches_truncated": self.matches_truncated,
            "wins": self.wins,
            "hand_wins": self.hand_wins,
            "hand_ties": self.hand_ties,
            "actions": self.actions,
        }

    def copy(self):  # Copie les données avant modification
        return copy.copy(self)

    def __eq__(self, other):  # Compare les valeurs
        if not isinstance(other, SessionStats):
            return False
        return self.__dict__ == other.__dict__

    @property
    def hands(self):  # Compte les mains jouées
        return self.completed + self.truncated

    @property
    def mean_bb(self):  # Calcule le gain moyen
        means = []
        for value in self.net_bb:
            if self.completed > 0:
                means.append(value / self.completed)
            else:
                means.append(0.0)
        return tuple(means)

    @property
    def matches(self):  # Compte les parties jouées
        return self.matches_completed + self.matches_truncated


class PendingDecision:
    def __init__(self,
                 observation, action_mask, action,
                 reward=0.0):  # Enregistre les données
        self.observation = observation
        self.action_mask = action_mask
        self.action = action
        self.reward = reward


class TableSession:
    def __init__(self, registry, selections, *, mode="infer", training_seats=(),
                 seed=42, game_kwargs=None, defer_callbacks=False, continuous=False,
                 max_hands_per_match=1000, record=False, records_dir=None, record_kind=None,
                 agents=None):  # Prépare les joueurs et la session
        if not 2 <= len(selections) <= 5:
            raise ValueError("Choisissez entre 2 et 5 joueurs.")
        if mode not in ("train", "infer"):
            raise ValueError("Mode attendu : train ou infer.")
        self.mode = mode
        if type(max_hands_per_match) is not int or max_hands_per_match < 1:
            raise ValueError("La limite de mains par partie doit être un entier positif.")
        self.continuous = continuous
        self.max_hands_per_match = max_hands_per_match
        self.training_seats = set()
        if mode == "train":
            self.training_seats = set(training_seats)
        if not self.training_seats.issubset(range(len(selections))):
            raise ValueError("Les sièges à entraîner doivent participer à la partie.")
        if mode == "train" and not self.training_seats:
            raise ValueError("Sélectionnez au moins un agent à entraîner.")
        for name, _ in selections:
            if hasattr(registry, "resolve_model"):
                model = registry.resolve_model(name)
            else:
                model = getattr(registry, "saved_models", {}).get(name)
            if model and model["num_players"] != len(selections):
                raise ValueError(f"{name} attend une table à {model['num_players']} joueurs.")
        self.registry = registry
        self.selections = []
        for name, kwargs in selections:
            self.selections.append((name, dict(kwargs)))
        if agents is None:
            self.agents = []
            for name, kwargs in selections:
                self.agents.append(registry.load(name, kwargs))
        else:
            self.agents = list(agents)
        if len(self.agents) != len(selections):
            raise ValueError("Fournissez un agent par joueur sélectionné.")
        self.names = []
        for name, kwargs in selections:
            self.names.append(name)
        for seat in self.training_seats:
            if not callable(getattr(self.agents[seat], "observe", None)):
                raise TypeError(f"Joueur {seat + 1} ({self.names[seat]}) : observe(transition) est requis pour entraîner cet agent.")
        for agent in self.agents:
            enable = getattr(agent, "enable_window_input", None)
            if callable(enable):
                enable()
        settings = dict(game_kwargs or {})
        settings["num_players"] = len(selections)
        self.game = PokerGame(**settings)
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
        zero = (0,) * len(selections)
        self.stats = SessionStats(0, 0, (0.0,) * len(selections), zero, wins=zero,
                                  hand_wins=zero, hand_ties=zero, actions=zero)
        self.decisions = [None] * len(selections)
        self._hand_recorded = True
        self.record = record
        if records_dir is None:
            self.records_dir = getattr(registry, "records_dir", DEFAULT_RECORDS_DIR)
        else:
            self.records_dir = records_dir
        self.record_kind = record_kind
        self.recorder = None
        self.last_archive = None
        self.validation_report = None
        self.action_observers = []
        self.hand_observers = []
        self.record_baseline = self.stats
        self.game_kwargs = dict(game_kwargs or {})
        if record:
            self._start_recording()
        try:
            self.next_hand()
        except BaseException:
            self.close()
            raise

    def next_hand(self):  # Prépare une nouvelle main
        if self.stop_event.is_set():
            raise RuntimeError("La session est arrêtée.")
        if self.hand and (not self.game.finished or not self._hand_recorded):
            raise RuntimeError("Terminez la main et ses callbacks avant la suivante.")
        if self.record and self.recorder.closed:
            self._start_recording()
        if not self.continuous or self.match == 0:
            self._begin_match()
        elif self.match_finished:
            raise RuntimeError("La partie est terminée ; advance() démarre la suivante.")
        self.hand += 1
        self.hand_in_match += 1
        for seat, agent in enumerate(self.agents):
            callback = getattr(agent, "on_episode_start", None)
            if callable(callback):
                callback(seed=self.seed + (self.hand - 1) * len(self.agents) + seat,
                         training=seat in self.training_seats)
        if self.hand_in_match == 1:
            self.game.reset(seed=self.seed + self.hand - 1,
                            options={"dealer_seat": (self.match - 1) % len(self.agents)})
        else:
            self.game.start_next_hand(seed=self.seed + self.hand - 1)
        self.decisions = [None] * len(self.agents)
        self.game._analysis_preflop_raises = 0
        self._hand_recorded = False
        for seat, name in enumerate(self.names):
            self.game.table.players[seat].name = f"{seat + 1}. {name}"
        if self.recorder:
            self.recorder.hand_start(self.game, hand=self.hand, match=self.match,
                                     seed=self.seed + self.hand - 1, hand_in_match=self.hand_in_match)
        if self.game.finished and not self.defer_callbacks:
            self.finish_hand()

    def _begin_match(self):  # Commence une nouvelle partie
        self.match += 1
        self.hand_in_match = 0
        self.match_finished = False
        self.match_truncated = False
        self.match_winner = None

    def advance(self):  # Passe à la main suivante

        if self.stop_event.is_set() or not self.game.finished or not self._hand_recorded:
            raise RuntimeError("Terminez la main et ses callbacks avant de continuer.")
        if self.continuous and self.match_finished:
            self._begin_match()
        self.next_hand()

    @property
    def human_turn(self):  # Indique si un humain doit jouer
        seat = self.game.current_seat
        return seat is not None and getattr(self.agents[seat], "input_mode", None) == "window"

    @property
    def has_human(self):  # Repère les joueurs humains
        for agent in self.agents:
            if getattr(agent, "input_mode", None) == "window":
                return True
        return False

    def _deliver(self, seat):  # Transmet la transition précédente
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
            self.stats = self.stats.copy()
            self.stats.transitions = tuple(counts)
        self.decisions[seat] = None

    def choose_action(self):  # Demande une action au joueur
        seat = self.game.current_seat
        if seat is None:
            raise RuntimeError("La main est terminée.")


        self._deliver(seat)
        return self.agents[seat].act(self.game.observe(seat), self.game.action_masks(seat),
                                     training=seat in self.training_seats)

    def submit_human_action(self, action):  # Transmet le choix humain
        if not self.human_turn or not self.game.action_masks()[action]:
            raise ValueError("Action humaine indisponible.")
        self.agents[self.game.current_seat].submit_action(action)

    def play(self, action):  # Joue et enregistre une action
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
        before = None
        if self.recorder:
            before = snapshot(game)
        from poker_rl.hand_analysis import decision_snapshot
        decision_analysis = None
        if self.action_observers:
            decision_analysis = decision_snapshot(game, seat, int(action))
        for observer in self.action_observers:
            observer(self, decision_analysis)
        if decision_analysis and decision_analysis['stage'] == 'PREFLOP' and decision_analysis['is_raise']:
            game._analysis_preflop_raises += 1
        _, rewards, _, _, _ = game.step(action)
        counts = list(self.stats.actions)
        counts[seat] += 1
        self.stats = self.stats.copy()
        self.stats.actions = tuple(counts)
        if self.recorder:
            self.recorder.action(before, game, action, curriculum_prefix=getattr(self, 'curriculum_prefix', False))
        for player, decision in enumerate(self.decisions):
            if decision is not None:
                decision.reward += float(rewards[player])
        if game.finished and not self.defer_callbacks:
            self.finish_hand()

    def finish_hand(self):  # Enregistre le résultat de la main
        if not self.game.finished or self._hand_recorded:
            return
        survivors = []
        for seat, player in enumerate(self.game.table.players):
            if player.stack > 0:
                survivors.append(seat)
        self.match_finished = (not self.continuous or self.game.truncated or len(survivors) == 1
                               or self.hand_in_match >= self.max_hands_per_match)
        self.match_truncated = self.game.truncated or (self.continuous and len(survivors) > 1
                                                      and self.hand_in_match >= self.max_hands_per_match)
        self.match_winner = None
        if self.continuous and len(survivors) == 1 and self.game.terminated:
            self.match_winner = survivors[0]
        for seat in range(len(self.agents)):
            self._deliver(seat)
        rewards = self.game.rewards()
        for seat, agent in enumerate(self.agents):
            callback = getattr(agent, "on_episode_end", None)
            if callable(callback):
                callback(reward=float(rewards[seat]), terminated=self.game.terminated,
                         truncated=self.game.truncated, training=seat in self.training_seats)
        totals = []
        for seat, total in enumerate(self.stats.net_bb):
            totals.append(total + float(rewards[seat]))
        wins = list(self.stats.wins)
        hand_wins, hand_ties = list(self.stats.hand_wins), list(self.stats.hand_ties)
        winners = []
        if self.game.terminated:
            winners = self.game.table.winner_indices
        for seat in winners:
            if len(winners) == 1:
                hand_wins[seat] += 1
            else:
                hand_ties[seat] += 1
        if self.match_finished and self.match_winner is not None:
            wins[self.match_winner] += 1
        stats = self.stats.copy()
        stats.completed += int(self.game.terminated)
        stats.truncated += int(self.game.truncated)
        stats.net_bb = tuple(totals)
        stats.matches_completed += int(self.match_finished and not self.match_truncated)
        stats.matches_truncated += int(self.match_finished and self.match_truncated)
        stats.wins = tuple(wins)
        stats.hand_wins = tuple(hand_wins)
        stats.hand_ties = tuple(hand_ties)
        self.stats = stats
        self._hand_recorded = True
        for observer in self.hand_observers:
            observer(self)
        if self.recorder:
            self.recorder.hand_end(self.game, match_finished=self.match_finished,
                                   match_truncated=self.match_truncated, match_winner=self.match_winner)

    def run_batch(self, target, *, max_actions=128):  # Joue plusieurs actions
        for seat in self.training_seats:
            if getattr(self.agents[seat], "native_training", False):
                raise ValueError("PPO utilise model.learn() : lancez train_sb3_session ou le mode entraînement de l'interface.")
        if self.has_human:
            raise ValueError("Le mode rapide nécessite uniquement des agents automatiques.")
        for _ in range(max_actions):
            if self.continuous:
                done = self.stats.matches
            else:
                done = self.stats.hands
            if self.stop_event.is_set() or self.pause_event.is_set() or done >= target:
                return
            if self.game.finished:
                self.finish_hand()
                if self.continuous:
                    done = self.stats.matches
                else:
                    done = self.stats.hands
                if done >= target:
                    return
                self.advance()
                if self.game.finished:
                    self.finish_hand()
                    continue
            self.play(self.choose_action())
            if self.game.finished:
                self.finish_hand()

    def use_inference(self):  # Passe en observation
        if not self.game.finished or not self._hand_recorded:
            raise RuntimeError("Terminez la main avant de passer en observation.")
        self.finish_recording()
        if self.mode == 'train':
            for agent in self.agents:
                if getattr(agent, 'native_training', False):
                    self.continuous = True
        self.mode = "infer"
        self.training_seats = set()
        self.advance()

    def _start_recording(self):  # Crée une archive de session
        self.record_baseline = self.stats
        self.recorder = SessionRecorder(self.selections, kind=self.record_kind or self.mode,
                                        training_seats=self.training_seats, seed=self.seed,
                                        game_kwargs=self.game_kwargs, continuous=self.continuous,
                                        max_hands_per_match=self.max_hands_per_match,
                                        root=self.records_dir, registry=self.registry)
        if self.last_archive is not None:
            previous = json.loads((self.last_archive / "manifest.json").read_text(encoding="utf-8"))
            self.recorder.manifest["parent_run"] = previous["id"]
            for entry, source in zip(self.recorder.manifest["agents"], previous["agents"]):
                if source.get("model_id"):
                    entry.update(source_model=source["model_id"], source_run=previous["id"])
            self.recorder.flush()

    def finish_recording(self, *, status="completed", error=None):  # Sauvegarde les résultats
        if self.recorder is None:
            return None
        stats = self.stats.to_dict()
        baseline = self.record_baseline.to_dict()
        for name, value in stats.items():
            if isinstance(value, tuple):
                differences = []
                for current, start in zip(value, baseline[name]):
                    differences.append(current - start)
                stats[name] = differences
            else:
                stats[name] = value - baseline[name]
        stats["mean_bb"] = []
        for value in stats["net_bb"]:
            if stats["completed"] > 0:
                stats["mean_bb"].append(value / stats["completed"])
            else:
                stats["mean_bb"].append(0)
        self.last_archive = self.recorder.finish(stats, agents=self.agents, status=status, error=error)
        return self.last_archive

    def complete_training(self, *, validation_hands=500, on_validation_progress=None):  # Valide et sauvegarde le modèle
        from poker_rl.evaluation import evaluate_agents, attach_validation
        if self.mode == "train" and self.validation_report is None and not self.stop_event.is_set():
            self.validation_report = evaluate_agents(self.agents, self.training_seats,
                game_kwargs=self.game_kwargs, hands_per_opponent=validation_hands,
                stop_event=self.stop_event, on_progress=on_validation_progress)
            attach_validation(self.recorder, self.validation_report)
        status = "completed"
        if self.stop_event.is_set():
            status = "interrupted"
        return self.finish_recording(status=status)

    def request_stop(self):  # Demande l'arrêt de la session
        self.stop_event.set()
        for agent in self.agents:
            if getattr(agent, "input_mode", None) == "window":
                agent.submit_action(None)

    def close(self):  # Ferme la session
        self.request_stop()
        self.finish_recording(status="interrupted")
        self.game.close()
