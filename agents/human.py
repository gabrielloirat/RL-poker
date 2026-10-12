from __future__ import annotations

import numpy as np
from queue import Queue

from poker_rl import CARD_ORDER, observation_fields


STAGES = ("Préflop", "Flop", "Turn", "River")
ACTION_LABELS = (
    ("f", "Se coucher / fold"),
    ("e", "Parole / check"),
    ("c", "Suivre / call"),
    ("b", "Relancer à 3 BB"),
    ("h", "Relancer d'un demi-pot"),
    ("p", "Relancer d'un pot"),
    ("d", "Relancer de deux pots"),
    ("a", "Tapis / all-in"),
)


def _cards(values):
    return " ".join(CARD_ORDER[i] for i in np.flatnonzero(values)) or "—"


class HumanAgent:


    def __init__(self, name="Utilisateur", input_mode="terminal"):
        if input_mode not in ("terminal", "window"):
            raise ValueError("input_mode must be terminal or window.")
        self.name = name
        self.seat = None
        self.input_mode = input_mode
        self._actions = Queue()

    def enable_window_input(self):
        self.input_mode = "window"

    def submit_action(self, action):

        self._actions.put(action)

    def act(self, observation, action_mask, *, training=False):
        legal = [int(action) for action in np.flatnonzero(action_mask)]
        if not legal:
            raise ValueError("L'agent humain doit être appelé à un tour avec des actions légales.")
        if self.input_mode == "window":
            while True:
                action = self._actions.get()
                if action is None:
                    raise KeyboardInterrupt
                if action in legal:
                    return int(action)


        fields = {}
        offset = 0
        num_players = 2 if len(observation) == 147 else (len(observation) - 119) // 15
        for field, size in observation_fields(num_players):
            fields[field] = observation[offset:offset + size]
            offset += size
        self.seat = int(np.argmax(fields["seat"]))
        print(f"\n{self.name} — siège {self.seat} — {STAGES[int(np.argmax(fields['stage']))]}")
        print(f"Vos cartes : {_cards(fields['hole_cards'])} | Board : {_cards(fields['board'])}")


        scale = num_players * 100
        stacks = fields["stacks"] * scale
        bets = fields["street_bets"] * scale
        pot = float(np.sum(fields["pots"])) * scale
        to_call = float(fields["to_call"][0]) * scale
        print("Montants en % du stack initial :")
        print(f"  Votre stack : {stacks[0]:.1f}% | Pot : {pot:.1f}% | À suivre : {to_call:.1f}%")
        print("  Stacks / mises du tour (vous, puis les autres sièges dans l'ordre) : " +
              " | ".join(f"{stack:.1f}% / {bet:.1f}%" for stack, bet in zip(stacks, bets)))
        for action in legal:
            shortcut, label = ACTION_LABELS[action]
            print(f"  {action} [{shortcut}] : {label}")
        print("  q : quitter la session")

        shortcuts = {shortcut: action for action, (shortcut, _) in enumerate(ACTION_LABELS)}
        while True:
            try:
                choice = input("Votre action (numéro ou lettre) : ").strip().lower()
            except EOFError:
                raise KeyboardInterrupt from None
            if choice in ("q", "quit", "quitter"):
                raise KeyboardInterrupt
            if choice in shortcuts:
                action = shortcuts[choice]
            else:
                try:
                    action = int(choice)
                except ValueError:
                    print("Saisie invalide. Entrez un numéro ou une lettre affichée.")
                    continue
            if action in legal:
                return action
            print("Action indisponible. Choisissez une des actions affichées.")

    def on_episode_start(self, *, seed, training):
        self.seat = None
        while not self._actions.empty():
            self._actions.get_nowait()

    def on_episode_end(self, *, reward, terminated, truncated, training):
        if self.input_mode == "window":
            return
        seat = f" (siège {self.seat})" if self.seat is not None else ""
        if truncated:
            print(f"{self.name}{seat} : main interrompue par la limite d'actions.")
        else:
            print(f"{self.name}{seat} : main terminée, gain net {reward:+.2f} BB.")
