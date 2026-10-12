from concurrent.futures import Future
from dataclasses import dataclass
import json
from threading import Thread
import time

import pyglet
from pyglet.window import key, mouse

from gym_env.rendering import PygletWindow
from poker_rl.plugins import AgentRegistry, DEFAULT_AGENTS_DIR
from poker_rl.table_session import TableSession
from poker_rl.display_equity import EquitySnapshot, estimate_equities

BACKGROUND = (15, 23, 34, 255)
PANEL = (28, 40, 54, 255)
TEXT = (231, 238, 246, 255)
MUTED = (161, 179, 199, 255)
ACCENT = (41, 174, 147, 255)
BUTTON = (42, 60, 78, 255)
HOVER = (58, 83, 105, 255)
PRESSED = (25, 111, 96, 255)
BORDER = (83, 105, 128, 255)
DISABLED = (24, 32, 43, 255)
DISABLED_TEXT = (111, 129, 149, 255)
ERROR = (247, 145, 139, 255)
UI_WIDTH = 1040
UI_HEIGHT = 860
LABELS = ("Se coucher", "Parole", "Suivre", "Relancer à 3 BB",
          "Relancer · demi-pot", "Relancer · pot", "Relancer · 2 pots", "Tapis")
SHORTCUTS = ("F", "Espace", "C", "B", "H", "P", "D", "A")
KEY_ACTIONS = {key.F: 0, key.SPACE: 1, key.C: 2, key.B: 3, key.H: 4, key.P: 5, key.D: 6, key.A: 7}
STAGE_LABELS = {"PREFLOP": "Préflop", "FLOP": "Flop", "TURN": "Turn", "RIVER": "River",
                "SHOWDOWN": "Abattage", "END_HIDDEN": "Fin de main"}


@dataclass
class Button:
    x: int
    y: int
    width: int
    height: int
    callback: object
    enabled: bool
    hint: str
    field: bool = False

    @property
    def identity(self):
        return self.x, self.y, self.width, self.height

    def contains(self, x, y):
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height


def _background(function):
    future = Future()

    def work():
        try:
            future.set_result(function())
        except BaseException as exc:
            future.set_exception(exc)

    Thread(target=work, daemon=True).start()
    return future


class PokerApp:
    def __init__(self, agents_dir=DEFAULT_AGENTS_DIR):
        self.agents_dir = agents_dir
        self.registry = AgentRegistry(agents_dir)
        self.available = list(self.registry.agents)
        self.count = 2
        default = "human" if "human" in self.available else (self.available[0] if self.available else "")
        self.selected = [default] * 5
        self.kwargs = ["{}"] * 5
        self.mode = "infer"
        self.training = [False] * 5
        self.episodes_text = "1"
        self.max_hands_text = "1000"
        self.auto_continue = True
        self.fast = False
        self.requested_visual = False
        self.pending_kind = None
        self.target_matches = 0
        self.run_baseline = None
        self.completed_notified = False
        self.editing = None
        self.replace_text = False
        self.session = None
        self.pending = None
        self.loading = False
        self.submitted = False
        self.paused = False
        self.reveal = False
        self.next_action_at = 0
        self.message = "Choisissez les agents, puis lancez la partie."
        self.error = ""
        self.buttons = []
        self.mouse_position = (-1, -1)
        self.pressed_button = None
        self.flash_button = None
        self.flash_until = 0
        self.toast = ""
        self.toast_until = 0
        self.equity_pending = None
        self.equity_job_snapshot = None
        self.equity_cache = {}
        self.equity_values = None
        self.equity_error = ""
        self.ui_scale = 1.0
        self.ui_offset_x = 0.0
        self.ui_offset_y = 0.0
        self.viewer = PygletWindow(UI_WIDTH, UI_HEIGHT, resizable=True)
        self.window = self.viewer.display_surface
        self.window.set_size(UI_WIDTH, UI_HEIGHT)
        self.window.set_minimum_size(760, 600)
        self.window.set_caption("RL Poker — parties complètes")
        self.window.push_handlers(on_draw=self.on_draw, on_mouse_press=self.on_mouse_press,
                                  on_mouse_release=self.on_mouse_release, on_mouse_motion=self.on_mouse_motion,
                                  on_mouse_leave=self.on_mouse_leave, on_mouse_drag=self.on_mouse_drag,
                                  on_key_press=self.on_key_press, on_text=self.on_text,
                                  on_resize=self.on_resize, on_close=self.on_close)
        self.on_resize(*self.window.get_size())
        pyglet.clock.schedule_interval(self.tick, 1 / 30)

    def text(self, content, x, y, size=12, color=TEXT):
        self.viewer.text(content, x, y, font_size=size, color=color)

    def rect(self, x, y, width, height, color=PANEL):
        self.viewer.rectangle(x, y, width, -height, color)

    def button(self, label, x, y, width, callback, *, enabled=True, active=False,
               primary=False, field=False, shortcut=None, hint=None, height=38):
        control = Button(x, y, width, height, callback, enabled, hint or label, field)
        self.buttons.append(control)
        hovered = control.contains(*self.mouse_position)
        held = self.pressed_button == control.identity and hovered
        flashed = self.flash_button == control.identity and time.monotonic() < self.flash_until
        fill = DISABLED if not enabled else (ACCENT if primary else BUTTON)
        border = PANEL if not enabled else (ACCENT if active else BORDER)
        if enabled and hovered:
            fill, border = HOVER, ACCENT
        if enabled and (held or flashed):
            fill, border = PRESSED, TEXT
        if active and enabled:
            border = ACCENT
        shift = 2 if held and enabled else 0
        self.rect(x + 2, y + 3, width, height, (8, 13, 20, 255))
        self.rect(x, y + shift, width, height, border)
        self.rect(x + 2, y + 2 + shift, width - 4, height - 4, fill)
        if active and enabled:
            self.rect(x + 3, y + 3 + shift, 4, height - 6, ACCENT)
        color = TEXT if enabled else DISABLED_TEXT
        self.text(label, x + 12, y + 10 + shift, 11, color)
        if shortcut:
            self.rect(x + width - 62, y + 8 + shift, 52, 23, BACKGROUND)
            self.text(shortcut, x + width - 56, y + 12 + shift, 9, color)
        elif active and not field and enabled:
            self.text("✓", x + width - 24, y + 10 + shift, 12, TEXT)
        if field and active and self.editing is not None and int(time.monotonic() * 2) % 2 == 0:
            caret_x = x + 14 + min(len(label) * 7, width - 28)
            self.rect(caret_x, y + 9, 2, 20, TEXT)

    def notify(self, message):
        self.message = self.toast = message
        self.toast_until = time.monotonic() + 2.8

    def button_at(self, x, y):
        return next((control for control in reversed(self.buttons) if control.contains(x, y)), None)

    def on_draw(self):
        from pyglet.gl import (glClearColor, glViewport, glMatrixMode, glLoadIdentity,
                               glOrtho, GL_PROJECTION, GL_MODELVIEW)
        width, height = self.window.get_size()
        self.update_layout(width, height)
        framebuffer = getattr(self.window, "get_framebuffer_size", self.window.get_size)()
        ratio_x, ratio_y = framebuffer[0] / width, framebuffer[1] / height
        glViewport(round(self.ui_offset_x * ratio_x), round(self.ui_offset_y * ratio_y),
                   round(UI_WIDTH * self.ui_scale * ratio_x), round(UI_HEIGHT * self.ui_scale * ratio_y))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, UI_WIDTH, 0, UI_HEIGHT, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glClearColor(*(value / 255 for value in BACKGROUND))
        self.window.clear()
        self.rect(0, 0, UI_WIDTH, UI_HEIGHT, BACKGROUND)
        self.buttons = []
        self.text("RL POKER", 34, 26, 25)
        self.button("Fenêtre [F11]" if self.window.fullscreen else "Plein écran [F11]", 330, 30, 190,
                    self.toggle_fullscreen, active=self.window.fullscreen,
                    hint="F11 bascule le plein écran ; Échap revient au mode fenêtré.")
        if self.session is None:
            self.draw_lobby()
        else:
            self.draw_table()
        fresh = time.monotonic() < self.toast_until
        self.rect(34, 792, 972, 34, PANEL)
        self.rect(34, 792, 4, 34, ERROR if self.error else (ACCENT if fresh else BORDER))
        message = self.error or (self.toast if fresh else self.message)
        prefix = "! " if self.error else ("✓ " if fresh else "")
        self.text((prefix + message)[:125], 48, 802, 11, ERROR if self.error else TEXT)
        hovered = self.button_at(*self.mouse_position)
        hint = hovered.hint if hovered else "F11 : plein écran · Échap : retour au mode fenêtré / menu"
        self.text(hint[:125], 34, 835, 10, MUTED)

    def draw_lobby(self):
        self.text("Configurez votre table", 34, 72, 17)
        self.text("Mode", 34, 120, 12, MUTED)
        self.button("Voir jouer", 176, 104, 186, lambda: self.set_mode("infer"), active=self.mode == "infer",
                    enabled=not self.loading, hint="Faire jouer les agents avec training=False, sans apprentissage.")
        self.button("Entraînement", 384, 104, 210, lambda: self.set_mode("train"), active=self.mode == "train",
                    enabled=not self.loading, hint="Envoyer les transitions aux agents cochés pour qu'ils apprennent.")
        self.text("Nombre de joueurs", 34, 170, 12, MUTED)
        for count in range(2, 6):
            self.button(f"{count} joueurs", 218 + (count - 2) * 116, 154, 104,
                        lambda count=count: self.set_count(count), active=count == self.count,
                        enabled=not self.loading, hint=f"Activer uniquement {count} sièges à la table.")
        self.button("Actualiser les agents", 786, 154, 220, self.refresh, enabled=not self.loading,
                    hint="Rechercher les nouveaux agents déposés dans agents/.")
        selector_hint = "Agent : choisissez avec les flèches" if len(self.available) > 1 else "Un seul agent disponible : human" if self.available == ["human"] else "Agents disponibles"
        self.text(selector_hint, 130, 211, 11, MUTED)
        self.text("Paramètres (JSON)", 450, 211, 11, MUTED)
        self.text("Apprentissage", 832, 211, 11, MUTED)
        for seat in range(self.count):
            y = 238 + seat * 74
            self.rect(34, y, 972, 64)
            self.text(f"Joueur {seat + 1}", 48, y + 22, 12)
            can_cycle = len(self.available) > 1 and not self.loading
            reason = "Un seul agent disponible. Ajoutez un plugin dans agents/ puis actualisez."
            self.button("‹", 130, y + 13, 34, lambda seat=seat: self.cycle(seat, -1), enabled=can_cycle,
                        hint=f"Agent précédent pour le joueur {seat + 1}." if can_cycle else reason)
            self.text(self.selected[seat][:24] or "Aucun agent", 176, y + 23, 12)
            self.button("›", 392, y + 13, 34, lambda seat=seat: self.cycle(seat, 1), enabled=can_cycle,
                        hint=f"Agent suivant pour le joueur {seat + 1}." if can_cycle else reason)
            self.button(self.kwargs[seat][-38:], 450, y + 13, 360, lambda seat=seat: self.edit(seat),
                        enabled=not self.loading, active=self.editing == seat, field=True,
                        hint=f"Modifier les paramètres JSON du joueur {seat + 1} ; Entrée termine la saisie.")
            eligible = self.can_train(seat)
            self.button("Entraîner" if eligible else "Non entraînable", 832, y + 13, 155,
                        lambda seat=seat: self.toggle_training(seat),
                        active=self.mode == "train" and self.training[seat],
                        enabled=self.mode == "train" and eligible and not self.loading,
                        hint="Activer l'apprentissage pour ce siège." if eligible and self.mode == "train"
                        else "Le mode Voir jouer utilise tous les agents en inférence." if eligible
                        else f"{self.selected[seat]} ne fournit pas observe(transition) et sert uniquement d'adversaire.")
        self.text("Parties", 34, 637, 12, MUTED)
        self.button(self.episodes_text, 104, 625, 100, self.edit_episodes, field=True,
                    active=self.editing == "episodes", enabled=not self.loading, hint="Nombre de parties complètes : chaque partie se termine lorsqu'il reste un seul joueur.")
        self.button("Enchaîner les mains", 226, 625, 224, self.toggle_auto,
                    active=self.auto_continue, enabled=not self.loading, hint="Enchaîner les mains avec les mêmes stacks, puis démarrer les parties suivantes.")
        self.button("Rapide", 472, 625, 140, self.toggle_fast, active=self.fast,
                    enabled=not self.loading and self.auto_continue, hint="Simuler rapidement sans dessiner chaque main ; les sièges humains imposent le mode visuel.")
        self.text("Max. mains / partie", 640, 637, 11, MUTED)
        self.button(self.max_hands_text, 820, 625, 186, self.edit_max_hands, field=True,
                    active=self.editing == "max_hands", enabled=not self.loading,
                    hint="Au-delà de cette limite, une partie est comptée comme interrompue, sans gagnant.")
        self.text("Une partie = plusieurs mains, stacks conservés, jusqu'au dernier joueur.", 34, 674, 11, MUTED)
        launch_label = "Démarrer l'entraînement" if self.mode == "train" else "Voir jouer"
        self.button("Chargement…" if self.loading else launch_label, 34, 710, 242, self.start,
                    enabled=bool(self.available) and not self.loading, primary=True,
                    hint="Chargement des agents en cours…" if self.loading else "Démarrer avec les sièges et agents sélectionnés.")
        self.text(f"{len(self.registry.agents)} plugin(s) disponible(s) · {len(self.registry.errors)} indisponible(s)",
                  304, 722, 12, MUTED)
        if self.registry.errors:
            name, reason = next(iter(self.registry.errors.items()))
            self.text(f"{name} : {reason}"[:125], 34, 762, 10, ERROR)

    def draw_table(self):
        if self.fast:
            self.draw_fast_table()
            return
        if self.pending_kind == "reset":
            self.text("Préparation de la prochaine main…", 34, 190, 20)
            self.button("Choisir les agents", 786, 30, 220, self.back)
            self.draw_stats()
            return
        game = self.session.game
        table = game.table
        mode = "Entraînement" if self.session.mode == "train" else "Voir jouer"
        current = self.session.match - self.run_baseline.matches
        goal = self.target_matches - self.run_baseline.matches
        alive = sum(player.stack > 0 for player in table.players) if game.finished else sum(value > 0 for value in game.hand_start_stacks)
        stage = STAGE_LABELS.get(table.stage.name, table.stage.name)
        self.text(f"{mode} · Partie {current}/{goal} · Main {self.session.hand_in_match} · {alive} en lice · {stage}", 34, 78, 13, MUTED)
        self.button("Choisir les agents", 786, 30, 220, self.back, hint="Quitter la main et revenir à la configuration de la table.")
        self.button("Reprendre" if self.paused else "Pause", 554, 30, 120,
                    self.toggle_pause, active=self.paused, hint="Suspendre les prochains tours automatiques / reprendre la partie.")
        next_label = "Partie suivante" if self.session.match_finished else "Main suivante"
        self.button(next_label, 34, 114, 180, self.next_hand, enabled=game.finished and self.pending is None,
                    hint="Démarrer une nouvelle partie et réinitialiser les stacks." if self.session.match_finished
                    else "Continuer la même partie : les stacks sont conservés et le dealer avance.")
        self.button("Masquer les cartes" if self.reveal else "Afficher les cartes", 236, 114, 200,
                    self.toggle_cards, active=self.reveal, hint="Afficher ou masquer les cartes de tous les joueurs pour le débogage.")
        if self.session.mode == "train" and self.is_complete:
            self.button("Voir ces agents jouer", 458, 114, 260, self.watch_trained,
                        enabled=self.pending is None, primary=True,
                        hint="Passer en inférence en conservant exactement les instances entraînées.")
        for seat, player in enumerate(table.players):
            x = 34 + seat * 196
            current = seat == game.current_seat
            eliminated = game.hand_start_stacks[seat] == 0 or (game.terminated and player.stack == 0)
            self.rect(x, 186, 184, 184, ACCENT if current else DISABLED if eliminated else PANEL)
            self.text(f"Joueur {seat + 1}" + (" · Dealer" if seat == table.dealer_pos else ""), x + 12, 200, 12)
            label = self.session.names[seat][:16] + " · train" if seat in self.session.training_seats else self.session.names[seat][:22]
            self.text(label, x + 12, 227, 11)
            show = self.reveal or (self.session.human_turn and current) or (
                game.terminated and table.stage.name == "SHOWDOWN" and not table.player_cycle.folder[seat])
            self.text("—" if not player.cards else " ".join(player.cards) if show else "??  ??", x + 12, 263, 20)
            self.text(f"Stack : {player.stack:g}", x + 12, 303, 12)
            if self.equity_values is not None:
                prefix = "" if game.terminated else "≈ "
                equity_label = f"EQ : {prefix}{self.equity_values[seat] * 100:.1f}%"
            else:
                equity_label = "EQ : indisponible" if self.equity_error else "EQ : calcul…"
            self.text(equity_label, x + 12, 325, 11, TEXT if current else ACCENT)
            status = "Éliminé" if eliminated else "Couché" if table.player_cycle.folder[seat] else ("Tapis" if player.stack == 0 else player.last_action_in_stage)
            self.text(status[:22], x + 12, 351, 10, TEXT if current else MUTED)
        self.rect(34, 397, 972, 116)
        self.text("BOARD", 54, 416, 11, MUTED)
        self.text("  ".join(table.table_cards) or "En attente du flop", 54, 445, 23)
        pot = 0 if game.terminated else table.community_pot + table.current_round_pot
        self.text(f"Pot : {pot:g} jetons", 708, 418, 16)
        if not game.finished:
            player = table.current_player
            to_call = min(player.stack, max(0, table.min_call - table.player_pots[player.seat]))
            self.text(f"À suivre : {to_call:g}", 708, 456, 13, MUTED)
        if game.finished:
            if self.session.match_finished:
                heading = "Partie interrompue" if self.session.match_truncated else f"Partie gagnée par le joueur {self.session.match_winner + 1}"
            else:
                heading = "Main terminée — la partie continue"
            self.text(heading, 34, 547, 17)
            summary = " | ".join(f"J{seat + 1} : {reward:+g} BB" for seat, reward in enumerate(game.rewards()))
            self.text(summary, 34, 587, 13)
        else:
            human = self.session.human_turn
            status = "action envoyée…" if self.submitted else "cliquez sur une action" if human else "l'agent réfléchit"
            self.text(f"Joueur {game.current_seat + 1} — {status}", 34, 547, 17)
            if human:
                mask = game.action_masks()
                for action in range(8):
                    self.button(LABELS[action], 34 + (action % 4) * 246, 583 + (action // 4) * 50, 234,
                                lambda action=action: self.human_action(action),
                                enabled=bool(mask[action]) and self.pending_kind == "action" and not self.submitted,
                                shortcut=SHORTCUTS[action],
                                hint=f"{LABELS[action]} — touche {SHORTCUTS[action]}." if mask[action] and not self.submitted
                                else "Votre action a été envoyée. Attendez le prochain tour." if self.submitted
                                else "Cette action n'est pas autorisée à ce tour.")
            else:
                self.text("Les décisions de cet agent sont exécutées automatiquement.", 34, 592, 12, MUTED)
        self.draw_stats()

    @property
    def is_complete(self):
        return self.session is not None and self.session.stats.matches >= self.target_matches

    def run_report(self):
        stats, baseline = self.session.stats, self.run_baseline
        completed = stats.completed - baseline.completed
        truncated = stats.truncated - baseline.truncated
        mean = tuple((total - baseline.net_bb[i]) / completed if completed else 0.0
                     for i, total in enumerate(stats.net_bb))
        transitions = tuple(value - baseline.transitions[i] for i, value in enumerate(stats.transitions))
        return completed, truncated, mean, transitions

    def draw_stats(self):
        completed, truncated, mean, transitions = self.run_report()
        stats = self.session.stats
        matches = stats.matches - self.run_baseline.matches
        cut = stats.matches_truncated - self.run_baseline.matches_truncated
        goal = self.target_matches - self.run_baseline.matches
        self.rect(34, 712, 972, 64)
        self.text(f"Parties : {matches}/{goal} · Mains jouées : {completed + truncated} · Parties interrompues : {cut}", 48, 723, 11)
        self.text("Gain moyen (BB/main) : " + " | ".join(f"J{i + 1} {value:+.2f}" for i, value in enumerate(mean)),
                  48, 748, 11, MUTED)

    def draw_fast_table(self):
        mode = "Entraînement" if self.session.mode == "train" else "Voir jouer"
        self.text(f"{mode} · Mode rapide", 34, 78, 16, MUTED)
        self.button("Choisir les agents", 786, 30, 220, self.back, hint="Arrêter la session après l'action en cours et revenir au menu.")
        self.button("Reprendre" if self.paused else "Pause", 554, 30, 120, self.toggle_pause, active=self.paused)
        self.button("Passer au mode visuel", 34, 114, 260, self.show_table,
                    hint="Continuer cette session avec la table et l'equity visibles.")
        if self.session.mode == "train" and self.is_complete:
            self.button("Voir ces agents jouer", 316, 114, 260, self.watch_trained,
                        enabled=self.pending is None, primary=True,
                        hint="Observer les mêmes agents après leur entraînement, sans les recréer.")
        completed, truncated, mean, transitions = self.run_report()
        goal = self.target_matches - self.run_baseline.matches
        done = self.session.stats.matches - self.run_baseline.matches
        self.rect(34, 186, 972, 414)
        self.text(f"Progression : {done}/{goal} parties · {completed + truncated} mains jouées", 54, 205, 19)
        self.rect(54, 249, 930, 10, BACKGROUND)
        self.rect(54, 249, 930 * min(1, done / goal), 10, ACCENT)
        for seat, name in enumerate(self.session.names):
            y = 289 + seat * 58
            role = "Apprentissage" if seat in self.session.training_seats else "Inférence"
            self.text(f"Joueur {seat + 1} · {name} · {role}", 54, y, 13)
            wins = self.session.stats.wins[seat] - self.run_baseline.wins[seat]
            self.text(f"Victoires : {wins} · Gain moyen : {mean[seat]:+.2f} BB/main · Transitions reçues : {transitions[seat]}", 54, y + 24, 11, MUTED)
        status = "Session terminée" if self.is_complete else "Session en pause" if self.paused else "Simulation en cours…"
        self.text(status, 34, 633, 19)
        self.text("Le mode visuel permet de regarder les mains et leurs equities.", 34, 672, 12, MUTED)
        self.draw_stats()

    def can_train(self, seat):
        plugin = self.registry.agents.get(self.selected[seat])
        return plugin is not None and plugin.trainable is not False

    def set_mode(self, mode):
        self.mode = mode
        if self.episodes_text in ("1", "100"):
            self.episodes_text = "100" if mode == "train" else "1"
        self.fast = mode == "train"
        if mode == "train":
            for seat in range(self.count):
                if not self.can_train(seat):
                    self.training[seat] = False
            if not any(self.training[:self.count]):
                eligible = next((seat for seat in range(self.count) if self.can_train(seat)), None)
                if eligible is not None:
                    self.training[eligible] = True
            self.notify("Entraînement sélectionné : cochez les sièges qui doivent apprendre.")
        else:
            self.notify("Voir jouer sélectionné : tous les agents restent en inférence.")
        self.editing = None

    def toggle_training(self, seat):
        if self.mode == "train" and self.can_train(seat):
            self.training[seat] = not self.training[seat]
            self.notify(f"Joueur {seat + 1} : " + ("apprentissage activé." if self.training[seat] else "adversaire en inférence."))

    def edit_episodes(self):
        self.editing = "episodes"
        self.replace_text = True
        self.notify("Tapez le nombre de parties complètes, puis Entrée.")

    def edit_max_hands(self):
        self.editing = "max_hands"
        self.replace_text = True
        self.notify("Tapez la limite de mains par partie. Une partie à la limite est interrompue sans gagnant.")

    def toggle_auto(self):
        self.auto_continue = not self.auto_continue
        if not self.auto_continue:
            self.fast = False
        self.notify("Enchaînement automatique activé." if self.auto_continue else "Les mains suivantes seront lancées manuellement.")

    def toggle_fast(self):
        self.fast = not self.fast
        self.notify("Mode rapide activé : suivi des statistiques." if self.fast else "Mode visuel activé : table et equity visibles.")

    def set_count(self, count):
        self.count = count
        self.editing = None
        self.notify(f"{count} joueurs sélectionnés. " + ("Tous les sièges participent." if count == 5 else "Les autres sièges ne participeront pas."))

    def cycle(self, seat, offset):
        if self.available:
            index = self.available.index(self.selected[seat]) if self.selected[seat] in self.available else 0
            self.selected[seat] = self.available[(index + offset) % len(self.available)]
            if not self.can_train(seat):
                self.training[seat] = False
            self.notify(f"Joueur {seat + 1} : agent {self.selected[seat]} sélectionné.")
        self.editing = None

    def edit(self, seat):
        self.editing = seat
        self.replace_text = True
        self.notify(f"Paramètres du joueur {seat + 1} sélectionnés. Tapez le JSON, puis Entrée.")

    def refresh(self):
        try:
            self.registry = AgentRegistry(self.agents_dir)
            self.available = list(self.registry.agents)
            for seat, name in enumerate(self.selected):
                if name not in self.available:
                    self.selected[seat] = self.available[0] if self.available else ""
                if not self.can_train(seat):
                    self.training[seat] = False
            self.error = ""
            self.notify(f"Catalogue actualisé : {len(self.available)} agent(s) disponible(s).")
        except Exception as exc:
            self.error = str(exc)

    def start(self):
        try:
            if not self.episodes_text.isdecimal():
                raise ValueError("Le nombre de parties doit être un entier supérieur à zéro.")
            episodes = int(self.episodes_text)
            if episodes < 1:
                raise ValueError("Le nombre de parties doit être supérieur à zéro.")
            if not self.max_hands_text.isdecimal() or int(self.max_hands_text) < 1:
                raise ValueError("La limite de mains par partie doit être un entier supérieur à zéro.")
            max_hands = int(self.max_hands_text)
            selections = [(self.selected[i], json.loads(self.kwargs[i])) for i in range(self.count)]
            if not all(isinstance(kwargs, dict) for _, kwargs in selections):
                raise ValueError("Les paramètres de chaque joueur doivent être un objet JSON, par exemple {}.")
            seats = [i for i in range(self.count) if self.training[i]] if self.mode == "train" else []
            if self.mode == "train" and not seats:
                raise ValueError("Cochez au moins un agent entraînable. human et random n'ont pas de méthode observe().")
            self.target_matches = episodes
            self.completed_notified = False
            self.loading = True
            self.editing = None
            self.error = ""
            self.pending_kind = "load"
            mode = self.mode
            self.pending = _background(lambda: TableSession(self.registry, selections, mode=mode,
                                                           training_seats=seats, defer_callbacks=True, continuous=True,
                                                           max_hands_per_match=max_hands))
            self.notify("Clic reçu : chargement des agents et préparation de la table…")
        except Exception as exc:
            self.error = str(exc)

    def tick(self, dt):
        if self.pending is not None and self.pending.done():
            future, self.pending = self.pending, None
            kind, self.pending_kind = self.pending_kind, None
            try:
                result = future.result()
                if kind == "load":
                    self.session = result
                    self.run_baseline = result.stats
                    self.paused = False
                    self.loading = False
                    if self.session.has_human and self.fast:
                        self.fast = False
                        self.notify("Session lancée en mode visuel : un siège humain attend vos actions.")
                    else:
                        self.notify(f"{'Entraînement' if self.mode == 'train' else 'Observation'} lancé : {self.count} joueurs, {self.target_matches} parties.")
                elif kind == "action":
                    seat = self.session.game.current_seat
                    self.session.play(result)
                    self.message = f"Joueur {seat + 1} : {LABELS[int(getattr(result, 'value', result))]}"
                    if self.session.game.finished:
                        self.pending_kind = "finish"
                        self.pending = _background(self.session.finish_hand)
                elif kind == "batch" and self.requested_visual:
                    self.fast = False
                    self.requested_visual = False
                    self.session.pause_event.clear()
                    self.clear_equity()
                    self.notify("Mode visuel activé : la session continue avec les mêmes agents.")
                elif kind == "reset":
                    self.clear_equity()
                    self.notify(f"Partie {self.session.match}, main {self.session.hand_in_match} distribuée.")
                    if self.session.game.finished:
                        self.pending_kind = "finish"
                        self.pending = _background(self.session.finish_hand)
                self.submitted = False
                delay = 1.0 if kind == "finish" and self.session.mode == "infer" else 0.4
                self.next_action_at = time.monotonic() + (0 if self.fast else delay)
            except BaseException as exc:
                self.loading = False
                self.paused = True
                if self.session:
                    self.session.pause_event.set()
                self.error = f"{type(exc).__name__} : {exc}"
        if self.session is not None and self.pending is None:
            if self.is_complete:
                if not self.completed_notified:
                    self.completed_notified = True
                    self.notify("Session terminée. " + ("Vous pouvez regarder les agents entraînés jouer." if self.session.mode == "train" else "Toutes les parties demandées ont été jouées."))
            elif not self.paused and time.monotonic() >= self.next_action_at:
                if self.fast:
                    session, target = self.session, self.target_matches
                    self.pending_kind = "batch"
                    self.pending = _background(lambda: session.run_batch(target))
                elif self.session.game.finished and self.auto_continue:
                    self.pending_kind = "reset"
                    self.pending = _background(self.session.advance)
                elif not self.session.game.finished:
                    self.pending_kind = "action"
                    self.pending = _background(self.session.choose_action)
        self.update_equity()

    def update_equity(self):
        if self.session is None or self.fast or self.pending_kind == "reset":
            return
        snapshot = EquitySnapshot.from_game(self.session.game)
        if self.equity_pending is not None and self.equity_pending.done():
            try:
                result = self.equity_pending.result()
            except Exception as exc:
                result = str(exc)
            if self.equity_job_snapshot == snapshot:
                self.equity_cache[snapshot] = result
            self.equity_pending = None
            self.equity_job_snapshot = None
        cached = self.equity_cache.get(snapshot)
        self.equity_values = cached if isinstance(cached, tuple) else None
        self.equity_error = cached if isinstance(cached, str) else ""
        if snapshot not in self.equity_cache and self.equity_pending is None:
            self.equity_job_snapshot = snapshot
            self.equity_pending = _background(lambda: estimate_equities(snapshot))

    def clear_equity(self):
        self.equity_pending = None
        self.equity_job_snapshot = None
        self.equity_cache.clear()
        self.equity_values = None
        self.equity_error = ""

    def human_action(self, action):
        if self.pending_kind == "action" and self.session and self.session.human_turn and not self.submitted:
            try:
                self.session.submit_human_action(action)
                self.submitted = True
                self.notify(f"Action envoyée pour le joueur {self.session.game.current_seat + 1} : {LABELS[action]}.")
            except ValueError as exc:
                self.error = str(exc)

    def next_hand(self):
        if self.session and self.session.game.finished and self.pending is None:
            try:
                if self.is_complete:
                    self.target_matches += 1
                    self.completed_notified = False
                self.pending_kind = "reset"
                self.pending = _background(self.session.advance)
                self.clear_equity()
                self.error = ""
                self.paused = False
                self.session.pause_event.clear()
                self.notify("Préparation de la prochaine main…")
            except Exception as exc:
                self.error = str(exc)
                self.paused = True

    def back(self):
        if self.session:
            self.session.close()
        self.session = None
        self.clear_equity()
        self.pending = None
        self.pending_kind = None
        self.loading = False
        self.submitted = False
        self.requested_visual = False
        self.error = ""
        self.notify("Retour au choix des agents. Vous pouvez modifier la table.")

    def toggle_pause(self):
        self.paused = not self.paused
        if self.session:
            if self.paused:
                self.session.pause_event.set()
            else:
                self.session.pause_event.clear()
        self.notify("Pause activée : les prochains tours automatiques sont suspendus." if self.paused else "Partie reprise.")

    def show_table(self):
        if self.pending_kind == "batch":
            self.requested_visual = True
            self.session.pause_event.set()
            self.notify("Passage au mode visuel après l'action en cours…")
        else:
            self.fast = False
            self.clear_equity()
            self.notify("Mode visuel activé.")

    def watch_trained(self):
        if self.session and self.is_complete and self.pending is None:
            self.run_baseline = self.session.stats
            self.target_matches = self.run_baseline.matches + 1
            self.completed_notified = False
            self.mode = "infer"
            self.fast = False
            self.paused = False
            self.auto_continue = True
            self.session.pause_event.clear()
            self.pending_kind = "reset"
            self.pending = _background(self.session.use_inference)
            self.clear_equity()
            self.notify("Observation des agents entraînés : leur apprentissage est conservé, sans mise à jour supplémentaire.")

    def toggle_cards(self):
        self.reveal = not self.reveal
        self.notify("Toutes les cartes sont affichées." if self.reveal else "Cartes adverses masquées.")

    def update_layout(self, width, height):
        self.ui_scale = max(0.01, min(width / UI_WIDTH, height / UI_HEIGHT))
        self.ui_offset_x = (width - UI_WIDTH * self.ui_scale) / 2
        self.ui_offset_y = (height - UI_HEIGHT * self.ui_scale) / 2

    def on_resize(self, width, height):
        self.update_layout(width, height)
        self.pressed_button = None
        self.mouse_position = (-1, -1)

    def screen_to_ui(self, x, y):
        return ((x - self.ui_offset_x) / self.ui_scale,
                UI_HEIGHT - (y - self.ui_offset_y) / self.ui_scale)

    def toggle_fullscreen(self):
        self.window.set_fullscreen(not self.window.fullscreen)
        self.on_resize(*self.window.get_size())
        self.notify("Plein écran activé. F11 ou Échap pour revenir." if self.window.fullscreen else "Mode fenêtré activé.")

    def on_mouse_press(self, x, y, button, modifiers):
        if button != mouse.LEFT:
            return
        self.mouse_position = self.screen_to_ui(x, y)
        control = self.button_at(*self.mouse_position)
        if control and control.enabled:
            self.pressed_button = control.identity
        elif control:
            self.notify(control.hint)
        else:
            self.editing = None

    def on_mouse_release(self, x, y, button, modifiers):
        if button != mouse.LEFT:
            return
        self.mouse_position = self.screen_to_ui(x, y)
        control = self.button_at(*self.mouse_position)
        pressed, self.pressed_button = self.pressed_button, None
        if control and control.enabled and control.identity == pressed:
            self.flash_button = control.identity
            self.flash_until = time.monotonic() + 0.35
            control.callback()

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_position = self.screen_to_ui(x, y)
        control = self.button_at(*self.mouse_position)
        cursor = self.window.CURSOR_DEFAULT
        if control and control.enabled:
            cursor = self.window.CURSOR_TEXT if control.field else self.window.CURSOR_HAND
        self.window.set_mouse_cursor(self.window.get_system_mouse_cursor(cursor))

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.on_mouse_motion(x, y, dx, dy)

    def on_mouse_leave(self, x, y):
        self.mouse_position = (-1, -1)
        self.window.set_mouse_cursor(self.window.get_system_mouse_cursor(self.window.CURSOR_DEFAULT))

    def on_text(self, text):
        if self.editing is not None and not self.loading and text.isprintable():
            if self.editing in ("episodes", "max_hands"):
                if text.isdecimal():
                    attribute = "episodes_text" if self.editing == "episodes" else "max_hands_text"
                    setattr(self, attribute, text if self.replace_text else getattr(self, attribute) + text)
                    self.replace_text = False
                return
            if self.replace_text:
                self.kwargs[self.editing] = ""
                self.replace_text = False
            self.kwargs[self.editing] += text

    def on_key_press(self, symbol, modifiers):
        if symbol == key.F11:
            self.toggle_fullscreen()
            return
        if symbol == key.ESCAPE and self.window.fullscreen:
            self.toggle_fullscreen()
            return
        if self.editing is not None:
            if symbol == key.BACKSPACE:
                if self.editing in ("episodes", "max_hands"):
                    attribute = "episodes_text" if self.editing == "episodes" else "max_hands_text"
                    setattr(self, attribute, "" if self.replace_text else getattr(self, attribute)[:-1])
                else:
                    self.kwargs[self.editing] = "" if self.replace_text else self.kwargs[self.editing][:-1]
                self.replace_text = False
            elif symbol == key.A and modifiers & key.MOD_CTRL:
                self.replace_text = True
            elif symbol in (key.ENTER, key.ESCAPE):
                self.editing = None
            return
        if symbol == key.ESCAPE:
            if self.session:
                self.back()
            elif not self.loading:
                self.window.close()
        elif self.session and symbol in KEY_ACTIONS:
            self.human_action(KEY_ACTIONS[symbol])

    def on_close(self):
        pyglet.clock.unschedule(self.tick)
        self.back()

    def run(self):
        try:
            pyglet.app.run()
        finally:
            self.on_close()
            self.viewer.close()


def main():
    PokerApp().run()


if __name__ == "__main__":
    main()

