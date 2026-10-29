from concurrent.futures import Future
from collections import deque
import json
from threading import Event, Thread
import time

import pyglet
from pyglet.window import key, mouse

from gym_env.rendering import PygletWindow
from poker_rl.plugins import AgentRegistry, DEFAULT_AGENTS_DIR
from poker_rl.table_session import TableSession
from poker_rl.display_equity import EquitySnapshot, estimate_equities
from poker_rl.arena import run_arena
from poker_rl.sb3_training import train_sb3_session
from poker_rl.records import SessionRecorder, save_model_version
from poker_rl.hand_analysis import DIMENSION_LABELS, CATEGORY_LABELS, ACTION_LABELS, STREET_LABELS, filter_hands

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


class Button:
    def __init__(self,
                 x, y, width,
                 height, callback, enabled,
                 hint, field=False):  # Enregistre les données
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.callback = callback
        self.enabled = enabled
        self.hint = hint
        self.field = field

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
        self.count = 2
        self.available = self.registry.choices(self.count)
        default = "human" if "human" in self.available else (self.available[0] if self.available else "")
        self.selected = [default] * 5
        self.kwargs = ["{}"] * 5
        self.mode = "infer"
        self.native_training_complete = False
        self.record_sessions = True
        self.library_open = False
        self.library_index = 0
        self.history_lines = None
        self.history_page = 0
        self.metrics_open = False
        self.metrics_report = None
        self.metrics_source = None
        self.metrics_index = 0
        self.metrics_tab = "summary"
        self.metrics_page = 0
        from poker_rl.campaign_ui import CampaignUI
        self.campaign = CampaignUI(self)
        self.analysis_parent = 'chart'
        self.analysis_view = "chart"
        self.analysis_benchmark = 0
        self.analysis_dimension = "combination"
        self.analysis_unit = "count"
        self.analysis_category = None
        self.analysis_outcome = "loss"
        self.analysis_hands = None
        self.analysis_page = 0
        self.analysis_action_page = 0
        self.analysis_selected = None
        self.picker_seat = None
        self.picker_page = 0
        self.validation_text = "500"
        self.arena_report = None
        self.arena_stop = Event()
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
        if len(color) == 3:
            color = (*color, 255)
        self.viewer.text(content, x, y, font_size=size, color=color)

    def rect(self, x, y, width, height, color=PANEL):
        if len(color) == 3:
            color = (*color, 255)
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
        if self.metrics_open:
            if self.metrics_tab == "hands":
                self.draw_hand_analysis()
            else:
                self.draw_metrics()
        elif self.campaign.visible:
            self.campaign.draw()
        elif self.picker_seat is not None:
            self.draw_picker()
        elif self.library_open:
            self.draw_library()
        elif self.arena_report is not None:
            self.draw_arena_results()
        elif self.session is None:
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
        self.button('Campagnes / ligue', 548, 30, 220, self.campaign.open, enabled=not self.loading,
                    hint='Entraîner 3 à 5 seeds, suivre la ligue et comparer les profondeurs de stack.')
        self.text("Mode", 34, 120, 12, MUTED)
        self.button("Voir jouer", 176, 104, 186, lambda: self.set_mode("infer"), active=self.mode == "infer",
                    enabled=not self.loading, hint="Faire jouer les agents avec training=False, sans apprentissage.")
        self.button("Entraînement", 384, 104, 210, lambda: self.set_mode("train"), active=self.mode == "train",
                    enabled=not self.loading, hint="Envoyer les transitions aux agents cochés pour qu'ils apprennent.")
        self.button("Arène", 614, 104, 170, lambda: self.set_mode("arena"), active=self.mode == "arena",
                    enabled=not self.loading, hint="Comparer les agents sur des parties complètes, sans apprentissage.")
        self.button("Bibliothèque", 804, 104, 202, self.open_library, enabled=not self.loading,
                    hint="Retrouver les modèles sauvegardés, leurs adversaires et leurs coups.")
        self.text("Nombre de joueurs", 34, 170, 12, MUTED)
        for count in range(2, 6):
            self.button(f"{count} joueurs", 218 + (count - 2) * 116, 154, 104,
                        lambda count=count: self.set_count(count), active=count == self.count,
                        enabled=not self.loading, hint=f"Activer uniquement {count} sièges à la table.")
        self.button("Actualiser les agents", 786, 154, 220, self.refresh, enabled=not self.loading,
                    hint="Rechercher les nouveaux agents déposés dans agents/.")
        selector_hint = "Cliquez sur un bot pour ouvrir la liste" if len(self.available) > 1 else "Agents disponibles"
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
            self.button(self.registry.label(self.selected[seat])[:24] or "Aucun agent", 170, y + 13, 212,
                        lambda seat=seat: self.open_picker(seat), enabled=not self.loading,
                        hint="Ouvrir la liste des bots et consulter leurs métriques.")
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
        native = self.native_ppo_selected()
        self.text("Décisions" if native else "Parties", 34, 637, 12, MUTED)
        self.button(self.episodes_text, 104, 625, 100, self.edit_episodes, field=True,
                    active=self.editing == "episodes", enabled=not self.loading,
                    hint="Nombre de décisions PPO ; SB3 termine son lot de collecte." if native else "Nombre de parties complètes.")
        self.button("Enchaîner les mains", 226, 625, 224, self.toggle_auto,
                    active=self.auto_continue, enabled=not self.loading, hint="Enchaîner les mains avec les mêmes stacks, puis démarrer les parties suivantes.")
        self.button("Rapide", 472, 625, 140, self.toggle_fast, active=self.fast,
                    enabled=not self.loading and self.auto_continue, hint="Simuler rapidement sans dessiner chaque main ; les sièges humains imposent le mode visuel.")
        self.text("Max. mains / partie", 640, 637, 11, MUTED)
        self.button(self.max_hands_text, 820, 625, 186, self.edit_max_hands, field=True,
                    active=self.editing == "max_hands", enabled=not self.loading,
                    hint="Au-delà de cette limite, une partie est comptée comme interrompue, sans gagnant.")
        if self.mode == "train":
            self.text("Validation : mains / adversaire", 34, 674, 11, MUTED)
            self.button(self.validation_text, 302, 664, 102, lambda: self.edit_field("validation"),
                        height=30, field=True, active=self.editing == "validation", enabled=not self.loading,
                        hint="Nouvelles mains après entraînement, contre random puis check/call ; aucune mise à jour.")
            self.text("Calcul automatique, sans arène", 426, 674, 11, ACCENT)
        else:
            self.text("Une partie = plusieurs mains, stacks conservés, jusqu'au dernier joueur.", 34, 674, 11, MUTED)
        self.button("Archives : oui" if self.record_sessions or self.mode == "arena" else "Archives : non",
                    820, 666, 186, self.toggle_recording, height=30,
                    enabled=not self.loading and self.mode != "arena",
                    hint="Conserver l'historique et actualiser le bot courant. Une version supplémentaire exige Enregistrer une version.")
        launch_label = "Démarrer l'entraînement" if self.mode == "train" else "Lancer l'arène" if self.mode == "arena" else "Voir jouer"
        self.button("Chargement…" if self.loading else launch_label, 34, 710, 242, self.start,
                    enabled=bool(self.available) and not self.loading, primary=True,
                    hint="Chargement des agents en cours…" if self.loading else "Démarrer avec les sièges et agents sélectionnés.")
        human_opponents = self.native_human_opponents()
        if human_opponents and "random" in self.registry.agents:
            self.button("Entraîner contre random", 304, 710, 350, self.train_against_random,
                        enabled=not self.loading,
                        hint="Remplacer les adversaires humains par random et lancer l'entraînement PPO.")
            if 'mixed' in self.registry.agents:
                self.button('Styles variés', 676, 710, 330, self.train_against_mixed, enabled=not self.loading,
                            hint='Entraîner face à un mélange aléatoire, passif, serré et agressif.')
        elif self.native_ppo_selected() and 'mixed' in self.registry.agents:
            self.button('Entraîner contre des styles variés', 304, 710, 440,
                        self.train_against_mixed, enabled=not self.loading)
        else:
            self.text(f"{len(self.available)} choix pour cette table · {len(self.registry.errors)} indisponible(s)",
                      304, 722, 12, MUTED)
        if self.loading and self.pending_kind in ("arena", "ppo_train"):
            self.button("Arrêter l'arène" if self.pending_kind == "arena" else "Arrêter l'entraînement",
                        786, 710, 220, self.arena_stop.set,
                        hint="Conserver les parties déjà terminées et enregistrer le résultat partiel.")
        if self.error:
            self.text(self.error[:125], 34, 762, 11, ERROR)
        elif human_opponents:
            seats = ", ".join(str(seat + 1) for seat in human_opponents)
            self.text(f"PPO attend des adversaires automatiques. Joueur(s) {seats} : human ; choisissez random.",
                      34, 762, 11, ERROR)
        elif self.registry.errors:
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
        heading = "PPO terminé" if self.native_training_complete else f"{mode} · Partie {current}/{goal}"
        self.text(f"{heading} · Main {self.session.hand_in_match} · {alive} en lice · {stage}", 34, 78, 13, MUTED)
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
            self.button("Métriques", 740, 114, 110, self.open_training_metrics,
                        enabled=self.pending is None)
            self.button("Bibliothèque", 862, 114, 144, self.open_library,
                        enabled=self.pending is None)
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
        return self.session is not None and (self.native_training_complete or self.session.stats.matches >= self.target_matches)

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
        progress = f"PPO : {self.session.sb3_timesteps} décisions" if self.native_training_complete else f"Parties : {matches}/{goal}"
        self.text(f"{progress} · Mains jouées : {completed + truncated} · Parties interrompues : {cut}", 48, 723, 11)
        self.text("Gain moyen (BB/main) : " + " | ".join(f"J{i + 1} {value:+.2f}" for i, value in enumerate(mean)),
                  48, 748, 11, MUTED)
        self.text("Victoires : " + " · ".join(f"J{i + 1} {value - self.run_baseline.wins[i]}"
                  for i, value in enumerate(stats.wins)), 650, 748, 11, ACCENT)

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
            self.button("Métriques du bot", 596, 114, 182, self.open_training_metrics,
                        enabled=self.pending is None)
            self.button("Bibliothèque", 800, 114, 206, self.open_library,
                        enabled=self.pending is None)
        completed, truncated, mean, transitions = self.run_report()
        goal = self.target_matches - self.run_baseline.matches
        done = self.session.stats.matches - self.run_baseline.matches
        self.rect(34, 186, 972, 414)
        progress = f"PPO : {self.session.sb3_timesteps} décisions" if self.native_training_complete else f"Progression : {done}/{goal} parties"
        self.text(f"{progress} · {completed + truncated} mains jouées", 54, 205, 17)
        self.rect(54, 249, 930, 10, BACKGROUND)
        fraction = 1 if self.native_training_complete else min(1, done / max(1, goal))
        self.rect(54, 249, 930 * fraction, 10, ACCENT)
        for seat, name in enumerate(self.session.names):
            y = 289 + seat * 58
            role = "Apprentissage" if seat in self.session.training_seats else "Inférence"
            self.text(f"Joueur {seat + 1} · {name} · {role}", 54, y, 13)
            wins = self.session.stats.wins[seat] - self.run_baseline.wins[seat]
            self.text(f"Victoires : {wins} · Gain moyen : {mean[seat]:+.2f} BB/main · Transitions reçues : {transitions[seat]}", 54, y + 24, 11, MUTED)
        status = "Session terminée" if self.is_complete else "Session en pause" if self.paused else "Simulation en cours…"
        self.text(status, 34, 633, 19)
        self.text("Les gains affichés ici sont ceux de l'entraînement ; la validation figure dans Métriques du bot."
                  if self.session.mode == "train" else "Le mode visuel permet de regarder les mains et leurs equities.",
                  34, 672, 12, MUTED)
        self.draw_stats()

    def can_train(self, seat):
        plugin = self.registry.agents.get(self.selected[seat])
        return plugin is not None and plugin.trainable is not False

    def native_ppo_selected(self):
        return self.mode == "train" and any(self.training[i] and self.registry.native_training(self.selected[i])
                                            for i in range(self.count))

    def native_human_opponents(self):
        if not self.native_ppo_selected():
            return []
        seats = []
        for seat in range(self.count):
            name = self.selected[seat]
            saved = self.registry.saved_models.get(name)
            plugin = self.registry.agents.get(saved["name"] if saved else name)
            if plugin and callable(getattr(plugin.factory, "enable_window_input", None)):
                seats.append(seat)
        return seats

    def train_against_random(self):
        if self.loading:
            return
        for seat in self.native_human_opponents():
            self.selected[seat] = "random"
            self.kwargs[seat] = "{}"
            self.training[seat] = False
        self.start()

    def train_against_mixed(self):
        if self.loading:
            return
        for seat in range(self.count):
            if not self.training[seat]:
                self.selected[seat] = 'mixed'
                self.kwargs[seat] = json.dumps({'seed':42 + seat})
        self.start()

    def set_mode(self, mode):
        self.mode = mode
        if self.episodes_text in ("1", "100"):
            self.episodes_text = "100" if mode == "train" else "1"
        self.fast = mode in ("train", "arena")
        if mode == "train":
            for seat in range(self.count):
                if not self.can_train(seat):
                    self.training[seat] = False
            if not any(self.training[:self.count]):
                eligible = next((seat for seat in range(self.count) if self.can_train(seat)), None)
                if eligible is not None:
                    self.training[eligible] = True
            self.notify("Entraînement sélectionné : cochez les sièges qui doivent apprendre.")
            if self.native_ppo_selected() and self.episodes_text == "100":
                self.episodes_text = "10000"
        elif mode == "arena":
            self.episodes_text = "100" if self.episodes_text == "1" else self.episodes_text
            self.notify("Arène : sélectionnez les modèles ou agents à comparer ; aucune mise à jour des poids.")
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
        self.notify("Tapez le nombre de décisions PPO, puis Entrée." if self.native_ppo_selected()
                    else "Tapez le nombre de parties complètes, puis Entrée.")

    def edit_field(self, name):
        self.editing = name
        self.replace_text = True

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
        self.refresh()
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
            self.registry = AgentRegistry(self.agents_dir, records_dir=self.registry.records_dir)
            self.available = self.registry.choices(self.count)
            for seat, name in enumerate(self.selected):
                if name not in self.available:
                    base = self.registry.resolve_model(name)
                    base_name = base["name"] if base else name
                    replacement = next((key for key in self.available if
                        self.registry.saved_models.get(key, {}).get("name") == base_name), None)
                    self.selected[seat] = replacement or (base_name if base_name in self.available else
                        self.available[0] if self.available else "")
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
            validation_text = getattr(self, "validation_text", "500")
            if not validation_text.isdecimal() or int(validation_text) < 1:
                raise ValueError("La validation nécessite au moins une main par adversaire.")
            validation_hands = int(validation_text)
            selections = [(self.selected[i], json.loads(self.kwargs[i])) for i in range(self.count)]
            if not all(isinstance(kwargs, dict) for _, kwargs in selections):
                raise ValueError("Les paramètres de chaque joueur doivent être un objet JSON, par exemple {}.")
            seats = [i for i in range(self.count) if self.training[i]] if self.mode == "train" else []
            if self.mode == "train" and not seats:
                raise ValueError("Cochez au moins un agent entraînable. human et random n'ont pas de méthode observe().")
            if self.native_ppo_selected():
                if len(seats) != 1:
                    raise ValueError("Sélectionnez un seul siège à entraîner avec PPO natif SB3.")
                if episodes < 2:
                    raise ValueError("Choisissez au moins 2 décisions PPO.")
                human_opponents = self.native_human_opponents()
                if human_opponents:
                    players = ", ".join(str(seat + 1) for seat in human_opponents)
                    raise ValueError(f"Joueur(s) {players} : human. PPO exige un adversaire automatique. Cliquez sur Entraîner contre random.")
            self.target_matches = episodes
            self.native_training_complete = False
            self.metrics_open = False
            self.metrics_report = None
            self.completed_notified = False
            self.loading = True
            self.editing = None
            self.error = ""
            if self.mode == "arena":
                self.arena_stop = Event()
                self.pending_kind = "arena"
                registry = self.registry
                self.pending = _background(lambda: run_arena(registry, selections, matches=episodes,
                    max_hands_per_match=max_hands, stop_event=self.arena_stop,
                    on_progress=lambda stats: setattr(self, "message", f"Arène : {stats.matches}/{episodes} parties · {stats.hands} mains")))
                self.notify("Arène lancée : les modèles jouent sans apprendre.")
                return
            if self.native_ppo_selected():
                self.arena_stop = Event()
                self.pending_kind = "ppo_train"
                registry, record = self.registry, self.record_sessions
                self.pending = _background(lambda: train_sb3_session(registry, selections,
                    training_seats=seats, total_timesteps=episodes, record=record,
                    validation_hands=validation_hands,
                    on_validation_progress=self.validation_progress,
                    max_hands_per_match=max_hands, stop_event=self.arena_stop,
                    on_progress=lambda done, total, stats: setattr(self, "message",
                        f"PPO Stable-Baselines3 : {done}/{total} décisions · {stats.hands} mains")))
                self.notify("Entraînement PPO lancé avec model.learn() de Stable-Baselines3.")
                return
            self.pending_kind = "load"
            mode = self.mode
            self.pending = _background(lambda: TableSession(self.registry, selections, mode=mode,
                                                           training_seats=seats, defer_callbacks=True, continuous=True,
                                                           max_hands_per_match=max_hands, record=self.record_sessions))
            self.notify("Clic reçu : chargement des agents et préparation de la table…")
        except Exception as exc:
            self.loading = False
            self.error = str(exc)

    def tick(self, dt):
        self.campaign.poll()
        if self.pending is not None and self.pending.done():
            future, self.pending = self.pending, None
            kind, self.pending_kind = self.pending_kind, None
            try:
                result = future.result()
                if kind == "ppo_train":
                    self.session = result
                    self.run_baseline = result.record_baseline
                    self.target_matches = result.stats.matches
                    self.native_training_complete = True
                    self.completed_notified = True
                    self.loading = False
                    self.paused = True
                    self.fast = True
                    self.training_finished_ui()
                elif kind == "arena":
                    self.arena_report = result
                    self.loading = False
                    self.refresh()
                    self.notify("Arène terminée : résultats et historique enregistrés.")
                elif kind == "history":
                    self.history_lines = result
                    self.loading = False
                    self.notify("Les 200 derniers coups sont consultables ; l'archive conserve l'historique complet.")
                elif kind == "archive":
                    self.refresh()
                    errors = [p for p in self.session.recorder.manifest["agents"]
                              if p["training"] and p["checkpoint_status"] != "saved"]
                    if self.session.mode == "train":
                        self.open_training_metrics()
                    if errors:
                        self.error = "Historique enregistré ; modèle non sauvegardé : " + ", ".join(p["label"] for p in errors)
                    else:
                        self.notify("Bot courant actualisé. Les métriques de validation sont consultables directement.")
                elif kind == "validation":
                    self.training_finished_ui()
                elif kind == "save_version":
                    self.loading = False
                    self.refresh()
                    self.notify(f"Version enregistrée : {result}. Elle sera conservée après les prochains entraînements.")
                elif kind == "validate_model":
                    model_id = self.metrics_source["model_id"]
                    self.loading = False
                    self.refresh()
                    self.open_model_metrics(model_id)
                    self.notify("Validation du checkpoint terminée ; métriques enregistrées sans entraînement.")
                elif kind == "analysis_hands":
                    self.analysis_hands = result
                    self.loading = False
                elif kind == "load":
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
                delay = 1.0 if kind == "finish" and self.session and self.session.mode == "infer" else 0.4
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
                    if self.session.mode == "train":
                        self.pending_kind = "validation"
                        session = self.session
                        self.pending = _background(lambda: session.complete_training(
                            validation_hands=int(self.validation_text), on_validation_progress=self.validation_progress))
                        self.notify("Entraînement terminé : validation sur de nouvelles mains…")
                    elif self.session.recorder and not self.session.recorder.closed:
                        self.pending_kind = "archive"
                        self.pending = _background(self.session.finish_recording)
                        self.notify("Session terminée : enregistrement des modèles et statistiques…")
                    else:
                        self.notify("Session terminée. Vous pouvez regarder les agents jouer.")
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
        if self.native_training_complete:
            self.watch_trained()
            return
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
        if self.metrics_open:
            if self.metrics_tab == "hands":
                if self.analysis_view == "detail":
                    self.analysis_view = "list"
                elif self.analysis_view == "list":
                    self.analysis_view = self.analysis_parent
                else:
                    self.metrics_tab = "summary"
                return
            self.metrics_open = False
            return
        if self.campaign.visible:
            self.campaign.leave()
            return
        if self.picker_seat is not None:
            self.picker_seat = None
            return
        self.arena_stop.set()
        session, pending = self.session, self.pending
        if session:
            session.request_stop()
            if pending is not None:
                def cleanup():
                    try:
                        pending.result()
                    except BaseException:
                        pass
                    finally:
                        session.close()
                _background(cleanup)
            else:
                session.close()
        elif pending is not None and self.pending_kind in ("load", "ppo_train"):
            def cleanup_load():
                try:
                    pending.result().close()
                except BaseException:
                    pass
            _background(cleanup_load)
        self.session = None
        self.clear_equity()
        self.pending = None
        self.pending_kind = None
        self.loading = False
        self.submitted = False
        self.requested_visual = False
        self.native_training_complete = False
        self.library_open = False
        self.arena_report = None
        self.error = ""
        self.notify("Retour au choix des agents. Vous pouvez modifier la table.")

    def validation_progress(self, seat, opponent, done, total):
        self.message = f"Validation J{seat + 1} contre {opponent} : {done}/{total} nouvelles mains · poids figés"

    def open_picker(self, seat):
        self.picker_seat, self.picker_page = seat, 0

    def choose_bot(self, name):
        seat = self.picker_seat
        self.selected[seat], self.kwargs[seat] = name, "{}"
        if not self.can_train(seat):
            self.training[seat] = False
        self.picker_seat = None
        self.notify(f"Joueur {seat + 1} : {self.registry.label(name)}")

    def draw_picker(self):
        self.text(f"Choisir le bot du joueur {self.picker_seat + 1}", 34, 86, 19)
        self.button("Retour", 786, 30, 220, self.back)
        self.text("Un bot courant par type et taille de table. Les versions supplémentaires sont enregistrées à votre demande.",
                  34, 128, 11, MUTED)
        pages = max(1, (len(self.available) + 7) // 8)
        self.picker_page %= pages
        for row, name in enumerate(self.available[self.picker_page * 8:(self.picker_page + 1) * 8]):
            y = 175 + row * 64
            self.rect(34, y, 972, 54)
            self.button(self.registry.label(name)[:48], 44, y + 8, 530,
                        lambda name=name: self.choose_bot(name), height=38,
                        active=self.selected[self.picker_seat] == name)
            self.button("Métriques", 596, y + 8, 180, lambda name=name: self.open_model_metrics(name),
                        enabled=name in self.registry.saved_models,
                        hint="Voir la validation de ce checkpoint." if name in self.registry.saved_models
                        else "Ce bot n'a pas encore de modèle entraîné pour cette taille de table.")
            if name in self.registry.saved_models and self.registry.native_training(name):
                self.button("Nouveau modèle", 798, y + 8, 190, lambda name=name: self.choose_fresh_bot(name),
                            hint="Préparer un entraînement depuis zéro. Le modèle courant reste conservé jusqu'au prochain entraînement terminé.")
        self.button("‹ Page", 34, 720, 160, lambda: setattr(self, "picker_page", self.picker_page - 1), enabled=pages > 1)
        self.button("Page ›", 214, 720, 160, lambda: setattr(self, "picker_page", self.picker_page + 1), enabled=pages > 1)
        self.text(f"{len(self.available)} choix · Page {self.picker_page + 1}/{pages}", 420, 733, 12, MUTED)

    def choose_fresh_bot(self, name):
        seat = self.picker_seat
        self.choose_bot(self.registry.saved_models[name]["name"])
        self.kwargs[seat] = '{"autoload":false}'
        if self.mode == "train":
            self.training[seat] = True
        self.notify("Nouveau modèle préparé : les poids seront initialisés au prochain entraînement.")

    def open_training_metrics(self):
        self.metrics_source = None
        self.metrics_report = self.session.validation_report
        self.metrics_index = 0
        self.metrics_tab = "summary"
        self.metrics_open = True

    def training_finished_ui(self):
        self.refresh()
        self.open_training_metrics()
        if self.session.stop_event.is_set():
            self.notify("Entraînement interrompu. La validation complète n'est pas disponible.")
        elif self.session.validation_report["status"] != "completed":
            self.notify("Entraînement terminé ; la fiche indique la validation incomplète ou en échec.")
        elif self.session.last_archive is None:
            self.notify("Validation terminée. Modèle et métriques en mémoire ; Enregistrer une version permet de les conserver.")
        else:
            errors = [entry for entry in self.session.recorder.manifest["agents"]
                      if entry["training"] and entry["checkpoint_status"] != "saved"]
            if errors:
                self.error = "Validation disponible ; checkpoint non sauvegardé : " + ", ".join(entry["label"] for entry in errors)
            else:
                self.notify("Entraînement terminé. Validation séparée disponible ; le bot courant est actualisé.")

    def open_model_metrics(self, name, *, validation_key='validation'):
        model = self.registry.resolve_model(name)
        if model is None:
            return
        self.metrics_source = model
        validation = model.get(validation_key)
        self.metrics_report = ({"protocol": validation["protocol"], "agents": [validation],
                                "status": validation["status"]} if validation else None)
        self.metrics_index = 0
        self.metrics_tab = "summary"
        self.metrics_open = True

    def draw_metrics(self):
        self.text("Métriques du bot · Validation après entraînement", 34, 86, 19)
        self.button("Retour", 786, 30, 220, self.back, enabled=self.pending is None)
        report = self.metrics_report
        source = self.metrics_source
        if not report or not report["agents"]:
            self.text("Aucune validation disponible pour ce checkpoint.", 34, 190, 16)
            self.text("Les anciens résultats d'entraînement ne sont pas des données de validation.", 34, 235, 12, MUTED)
            self.text("La validation est automatique après entraînement. Ce checkpoint ancien peut aussi être validé directement.",
                      34, 272, 11, MUTED)
            if source:
                self.button("Valider ce checkpoint", 34, 330, 290, self.validate_existing_model,
                            enabled=self.pending is None, primary=True)
                if self.pending_kind == "validate_model":
                    self.button("Arrêter la validation", 346, 330, 260, self.arena_stop.set)
            return
        self.metrics_index %= len(report["agents"])
        candidate = report["agents"][self.metrics_index]
        seat = candidate["seat"]
        title = source.get('display_label') or self.registry.label(source["model_id"]) if source else self.session.names[seat]
        self.text(f"J{seat + 1} · {title}"[:95], 34, 142, 15, ACCENT)
        if len(report["agents"]) > 1:
            self.button("Bot suivant ›", 786, 124, 220,
                        lambda: setattr(self, "metrics_index", self.metrics_index + 1))
        protocol = report["protocol"]
        status = {"completed": "Terminée", "interrupted": "Interrompue · résultats partiels", "failed": "Échec · voir le détail"}
        self.text(f"Validation : {status.get(candidate['status'], candidate['status'])} · "
                  f"{protocol['num_players']} joueurs · seed {protocol['seed']}", 34, 192, 12)
        policy = candidate.get("policy", {})
        if policy.get("algorithm"):
            self.text(f"Politique : {policy['algorithm']} · " + ("mémoire récurrente" if policy.get("recurrent") else "sans mémoire récurrente"),
                      34, 217, 11, MUTED)
            self.text('Actions déterministes' if policy.get('deterministic') else 'Actions échantillonnées (seed fixée)',
                      610, 217, 10, MUTED)
        self.text("Poids figés · nouvelles mains · stacks réinitialisés · positions alternées · aucune transition d'apprentissage",
                  34, 241, 10, MUTED)
        pages = max(1, (len(candidate['benchmarks']) + 1) // 2)
        self.metrics_page %= pages
        if pages > 1:
            self.button(f'‹ {self.metrics_page + 1}/{pages}', 770, 170, 108,
                        lambda: setattr(self, 'metrics_page', self.metrics_page - 1))
            self.button('Suivant ›', 890, 170, 116,
                        lambda: setattr(self, 'metrics_page', self.metrics_page + 1))
        for index, benchmark in enumerate(candidate["benchmarks"][self.metrics_page * 2:self.metrics_page * 2 + 2]):
            y = 267 + index * 155
            self.rect(34, y, 972, 140)
            opponent = self.benchmark_label(benchmark)
            self.text(f"Face à {opponent}", 54, y + 15, 15)
            mean, rate = benchmark["bb_per_100"], benchmark["hand_win_rate"]
            self.text(f"{mean:+.2f} BB/100" if mean is not None else "BB/100 indisponible", 54, y + 52, 23,
                      ERROR if mean is not None and mean < 0 else ACCENT)
            self.text(f"{benchmark['completed']} mains terminées · {benchmark['truncated']} tronquées · "
                      + (f"{rate:.1%} de mains gagnées · " if rate is not None else "")
                      + f"{benchmark['ties']} égalités", 385, y + 58, 11)
            ci = benchmark["ci95_bb_per_hand"]
            self.text(f"Intervalle à 95 % approximatif : [{ci[0] * 100:+.2f} ; {ci[1] * 100:+.2f}] BB/100"
                      if ci else "Intervalle indisponible : au moins 2 mains terminées nécessaires.",
                      54, y + 105, 11, MUTED)
        if candidate.get("error"):
            if not candidate["benchmarks"]:
                self.rect(34, 267, 972, 140)
                self.text("Validation impossible pour ce checkpoint", 54, 287, 15, ERROR)
                self.text(candidate["error"][:120], 54, 330, 10, ERROR)
                self.text("Le checkpoint est conservé. Aucun résultat de performance ne lui est attribué.", 54, 375, 11, MUTED)
            else:
                self.text(candidate["error"][:120], 34, 590, 11, ERROR)
        else:
            self.text("Les mains tronquées sont exclues des gains et taux. Un intervalle large appelle davantage de mains.",
                      34, 590, 11, MUTED)
        self.text("Ces références simples ne mesurent ni la force contre tous les joueurs, ni l'exploitabilité / GTO.",
                  34, 620, 11, MUTED)
        training = source.get("training_metrics", {}) if source else getattr(self.session.agents[seat], "training_stats", {})
        self.text(f"Diagnostic entraînement : {training.get('timesteps', '—')} décisions · "
                  f"{training.get('ppo_epochs', '—')} époques PPO", 34, 649, 11, MUTED)
        def values(labels):
            return " · ".join(f"{label} : {training[name]:.3g}" for name, label in labels if name in training)
        self.text(values((("policy_gradient_loss", "Perte politique"), ("value_loss", "Perte valeur"),
                          ("entropy_loss", "Perte entropie"))), 34, 672, 10, MUTED)
        self.text(values((("approx_kl", "KL"), ("clip_fraction", "Fraction écrêtée"),
                          ("explained_variance", "Variance expliquée"))), 34, 693, 10, MUTED)
        self.button("Enregistrer une version", 34, 717, 290, self.save_metrics_version,
                    enabled=self.pending is None and not (source and source.get('campaign_path') and
                        (self.campaign.running or (self.campaign.data or {}).get('status') == 'running')), primary=True,
                    hint="Conserver explicitement ces poids et leur validation, même après un nouvel entraînement.")
        if source is None:
            self.button("Voir ce bot jouer", 346, 717, 260, self.watch_from_metrics,
                        enabled=self.pending is None and self.session.game.finished)
        elif not source.get('campaign_path'):
            self.button("Valider à nouveau" if self.pending is None else "Arrêter la validation",
                        346, 717, 260, self.validate_existing_model if self.pending is None else self.arena_stop.set,
                        enabled=self.pending is None or self.pending_kind == "validate_model")
            if candidate["status"] == "failed" and self.registry.native_training(source["model_id"]):
                self.button("Nouveau modèle", 628, 717, 260, self.new_from_metrics, enabled=self.pending is None)
        if candidate["benchmarks"]:
            self.button("Analyse des coups", 628, 717, 260, self.open_hand_analysis,
                        enabled=self.pending is None, hint="Comparer les mains gagnantes et perdantes par catégorie et inspecter leurs actions.")
        if source and not source.get('campaign_path') and source.get('champion_validation'):
            self.button('Courant', 906, 717, 100,
                lambda: self.open_model_metrics(source['model_id'], validation_key='champion_validation'),
                enabled=self.pending is None, hint='Consulter la confrontation longue au bot courant et ses mains.')
        self.text("Cette seed reste dans la campagne. Enregistrer ajoute une version ; le bot courant est conservé."
                  if source and source.get('campaign_path') else
                  "Le modèle courant est remplacé au prochain entraînement. Une version enregistrée reste disponible.",
                  34, 765, 10, MUTED)

    def analysis_candidate(self):
        return self.metrics_report["agents"][self.metrics_index]

    def benchmark_label(self, benchmark):
        name = benchmark['opponent']
        name = {'random': 'Aléatoire', 'check_call': 'Check / call', 'tight': 'Serré', 'aggressive': 'Agressif'}.get(name,
            self.metrics_report.get('protocol', {}).get('frozen_opponents', {}).get(name, {}).get('label', name))
        return (f"{benchmark['depth_bb']:g} BB · " if 'depth_bb' in benchmark else '') + name

    def analysis_benchmark_data(self):
        return self.analysis_candidate()["benchmarks"][self.analysis_benchmark]

    def open_hand_analysis(self):
        self.metrics_tab = "hands"
        self.analysis_view = "chart"
        self.analysis_parent = 'chart'
        self.analysis_benchmark = 0
        self.analysis_dimension, self.analysis_unit = "combination", "count"
        self.analysis_category = None
        self.analysis_outcome = "loss"
        self.analysis_page = self.analysis_action_page = 0
        self.load_analysis_hands()
        self.notify("Analyse de validation : cliquez sur une barre ou consultez les coups perdants / gagnants.")

    def load_analysis_hands(self):
        self.analysis_hands = None
        analysis = self.analysis_benchmark_data().get("hand_analysis")
        if analysis is None:
            return
        if "hands" in analysis:
            self.analysis_hands = analysis["hands"]
            return
        source = self.metrics_source
        filename = analysis.get("hands_file")
        if source is None or not filename:
            return
        root = source["run_path"].resolve()
        path = (root / filename).resolve()
        if not path.is_relative_to(root):
            self.error = "Chemin de rapport de validation invalide."
            return
        seat = self.analysis_candidate()["seat"]
        opponent = self.analysis_benchmark_data()["opponent"]
        depth = self.analysis_benchmark_data().get('depth_bb')
        def read():
            report = json.loads(path.read_text(encoding="utf-8"))
            candidate = next(item for item in report["agents"] if item["seat"] == seat)
            benchmark = next(item for item in candidate["benchmarks"] if item["opponent"] == opponent and item.get('depth_bb') == depth)
            return benchmark["hand_analysis"]["hands"]
        self.loading = True
        self.pending_kind = "analysis_hands"
        self.pending = _background(read)

    def switch_analysis_benchmark(self):
        self.analysis_benchmark = (self.analysis_benchmark + 1) % len(self.analysis_candidate()["benchmarks"])
        self.analysis_category = None
        self.analysis_page = 0
        if self.analysis_view not in ('ranges', 'style', 'policy'):
            self.analysis_view = 'chart'
        self.load_analysis_hands()

    def show_analysis_hands(self, *, category=None, outcome=None):
        self.analysis_category = category
        if outcome is not None:
            self.analysis_outcome = outcome
        self.analysis_page = 0
        self.analysis_view = "list"

    def draw_hand_analysis(self):
        if self.analysis_view == "detail":
            self.draw_hand_detail()
            return
        self.text("Analyse des mains de validation", 34, 86, 19)
        self.button("Retour", 786, 30, 220, self.back, enabled=self.pending is None)
        candidate = self.analysis_candidate()
        title = ((self.metrics_source.get('display_label') or self.registry.label(self.metrics_source["model_id"])) if self.metrics_source
                 else self.session.names[candidate["seat"]])
        benchmark = self.analysis_benchmark_data()
        opponent = self.benchmark_label(benchmark)
        self.text(title[:24], 34, 137, 12, ACCENT)
        def change_view(view):
            self.analysis_view = self.analysis_parent = view
            if view == 'chart':
                self.analysis_dimension = 'combination'
        self.button('Catégories', 260, 123, 110, lambda: change_view('chart'), active=self.analysis_view == 'chart')
        self.button('Départ', 380, 123, 108, lambda: change_view('ranges'), active=self.analysis_view == 'ranges')
        self.button('Style', 500, 123, 108, lambda: change_view('style'), active=self.analysis_view == 'style')
        self.button('Politique', 620, 123, 106, lambda: change_view('policy'), active=self.analysis_view == 'policy')
        opponent_label = f"Face à {opponent} ›"
        self.button(opponent_label[:35] + ('…' if len(opponent_label) > 35 else ''), 736, 123, 270,
                    self.switch_analysis_benchmark, hint=opponent_label,
                    enabled=self.pending is None and len(candidate["benchmarks"]) > 1)
        analysis = benchmark.get("hand_analysis")
        if self.analysis_view == 'policy':
            from poker_rl.strategy_ui import draw_policy
            draw_policy(self)
            return
        if analysis is None:
            self.text("Cette validation ancienne ne contient pas encore l'analyse des mains.", 34, 222, 14)
            self.text("Relancez sa validation pour calculer les catégories et conserver les actions, sans apprentissage.",
                      34, 267, 11, MUTED)
            if self.metrics_source:
                self.button("Calculer l'analyse", 34, 330, 290, self.validate_existing_model,
                            enabled=self.pending is None, primary=True)
                if self.pending_kind == "validate_model":
                    self.button("Arrêter", 346, 330, 200, self.arena_stop.set)
            return
        if self.analysis_view == "list":
            self.draw_hand_list()
            return
        if self.analysis_view in ('ranges', 'style'):
            from poker_rl.strategy_ui import draw_strategy
            draw_strategy(self)
            return
        for label, dimension, x, width in (("Combinaison", "combination", 34, 208),
                ("Dernière action", "action", 260, 208), ("Phase de jeu", "street", 486, 160)):
            self.button(label, x, 174, width, lambda dimension=dimension: setattr(self, "analysis_dimension", dimension),
                        active=self.analysis_dimension == dimension, enabled=self.pending is None)
        self.button("Nombre", 664, 174, 160, lambda: setattr(self, "analysis_unit", "count"),
                    active=self.analysis_unit == "count", enabled=self.pending is None)
        self.button("Gains / pertes BB", 842, 174, 164, lambda: setattr(self, "analysis_unit", "bb"),
                    active=self.analysis_unit == "bb", enabled=self.pending is None)
        self.text("Gagnantes : gain net > 0 · Perdantes : gain net < 0 · Neutres : 0 BB · Mains tronquées exclues",
                  34, 229, 11, MUTED)
        notes = {"combination": "Combinaison du bot à l'abattage ; folds et fins sans abattage sont séparés.",
                 "action": "Une main compte une fois, selon la dernière action du bot. Ce regroupement ne prouve pas une erreur.",
                 "street": "Phase de la dernière décision du bot dans chaque main, même si la main se termine plus tard."}
        self.text(notes[self.analysis_dimension], 34, 254, 10, MUTED)
        rows = analysis["breakdown"][self.analysis_dimension]
        win_key, loss_key = ("wins", "losses") if self.analysis_unit == "count" else ("won_bb", "lost_bb")
        maximum = max((max(row[win_key], row[loss_key]) for row in rows), default=0)
        scale = max(1, maximum)
        self.rect(34, 281, 972, 369)
        self.text("Catégorie", 48, 290, 10, MUTED)
        self.text("0", 264, 290, 9, MUTED)
        self.text(f"{scale / 2:g}", 488, 290, 9, MUTED)
        self.text(f"{scale:g}" + (" BB" if self.analysis_unit == "bb" else " mains"), 683, 290, 9, MUTED)
        self.text("Gains", 782, 290, 10, ACCENT)
        self.text("Pertes", 858, 290, 10, ERROR)
        self.text("Neutres", 932, 290, 10, MUTED)
        for index, row in enumerate(rows):
            y = 317 + index * 27
            control = Button(42, y - 1, 954, 26,
                lambda category=row["category"], y=y: self.select_analysis_bar(category, y),
                self.pending is None and self.analysis_hands is not None,
                "Voir les mains de cette catégorie, triées par gain ou perte.")
            self.buttons.append(control)
            if control.contains(*self.mouse_position):
                self.rect(42, y - 1, 954, 26, HOVER)
            self.text(DIMENSION_LABELS[self.analysis_dimension][row["category"]], 48, y + 4, 10)
            self.rect(264, y + 2, 472, 7, BACKGROUND)
            self.rect(264, y + 12, 472, 7, BACKGROUND)
            if row[win_key]:
                self.rect(264, y + 2, max(1, 472 * row[win_key] / scale), 7, ACCENT)
            if row[loss_key]:
                self.rect(264, y + 12, max(1, 472 * row[loss_key] / scale), 7, ERROR)
            self.text(f"{row[win_key]:g}" if self.analysis_unit == "count" else
                      f"+{row[win_key]:.1f}" if row[win_key] else "0", 782, y + 4, 10, ACCENT)
            self.text(f"{row[loss_key]:g}" if self.analysis_unit == "count" else
                      f"-{row[loss_key]:.1f}" if row[loss_key] else "0", 858, y + 4, 10, ERROR)
            self.text(str(row["neutral"]), 940, y + 4, 10, MUTED)
        for label, outcome, x, width in (("Coups perdants", "loss", 34, 220), ("Coups gagnants", "win", 276, 220),
                ("Coups neutres", "neutral", 518, 220), ("Toutes les mains", "all", 760, 246)):
            self.button(label, x, 676, width, lambda outcome=outcome: self.show_analysis_hands(outcome=outcome),
                        enabled=self.pending is None and self.analysis_hands is not None)
        self.text("Cliquez sur une catégorie pour filtrer les mains, puis sur une main pour voir toutes ses actions.",
                  34, 735, 11, MUTED)
        self.text("BB : montants cumulés gagnés / perdus, pas BB/100. Un résultat perdant ne prouve pas une mauvaise décision."
                  if self.analysis_unit == "bb" else
                  "Les pertes et gains décrivent le résultat financier d'une main ; un fold perdant n'est pas forcément une mauvaise décision.",
                  34, 765, 10, MUTED)

    def select_analysis_bar(self, category, y):
        x, mouse_y = self.mouse_position
        if x < 264:
            outcome = "all"
        elif x < 760:
            outcome = "win" if mouse_y - y < 11 else "loss"
        else:
            outcome = "neutral" if x >= 932 else "loss" if x >= 850 else "win"
        self.show_analysis_hands(category=category, outcome=outcome)

    def draw_hand_list(self):
        labels = {"loss": "Perdantes", "win": "Gagnantes", "neutral": "Neutres", "all": "Toutes"}
        for index, (outcome, label) in enumerate(labels.items()):
            self.button(label, 34 + index * 248, 174, 228,
                        lambda outcome=outcome: self.show_analysis_hands(category=self.analysis_category, outcome=outcome),
                        active=self.analysis_outcome == outcome)
        category = DIMENSION_LABELS.get(self.analysis_dimension, {}).get(self.analysis_category,
            self.analysis_category or "Toutes catégories")
        self.text(category + " · Résultat net par main en BB", 34, 226, 12, MUTED)
        for label, x in (("Main", 46), ("Cartes du bot", 112), ("Board final", 205),
                         ("Combinaison / fin", 424), ("Dernière action", 636), ("Gain net", 862)):
            self.text(label, x, 246, 9, MUTED)
        hands = filter_hands(self.analysis_hands or [], dimension=self.analysis_dimension,
                             category=self.analysis_category, outcome=self.analysis_outcome)
        pages = max(1, (len(hands) + 11) // 12)
        self.analysis_page %= pages
        if not hands:
            self.text("Aucune main correspondant à ces filtres.", 48, 305, 14, MUTED)
        for index, hand in enumerate(hands[self.analysis_page * 12:(self.analysis_page + 1) * 12]):
            y = 261 + index * 35
            control = Button(34, y, 972, 31, lambda hand=hand: self.inspect_analysis_hand(hand), True,
                             "Afficher les cartes, le board et les actions de cette main.")
            self.buttons.append(control)
            self.rect(34, y, 972, 31, HOVER if control.contains(*self.mouse_position) else PANEL)
            self.text(f"#{hand['hand']}", 46, y + 9, 10)
            self.text(" ".join(hand["hole_cards"]), 112, y + 9, 10)
            self.text(" ".join(hand["board"]) or "Sans board", 205, y + 9, 10, MUTED)
            self.text(CATEGORY_LABELS[hand["category"]], 424, y + 9, 10)
            self.text(ACTION_LABELS[hand["last_action"]], 636, y + 9, 10, MUTED)
            self.text(f"{hand['net_bb']:+.2f} BB", 862, y + 9, 10,
                      ERROR if hand["outcome"] == "loss" else ACCENT if hand["outcome"] == "win" else MUTED)
        self.button("‹ Page", 34, 717, 160, lambda: setattr(self, "analysis_page", self.analysis_page - 1), enabled=pages > 1)
        self.button("Page ›", 214, 717, 160, lambda: setattr(self, "analysis_page", self.analysis_page + 1), enabled=pages > 1)
        self.text(f"{len(hands)} mains · Page {self.analysis_page + 1}/{pages} · Tri par montant du gain / de la perte",
                  398, 730, 11, MUTED)
        self.text("Cliquez sur une main pour inspecter les décisions du bot et des adversaires.", 34, 765, 11, MUTED)

    def inspect_analysis_hand(self, hand):
        self.analysis_selected = hand
        self.analysis_action_page = 0
        self.analysis_view = "detail"

    def draw_hand_detail(self):
        hand = self.analysis_selected
        self.text(f"Détail de la main #{hand['hand']} · {hand['net_bb']:+.2f} BB", 34, 86, 19,
                  ERROR if hand["outcome"] == "loss" else ACCENT if hand["outcome"] == "win" else TEXT)
        self.button("Retour aux mains", 786, 30, 220, self.back)
        self.text(f"Bot J{hand['bot_seat'] + 1} : {' '.join(hand['hole_cards'])} · Dealer J{hand['dealer'] + 1} · Seed {hand['seed']}",
                  34, 141, 13)
        self.text("Board final : " + (" ".join(hand["board"]) or "Non distribué"), 34, 179, 15)
        self.text("Catégorie : " + CATEGORY_LABELS[hand["category"]] + " · Dernière action : "
                  + ACTION_LABELS[hand["last_action"]], 34, 217, 12, MUTED)
        self.text("Cartes adverses (analyse après la main) : " + " · ".join(f"J{seat + 1} {' '.join(cards)}"
                  for seat, cards in enumerate(hand["player_cards"]) if seat != hand["bot_seat"]), 34, 253, 11, MUTED)
        for label, x in (("#", 46), ("Joueur", 90), ("Phase", 190), ("Action", 285), ("Board", 462),
                         ("Pot avant", 686), ("Stack avant", 804), ("À suivre", 928)):
            self.text(label, x, 296, 10, MUTED)
        actions = hand["actions"]
        pages = max(1, (len(actions) + 11) // 12)
        self.analysis_action_page %= pages
        for index, action in enumerate(actions[self.analysis_action_page * 12:(self.analysis_action_page + 1) * 12]):
            y = 326 + index * 29
            self.rect(34, y, 972, 25)
            self.text(str(self.analysis_action_page * 12 + index + 1), 46, y + 6, 10)
            self.text(f"J{action['seat'] + 1}" + (" (bot)" if action["seat"] == hand["bot_seat"] else ""),
                      90, y + 6, 10, ACCENT if action["seat"] == hand["bot_seat"] else MUTED)
            self.text(STREET_LABELS[action["stage"]], 190, y + 6, 10)
            self.text(ACTION_LABELS[action["action"]], 285, y + 6, 10)
            self.text(" ".join(action["board"]) or "—", 462, y + 6, 10, MUTED)
            self.text(f"{action['pot']:g}", 686, y + 6, 10)
            self.text(f"{action['stack']:g}", 804, y + 6, 10)
            self.text(f"{action['to_call']:g}", 928, y + 6, 10)
        if not actions:
            self.text("La main s'est terminée sans décision.", 46, 335, 12, MUTED)
        self.button("‹ Actions", 34, 717, 160, lambda: setattr(self, "analysis_action_page", self.analysis_action_page - 1), enabled=pages > 1)
        self.button("Actions ›", 214, 717, 160, lambda: setattr(self, "analysis_action_page", self.analysis_action_page + 1), enabled=pages > 1)
        self.text(f"{len(actions)} décisions · Page {self.analysis_action_page + 1}/{pages}", 398, 730, 11, MUTED)
        self.text("Pot, stack et montant à suivre en jetons, avant l'action. Les blinds sont automatiques. Résultat final en BB.",
                  34, 765, 10, MUTED)

    def watch_from_metrics(self):
        self.metrics_open = False
        self.watch_trained()

    def validate_existing_model(self):
        from poker_rl.evaluation import validate_model
        model_id = self.metrics_source["model_id"]
        self.arena_stop = Event()
        self.loading = True
        self.pending_kind = "validate_model"
        self.pending = _background(lambda: validate_model(self.registry, model_id,
            hands_per_opponent=int(self.validation_text), stop_event=self.arena_stop,
            on_progress=self.validation_progress))

    def new_from_metrics(self):
        model_id = self.metrics_source["model_id"]
        self.count = self.metrics_source["num_players"]
        self.metrics_open = self.library_open = False
        self.training = [False] * 5
        self.refresh()
        self.picker_seat = 0
        self.choose_fresh_bot(model_id)
        self.set_mode("train")

    def save_metrics_version(self):
        source, session = self.metrics_source, self.session
        if source and source.get('campaign_path'):
            self.campaign.save()
            return
        seat = self.metrics_report["agents"][self.metrics_index]["seat"]
        def save():
            if source:
                return save_model_version(self.registry.records_dir, source["model_id"])
            if session.last_archive is None:
                recorder = SessionRecorder(session.selections, kind="train", training_seats=session.training_seats,
                    seed=session.seed, game_kwargs=session.game_kwargs, root=session.records_dir, registry=session.registry)
                from poker_rl.evaluation import attach_validation
                attach_validation(recorder, session.validation_report)
                session.last_archive = recorder.finish(session.stats.to_dict(), agents=session.agents)
            manifest = json.loads((session.last_archive / "manifest.json").read_text(encoding="utf-8"))
            model_id = manifest["agents"][seat].get("model_id")
            if model_id is None:
                raise ValueError("Cet agent doit fournir save_checkpoint/load_checkpoint pour enregistrer une version.")
            return save_model_version(self.registry.records_dir, model_id)
        self.loading = True
        self.pending_kind = "save_version"
        self.pending = _background(save)

    def toggle_recording(self):
        self.record_sessions = not self.record_sessions
        self.notify("Archives activées." if self.record_sessions else "Archives désactivées pour cette session.")

    def open_library(self):
        if self.pending is not None:
            return
        if self.session:
            self.back()
        self.refresh()
        self.library_open = True
        self.history_lines = None
        self.library_index = 0

    def library_model(self):
        models = list(self.registry.saved_models.values())
        return models[self.library_index % len(models)] if models else None

    def move_model(self, offset):
        self.library_index += offset
        self.history_lines = None

    def select_model(self, seat):
        model = self.library_model()
        self.selected[seat] = model["model_id"]
        self.kwargs[seat] = "{}"
        self.count = max(seat + 1, model["num_players"])
        self.available = self.registry.choices(self.count)
        self.library_open = False
        self.arena_report = None
        self.set_mode("infer")
        self.notify(f"Modèle sélectionné pour J{seat + 1}. Ses métriques sont déjà consultables sans arène.")

    def load_history(self):
        model = self.library_model()
        if model['stats'].get('training_history_recorded') is False:
            self.open_model_metrics(model['model_id'])
            self.open_hand_analysis()
            self.notify("Cette expérience conserve les mains de validation ; l'historique d'entraînement n'a pas été archivé.")
            return
        path = model["run_path"] / "history.jsonl"
        def read():
            lines = deque(maxlen=200)
            with path.open(encoding="utf-8") as history:
                for line in history:
                    item = json.loads(line)
                    if item["event"] == "action":
                        before = item["before"]
                        lines.append(f"P{item['match']} M{item['hand']} · J{item['seat'] + 1} · "
                            f"{' '.join(before['hole_cards'][item['seat']])} · {before['stage']} · "
                            f"{item['action_name']} · pot {before['pot']:g} · "
                            f"board {' '.join(before['board']) or '—'}")
            return list(lines)
        self.history_page = 0
        self.loading = True
        self.pending_kind = "history"
        self.pending = _background(read)

    def draw_library(self):
        self.text("Mes bots · modèles courants et versions enregistrées", 34, 86, 19)
        self.button("Retour", 786, 30, 220, self.back, enabled=not self.loading)
        model = self.library_model()
        if model is None:
            self.text("Aucun modèle enregistré. Lancez un entraînement avec Archives : oui.", 34, 180, 13)
            return
        self.button("‹ Précédent", 34, 126, 180, lambda: self.move_model(-1), enabled=not self.loading)
        self.button("Suivant ›", 236, 126, 180, lambda: self.move_model(1), enabled=not self.loading)
        self.text(self.registry.label(model["model_id"]), 450, 140, 14, ACCENT)
        self.text(f"Identifiant : {model['model_id']} · {model['num_players']} joueurs · {model['created_at'][:19]} UTC", 34, 190, 12)
        opponents = ", ".join(p["label"] for p in model["opponents"] if p["seat"] != model["seat"])
        self.text(f"Adversaires : {opponents}"[:125], 34, 222, 12, MUTED)
        stats = model["stats"]
        if stats.get('training_history_recorded') is False:
            self.text(f"Entraînement : {model.get('training_metrics', {}).get('timesteps', '—')} décisions · Mains de validation consultables",
                      34, 254, 12, MUTED)
        else:
            self.text(f"Entraînement : {stats.get('completed', 0)} mains terminées · Transitions reçues : {stats.get('transitions', [])}", 34, 254, 12, MUTED)
        self.button("Métriques", 34, 290, 200, lambda: self.open_model_metrics(model["model_id"]), enabled=not self.loading)
        for seat in range(model["num_players"]):
            self.button(f"Choisir pour J{seat + 1}", 256 + seat * 150, 290, 142,
                        lambda seat=seat: self.select_model(seat), enabled=not self.loading)
        if self.history_lines is not None:
            pages = max(1, (len(self.history_lines) + 17) // 18)
            self.history_page %= pages
            for index, line in enumerate(self.history_lines[self.history_page * 18:(self.history_page + 1) * 18]):
                self.text(line[:138], 34, 352 + index * 21, 10)
            self.button("‹ Coups", 34, 742, 150, lambda: setattr(self, "history_page", self.history_page - 1))
            self.button("Coups ›", 206, 742, 150, lambda: setattr(self, "history_page", self.history_page + 1))
            self.text(f"Page {self.history_page + 1}/{pages} · 200 derniers coups · archive complète : history.jsonl", 382, 755, 10, MUTED)
        else:
            validation = model.get("validation")
            self.text("Validation : " + ({"completed": "terminée", "interrupted": "interrompue", "failed": "échec"}.get(
                      validation["status"], validation["status"]) if validation else "indisponible pour cet ancien checkpoint"),
                      34, 370, 12, ACCENT)
            self.text("Un modèle courant par type et taille de table. Seules vos sauvegardes créent des versions supplémentaires.",
                      34, 410, 11, MUTED)
            self.button("Enregistrer une version", 34, 470, 290,
                        lambda: self.save_library_version(model["model_id"]), enabled=not self.loading)
            self.button("Historique des coups", 346, 470, 260, self.load_history, enabled=not self.loading)

    def save_library_version(self, model_id):
        self.loading = True
        self.pending_kind = "save_version"
        self.pending = _background(lambda: save_model_version(self.registry.records_dir, model_id))

    def draw_arena_results(self):
        report = self.arena_report
        self.text("Résultats de l'arène", 34, 86, 21)
        self.button("Retour", 786, 30, 220, self.back)
        self.text(f"{report['matches_completed']} parties terminées · {report['matches_truncated']} interrompues · "
                  f"{report['completed']} mains terminées", 34, 140, 13, MUTED)
        for index, player in enumerate(report["players"]):
            y = 195 + index * 92
            self.rect(34, y, 972, 80)
            rate = f"{player['win_rate']:.1%}" if player["win_rate"] is not None else "—"
            self.text(f"J{index + 1} · {self.registry.label(player['name'])}", 54, y + 14, 14)
            self.text(f"Victoires {player['wins']} · Défaites {player['losses']} · Taux {rate} · "
                      f"Mains gagnées {player['hand_wins']} · Égalités {player['hand_ties']} · "
                      f"{player['bb_per_hand']:+.2f} BB/main", 54, y + 48, 11, MUTED)
        self.text("Les parties interrompues sont exclues du taux de victoire.", 34, 690, 12, MUTED)
        self.button("Nouvelle arène", 34, 730, 230, self.back)
        self.button("Bibliothèque", 286, 730, 230, self.open_library)

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
            self.native_training_complete = False
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
        if self.campaign.visible and not self.metrics_open and self.campaign.editing:
            self.campaign.input_text(text)
            return
        if self.editing is not None and not self.loading and text.isprintable():
            if self.editing in ("episodes", "max_hands", "validation"):
                if text.isdecimal():
                    attribute = {"episodes": "episodes_text", "max_hands": "max_hands_text", "validation": "validation_text"}[self.editing]
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
        if self.campaign.visible and not self.metrics_open and self.campaign.input_key(symbol, modifiers):
            return
        if self.editing is not None:
            if symbol == key.BACKSPACE:
                if self.editing in ("episodes", "max_hands", "validation"):
                    attribute = {"episodes": "episodes_text", "max_hands": "max_hands_text", "validation": "validation_text"}[self.editing]
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
            if self.session or self.library_open or self.arena_report is not None or self.metrics_open or self.picker_seat is not None or self.campaign.visible:
                self.back()
            elif not self.loading:
                self.window.close()
        elif self.session and symbol in KEY_ACTIONS:
            self.human_action(KEY_ACTIONS[symbol])

    def on_close(self):
        self.campaign.stop.set()
        if self.campaign.running:
            try:
                self.campaign.future.result(timeout=10)
            except Exception:
                pass
        pyglet.clock.unschedule(self.tick)
        self.metrics_open = False
        self.picker_seat = None
        self.arena_stop.set()
        if self.session:
            self.session.request_stop()
        if self.pending is not None:
            try:
                result = self.pending.result(timeout=10)
                if self.pending_kind in ("load", "ppo_train"):
                    result.close()
                self.pending = None
                self.pending_kind = None
            except BaseException:
                pass
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

