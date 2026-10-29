"""Native Pyglet monitoring for independent training campaigns."""
from concurrent.futures import Future
from pathlib import Path
from threading import Event, Thread
import time

from poker_rl.campaigns import (CampaignConfig, campaign_paths, read_campaign,
                                run_campaign, save_campaign_version, scenario_scores)

TEXT = (230, 236, 244)
MUTED = (154, 169, 188)
GREEN = (78, 213, 164)
RED = (240, 112, 116)
PANEL = (25, 36, 51)
COLORS = (GREEN, (111, 168, 255), (244, 190, 89), (202, 139, 245), RED)
PHASE_LABELS = {'training': 'apprentissage', 'selection_validation': 'validation de sélection',
                'selection_complete': 'sélection terminée', 'final_test': 'test final',
                'champion_test': 'confrontation', 'completed': 'terminé', 'interrupted': 'interrompu'}
STATUS_LABELS = {'running': 'en cours', 'selection_only': 'sélection seule', 'completed': 'terminé',
                 'failed': 'échec', 'interrupted': 'interrompu'}


class CampaignUI:
    def __init__(self, app):
        self.app, self.visible = app, False
        self.tab, self.data, self.future = 'config', None, None
        self.stop = Event()
        self.fields = {'seeds': '101,202,303', 'steps': '100000', 'interval': '50000',
                       'selection': '500', 'final': '1000', 'depths': '20,50,100',
                       'rates': '0.0003,0.0001', 'champion': '5000'}
        self.curriculum_enabled = self.candidate_enabled = True
        self.trial_index, self.validation_mode = 0, 'best'
        self.editing, self.replace = None, False
        self.paths, self.path_index, self.seed_index, self.scenario_index = [], 0, 0, 0
        self.metric = 'rolling_bb100'
        self.depth_column_page = self.depth_row_page = 0
        self.poll_time = 0
        self.disk_path = None

    @property
    def running(self):
        return self.future is not None and not self.future.done()

    def open(self):
        self.visible = True
        self.paths = campaign_paths()
        if self.data is None and self.paths:
            self.disk_path = self.paths[0]
            self.data = read_campaign(self.disk_path)

    def leave(self):
        if self.running:
            self.stop.set()
            self.app.notify('Arrêt demandé : la campagne sera conservée avec ses résultats partiels.')
        else:
            self.visible = False
            self.editing = None

    def launch(self):
        if self.running:
            return
        try:
            config = CampaignConfig(seeds=tuple(int(v.strip()) for v in self.fields['seeds'].split(',')),
                total_timesteps=int(self.fields['steps']), checkpoint_interval=int(self.fields['interval']),
                selection_hands=int(self.fields['selection']), final_hands=int(self.fields['final']),
                stack_depths=tuple(int(v.strip()) for v in self.fields['depths'].split(',')), num_players=self.app.count)
            rates = []
            for value in self.fields['rates'].split(','):
                rates.append(float(value.strip()))
            config.learning_rates = tuple(rates)
            config.champion_hands = int(self.fields['champion'])
            config.include_seed202 = self.candidate_enabled
            config.curriculum_probability = 0
            if self.curriculum_enabled:
                config.curriculum_probability = .2
            config.validate()
        except (ValueError, TypeError) as exc:
            self.app.error = str(exc)
            return
        self.stop = Event()
        self.data, self.disk_path, self.editing = None, None, None
        self.tab, self.seed_index = 'train', 0
        self.future = Future()
        def work():
            try:
                self.future.set_result(run_campaign(self.app.registry, config, stop_event=self.stop,
                    on_update=lambda state: setattr(self, 'data', state)))
            except BaseException as exc:
                self.future.set_exception(exc)
        Thread(target=work, daemon=True, name='poker-campaign').start()
        self.app.error = ''

    def poll(self):
        if self.future is not None and self.future.done():
            try:
                self.data = self.future.result()
                self.disk_path = Path(self.data['path']) / 'campaign.json'
                self.app.notify('Campagne : ' + self.data['status'])
                self.paths = campaign_paths()
            except Exception as exc:
                self.app.error = str(exc)
            self.future = None
        if not self.running and self.disk_path and time.monotonic() - self.poll_time > 2:
            self.poll_time = time.monotonic()
            try:
                self.data = read_campaign(self.disk_path)
            except (OSError, ValueError):
                pass

    def previous(self):
        if self.paths:
            self.path_index = (self.path_index + 1) % len(self.paths)
            self.disk_path = self.paths[self.path_index]
            self.data = read_campaign(self.disk_path)
            self.seed_index = 0

    def field(self, name):
        self.editing, self.replace = name, True

    def select_tab(self, tab):
        self.tab, self.editing = tab, None

    def input_text(self, text):
        if self.editing and text.isprintable():
            self.fields[self.editing] = text if self.replace else self.fields[self.editing] + text
            self.replace = False

    def input_key(self, symbol, modifiers):
        from pyglet.window import key
        if self.editing:
            if symbol == key.BACKSPACE:
                self.fields[self.editing] = '' if self.replace else self.fields[self.editing][:-1]
                self.replace = False
            elif symbol == key.A and modifiers & key.MOD_CTRL:
                self.replace = True
            elif symbol in (key.ENTER, key.ESCAPE):
                self.editing = None
            return True
        return False

    def selected_run(self):
        runs = self.visible_runs()
        return runs[self.seed_index % len(runs)] if runs else None

    def selected_trial(self):
        trials = (self.data or {}).get('trials', [])
        return trials[self.trial_index % len(trials)] if trials else None

    def visible_runs(self):
        runs = (self.data or {}).get('runs', [])
        trial = self.selected_trial()
        return [r for r in runs if not trial or r.get('trial_id') == trial['id']]

    def selected_aggregate(self):
        trial = self.selected_trial()
        return (self.data.get('aggregates', {}).get(trial['id'], self.data['aggregate'])
                if trial else self.data['aggregate'])

    def next_trial(self):
        self.trial_index += 1
        self.seed_index = 0

    def inspect(self, kind='final_validation'):
        run = self.selected_run()
        if run and run.get(kind):
            app = self.app
            app.metrics_source = {'model_id': f"campaign_{run['seed']}", 'display_label': f"{run.get('trial_id', 'Campagne')} · seed {run['seed']}",
                'run_path': Path(run['path']), 'training_metrics': run.get('training_metrics', {}),
                'campaign_path': Path(self.data['path']) / 'campaign.json', 'seed': run['seed'], 'trial_id': run.get('trial_id')}
            app.metrics_report = run[kind]
            app.metrics_index, app.metrics_tab, app.metrics_open = 0, 'summary', True
            app.metrics_page = 0

    def save(self):
        if self.running or (self.data and self.data.get('status') == 'running'):
            self.app.notify('L’enregistrement sera disponible à la fin de la campagne ; les métriques restent consultables.')
            return
        run = self.selected_run()
        if run:
            try:
                version = save_campaign_version(self.app.registry, Path(self.data['path']) / 'campaign.json', run['seed'], trial_id=run.get('trial_id'))
                self.data = read_campaign(Path(self.data['path']) / 'campaign.json')
                self.app.refresh()
                self.app.notify('Version enregistrée : ' + version)
            except Exception as exc:
                self.app.error = str(exc)

    def draw(self):
        self.poll()
        a = self.app
        a.text('Campagnes · seeds, ligue et profondeurs', 34, 86, 19)
        a.button('Arrêter' if self.running else 'Retour', 786, 30, 220, self.leave)
        tabs = [('Réglages', 'config'), ('Apprentissage', 'train'), ('Validation', 'validation'), ('Modes / PPO', 'modes'),
                ('Seeds / final', 'seeds'), ('Profondeurs', 'depths'), ('Ligue', 'league')]
        for i, (label, tab) in enumerate(tabs):
            a.button(label, 34 + i * 140, 126, 130, lambda tab=tab: self.select_tab(tab), active=self.tab == tab)
        if self.tab == 'config':
            self.draw_config()
            return
        if not self.data:
            a.text('Préparation des adversaires figés…' if self.running else 'Aucune campagne : configurez puis lancez les entraînements.', 34, 220, 14)
            return
        data, run = self.data, self.selected_run()
        total = len(data['config']['seeds']) * max(1, len(data.get('trials', [])))
        a.text(f"{data['id']} · {STATUS_LABELS.get(data['status'], data['status'])} · {len(data.get('runs', []))}/{total} entraînements démarrés", 34, 184, 11, MUTED)
        trial = self.selected_trial()
        if trial:
            a.button(trial['label'] + ' ›', 786, 170, 220, self.next_trial, enabled=len(data['trials']) > 1)
        if run:
            a.button(f"Seed {run['seed']} ›", 34, 214, 195,
                lambda: setattr(self, 'seed_index', self.seed_index + 1))
            phase = PHASE_LABELS.get(run.get('phase'), run.get('phase', ''))
            a.text(f"{phase} · {run['steps']:,}/{data['config']['total_timesteps']:,} décisions", 252, 227, 12)
        a.button('Autre campagne ›', 786, 214, 220, self.previous, enabled=not self.running and len(self.paths) > 1)
        if data.get('error'):
            a.text(data['error'][:130], 34, 766, 10, RED)
        if self.tab == 'train':
            self.draw_train(run)
        elif self.tab == 'validation':
            self.draw_validation()
        elif self.tab == 'seeds':
            self.draw_seeds()
        elif self.tab == 'depths':
            self.draw_depths()
        elif self.tab == 'modes':
            self.draw_modes(run)
        else:
            self.draw_league(run)

    def draw_config(self):
        a = self.app
        labels = [('seeds', 'Seeds indépendantes (3 à 5)', 34, 187), ('steps', 'Décisions par seed et essai PPO', 530, 187),
                  ('interval', 'Intervalle des validations de sélection', 34, 277), ('depths', 'Profondeurs en BB (CSV)', 530, 277),
                  ('selection', 'Mains / scénario / mode · sélection', 34, 367),
                  ('final', 'Mains / scénario · test final', 530, 367),
                  ('rates', 'Taux d’apprentissage comparés (CSV)', 34, 457),
                  ('champion', 'Mains / profondeur contre le bot courant', 530, 457)]
        for name, label, x, y in labels:
            a.text(label, x, y, 12, MUTED)
            a.button(self.fields[name], x, y + 27, 470, lambda name=name: self.field(name),
                field=True, active=self.editing == name, enabled=not self.running)
        a.text(f"{a.count} joueurs · mêmes seeds et budgets par taux · comparaison déterministe / probabiliste", 34, 551, 11)
        a.button('Curriculum postflop : ' + ('oui' if self.curriculum_enabled else 'non'), 34, 580, 470,
            lambda: setattr(self, 'curriculum_enabled', not self.curriculum_enabled), enabled=not self.running)
        a.button('Référence seed 202 : ' + ('oui' if self.candidate_enabled else 'non'), 530, 580, 470,
            lambda: setattr(self, 'candidate_enabled', not self.candidate_enabled), enabled=not self.running)
        a.text('Ligue : snapshots tous les 5 000 pas, filtrés par diversité ; références figées conservées.', 34, 638, 11, MUTED)
        a.button('Lancer les entraînements', 34, 685, 360, self.launch, enabled=not self.running, primary=True)
        a.button('Voir la dernière campagne', 416, 685, 360, lambda: self.select_tab('seeds'), enabled=self.data is not None)
        if self.running:
            a.button('Arrêter', 798, 685, 208, self.stop.set)
        a.text('Le réglage, les poids et le mode sont figés avant le test final. Aucune promotion automatique du bot courant.', 34, 748, 10, MUTED)

    def chart(self, series, *, y=330, height=325, caption=''):
        a = self.app
        a.rect(34, y, 972, height, PANEL)
        points = [(x, v) for _, values, _ in series for x, v in values if v is not None]
        if not points:
            a.text('En attente des premières mesures…', 58, y + 55, 13, MUTED)
            return
        low, high = min(v for _, v in points), max(v for _, v in points)
        if low == high:
            low, high = low - 1, high + 1
        max_x = max(x for x, _ in points) or 1
        from pyglet.gl import glBegin, glEnd, glColor3f, glVertex2f, GL_LINE_STRIP
        for index, (label, values, color) in enumerate(series):
            a.text(label, 65 + index * 180, y + 15, 10, color)
            glColor3f(*(c / 255 for c in color))
            glBegin(GL_LINE_STRIP)
            for x, value in values:
                if value is not None:
                    glVertex2f(125 + 850 * x / max_x, 860 - (y + height - 40 - (height - 95) * (value - low) / (high - low)))
            glEnd()
            for x, value in values:
                if value is not None:
                    px = 125 + 850 * x / max_x
                    py = y + height - 40 - (height - 95) * (value - low) / (high - low)
                    a.rect(px - 2, py - 2, 4, 4, color)
        a.text(f'{high:+.2f}', 47, y + 54, 10, MUTED)
        a.text(f'{low:+.2f}', 47, y + height - 53, 10, MUTED)
        a.text('0', 125, y + height - 22, 10, MUTED)
        a.text('Décisions PPO', 495, y + height - 22, 10, MUTED)
        a.text(f'{max_x:,}', 900, y + height - 22, 10, MUTED)
        a.text(caption, 34, y + height + 17, 10, MUTED)

    def draw_train(self, run):
        a = self.app
        keys = [('Gain BB/100', 'rolling_bb100'), ('Entropie (perte)', 'entropy_loss'),
                ('Perte valeur', 'value_loss'), ('KL', 'approx_kl'), ('Postflop (%)', 'postflop_fraction')]
        for i, (label, metric) in enumerate(keys):
            a.button(label, 34 + i * 196, 273, 184, lambda metric=metric: setattr(self, 'metric', metric), active=self.metric == metric)
        def value(point):
            result = point.get(self.metric)
            return 100 * result if result is not None and self.metric == 'postflop_fraction' else result
        series = [(f"Seed {r['seed']}", [(p['steps'], value(p)) for p in r['curves']], COLORS[i])
                  for i, r in enumerate(self.visible_runs())]
        self.chart(series, caption='APPRENTISSAGE · gain sur les 500 dernières mains complètes ; adversaires de ligue évolutifs.')
        if run and run.get('curves'):
            p = run['curves'][-1]
            a.text(f"Mesure à {p['steps']} décisions : {p['completed']} mains complètes / {p['truncated']} tronquées · fenêtre : {p['rolling_hands']} mains", 34, 702, 10, MUTED)
            coverage = run.get('training_coverage', {})
            if coverage:
                total = sum(coverage.get('decisions_by_street', {}).values())
                for i, street in enumerate(('PREFLOP', 'FLOP', 'TURN', 'RIVER')):
                    n = coverage.get('decisions_by_street', {}).get(street, 0)
                    natural = coverage.get('natural_decisions_by_street', {}).get(street, 0)
                    targeted = coverage.get('curriculum_decisions_by_street', {}).get(street, 0)
                    x = 34 + i * 244
                    a.rect(x, 724, 230, 44, PANEL)
                    a.text(f'{street} · {n} ({n / total:.0%})' if total else street, x + 8, 730, 10)
                    a.text(f'Naturel {natural} / ciblé {targeted}', x + 8, 749, 9, MUTED)
                return
            for i, depth in enumerate(self.data['config']['stack_depths'][:3]):
                row = run.get('training_by_depth', {}).get(str(int(depth)), {})
                n, net = row.get('hands', 0), row.get('net_bb', 0.)
                x = 34 + i * 328
                a.rect(x, 724, 314, 44, PANEL)
                a.text(f'{depth} BB · {n} mains', x + 10, 731, 10)
                a.text(f'{100 * net / n:+.1f} BB/100 train' if n else '—', x + 10, 749, 9, MUTED)

    def draw_validation(self):
        a = self.app
        scenarios = sorted({k for r in self.visible_runs() for p in r['selection'] for k in p['scores']})
        if scenarios:
            self.scenario_index %= len(scenarios)
            scenario = scenarios[self.scenario_index]
            a.button(f'Scénario : {scenario} ›', 34, 273, 620,
                lambda: setattr(self, 'scenario_index', self.scenario_index + 1))
        else:
            scenario = ''
        modes = ('best', 'deterministic', 'stochastic')
        a.button({'best': 'Meilleur mode', 'deterministic': 'Déterministe', 'stochastic': 'Probabiliste'}[self.validation_mode] + ' ›',
            674, 273, 332, lambda: setattr(self, 'validation_mode', modes[(modes.index(self.validation_mode) + 1) % len(modes)]))
        def value(point):
            values = point if self.validation_mode == 'best' else point.get('modes', {}).get(self.validation_mode, {})
            return values.get('scores', {}).get(scenario)
        series = [(f"Seed {r['seed']}", [(p['steps'], value(p)) for p in r['selection']], COLORS[i])
                  for i, r in enumerate(self.visible_runs())]
        self.chart(series, caption='VALIDATION DE SÉLECTION · même scénario et mêmes cartes pour toutes les seeds ; BB/100.')
        a.text('Checkpoint retenu : meilleur scénario le plus faible, puis moyenne des scénarios.', 34, 715, 11)
        a.text('Le test final utilise une autre seed de distribution et apparaît dans « Seeds / test final ».', 34, 746, 11, MUTED)

    def draw_seeds(self):
        a = self.app
        for label, x in (('Seed', 48), ('Statut', 145), ('Décisions', 330),
                         ('Checkpoint retenu', 480), ('Test BB/100', 760)):
            a.text(label, x, 282, 12, MUTED)
        for i, run in enumerate(self.visible_runs()):
            scores = scenario_scores(run.get('final_validation'))
            mean = sum(scores.values()) / len(scores) if scores else None
            a.rect(34, 315 + i * 49, 972, 43, PANEL)
            for value, x in ((str(run['seed']), 48), (STATUS_LABELS.get(run['status'], run['status']), 145), (f"{run['steps']:,}", 330),
                             (str(run.get('selected_steps', '—')), 480),
                             (f'{mean:+.2f}' if mean is not None else '—', 760)):
                a.text(value, x, 329 + i * 49, 12)
        aggregate = self.selected_aggregate()
        stats = aggregate['overall']
        a.text(f"Test final : {aggregate['completed_seeds']}/{aggregate['expected_seeds']} seeds complètes", 34, 585, 14)
        if stats['mean'] is not None:
            sd = f"{stats['std']:.2f}" if stats['std'] is not None else '—'
            a.text(f"Moyenne {stats['mean']:+.2f} · écart-type entre seeds {sd} · min {stats['min']:+.2f} / max {stats['max']:+.2f} BB/100", 34, 622, 13, GREEN)
        elif aggregate.get('scenario_mismatch'):
            a.text('Agrégation indisponible : scénarios différents entre seeds.', 34, 622, 13, RED)
        run = self.selected_run()
        if run and run.get('selected_mode'):
            a.text('Mode retenu : ' + ('déterministe' if run['selected_mode'] == 'deterministic' else 'probabiliste'), 34, 650, 11, MUTED)
        a.button('Métriques / analyse de cette seed', 34, 682, 450, self.inspect,
            enabled=bool(run and run.get('final_validation')))
        a.button('Version enregistrée' if run and run.get('saved_version') else 'Enregistrer cette seed', 506, 682, 500,
            self.save, enabled=bool(run and run['status'] == 'completed' and not run.get('saved_version'))
                and not self.running and self.data.get('status') != 'running')
        a.text('Moyenne : poids égal par scénario, puis par seed. Dispersion calculée entre entraînements indépendants.', 34, 746, 10, MUTED)

    def draw_modes(self, run):
        a = self.app
        trials = self.data.get('trials', [])
        if not trials:
            a.text('Cette ancienne campagne ne comparait pas les réglages et les modes de jeu.', 34, 290, 13, MUTED)
            return
        a.text('Essais PPO · mêmes seeds et budgets · sélection uniquement', 34, 277, 13)
        for label, x in (('Taux', 48), ('Seeds mesurées', 230), ('Moyenne BB/100', 465), ('Écart-type', 700), ('Retenu', 880)):
            a.text(label, x, 309, 11, MUTED)
        for i, trial in enumerate(trials[:3]):
            y = 338 + i * 35
            summary = self.data.get('trial_summaries', {}).get(trial['id'], {})
            stat = summary.get('selection_mean', {})
            a.text(f"{trial['learning_rate']:g}", 48, y, 12)
            a.text(str(summary.get('trained_seeds', 0)), 230, y, 12)
            a.text(f"{stat['mean']:+.1f}" if stat.get('mean') is not None else '—', 465, y, 12)
            a.text(f"{stat['std']:.1f}" if stat.get('std') is not None else '—', 700, y, 12)
            a.text('oui' if self.data.get('selected_trial') == trial['id'] else '—', 880, y, 12, GREEN)
        if not run:
            return
        step = run.get('selected_steps')
        point = next((p for p in run['selection'] if p['steps'] == step), run['selection'][-1] if run['selection'] else {})
        a.text(f"Seed {run['seed']} · checkpoint {step or '—'} · comparaison des modes", 34, 458, 13)
        for label, x in (('Mode', 48), ('Gain moyen', 300), ('Pire scénario', 475), ('Entropie norm.', 665), ('P(tapis)', 880)):
            a.text(label, x, 490, 11, MUTED)
        for i, (mode, values) in enumerate(point.get('modes', {}).items()):
            y = 525 + i * 40
            a.text('Déterministe' if mode == 'deterministic' else 'Probabiliste', 48, y, 12)
            a.text(f"{values['mean_bb100']:+.1f}" if 'mean_bb100' in values else '—', 300, y, 12)
            a.text(f"{values['worst_bb100']:+.1f}" if 'worst_bb100' in values else '—', 475, y, 12)
            diagnostics = [d['overall'] for d in values.get('diagnostics', {}).values() if d]
            n = sum(d['decisions'] for d in diagnostics)
            if n:
                entropy = sum(d['decisions'] * d['normalized_entropy'] for d in diagnostics) / n
                all_in = sum(d['decisions'] * d['mean_probabilities']['ALL_IN'] for d in diagnostics) / n
                a.text(f'{entropy:.1%}', 665, y, 12)
                a.text(f'{all_in:.1%}', 880, y, 12)
        mode_label = {'deterministic': 'déterministe', 'stochastic': 'probabiliste'}.get(run.get('selected_mode'), 'en attente')
        a.text('Choix enregistré : ' + mode_label, 34, 620, 12, GREEN)
        a.button('Analyser la confrontation', 674, 607, 332, lambda: self.inspect('champion_validation'),
            enabled=bool(run.get('champion_validation')))
        review = run.get('promotion_review', {})
        labels = {'inconclusive': 'avantage non démontré', 'supported_on_tested_depths': 'avantage appuyé aux profondeurs testées',
                  'no_compatible_champion': 'aucun bot courant compatible', 'incomplete': 'test incomplet'}
        a.text('Face au bot courant au départ : ' + labels.get(review.get('status'), 'en attente du test final'), 34, 661, 12)
        for i, row in enumerate(review.get('scenarios', [])[:3]):
            low, high = row['simultaneous_ci95_bb100']
            a.text(f"{row['depth_bb']} BB : {row['bb100']:+.0f} [{low:+.0f} ; {high:+.0f}]", 34 + i * 328, 701, 10, MUTED)
        a.text('Intervalles simultanés approximatifs à 95 %. Aucun score ne déclenche une promotion automatique.', 34, 758, 10, MUTED)

    def draw_depths(self):
        a = self.app
        scenarios = self.selected_aggregate()['scenarios']
        all_depths = self.data['config']['stack_depths']
        column_pages = max(1, (len(all_depths) + 2) // 3)
        self.depth_column_page %= column_pages
        depths = all_depths[self.depth_column_page * 3:self.depth_column_page * 3 + 3]
        opponents = [*self.data.get('anchors', {})]
        opponents = ['random', 'check_call', 'tight', 'aggressive', *opponents]
        row_pages = max(1, (len(opponents) + 6) // 7)
        self.depth_row_page %= row_pages
        a.text('TEST FINAL · moyenne ± écart-type entre seeds · BB/100 · N = seeds complètes', 34, 280, 12, MUTED)
        run = self.selected_run()
        if run:
            coverage = ' · '.join(f"{depth} BB : {run.get('training_by_depth', {}).get(str(int(depth)), {}).get('hands', 0)} mains"
                                  for depth in depths)
            a.text(f"Couverture train seed {run['seed']} · {coverage}"[:130], 34, 305, 10, MUTED)
        for j, depth in enumerate(depths):
            a.text(f'{depth} BB', 355 + j * 215, 325, 15)
        for i, opponent in enumerate(opponents[self.depth_row_page * 7:self.depth_row_page * 7 + 7]):
            y = 365 + i * 48
            label = self.data.get('anchors', {}).get(opponent, {}).get('label', opponent)
            a.text(label[:31], 34, y + 10, 10)
            for j, depth in enumerate(depths):
                stat = scenarios.get(f'{depth}bb:{opponent}', {})
                mean, sd = stat.get('mean'), stat.get('std')
                color = (32, 83, 71) if mean is not None and mean >= 0 else (92, 45, 52) if mean is not None else PANEL
                a.rect(330 + j * 215, y, 200, 40, color)
                label = '—' if mean is None else f"{mean:+.0f} ± {sd:.0f} · N{stat['n']}" if sd is not None else f"{mean:+.0f} · N{stat['n']}"
                a.text(label, 345 + j * 215, y + 12, 11)
        if column_pages > 1:
            a.button(f'Profondeurs {self.depth_column_page + 1}/{column_pages} ›', 34, 710, 280,
                lambda: setattr(self, 'depth_column_page', self.depth_column_page + 1))
        if row_pages > 1:
            a.button(f'Adversaires {self.depth_row_page + 1}/{row_pages} ›', 336, 710, 280,
                lambda: setattr(self, 'depth_row_page', self.depth_row_page + 1))
        a.text('Les cases absentes attendent un test final complet. Les gains restent exprimés en BB, à chaque profondeur.', 34, 765, 10, MUTED)

    def draw_league(self, run):
        if not run:
            return
        a = self.app
        league = run.get('league', {})
        similar = sum(item.get('reason') == 'similar_behaviour' for item in league.get('retired', []))
        a.text(f"Références : {len(league.get('anchors', {}))} · snapshots actifs : {len(league.get('history', {}))} · retirés : {len(league.get('retired', []))} · similaires : {similar}", 34, 280, 13)
        a.text('APPRENTISSAGE · exposition par adversaire, toutes profondeurs · répartition des mains', 34, 319, 11, MUTED)
        totals = {}
        for row in league.get('exposure', []):
            target = totals.setdefault(row['opponent'], [0, 0.])
            target[0] += row['hands']
            target[1] += row['net_bb']
        maximum = max([v[0] for v in totals.values()] or [1]) or 1
        for i, (opponent, (hands, net)) in enumerate(sorted(totals.items(), key=lambda item: -item[1][0])[:11]):
            y = 355 + i * 28
            label = league.get('anchors', {}).get(opponent, {}).get('label', opponent)
            a.text(label[:40], 34, y + 4, 10)
            a.rect(390, y, 360 * hands / maximum, 20, COLORS[i % len(COLORS)])
            a.text(f'{hands} mains · {100 * net / hands:+.1f} BB/100' if hands else '0 main', 775, y + 4, 10)
        a.text('50 % heuristiques / 50 % pool figé (probabilités). Les fréquences observées fluctuent.', 34, 701, 11, MUTED)
        a.text('À plusieurs joueurs, une main peut compter pour plusieurs adversaires. Ces gains ne sont pas de la validation.', 34, 732, 10, MUTED)
        if self.data.get('skipped_anchors'):
            a.text(f"{len(self.data['skipped_anchors'])} ancienne(s) référence(s) incompatible(s), détaillées dans campaign.json.", 34, 760, 10, MUTED)
