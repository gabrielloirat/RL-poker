"""Starting-hand heatmap and position/style panels for native validation UI."""
from poker_rl.strategy_metrics import RANKS, matrix_key, analyze_strategy

MUTED = (154, 169, 188)
GREEN = (78, 213, 164)
RED = (240, 112, 116)
PANEL = (25, 36, 51)


def rate_text(metric):
    rate = metric.get('rate')
    return ('—' if rate is None else f'{rate:.1%}') + f" ({metric.get('numerator', 0)}/{metric.get('denominator', 0)})"


def strategy_data(app):
    analysis = app.analysis_benchmark_data()['hand_analysis']
    if 'strategy' in analysis and (app.analysis_view != 'ranges' or 'starting_hands' in analysis['strategy']):
        return analysis['strategy']
    return analyze_strategy(app.analysis_hands) if app.analysis_hands is not None else None


def draw_strategy(app):
    data = strategy_data(app)
    if data is None:
        app.text('Chargement des mains ou ancienne validation : relancez sa validation pour cette analyse.', 34, 230, 12, MUTED)
        return
    if app.analysis_view == 'ranges':
        draw_ranges(app, data)
    else:
        draw_positions(app, data)


def draw_policy(app):
    diagnostics = app.analysis_benchmark_data().get('policy_diagnostics')
    if not diagnostics:
        app.text('Cette validation ne contient pas encore les probabilités. Relancez sa validation.', 34, 230, 13, MUTED)
        return
    row = diagnostics['overall']
    cards = [('Entropie normalisée', f"{row['normalized_entropy']:.1%}"),
             ('Probabilité maximale', f"{row['max_probability']:.1%}"),
             ('Écart des deux premiers', f"{row['top_two_margin']:.1%}")]
    for i, (label, value) in enumerate(cards):
        x = 34 + i * 328
        app.rect(x, 183, 314, 91, PANEL)
        app.text(label, x + 12, 200, 12, MUTED)
        app.text(value, x + 12, 235, 19, GREEN)
    from poker_rl.game import ACTION_NAMES
    from poker_rl.hand_analysis import ACTION_LABELS
    app.text('Action', 46, 298, 11, MUTED)
    app.text('Probabilité moyenne prédite', 320, 298, 11, MUTED)
    app.text('Fréquence réellement jouée', 650, 298, 11, MUTED)
    for i, action in enumerate(ACTION_NAMES):
        y = 334 + i * 36
        probability = row['mean_probabilities'][action]
        frequency = row['action_counts'][action] / row['decisions']
        app.text(ACTION_LABELS[action], 46, y + 5, 11)
        app.rect(320, y, 220, 23, PANEL)
        app.rect(320, y, 220 * probability, 23, GREEN)
        app.text(f'{probability:.1%}', 553, y + 5, 10)
        app.rect(650, y, 220, 23, PANEL)
        app.rect(650, y, 220 * frequency, 23, (111, 168, 255))
        app.text(f'{frequency:.1%}', 883, y + 5, 10)
    app.text(f"{row['decisions']} décisions du bot · mains complètes de validation uniquement", 34, 650, 11, MUTED)
    for i, (street, values) in enumerate(diagnostics['by_street'].items()):
        x = 34 + i * 244
        app.text(street, x, 690, 11)
        app.text(f"{values['decisions']} décisions · H {values['normalized_entropy']:.0%}", x, 717, 10, MUTED)
    app.text('Une action toujours choisie en mode déterministe peut avoir une faible probabilité : elle arrive simplement en tête.', 34, 758, 10, MUTED)


def draw_ranges(app, data):
    metric = getattr(app, 'range_metric', 'bb_per_100')
    for i, (label, key) in enumerate((('Gain BB/100', 'bb_per_100'), ('VPIP', 'vpip'), ('PFR', 'pfr'), ('Effectif', 'hands'))):
        app.button(label, 34 + i * 244, 174, 230, lambda key=key: setattr(app, 'range_metric', key), active=metric == key)
    app.text('Suited au-dessus de la diagonale · offsuit en dessous · paires sur la diagonale', 34, 225, 11, MUTED)
    selected = getattr(app, 'range_selected', 'AA')
    values = data['starting_hands']
    max_abs = max([abs(v.get('bb_per_100') or 0) for v in values.values()] or [1]) or 1
    max_n = max([v['hands'] for v in values.values()] or [1]) or 1
    def choose(key):
        app.range_selected = key
    for i, rank in enumerate(RANKS):
        app.text(rank, 73, 280 + i * 36, 11, MUTED)
        app.text(rank, 107 + i * 39, 253, 11, MUTED)
    for row in range(13):
        for column in range(13):
            key = matrix_key(row, column)
            stats = values[key]
            n = stats['hands']
            value = stats.get(metric)
            if isinstance(value, dict):
                value = value['rate']
            x, y = 94 + column * 39, 271 + row * 36
            if n == 0 or value is None:
                color = (38, 46, 58)
            elif metric == 'bb_per_100':
                strength = .25 + .75 * abs(value) / max_abs
                if n < 20:
                    strength *= .45
                color = (int(38 + 95 * strength), 45, 53) if value < 0 else (30, int(58 + 105 * strength), 76)
            else:
                strength = value / max_n if metric == 'hands' else value
                color = (30, int(60 + 100 * strength), int(75 + 90 * strength))
            app.rect(x, y, 37, 34, color)
            from poker_rl.gui import Button
            app.buttons.append(Button(x, y, 37, 34, lambda key=key: choose(key), True,
                f"{key} · {n} mains · " + (f"{stats['bb_per_100']:+.1f} BB/100" if n else 'non observée')))
            app.text(key, x + 2, y + 3, 9, GREEN if key == selected else (230, 236, 244))
            label = '—' if value is None or n == 0 else f'{value:.0%}' if metric in ('vpip', 'pfr') else f'{value:+.0f}' if metric == 'bb_per_100' else str(n)
            app.text(label, x + 2, y + 19, 8, MUTED)
    stats = values[selected]
    app.text(selected, 646, 284, 24, GREEN)
    app.text(f"{stats['hands']} mains complètes", 646, 336, 13)
    app.text(f"Gain : {stats['bb_per_100']:+.1f} BB/100" if stats['hands'] else 'Gain : non observé', 646, 375, 12)
    app.text('VPIP : ' + rate_text(stats['vpip']), 646, 414, 12)
    app.text('PFR : ' + rate_text(stats['pfr']), 646, 453, 12)
    app.text('Gains < 20 mains : couleurs atténuées.', 646, 506, 10, MUTED)
    app.text('Gains observés ; précision souvent faible.', 646, 531, 10, MUTED)
    app.text('Aucune case vide ne vaut zéro.', 646, 556, 10, MUTED)
    def inspect():
        app.analysis_parent = 'ranges'
        app.analysis_dimension = 'starting_hand'
        app.show_analysis_hands(category=selected, outcome='all')
    app.button('Inspecter ces mains', 646, 606, 360, inspect, enabled=bool(stats['hands']) and app.analysis_hands is not None)
    app.text('Statistiques de validation · VPIP/PFR : mains avec au moins une décision préflop, blindes exclues.', 34, 758, 10, MUTED)


def draw_positions(app, data):
    stats = data['overall']
    for i, (label, key) in enumerate((('VPIP', 'vpip'), ('Relance préflop', 'pfr'), ('3-bet', 'three_bet'), ('Fold grosse mise', 'fold_large_bet'))):
        x = 34 + i * 244
        app.rect(x, 188, 230, 99, PANEL)
        app.text(label, x + 12, 204, 12, MUTED)
        app.text(rate_text(stats[key]), x + 12, 242, 15, GREEN)
    for label, x in (('Position', 48), ('Mains', 165), ('Gain net BB', 275),
                     ('Gain BB/100', 450), ('VPIP', 650), ('PFR', 845)):
        app.text(label, x, 321, 11, MUTED)
    for i, (position, row) in enumerate(data['positions'].items()):
        y = 358 + i * 44
        app.rect(34, y, 972, 38, PANEL)
        app.text(position, 48, y + 11, 11)
        app.text(str(row['hands']), 165, y + 11, 11)
        app.text(f"{row['net_bb']:+.1f}", 275, y + 11, 11)
        app.text(f"{row['bb_per_100']:+.1f}", 450, y + 11, 11, GREEN if row['net_bb'] >= 0 else RED)
        app.text(rate_text(row['vpip']), 650, y + 11, 10)
        app.text(rate_text(row['pfr']), 845, y + 11, 10)
    app.text('Folds face aux grosses mises · par décision et par street', 34, 592, 13)
    for i, street in enumerate(('PREFLOP', 'FLOP', 'TURN', 'RIVER')):
        row = stats['fold_large_bet_by_street'].get(street, {'rate': None, 'numerator': 0, 'denominator': 0})
        x = 34 + i * 244
        app.text(street, x, 628, 10, MUTED)
        app.rect(x, 656, 210, 15, PANEL)
        if row['rate'] is not None:
            app.rect(x, 656, 210 * row['rate'], 15, RED)
        app.text(rate_text(row), x, 686, 11)
    app.text('Grosse mise : montant à suivre / pot avant action ≥ 50 % ; le pot comprend la mise adverse.', 34, 731, 10, MUTED)
    app.text('3-bet : mains avec relance autorisée face à exactement une relance préflop. Ratios = numérateur / dénominateur.', 34, 758, 10, MUTED)
