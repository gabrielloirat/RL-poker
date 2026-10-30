"""Scientific figures and a local HTML readout of measured benchmark artifacts."""
import base64
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'ppo': '#2563eb', 'dqn': '#d97706', 'mcts': '#16805b'}


def render_results(root):
    root = Path(root)
    state = json.loads((root / 'benchmark.json').read_text(encoding='utf-8'))
    config = state['config']
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 140})
    outputs = []
    paired_effects = []

    def save(fig, name):
        fig.savefig(figures / name, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        outputs.append(figures / name)

    for players in config['players']:
        models = [m for m in state['models'] if m['players'] == players]
        if not models:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout='constrained')
        for ax, kind in zip(axes, COLORS):
            row = next((m for m in models if m['kind'] == kind and m['regime'] == 'curriculum'), None)
            if row is None:
                continue
            values, boundaries = [], []
            for block in row['blocks']:
                phase = block['phase']
                prefix = 'random' if phase == 'random' else 'selfplay'
                index = sum(b['phase'] == phase for b in row['blocks'][:row['blocks'].index(block)])
                filename = root / row['id'] / f'{prefix}_{index}' / 'hands.csv'
                if filename.exists():
                    with filename.open(encoding='utf-8') as stream:
                        values.extend(float(r['reward_bb']) for r in csv.DictReader(stream))
                boundaries.append((len(values), phase))
            if values:
                rewards = np.asarray(values)
                window = min(500, len(rewards))
                rolling = np.convolve(rewards, np.ones(window) / window, mode='valid') * 100
                ax.plot(np.arange(window, len(values) + 1), rolling, color=COLORS[kind], lw=1.2)
                transition = next((boundaries[i - 1][0] for i, (_, phase) in enumerate(boundaries)
                                   if phase == 'selfplay' and i > 0), None)
                if transition is not None:
                    ax.axvline(transition, color='#374151', ls='--', label='Passage au jeu contre soi-même')
                    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
                ax.set_title(f'{kind.upper()} — entraînement, fenêtre de {window} mains')
                ax.set_ylabel('Gain (bb/100)')
                ax.axhline(0, color='#6b7280', lw=.7)
                ax.grid(axis='y', alpha=.18)
        axes[-1].set_xlabel('Mains jouées pendant l’entraînement')
        fig.suptitle(f'{players} joueurs · seed {config["seed"]} · courbes d’apprentissage (hors validation)', fontsize=15)
        save(fig, f'training_{players}p.png')

        specialists = [m for m in models if m['regime'].startswith('vs_') and m['blocks']]
        if specialists:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout='constrained')
            for ax, kind in zip(axes, COLORS):
                for index, model in enumerate(m for m in specialists if m['kind'] == kind):
                    values = []
                    for block in range(len(model['blocks'])):
                        filename = root / model['id'] / f'block_{block}' / 'hands.csv'
                        if filename.exists():
                            with filename.open(encoding='utf-8') as stream:
                                values.extend(float(r['reward_bb']) for r in csv.DictReader(stream))
                    if not values:
                        continue
                    window = min(500, max(1, len(values) // 5))
                    rolling = np.convolve(values, np.ones(window) / window, mode='valid') * 100
                    ax.plot(np.arange(window, len(values) + 1), rolling, color=COLORS[kind],
                            ls='-' if index == 0 else '--', lw=1.3,
                            label=f"Contre {model['regime'].removeprefix('vs_').upper()} · fenêtre {window} mains")
                ax.set_title(kind.upper() + ' — entraînements spécialisés')
                ax.set_ylabel('Gain (bb/100)')
                ax.axhline(0, color='#6b7280', lw=.7)
                ax.grid(axis='y', alpha=.18)
                if ax.lines and ax.get_legend_handles_labels()[0]:
                    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
            axes[-1].set_xlabel('Mains supplémentaires pendant la spécialisation')
            fig.suptitle(f'Spécialisations à {players} joueurs · seed {config["seed"]} · hors validation', fontsize=15)
            save(fig, f'specialist_training_{players}p.png')

        tests = [r for r in state['tests'] if r.get('table_players', r['players']) == players]
        if not tests:
            continue
        opponents = ['random', 'ppo', 'dqn', 'mcts'] + (['table_diverse'] if players > 2 else [])
        fig, axes = plt.subplots(len(opponents), 1, figsize=(13, 6.2 * len(opponents)), layout='constrained')
        for ax, opponent in zip(np.atleast_1d(axes), opponents):
            rows = [r for r in tests if r['opponent'] == opponent]
            labels = [f"{r['kind'].upper()} {r['players']}j → {players}j · {r['regime'].replace('vs_', 'spéc. ')}" for r in rows]
            for i, row in enumerate(rows):
                ax.errorbar(row['bb100'], i,
                            xerr=[[row['bb100'] - row['ci_low']], [row['ci_high'] - row['bb100']]],
                            fmt='o', capsize=3, color=COLORS[row['kind']])
            ax.set_yticks(range(len(labels)), labels)
            ax.invert_yaxis()
            ax.axvline(0, color='#374151', ls='--', lw=.9)
            ax.grid(axis='x', alpha=.18)
            ax.set_title(f'Adversaire : {opponent} · {config["final_hands"]} mains par modèle')
            ax.set_xlabel('Gain (bb/100 mains) · intervalle normal ponctuel à 95 %')
        fig.suptitle(f'Évaluations sans apprentissage · {players} joueurs · seed racine {config["seed"]}', fontsize=15)
        save(fig, f'evaluation_{players}p.png')

        tested_models = [m for m in state['models'] if any(r['model'] == m['id'] for r in tests)]
        matrix = np.array([[next((r['bb100'] for r in tests if r['model'] == model['id'] and r['opponent'] == opp), np.nan)
                            for opp in opponents] for model in tested_models])
        fig, ax = plt.subplots(figsize=(12, 9), layout='constrained')
        bound = max(1., float(np.nanmax(np.abs(matrix))))
        palette = plt.get_cmap('RdBu').copy()
        palette.set_bad('#e5e7eb')
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=palette, vmin=-bound, vmax=bound, aspect='auto')
        ax.set_xticks(range(len(opponents)), opponents)
        ax.set_yticks(range(len(tested_models)), [f"{m['kind'].upper()} {m['players']}j → {players}j · {m['regime']}" for m in tested_models])
        for i in range(len(tested_models)):
            for j in range(len(opponents)):
                ax.text(j, i, '—' if np.isnan(matrix[i, j]) else f'{matrix[i, j]:.0f}', ha='center', va='center',
                        color='white' if abs(matrix[i, j]) > .6 * bound else '#111827')
        fig.colorbar(image, ax=ax, label='Gain moyen (bb/100)')
        ax.set_title(f'{players} joueurs — comparaison des modèles et entraînements')
        save(fig, f'comparison_{players}p.png')

        # Paired differences use the same held-out hand seeds, not independent
        # standard errors: a high-variance deal affects both checkpoints.
        effects = []
        available = {(row['model'], row['opponent']) for row in tests}
        for kind in COLORS:
            for training_players in config['players']:
                before_id = f'{kind}_{training_players}p_random_only'
                after_id = f'{kind}_{training_players}p_curriculum'
                for opponent in opponents:
                    if (before_id, opponent) not in available or (after_id, opponent) not in available:
                        continue
                    before_path = root / after_id / f'test_{players}p_{opponent}.csv'
                    initial_path = root / before_id / f'test_{players}p_{opponent}.csv'
                    if not before_path.exists() or not initial_path.exists():
                        continue
                    with before_path.open(encoding='utf-8') as stream:
                        after_hands = list(csv.DictReader(stream))
                    with initial_path.open(encoding='utf-8') as stream:
                        before_hands = list(csv.DictReader(stream))
                    if [r['seed'] for r in after_hands] != [r['seed'] for r in before_hands]:
                        raise ValueError('Les différences appariées exigent les mêmes deals.')
                    delta = np.array([float(a['reward_bb']) - float(b['reward_bb'])
                                      for a, b in zip(after_hands, before_hands)])
                    mean = float(delta.mean() * 100)
                    margin = float(1.96 * delta.std(ddof=1) / np.sqrt(len(delta)) * 100)
                    effect = dict(kind=kind, training_players=training_players, table_players=players,
                                  opponent=opponent, delta_bb100=mean, ci_low=mean - margin,
                                  ci_high=mean + margin, hands=len(delta))
                    effects.append(effect)
                    paired_effects.append(effect)
        if effects:
            fig, axes = plt.subplots(len(opponents), 1, figsize=(13, 3.1 * len(opponents)), layout='constrained')
            for ax, opponent in zip(np.atleast_1d(axes), opponents):
                rows = [r for r in effects if r['opponent'] == opponent]
                for i, row in enumerate(rows):
                    ax.errorbar(row['delta_bb100'], i,
                                xerr=[[row['delta_bb100'] - row['ci_low']], [row['ci_high'] - row['delta_bb100']]],
                                color=COLORS[row['kind']], fmt='o', capsize=3)
                ax.set_yticks(range(len(rows)), [f"{r['kind'].upper()} entraîné à {r['training_players']}j" for r in rows])
                ax.invert_yaxis()
                ax.axvline(0, color='#374151', ls='--')
                ax.set_title(f'Test à {players} joueurs contre {opponent}')
                ax.set_xlabel('Après self-play − avant self-play (bb/100) · IC 95 % apparié')
                ax.grid(axis='x', alpha=.18)
            fig.suptitle('Transfert du jeu contre soi-même vers les autres adversaires', fontsize=15)
            save(fig, f'selfplay_transfer_{players}p.png')

    if paired_effects:
        with (root / 'selfplay_transfer.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired_effects[0]))
            writer.writeheader()
            writer.writerows(paired_effects)

    table = []
    for row in state['tests']:
        table.append('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in (
            row['model'], f"{row.get('table_players', row['players'])}j / {row['opponent']}", row['hands'], f"{row['bb100']:.1f}",
            f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}]", f"{row['win_rate']:.1%}")) + '</tr>')
    gates = ''.join(f"<li>{html.escape(m['id'])} : {'seuil atteint, self-play effectué' if m.get('gate_passed') else 'seuil non atteint ; self-play non déclenché'}</li>"
                    for m in state['models'] if m['regime'] == 'curriculum')
    budgets = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in (
        m['id'], sum(b['decisions'] for b in m['blocks']),
        sum(b['training']['hands'] for b in m['blocks']),
        f"{sum(b['seconds'] for b in m['blocks']) / 60:.1f}")) + '</tr>'
        for m in state['models'] if m['regime'] != 'random_only')
    images = ''.join(f'<figure><img alt="{html.escape(p.stem)}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"></figure>'
                     for p in outputs)
    expected_tests = len(state['models']) * sum(4 + int(n > 2) for n in config['players'])
    lookup = {(r['model'], r.get('table_players', r['players']), r['opponent']): r for r in state['tests']}

    def score(row):
        return f"{row['bb100']:+.0f} bb/100 (IC 95 % [{row['ci_low']:+.0f} ; {row['ci_high']:+.0f}])"

    findings = []
    random_duel = lookup.get(('ppo_2p_curriculum', 2, 'random'))
    mcts_duel = lookup.get(('ppo_2p_curriculum', 2, 'mcts'))
    if random_duel and mcts_duel:
        findings.append(f"<strong>L’adversaire compte.</strong> Le même PPO issu du duel obtient {score(random_duel)} contre random et {score(mcts_duel)} contre MCTS en duel.")
    diverse = lookup.get(('ppo_2p_curriculum', 4, 'table_diverse'))
    if diverse:
        findings.append(f"<strong>Transfert du duel vers quatre joueurs.</strong> Sans changer ses poids, ce PPO obtient {score(diverse)} face à random, DQN et MCTS réunis.")
    effect = next((e for e in paired_effects if e['kind'] == 'ppo' and e['training_players'] == 4
                   and e['table_players'] == 4 and e['opponent'] == 'random'), None)
    if effect:
        change = f"a diminué de {abs(effect['delta_bb100']):.0f}" if effect['ci_high'] < 0 else f"a augmenté de {abs(effect['delta_bb100']):.0f}" if effect['ci_low'] > 0 else f"a varié de {effect['delta_bb100']:+.0f}"
        findings.append(f"<strong>Effet du jeu contre soi-même.</strong> Pour le PPO appris à quatre joueurs, le gain contre random {change} bb/100 après cette phase : IC 95 % apparié de la différence [{effect['ci_low']:+.0f} ; {effect['ci_high']:+.0f}].")
    specialist_dqn = lookup.get(('ppo_2p_vs_dqn', 4, 'dqn'))
    specialist_ppo = lookup.get(('ppo_2p_vs_dqn', 4, 'ppo'))
    if specialist_dqn and specialist_ppo:
        findings.append(f"<strong>Spécialisation et transfert.</strong> Le PPO spécialisé contre DQN en duel obtient, à quatre joueurs, {score(specialist_dqn)} contre trois DQN et {score(specialist_ppo)} contre trois PPO.")
    findings_html = (f'<h2>Ce que montre cette exécution</h2><p>Ces constats concernent la seed {config["seed"]} et les budgets exécutés. Les intervalles à 95 % sont ponctuels ; ils ne sont pas corrigés pour l’ensemble des comparaisons finales.</p><ul>'
                     + ''.join(f'<li>{f}</li>' for f in findings) + '</ul>') if findings else ''
    content = f'''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Poker : entraînements et évaluations — seed {config['seed']}</title>
<style>body{{font:16px system-ui;margin:36px auto;max-width:1280px;padding:0 24px;color:#172033;background:#f8fafc}}
h1{{font-size:32px}}h2{{margin-top:40px}}p,li{{line-height:1.6}}figure{{margin:24px 0;background:white;border:1px solid #dde3ec;border-radius:10px;padding:12px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #e5e7eb}}.table{{overflow:auto}}code{{background:#e2e8f0;padding:3px}}</style>
<h1>Poker — PPO, DQN et MCTS</h1>
<p>Deux expériences séparées : duel et tables à quatre joueurs. Chaque modèle est ensuite testé dans les deux formats, avec les mêmes poids. Seed racine : <strong>{config['seed']}</strong>.
État : <strong>{state['status']}</strong>. Random sert de référence ; human et mixed sont exclus.</p>
<p>Évaluations enregistrées à la génération de ce rapport : <strong>{len(state['tests'])}/{expected_tests}</strong>. Les cellules « — » indiquent des résultats encore absents.</p>
{findings_html}
<p>Lecture des noms : <code>curriculum</code> désigne le modèle après random puis jeu contre soi-même ; <code>random_only</code> est le checkpoint avant cette seconde phase ; <code>vs_X</code> est une spécialisation contre X. Les mentions « 2j → 4j » indiquent un entraînement en duel suivi d’un test à quatre joueurs.</p>
<h2>Passage de random au jeu contre soi-même</h2><ul>{gates}</ul>
<p>Seuil : au moins {config['gate_bb100']:.0f} bb/100 et borne inférieure positive, corrigée pour les contrôles répétés.
Les spécialistes reprennent les poids du curriculum et s’entraînent contre un autre modèle figé.</p>
<h2>Graphiques</h2>{images}
<h2>Budgets d’entraînement exécutés</h2><p>Pour les spécialistes, ces quantités sont ajoutées au curriculum hérité. Les checkpoints « random_only » sont des copies avant self-play, sans nouvel entraînement. Les temps ci-dessous excluent les validations.</p>
<div class="table"><table><thead><tr><th>Modèle</th><th>Décisions</th><th>Mains</th><th>Minutes</th></tr></thead><tbody>{budgets}</tbody></table></div>
<h2>Résultats finaux</h2><div class="table"><table><thead><tr><th>Modèle</th><th>Adversaire</th><th>Mains</th><th>bb/100</th><th>IC 95 %</th><th>Mains à gain positif</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<h2>Lecture et limites</h2><p>Les gains sont mesurés en grosses blindes, avec un stack remis à 50 bb à chaque main.
Les courbes d’entraînement décrivent une politique qui change ; seules les évaluations finales comparent des politiques figées.
Les sièges, boutons et placements des adversaires tournent. Les deals finaux sont identiques entre modèles d’un même format, sans être utilisés pour entraîner ou sélectionner les poids.</p>
<p>Les intervalles à 95 % décrivent la variabilité entre mains pour ce modèle et cette seed. Ils ne mesurent pas la variabilité entre entraînements ; les comparaisons multiples finales ne sont pas corrigées.
Un gain positif contre random ne prouve pas une stratégie optimale. Les budgets sont propres à chaque algorithme et ne constituent pas une comparaison à coût de calcul égal.
MCTS utilise {config['mcts_simulations']} simulations par décision ; son apprentissage modifie les préférences de sa recherche.
Les adversaires PPO, DQN et MCTS sont les checkpoints issus de leur curriculum, et restent inchangés pendant les tests.</p>
<p>Le transfert entre formats utilise une observation commune à cinq emplacements pour les réseaux PPO et DQN : tous les adversaires sont conservés, les places absentes sont nulles, et les montants utilisent une échelle commune. MCTS accepte directement le nombre réel de joueurs. Une table diverse réunit random et les deux autres approches, face au modèle évalué.</p>
<p>Les graphiques « après − avant self-play » mesurent le changement sur les mêmes deals. Une différence positive indique une amélioration du score contre l’adversaire testé, même si le gain absolu reste négatif. Un intervalle qui traverse zéro ne permet pas de conclure au seuil ponctuel de 95 %. Ces différences concernent une seule trajectoire d’entraînement.</p>
<p>Données : <a href="results.csv">results.csv</a> · <a href="benchmark.json">configuration, seeds et checkpoints</a> · <a href="source_provenance.json">empreintes du code et versions</a> · <a href="validation_summary.json">contrôles des données et calculs</a>.
Les CSV de chaque main se trouvent dans les dossiers de modèles. Les poids sont conservés dans les fichiers <code>.checkpoint</code>.</p>
</html>'''
    (root / 'report.html').write_text(content, encoding='utf-8')
    return root / 'report.html'
