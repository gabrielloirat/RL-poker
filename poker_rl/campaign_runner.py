"""Train matched PPO trials, choose on selection data, then open the final tests."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from statistics import NormalDist
from threading import Event
import time
from uuid import uuid4

import numpy as np

from poker_rl.campaigns import CampaignConfig, aggregate_seeds, distribution, scenario_scores
from poker_rl.evaluation import evaluate_depths
from poker_rl.hand_analysis import compact_report
from poker_rl.league import HEURISTICS, League, LeagueOpponent, freeze_anchors
from poker_rl.records import write_json
from poker_rl.sb3_training import train_sb3_session


def selection_score(report, expected):
    scores = scenario_scores(report)
    if len(scores) != expected:
        return None
    return {'scores': scores, 'worst_bb100': min(scores.values()),
            'mean_bb100': float(np.mean(list(scores.values())))}


def promotion_review(report, *, expected_depths):
    """Review only; never changes the current bot or creates a catalog entry."""
    if not report or report['status'] != 'completed':
        return {'status': 'incomplete', 'scenarios': [], 'automatic_promotion': False}
    benchmarks = report['agents'][0]['benchmarks']
    if len(benchmarks) != len(expected_depths) or any(b['standard_error_bb'] is None for b in benchmarks):
        return {'status': 'incomplete', 'scenarios': [], 'automatic_promotion': False}
    critical = NormalDist().inv_cdf(1 - .05 / (2 * len(benchmarks)))
    rows = []
    for b in benchmarks:
        margin = critical * b['standard_error_bb'] * 100
        rows.append({'depth_bb': b['depth_bb'], 'bb100': b['bb_per_100'], 'completed': b['completed'],
                     'simultaneous_ci95_bb100': [b['bb_per_100'] - margin, b['bb_per_100'] + margin]})
    supported = all(row['simultaneous_ci95_bb100'][0] > 0 for row in rows)
    return {'status': 'supported_on_tested_depths' if supported else 'inconclusive',
            'scenarios': rows, 'method': 'normal_approximation_bonferroni_familywise_95_percent',
            'automatic_promotion': False}


def _lean_report(report, filename):
    result = compact_report(report, hands_file=filename)
    for candidate in result['agents']:
        for benchmark in candidate['benchmarks']:
            analysis = benchmark.get('hand_analysis', {})
            if 'strategy' in analysis:
                analysis['strategy'] = {k: v for k, v in analysis['strategy'].items() if k != 'starting_hands'}
    return result


def run_campaign(registry, config=None, *, root='runs/campaigns', stop_event=None, on_update=None,
                 agent_kwargs=None, use_catalog_anchors=True):
    import torch
    import stable_baselines3
    config = config or CampaignConfig()
    config.validate()
    automatic_test_seeds = config.selection_seed is None and config.final_seed is None
    torch.set_num_threads(1)
    token = uuid4()
    generated = np.random.SeedSequence(token.int).generate_state(3, dtype=np.uint64)
    selection_seed = config.selection_seed if config.selection_seed is not None else int(generated[0]) % (2**62) + 2**32
    final_seed = config.final_seed if config.final_seed is not None else int(generated[1]) % (2**62) + 2**32
    if final_seed == selection_seed:
        final_seed += 1
    champion_seed = config.champion_seed if config.champion_seed is not None else int(generated[2]) % (2**62) + 2**32
    while champion_seed in (selection_seed, final_seed, *config.seeds):
        champion_seed += 1
    config = config.copy()
    config.selection_seed = selection_seed
    config.final_seed = final_seed
    config.champion_seed = champion_seed
    config.validate()
    stop = stop_event if stop_event is not None else Event()
    campaign_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + token.hex[:8]
    path = Path(root).resolve() / campaign_id
    path.mkdir(parents=True)
    trials = [{'id': f'lr_{rate:g}'.replace('.', 'p').replace('-', 'm'), 'learning_rate': rate,
               'label': f'LR {rate:g}'} for rate in config.learning_rates]
    state = {'version': 2, 'id': campaign_id, 'path': str(path), 'status': 'running', 'phase': 'training_selection',
        'created_at': datetime.now(timezone.utc).isoformat(), 'config': config.to_dict(), 'champion_seed': champion_seed,
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'stable_baselines3': stable_baselines3.__version__},
        'trials': trials, 'selected_trial': None, 'runs': [],
        'protocol': {'initialization': 'fresh_weights_same_seed_per_learning_rate',
            'selection': 'worst_scenario_then_mean; compare_inference_modes',
            'trial_selection': 'worst_seed_worst_scenario_then_mean; selection_data_only',
            'final_test': 'only_selected_trial; mode_and_checkpoint_locked_before_test',
            'fresh_test_seeds_per_campaign': automatic_test_seeds,
            'policy_mode': 'chosen_on_selection' if config.compare_modes else 'configured',
            'training_depth_sampling': 'uniform_per_independent_hand',
            'curriculum': 'adaptive_legal_check_call_prefix; natural_validation_hands_only',
            'training_metrics_are_validation': False}}
    last_publish = 0.

    def publish(*, force=True):
        nonlocal last_publish
        if not force and time.monotonic() - last_publish < 1:
            return
        last_publish = time.monotonic()
        summaries, aggregates = {}, {}
        for trial in trials:
            runs = [r for r in state['runs'] if r['trial_id'] == trial['id']]
            ready = [r for r in runs if r.get('selected_score')]
            summaries[trial['id']] = {'trained_seeds': len(ready),
                'selection_mean': distribution([r['selected_score']['mean_bb100'] for r in ready]),
                'worst_selection_scenario': min((r['selected_score']['worst_bb100'] for r in ready), default=None)}
            aggregates[trial['id']] = aggregate_seeds(runs, len(config.seeds))
        state['trial_summaries'], state['aggregates'] = summaries, aggregates
        state['aggregate'] = aggregates[state['selected_trial'] or trials[0]['id']]
        write_json(path / 'campaign.json', state)
        if on_update:
            on_update(json.loads(json.dumps(state)))

    publish()
    try:
        anchors, skipped = freeze_anchors(registry, path / 'anchors', num_players=config.num_players,
            candidate_root=root if config.include_seed202 else None, candidate_seed=202) if use_catalog_anchors else ({}, [])
        state.update(anchors=anchors, skipped_anchors=skipped)
        champion_ids = [key for key, spec in anchors.items() if spec.get('catalog_kind') == 'current']
        state['champion_opponents'] = champion_ids
        opponents = (*HEURISTICS, *anchors)
        expected = len(config.stack_depths) * len(opponents)
        for trial in trials:
            for seed in config.seeds:
                if stop.is_set():
                    break
                run_path = path / trial['id'] / f'seed_{seed}'
                run_path.mkdir(parents=True)
                league = League(run_path / 'league', seed=seed, anchors=anchors, agents_dir=registry.directory,
                                diversity_threshold=config.diversity_threshold, num_players=config.num_players)
                run = {'id': f"{trial['id']}_seed_{seed}", 'trial_id': trial['id'], 'learning_rate': trial['learning_rate'],
                    'seed': seed, 'status': 'running', 'phase': 'training', 'steps': 0, 'curves': [],
                    'selection': [], 'path': str(run_path), 'league': league.state()}
                state['runs'].append(run)
                publish()
                options = {'n_steps': config.n_steps, 'batch_size': min(128, config.n_steps), **(agent_kwargs or {}),
                    'learning_rate': trial['learning_rate'], 'seed': seed, 'autoload': False,
                    'deterministic': config.deterministic}
                options.pop('checkpoint', None)
                run['agent_options'] = options
                learner = registry.load('ppo', options)
                rivals = [LeagueOpponent(league, seed + 1000 + i) for i in range(config.num_players - 1)]
                agents = [learner, *rivals]
                selections = [('ppo', options), *[('mixed', {}) for _ in rivals]]
                recent, training_by_depth = [], {}
                last_checkpoint = last_snapshot = 0
                best_score = None

                def on_hand(session):
                    if session.game.terminated:
                        recent.append(float(session.game.rewards()[0]))
                        del recent[:-500]
                    depth = float(session.game.hand_start_stacks[0] / session.game.table.big_blind)
                    row = training_by_depth.setdefault(str(int(depth)), {'hands': 0, 'truncated': 0, 'net_bb': 0.})
                    row['hands'] += int(session.game.terminated)
                    row['truncated'] += int(session.game.truncated)
                    if session.game.terminated:
                        row['net_bb'] += float(session.game.rewards()[0])
                    for rival in rivals:
                        league.record(rival.current_id, depth, float(session.game.rewards()[0]), session.game.terminated)

                def on_session(session):
                    session.hand_observers.append(on_hand)

                def progress(done, total, stats):
                    run['steps'] = done
                    if not run['curves'] or done - run['curves'][-1]['steps'] >= 512 or done >= total:
                        coverage = getattr(active_session[0], 'training_coverage', {})
                        point = {'steps': done, 'completed': stats.completed, 'truncated': stats.truncated,
                            'rolling_hands': len(recent), 'rolling_bb100': float(np.mean(recent) * 100) if recent else None,
                            'postflop_fraction': coverage.get('postflop_fraction'), **learner.training_stats}
                        if run['curves'] and run['curves'][-1]['steps'] == done:
                            run['curves'][-1] = point
                        else:
                            run['curves'].append(point)
                        run.update(training_by_depth=training_by_depth, league=league.state(), training_coverage=coverage)
                        publish(force=done >= total)

                def validate(steps):
                    nonlocal best_score
                    run['phase'] = 'selection_validation'
                    publish()
                    point = {'steps': steps, 'modes': {}, 'scores': {}, 'status': 'completed'}
                    modes = (True, False) if config.compare_modes else (config.deterministic,)
                    for deterministic in modes:
                        mode = 'deterministic' if deterministic else 'stochastic'
                        report = evaluate_depths(agents, (0,), stack_depths=config.stack_depths,
                            hands_per_opponent=config.selection_hands, seed=config.selection_seed,
                            opponents=opponents, frozen_opponents=anchors, stop_event=stop,
                            deterministic=deterministic, record_hands=False)
                        score = selection_score(report, expected)
                        point['modes'][mode] = {'status': report['status'], **(score or {}),
                            'diagnostics': {b['scenario']: b.get('policy_diagnostics') for b in report['agents'][0]['benchmarks']}}
                        write_json(run_path / f'selection_{steps}_{mode}.json', report)
                        if score is not None and not stop.is_set():
                            rank = (score['worst_bb100'], score['mean_bb100'])
                            if best_score is None or rank > best_score:
                                best_score = rank
                                previous_mode = learner.deterministic
                                learner.deterministic = deterministic
                                try:
                                    learner.save_checkpoint(run_path / 'selected.checkpoint')
                                finally:
                                    learner.deterministic = previous_mode
                                run.update(selected_steps=steps, selected_mode=mode, selected_score=score,
                                    selected_sha256=hashlib.sha256((run_path / 'selected.checkpoint').read_bytes()).hexdigest())
                        if report['status'] != 'completed':
                            point['status'] = report['status']
                            if not stop.is_set():
                                error = report['agents'][0].get('error', 'rapport incomplet')
                                raise RuntimeError(f'Comparaison du mode {mode} incomplète : {error}')
                            break
                    valid_modes = [m for m in point['modes'].values() if m.get('scores')]
                    if valid_modes:
                        best = max(valid_modes, key=lambda m: (m['worst_bb100'], m['mean_bb100']))
                        point.update({k: best[k] for k in ('scores', 'worst_bb100', 'mean_bb100')})
                    run['selection'].append(point)
                    run['phase'] = 'training'
                    publish()

                def on_rollout(session, steps):
                    nonlocal last_checkpoint, last_snapshot
                    if stop.is_set() or steps == 0:
                        return
                    if steps - last_snapshot >= config.league_snapshot_interval:
                        league.snapshot(learner, steps)
                        last_snapshot = steps
                    if steps - last_checkpoint >= config.checkpoint_interval:
                        league.snapshot(learner, steps)
                        last_checkpoint = steps
                        validate(steps)

                active_session = [None]
                def attach(session):
                    active_session[0] = session
                    on_session(session)
                session = None
                try:
                    session = train_sb3_session(registry, selections, agents=agents, training_seats=(0,),
                        seed=seed, total_timesteps=config.total_timesteps, stack_depths=config.stack_depths,
                        record=False, stop_event=stop, on_session=attach, on_rollout=on_rollout,
                        on_progress=progress, normalize_rewards=True, perform_validation=False,
                        curriculum_probability=config.curriculum_probability,
                        curriculum_warmup=config.curriculum_warmup, postflop_target=config.postflop_target)
                    run.update(steps=session.sb3_timesteps, training_metrics_collected=learner.training_stats)
                    if not stop.is_set():
                        league.snapshot(learner, session.sb3_timesteps)
                        if not run['selection'] or run['selection'][-1]['steps'] != session.sb3_timesteps:
                            validate(session.sb3_timesteps)
                        if best_score is None:
                            raise RuntimeError('Aucun checkpoint entièrement validé ; test final refusé.')
                        run.update(status='selection_only', phase='selection_complete')
                    else:
                        run.update(status='interrupted', phase='interrupted')
                finally:
                    league.save()
                    run.update(league=league.state(), training_by_depth=training_by_depth, training_metrics=learner.training_stats,
                               training_coverage=getattr(active_session[0], 'training_coverage', {}))
                    if session is not None:
                        session.stop_event = Event()
                        session.close()
                    publish()
            if stop.is_set():
                break

        if not stop.is_set():
            eligible = []
            for trial in trials:
                runs = [r for r in state['runs'] if r['trial_id'] == trial['id'] and r['status'] == 'selection_only']
                if len(runs) == len(config.seeds):
                    eligible.append((min(r['selected_score']['worst_bb100'] for r in runs),
                                     float(np.mean([r['selected_score']['mean_bb100'] for r in runs])), trial['id']))
            if not eligible:
                raise RuntimeError('Aucun essai PPO complet ; test final refusé.')
            state.update(selected_trial=max(eligible)[2], phase='final_test',
                         trial_selection_locked_at=datetime.now(timezone.utc).isoformat())
            publish()
            for run in [r for r in state['runs'] if r['trial_id'] == state['selected_trial']]:
                if stop.is_set():
                    break
                run.update(status='running', phase='final_test')
                publish()
                learner = registry.load('ppo', {'autoload': False})
                checkpoint = Path(run['path']) / 'selected.checkpoint'
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != run['selected_sha256']:
                    raise ValueError('Checkpoint sélectionné modifié avant le test final.')
                learner.load_checkpoint(checkpoint)
                agents = [learner, *[None] * (config.num_players - 1)]
                report = evaluate_depths(agents, (0,), stack_depths=config.stack_depths,
                    hands_per_opponent=config.final_hands, seed=config.final_seed, opponents=opponents,
                    frozen_opponents=anchors, stop_event=stop, deterministic=run['selected_mode'] == 'deterministic')
                write_json(Path(run['path']) / 'final_validation.json', report)
                run['final_validation'] = _lean_report(report, 'final_validation.json')
                if selection_score(report, expected) is None:
                    if stop.is_set():
                        run.update(status='interrupted', phase='interrupted')
                        break
                    raise RuntimeError('Test final incomplet ; agrégation refusée.')
                if champion_ids:
                    run['phase'] = 'champion_test'
                    publish()
                    champion = evaluate_depths(agents, (0,), stack_depths=config.stack_depths,
                        hands_per_opponent=config.champion_hands, seed=champion_seed, opponents=tuple(champion_ids),
                        frozen_opponents=anchors, stop_event=stop, deterministic=run['selected_mode'] == 'deterministic')
                    write_json(Path(run['path']) / 'champion_validation.json', champion)
                    run['champion_validation'] = _lean_report(champion, 'champion_validation.json')
                    run['promotion_review'] = promotion_review(champion, expected_depths=config.stack_depths)
                    if champion['status'] != 'completed':
                        if stop.is_set():
                            run.update(status='interrupted', phase='interrupted')
                            break
                        raise RuntimeError('Confrontation au bot courant incomplète.')
                else:
                    run['promotion_review'] = {'status': 'no_compatible_champion', 'automatic_promotion': False, 'scenarios': []}
                run.update(status='completed', phase='completed', training_metrics=learner.training_stats)
                publish()
        state.update(status='interrupted' if stop.is_set() else 'completed',
                     phase='interrupted' if stop.is_set() else 'completed')
    except Exception as exc:
        state.update(status='failed', phase='failed', error=f'{type(exc).__name__}: {exc}')
        for run in state['runs']:
            if run['status'] == 'running':
                run.update(status='failed', error=state['error'])
    finally:
        state['finished_at'] = datetime.now(timezone.utc).isoformat()
        publish()
    return state
