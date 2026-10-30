"""Descriptive diagnosis from complete held-out hands; never pools training rewards."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poker_rl.campaigns import campaign_paths, read_campaign
from poker_rl.strategy_metrics import analyze_strategy
from poker_rl.plugins import AgentRegistry
from poker_rl.records import write_json


def summarize(hands):
    complete = [h for h in hands if h['terminated']]
    style = analyze_strategy(complete)
    pre, first, post, all_actions = Counter(), Counter(), Counter(), Counter()
    preflop_decisions = postflop_hands = all_in_hands = open_shove = 0
    holdings = defaultdict(Counter)
    unique_deals = set()
    for hand in complete:
        unique_deals.add(hand['seed'])
        actions = [a for a in hand['actions'] if a['seat'] == hand['bot_seat']]
        pre_actions = [a for a in actions if a['stage'] == 'PREFLOP']
        post_actions = [a for a in actions if a['stage'] != 'PREFLOP']
        pre.update(a['action'] for a in pre_actions)
        post.update(a['action'] for a in post_actions)
        all_actions.update(a['action'] for a in actions)
        preflop_decisions += bool(pre_actions)
        postflop_hands += bool(post_actions)
        all_in_hands += any(a['action'] == 'ALL_IN' for a in pre_actions)
        if pre_actions:
            first[pre_actions[0]['action']] += 1
            holdings[hand['starting_hand']][pre_actions[0]['action']] += 1
            open_shove += pre_actions[0]['action'] == 'ALL_IN' and pre_actions[0]['preflop_raises'] == 0
    return {'hands': len(complete), 'unique_deal_seeds': len(unique_deals),
        'bb100': sum(h['net_bb'] for h in complete) * 100 / len(complete) if complete else None,
        'style': style['overall'], 'positions': style['positions'],
        'preflop_actions': dict(pre), 'first_preflop_actions': dict(first), 'postflop_actions': dict(post),
        'all_actions': dict(all_actions), 'hands_with_preflop_decision': preflop_decisions,
        'hands_with_postflop_decision': postflop_hands, 'hands_with_preflop_all_in': all_in_hands,
        'open_shove_hands': open_shove, 'first_actions_by_holding': dict(holdings)}


def main():
    path = next(p for p in campaign_paths() if read_campaign(p)['status'] == 'completed')
    campaign = read_campaign(path)
    result = {'campaign': campaign['id'], 'executed_at': datetime.now(timezone.utc).isoformat(),
        'protocol': campaign['config'], 'anchors': {k: v['label'] for k, v in campaign['anchors'].items()},
        'runs': [], 'comparability': {'cards': 'Common deal schedule across seeds, references and depths.',
            'current_bot': 'Different test seed and 500 rather than 200 hands per scenario; no paired comparison.'}}
    for run in campaign['runs']:
        if campaign.get('selected_trial') and run.get('trial_id') != campaign['selected_trial']:
            continue
        full = read_campaign(Path(run['path']) / 'final_validation.json')
        all_hands = []
        benchmarks = []
        for benchmark in full['agents'][0]['benchmarks']:
            hands = benchmark['hand_analysis']['hands']
            net = sum(h['net_bb'] for h in hands if h['terminated'])
            assert abs(net * 100 / benchmark['completed'] - benchmark['bb_per_100']) < 1e-8
            assert len({h['hand'] for h in hands}) == len(hands)
            assert len(hands) == campaign['config']['final_hands']
            all_hands.extend(hands)
            benchmarks.append({'scenario': benchmark['scenario'], 'opponent': benchmark['opponent'],
                'depth': benchmark['depth_bb'], 'bb100': benchmark['bb_per_100'],
                'ci95_bb100': [x * 100 for x in benchmark['ci95_bb_per_hand']],
                **summarize(hands)})
        summary = summarize(all_hands)
        summary.update(seed=run['seed'], selected_steps=run['selected_steps'],
            mean_scenarios=statistics.mean(b['bb100'] for b in benchmarks),
            worst_scenario=min(benchmarks, key=lambda b: b['bb100'])['scenario'],
            worst_bb100=min(b['bb100'] for b in benchmarks),
            benchmarks=benchmarks,
            by_depth={str(depth): summarize([h for h in all_hands if h['depth_bb'] == depth])
                      for depth in campaign['config']['stack_depths']},
            selection=[{'steps': p['steps'], 'mean': p.get('mean_bb100'), 'worst': p.get('worst_bb100')}
                       for p in run['selection']])
        exposure = run['league']['exposure']
        groups = defaultdict(int)
        for row in exposure:
            group = ('heuristic' if row['opponent'] in ('random', 'check_call', 'tight', 'aggressive') else
                     'anchor' if row['opponent'] in campaign['anchors'] else 'own_snapshot')
            groups[group] += row['hands']
        summary['league_training_exposure'] = dict(groups)
        result['runs'].append(summary)
    registry = AgentRegistry('agents')
    current_id = next(key for key, model in registry.saved_models.items() if key.startswith('current_') and model['num_players'] == 2)
    model = registry.resolve_model(current_id)
    validation = model['validation']
    filename = validation['benchmarks'][0]['hand_analysis']['hands_file']
    current = read_campaign(model['run_path'] / filename)
    current_hands = [h for b in current['agents'][0]['benchmarks'] for h in b['hand_analysis']['hands']]
    result['current'] = {**summarize(current_hands), 'model_id': current_id,
        'training_steps': model['training_metrics']['timesteps'], 'protocol': current['protocol'],
        'benchmarks': [{'scenario': b['scenario'], **summarize(b['hand_analysis']['hands'])}
                       for b in current['agents'][0]['benchmarks']]}
    result['seed_distribution'] = {'mean': statistics.mean(r['mean_scenarios'] for r in result['runs']),
        'median': statistics.median(r['mean_scenarios'] for r in result['runs']),
        'std': statistics.stdev(r['mean_scenarios'] for r in result['runs'])}
    out = Path('runs/analysis') / campaign['id']
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'diagnosis.json', result)
    for run in result['runs']:
        print(json.dumps({k: run[k] for k in ('seed', 'mean_scenarios', 'worst_scenario', 'worst_bb100',
            'style', 'first_preflop_actions', 'postflop_actions', 'hands_with_preflop_all_in',
            'hands_with_postflop_decision', 'league_training_exposure', 'selection')}, ensure_ascii=True), flush=True)
        print('BENCHMARKS', run['seed'], [(b['scenario'], round(b['bb100'], 2)) for b in run['benchmarks']], flush=True)
        current_anchor = next(key for key, label in result['anchors'].items() if 'courant' in label)
        print('HEAD_TO_HEAD', run['seed'], [(b['depth'], b['bb100'], b['ci95_bb100']) for b in run['benchmarks'] if b['opponent'] == current_anchor], flush=True)
        chosen = next(b for b in run['benchmarks'] if b['scenario'] == '50bb:random')
        print('HOLDINGS_50_RANDOM', run['seed'], chosen['first_actions_by_holding'], flush=True)
        print('POSITIONS_50', run['seed'], run['by_depth']['50']['positions'], flush=True)
    print('CURRENT', json.dumps({k: result['current'][k] for k in ('style', 'first_preflop_actions', 'hands_with_postflop_decision')}, ensure_ascii=True), flush=True)
    print('OUTPUT', str((out / 'diagnosis.json').resolve()), flush=True)


if __name__ == '__main__':
    main()
