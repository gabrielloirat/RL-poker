"""Check persisted experiment evidence without promoting or updating a bot."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from poker_rl.campaigns import campaign_paths, read_campaign, aggregate_seeds
from poker_rl.league import load_frozen
from poker_rl.plugins import AgentRegistry
from poker_rl.records import write_json

torch.set_num_threads(1)
paths = [Path(sys.argv[1])] if len(sys.argv) > 1 else campaign_paths()
path = next(p for p in paths if read_campaign(p)['status'] == 'completed')
state = read_campaign(path)
registry = AgentRegistry('agents')
audit = {'campaign': state['id'], 'runs': [], 'catalog_entries': list(registry.saved_models)}
for run in state['runs']:
    if state.get('version', 1) >= 2 and run['status'] == 'selection_only':
        continue
    report = read_campaign(Path(run['path']) / 'final_validation.json')
    assert report['status'] == run['status'] == 'completed'
    assert report['protocol']['seed'] != state['config']['selection_seed']
    assert len(report['agents'][0]['benchmarks']) == len(state['config']['stack_depths']) * (4 + len(state['anchors']))
    assert all(b['completed'] + b['truncated'] == state['config']['final_hands'] for b in report['agents'][0]['benchmarks'])
    assert all(key.startswith(f"seed_{run['seed']}_") for key in run['league']['history'])
    assert {key: spec['sha256'] for key, spec in run['league']['anchors'].items()} == {
        key: spec['sha256'] for key, spec in state['anchors'].items()}
    history = {**run['league']['history'], **{item['id']: item for item in run['league']['retired']}}
    spec = history[f"seed_{run['seed']}_step_{run['selected_steps']}"]
    frozen = load_frozen(spec)
    selected = registry.load('ppo', {'autoload': False})
    checkpoint = Path(run['path']) / 'selected.checkpoint'
    selected.load_checkpoint(checkpoint)
    weights = selected.model.policy.state_dict()
    assert all(torch.equal(weights[k], tensor) for k, tensor in frozen.model.policy.state_dict().items())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if run.get('selected_sha256'):
        assert digest == run['selected_sha256']
    audit['runs'].append({'seed': run['seed'], 'steps': run['steps'], 'selected_steps': run['selected_steps'],
        'checkpoint_sha256': digest, 'selected_weights_match_frozen_snapshot': True,
        'training_by_depth': run['training_by_depth'],
        'final_hands': sum(b['completed'] for b in report['agents'][0]['benchmarks']),
        'final_truncated': sum(b['truncated'] for b in report['agents'][0]['benchmarks'])})
audited_runs = [r for r in state['runs'] if not state.get('selected_trial') or r.get('trial_id') == state['selected_trial']]
audit['aggregate'] = aggregate_seeds(audited_runs, len(state['config']['seeds']))
assert audit['aggregate']['overall']['n'] == len(state['config']['seeds'])
write_json(path.parent / 'audit.json', audit)
print(json.dumps(audit, indent=2, ensure_ascii=True), flush=True)
