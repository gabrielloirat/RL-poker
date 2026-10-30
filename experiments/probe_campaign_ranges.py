"""Controlled first-decision policy probes, not additional validation hands."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from poker_rl.campaigns import campaign_paths, read_campaign
from poker_rl.game import PokerGame, CARD_ORDER, ACTION_NAMES
from poker_rl.plugins import AgentRegistry
from poker_rl.records import write_json
from poker_rl.strategy_metrics import starting_hand

torch.set_num_threads(1)
campaign_path = next(p for p in campaign_paths() if read_campaign(p)['status'] == 'completed')
campaign = read_campaign(campaign_path)
registry = AgentRegistry('agents')
models = [(str(run['seed']), Path(run['path']) / 'selected.checkpoint') for run in campaign['runs']
          if not campaign.get('selected_trial') or run.get('trial_id') == campaign['selected_trial']]
current_id = next(key for key, value in registry.saved_models.items() if key.startswith('current_') and value['num_players'] == 2)
models.append(('current', registry.resolve_model(current_id)['path']))
result = {'purpose': 'Controlled first-decision inference only, not validation games or reward estimates.',
          'holdings_per_context': 1326, 'contexts': ['BTN/SB unopened', 'BB facing 3 BB'], 'models': {}}
for label, checkpoint in models:
    agent = registry.load('ppo', {'autoload': False, 'deterministic': True})
    agent.load_checkpoint(checkpoint)
    rows = []
    for depth in (20, 50, 100):
        for context in result['contexts']:
            game = PokerGame(initial_stacks=depth * 2)
            game.reset(seed=8765, options={'dealer_seat': 0})
            if context.startswith('BB'):
                game.step(3)
            seat = game.current_seat
            obs = game.observe(seat)
            mask = game.action_masks(seat)
            counts, by_holding = Counter(), defaultdict(Counter)
            for a, b in combinations(range(52), 2):
                probe = obs.copy()
                probe[:52] = 0
                probe[a] = probe[b] = 1
                agent.on_episode_start(seed=1, training=False)
                action = agent.act(probe, mask, training=False)
                counts[ACTION_NAMES[action]] += 1
                by_holding[starting_hand([CARD_ORDER[a], CARD_ORDER[b]])][ACTION_NAMES[action]] += 1
            rows.append({'depth': depth, 'context': context, 'seat': seat, 'actions': dict(counts),
                         'by_holding': dict(by_holding)})
            print(label, depth, context, dict(counts), flush=True)
            game.close()
    result['models'][label] = rows
out = Path('runs/analysis') / campaign['id'] / 'range-probe.json'
write_json(out, result)
print(out, flush=True)
