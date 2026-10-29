"""Compute the depth/style suite for the current bot without updating its weights."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poker_rl.evaluation import validate_model
from poker_rl.plugins import AgentRegistry
import torch

torch.set_num_threads(1)

registry = AgentRegistry('agents')
model_id = next(key for key, value in registry.saved_models.items()
                if key.startswith('current_') and value['num_players'] == 2 and value['name'] == 'ppo')
print('Validating', model_id, flush=True)
report = validate_model(registry, model_id, hands_per_opponent=500,
    on_progress=lambda seat, opponent, done, total: print(opponent, done, total, flush=True) if done == total else None)
print(report['status'], [(b['scenario'], round(b['bb_per_100'], 2)) for b in report['benchmarks']], flush=True)
