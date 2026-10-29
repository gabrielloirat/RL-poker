"""Run a bounded, fully held-out three-seed integration experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poker_rl.campaigns import CampaignConfig, run_campaign
from poker_rl.plugins import AgentRegistry

previous = None
def update(state):
    global previous
    run = state['runs'][-1] if state['runs'] else {}
    key = (state['status'], run.get('seed'), run.get('phase'), run.get('steps', 0) // 2048)
    if key != previous:
        print(state['id'], *key, flush=True)
        previous = key

result = run_campaign(AgentRegistry('agents'), CampaignConfig(total_timesteps=8192, checkpoint_interval=4096,
    selection_hands=32, final_hands=200, champion_hands=200, learning_rates=(3e-4,)), on_update=update)
print(result['path'], result['status'], result.get('error', ''), flush=True)
if result['status'] != 'completed':
    raise SystemExit(1)
