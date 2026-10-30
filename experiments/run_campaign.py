"""Run the same multi-seed protocol as the native interface."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poker_rl.campaigns import CampaignConfig, run_campaign
from poker_rl.plugins import AgentRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='101,202,303')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--interval', type=int, default=50000)
    parser.add_argument('--selection-hands', type=int, default=500)
    parser.add_argument('--final-hands', type=int, default=1000)
    parser.add_argument('--champion-hands', type=int, default=5000)
    parser.add_argument('--learning-rates', default='0.0003,0.0001')
    parser.add_argument('--league-interval', type=int, default=5000)
    parser.add_argument('--no-curriculum', action='store_true')
    parser.add_argument('--no-candidate-202', action='store_true')
    parser.add_argument('--selection-seed', type=int)
    parser.add_argument('--final-seed', type=int)
    parser.add_argument('--champion-seed', type=int)
    parser.add_argument('--depths', default='20,50,100')
    parser.add_argument('--players', type=int, default=2)
    args = parser.parse_args()
    config = CampaignConfig(seeds=tuple(map(int, args.seeds.split(','))), total_timesteps=args.steps,
        checkpoint_interval=args.interval, selection_hands=args.selection_hands, final_hands=args.final_hands,
        stack_depths=tuple(map(int, args.depths.split(','))), num_players=args.players,
        learning_rates=tuple(map(float, args.learning_rates.split(','))), champion_hands=args.champion_hands,
        league_snapshot_interval=args.league_interval, curriculum_probability=0. if args.no_curriculum else .2,
        include_seed202=not args.no_candidate_202, selection_seed=args.selection_seed, final_seed=args.final_seed,
        champion_seed=args.champion_seed)
    previous = None
    def progress(state):
        nonlocal previous
        run = state['runs'][-1] if state['runs'] else {}
        active = next((r for r in state['runs'] if r.get('status') == 'running'), run)
        key = (state['status'], active.get('trial_id'), active.get('seed'), active.get('phase'), (active.get('steps', 0) // 4096))
        if key != previous:
            print(state['id'], *key, flush=True)
            previous = key
    report = run_campaign(AgentRegistry('agents'), config, on_update=progress)
    print(report['path'], report['status'], report.get('error', ''), flush=True)
    if report['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
