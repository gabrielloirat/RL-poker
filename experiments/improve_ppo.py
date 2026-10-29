"""Train a candidate outside the library; score it on separate selection seeds."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from agents.ppo import PPOAgent
from poker_rl.evaluation import evaluate_agents
from poker_rl.plugins import AgentRegistry
from poker_rl.sb3_training import train_sb3_session
from poker_rl.records import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-hands', type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    registry = AgentRegistry()
    selections = [('ppo', {'autoload':False, 'seed':args.seed,
        'n_steps':1024, 'batch_size':128, 'n_epochs':5}), ('mixed', {'seed':args.seed + 1})]
    agents = [registry.load(name, kwargs) for name,kwargs in selections]
    start = time.monotonic()
    saved = set()
    def progress(done, total, stats):
        milestone = done // 25000
        if milestone and milestone not in saved:
            saved.add(milestone)
            agents[0].save_checkpoint(args.output / f'step_{done}.checkpoint')
            print(f'Train {done}/{total} decisions, {stats.hands} hands, {time.monotonic()-start:.1f}s',flush=True)
    session = train_sb3_session(registry, selections, training_seats=(0,), total_timesteps=args.timesteps,
        seed=args.seed, record=False, agents=agents, validation_hands=32, on_progress=progress)
    try:
        agents[0].save_checkpoint(args.output / 'final.checkpoint')
        write_json(args.output / 'training.json', {'selections':selections, 'seed':args.seed,
            'timesteps':session.sb3_timesteps, 'stats':session.stats.to_dict(),
            'training_metrics':agents[0].training_stats, 'elapsed_s':time.monotonic()-start})
        reports=[]
        for checkpoint in sorted(args.output.glob('*.checkpoint')):
            agent=PPOAgent(checkpoint=checkpoint, autoload=False)
            for deterministic in (False, True):
                report=evaluate_agents([agent,None], (0,), seed=2_000_000_019,
                    hands_per_opponent=args.selection_hands, deterministic=deterministic, record_hands=False)
                if report['status'] != 'completed':
                    raise RuntimeError(report)
                reports.append({'checkpoint':str(checkpoint), 'deterministic':deterministic, 'report':report})
                print('Select',checkpoint.name,'greedy' if deterministic else 'sampled',
                    [(b['opponent'],round(b['bb_per_100'],2)) for b in report['agents'][0]['benchmarks']],flush=True)
                write_json(args.output / 'selection.json', reports)
    finally:
        session.close()


if __name__ == '__main__':
    main()
