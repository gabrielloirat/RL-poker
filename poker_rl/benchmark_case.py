"""Execute one immutable benchmark case; called by benchmark_workers only."""
import json
from pathlib import Path
import sys

from poker_rl.benchmark import BenchmarkConfig
from poker_rl.benchmark_workers import evaluate_task
from poker_rl.records import write_json


def main():
    payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    config = BenchmarkConfig(**payload['config'])
    task = (config, Path(payload['root']), payload['model'], payload['table_players'],
            payload['opponent'], payload['rivals'])
    result = evaluate_task(task)
    write_json(payload['result_path'], {'task_sha256': payload['task_sha256'], 'result': result})
    print(f"{result['model']} / {result['table_players']}p {result['opponent']}: {result['bb100']:.1f}", flush=True)


if __name__ == '__main__':
    main()
