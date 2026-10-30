"""Reproducible all-agent experiment: python experiments/train_benchmark.py --help."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poker_rl.benchmark import BenchmarkConfig, run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--resume', action='store_true', help='Reprendre les modèles terminés et les tests enregistrés')
    parser.add_argument('--plots-only', action='store_true', help='Régénérer les graphiques depuis les résultats existants')
    for name, field in BenchmarkConfig.__dataclass_fields__.items():
        if name == 'players':
            parser.add_argument('--players', default='2,4', help='Formats séparés : 2,4')
        else:
            parser.add_argument('--' + name.replace('_', '-'), type=type(field.default), default=field.default)
    args = parser.parse_args()
    output = args.output or Path('runs/benchmarks') / datetime.now().strftime('%Y%m%d_%H%M%S_seed42')
    kwargs = vars(args).copy()
    kwargs.pop('output')
    kwargs.pop('resume')
    kwargs.pop('plots_only')
    kwargs['players'] = tuple(map(int, args.players.split(',')))
    if not args.plots_only:
        run_benchmark(BenchmarkConfig(**kwargs), output, resume=args.resume)
    from poker_rl.benchmark_plots import render_results
    render_results(output)
    print(output.resolve(), flush=True)


if __name__ == '__main__':
    main()
