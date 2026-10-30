"""Independent held-out test processes with bounded PyTorch thread counts."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from poker_rl.benchmark import ALGORITHMS, evaluate_lineup, fixed_seed, load_policy
from poker_rl.records import write_json

_registry = None


def initialize_worker():
    global _registry
    import torch
    from poker_rl.plugins import AgentRegistry
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from poker_rl.benchmark_fast_inference import install_fast_features
    install_fast_features()
    _registry = AgentRegistry()
    from poker_rl.benchmark_fast_inference import install_strength_cache
    install_strength_cache(_registry)


def evaluate_task(task, registry=None):
    import torch
    from poker_rl.plugins import AgentRegistry
    config, root, row, table_players, name, rivals = task
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from poker_rl.benchmark_fast_inference import install_fast_features
    install_fast_features()
    registry = registry if registry is not None else _registry if _registry is not None else AgentRegistry()
    from poker_rl.benchmark_fast_inference import install_strength_cache
    install_strength_cache(registry)
    learner = load_policy(registry, row['checkpoint'], config, config.seed)
    result = evaluate_lineup(registry, learner, rivals, config,
        seed=fixed_seed(config.seed, 3, table_players), hands=config.final_hands,
        output=root / row['id'] / f'test_{table_players}p_{name}.csv')
    result.update(model=row['id'], kind=row['kind'], players=row['players'],
                  table_players=table_players, regime=row['regime'], opponent=name)
    return result


def evaluate_subprocess(task):
    """One isolated process and committed result file per case, without IPC queues."""
    config, root, row, count, name, rivals = task
    row = {key: row[key] for key in ('id', 'kind', 'players', 'regime', 'checkpoint')}
    payload = {'config': asdict(config), 'root': str(root.resolve()), 'model': row,
               'table_players': count, 'opponent': name, 'rivals': rivals}
    semantic = {**payload, 'config': {k: v for k, v in payload['config'].items() if k != 'evaluation_workers'}}
    digest = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode('utf-8')).hexdigest()
    folder = root / row['id'] / 'evaluation_jobs'
    folder.mkdir(parents=True, exist_ok=True)
    prefix = folder / f'{count}p_{name}'
    result_path = prefix.with_suffix('.result.json')
    raw_path = root / row['id'] / f'test_{count}p_{name}.csv'
    if result_path.exists() and raw_path.exists():
        saved = json.loads(result_path.read_text(encoding='utf-8'))
        if saved.get('task_sha256') == digest:
            return saved['result']
    payload.update(task_sha256=digest, result_path=str(result_path.resolve()))
    task_path = prefix.with_suffix('.input.json')
    write_json(task_path, payload)
    environment = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        environment[key] = '1'
    log_path = prefix.with_suffix('.log')
    with log_path.open('wb') as log:
        completed = subprocess.run([sys.executable, '-m', 'poker_rl.benchmark_case', str(task_path.resolve())],
            cwd=Path(__file__).resolve().parents[1], env=environment, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding='utf-8', errors='replace')[-2500:]
        if 'WinError 1114' in tail or 'MemoryError' in tail:
            raise OSError(f'{row["id"]}/{count}p/{name}: mémoire de processus indisponible')
        raise RuntimeError(f'{row["id"]}/{count}p/{name}: {tail}')
    saved = json.loads(result_path.read_text(encoding='utf-8'))
    if saved['task_sha256'] != digest:
        raise ValueError('Le résultat ne correspond pas au cas demandé.')
    return saved['result']


def evaluate_suite(config, root, state, curricula, publish):
    done = {(r['model'], r['table_players'], r['opponent']) for r in state['tests']}
    tasks = []
    for row in state['models']:
        for count in config.players:
            scenarios = {op: [curricula[(count, op)]] * (count - 1) for op in ALGORITHMS}
            scenarios['random'] = ['random'] * (count - 1)
            if count == 4:
                scenarios['table_diverse'] = ['random', *[curricula[(count, op)] for op in ALGORITHMS if op != row['kind']]]
            for name, rivals in scenarios.items():
                if (row['id'], count, name) not in done:
                    tasks.append((config, root, row, count, name, rivals))
    total = len(tasks) + len(done)
    def collect(row):
        state['tests'].append(row)
        state['tests'].sort(key=lambda r: (r['model'], r['table_players'], r['opponent']))
        publish(f"Tests {len(state['tests'])}/{total} : {row['model']} / {row['table_players']}p {row['opponent']} = "
                f"{row['bb100']:.1f} [{row['ci_low']:.1f}, {row['ci_high']:.1f}] bb/100")

    publish(f'Évaluations finales : {len(done)}/{total}, {config.evaluation_workers} processus indépendants')
    state['runtime']['evaluation_backend'] = 'isolated_subprocess_per_case_with_result_files'
    if not tasks:
        return
    if config.evaluation_workers > 1:
        try:
            # Threads only supervise child processes. All inference and RNG
            # state live in separate interpreters; no multiprocessing queues.
            with ThreadPoolExecutor(max_workers=config.evaluation_workers) as pool:
                futures = [pool.submit(evaluate_subprocess, task) for task in tasks]
                for future in as_completed(futures):
                    collect(future.result())
            return
        except OSError as error:
            publish(f'Ressources des processus indisponibles ({error}) ; poursuite séquentielle des tests restants.')
    done = {(r['model'], r['table_players'], r['opponent']) for r in state['tests']}
    for task in tasks:
        if (task[2]['id'], task[3], task[4]) not in done:
            collect(evaluate_subprocess(task))
