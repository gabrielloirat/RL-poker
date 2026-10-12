from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_rl.interface import load_agent
from poker_rl.plugins import AgentRegistry, DEFAULT_AGENTS_DIR
from poker_rl.session import run_session


def main(argv=None):
    parser = argparse.ArgumentParser(description="Brancher vos agents au poker.")
    parser.add_argument("mode", choices=("train", "infer", "list"))
    parser.add_argument("--config", type=Path, help="Configuration JSON des agents et de l'environnement")
    parser.add_argument("--agents-dir", type=Path, help="Dossier des plugins (agents/ par défaut)")
    parser.add_argument("--agent0", help="Agent du siège 0 : nom du plugin ou module_python:ClasseOuFactory")
    parser.add_argument("--agent1", help="Agent du siège 1 : nom du plugin ou module_python:ClasseOuFactory")
    parser.add_argument("--train-seats", type=int, choices=(0, 1), nargs="+",
                        help="Sièges recevant les transitions (train uniquement)")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true", help="Afficher les deux mains pour le débogage")
    parser.add_argument("--delay", type=float, default=0.25, help="Pause entre actions affichées, en secondes")
    args = parser.parse_args(argv)
    if args.mode == "infer" and args.train_seats is not None:
        parser.error("--train-seats s'utilise uniquement en mode train.")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    agents_dir = args.agents_dir
    if agents_dir is None and "agents_dir" in config:
        agents_dir = Path(config["agents_dir"])
        if args.config and not agents_dir.is_absolute():
            agents_dir = args.config.resolve().parent / agents_dir
    if agents_dir is None:
        agents_dir = DEFAULT_AGENTS_DIR
    if args.mode == "list":
        try:
            registry = AgentRegistry(agents_dir)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Dossier d'agents : {registry.directory}")
        for name, plugin in registry.agents.items():
            print(f"  {name} : {plugin.path}")
        for name, error in registry.errors.items():
            print(f"  {name} [indisponible] : {error}")
        if not registry.agents:
            print("Aucun agent disponible. Déposez vos fichiers Python ou dossiers dans ce dossier.")
        return registry
    specs = config.get("agents", [{}, {}])
    if not isinstance(specs, list) or len(specs) != 2:
        parser.error("La configuration doit définir exactement deux agents.")
    specs = [dict(item) for item in specs]
    for seat, override in enumerate((args.agent0, args.agent1)):
        if override:
            specs[seat] = {"class": override}
        elif "name" in specs[seat]:
            specs[seat]["class"] = specs[seat]["name"]
        if not specs[seat].get("class"):
            parser.error(f"Agent manquant au siège {seat} : utilisez --agent{seat} ou --config.")
    if args.mode == "infer":
        training_seats = ()
    elif args.train_seats is not None:
        training_seats = args.train_seats
    elif any("train" in spec for spec in specs):
        training_seats = [seat for seat, spec in enumerate(specs) if spec.get("train", False)]
    else:
        training_seats = (0,)
    if args.mode == "train" and not training_seats:
        parser.error("Sélectionnez au moins un siège à entraîner.")
    try:
        registry = AgentRegistry(agents_dir) if any(":" not in spec["class"] for spec in specs) else None
        agents = [registry.load(spec["class"], spec.get("kwargs")) if ":" not in spec["class"]
                  else load_agent(spec["class"], spec.get("kwargs")) for spec in specs]
    except (ValueError, TypeError, ImportError, AttributeError) as exc:
        parser.error(str(exc))
    count = args.episodes if args.episodes is not None else config.get("episodes", 100)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    print(f"Mode {args.mode} | agents : {specs[0]['class']} / {specs[1]['class']} | "
          f"sièges entraînés : {list(training_seats)}", flush=True)

    def progress(result):
        if (result.episode + 1) % 100 == 0 or result.episode + 1 == count:
            print(f"Main {result.episode + 1}/{count}", flush=True)

    results = run_session(agents, episodes=count, training_seats=training_seats,
                          seed=seed, game_kwargs=config.get("environment", {}),
                          render=args.render, delay=args.delay, on_episode_end=progress)
    completed = [result for result in results if result.terminated]
    report = {
        "hands": len(results), "completed": len(completed),
        "truncated": sum(result.truncated for result in results),
        "illegal_actions": sum(result.illegal_actions for result in results),
        "mean_net_bb": [sum(r.rewards_bb[seat] for r in completed) / len(completed)
                        if completed else None for seat in (0, 1)],
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSession interrompue.")
