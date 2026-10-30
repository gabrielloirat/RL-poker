from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_rl.interface import load_agent
from poker_rl.plugins import AgentRegistry, DEFAULT_AGENTS_DIR
from poker_rl.session import run_session
from poker_rl.arena import run_arena
from poker_rl.records import SessionRecorder, save_model_version
from poker_rl.evaluation import evaluate_agents, attach_validation, validate_model
from poker_rl.hand_analysis import compact_report
from poker_rl.sb3_training import train_sb3_session


def main(argv=None):
    parser = argparse.ArgumentParser(description="Brancher vos agents au poker.")
    parser.add_argument("mode", choices=("train", "infer", "list", "models", "arena", "save", "metrics", "validate"))
    parser.add_argument("--model", help="Identifiant du bot pour save ou metrics")
    parser.add_argument("--name", help="Nom de la version enregistrée volontairement")
    parser.add_argument("--validation-hands", type=int, default=None,
                        help="Mains indépendantes par adversaire de validation (500 par défaut)")
    parser.add_argument("--records-dir", type=Path, help="Dossier des archives et modèles sauvegardés")
    parser.add_argument("--record", action="store_true", help="Enregistrer aussi les sessions d'inférence")
    parser.add_argument("--no-record", action="store_true", help="Désactiver l'enregistrement en entraînement")
    parser.add_argument("--matches", type=int, default=100, help="Nombre de parties complètes en arène")
    parser.add_argument("--max-hands-per-match", type=int, default=1000)
    parser.add_argument("--config", type=Path, help="Configuration JSON des agents et de l'environnement")
    parser.add_argument("--agents-dir", type=Path, help="Dossier des plugins (agents/ par défaut)")
    parser.add_argument("--agent0", help="Agent du siège 0 : nom du plugin ou module_python:ClasseOuFactory")
    parser.add_argument("--agent1", help="Agent du siège 1 : nom du plugin ou module_python:ClasseOuFactory")
    parser.add_argument("--train-seats", type=int, choices=(0, 1), nargs="+",
                        help="Sièges recevant les transitions (train uniquement)")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None, help="Décisions d'entraînement PPO natif SB3")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true", help="Afficher les deux mains pour le débogage")
    parser.add_argument("--delay", type=float, default=0.25, help="Pause entre actions affichées, en secondes")
    args = parser.parse_args(argv)
    if args.mode in ("infer", "arena") and args.train_seats is not None:
        parser.error("--train-seats s'utilise uniquement en mode train.")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    agents_dir = args.agents_dir
    if agents_dir is None and "agents_dir" in config:
        agents_dir = Path(config["agents_dir"])
        if args.config and not agents_dir.is_absolute():
            agents_dir = args.config.resolve().parent / agents_dir
    if agents_dir is None:
        agents_dir = DEFAULT_AGENTS_DIR
    records_dir = args.records_dir or config.get("records_dir")
    if records_dir is not None:
        records_dir = Path(records_dir)
        if args.records_dir is None and args.config and not records_dir.is_absolute():
            records_dir = args.config.resolve().parent / records_dir
    validation_hands = args.validation_hands if args.validation_hands is not None else config.get("validation_hands", 500)
    if type(validation_hands) is not int or validation_hands < 1:
        parser.error("--validation-hands doit être un entier positif.")
    if args.mode in ("list", "models", "save", "metrics", "validate"):
        try:
            registry = AgentRegistry(agents_dir, records_dir=records_dir)
        except ValueError as exc:
            parser.error(str(exc))
        if args.mode in ("save", "metrics", "validate"):
            if not args.model or not registry.resolve_model(args.model):
                parser.error("Choisissez un --model parmi les identifiants de la commande models.")
            if args.mode == "save":
                model_id = save_model_version(registry.records_dir, args.model, label=args.name)
                print(model_id)
                return model_id
            report = (validate_model(registry, args.model, hands_per_opponent=validation_hands)
                      if args.mode == "validate" else registry.resolve_model(args.model).get("validation"))
            print(json.dumps(report or {"status": "unavailable", "message": "Aucune validation pour ce checkpoint."}, indent=2))
            return report
        print(f"Dossier d'agents : {registry.directory}")
        for name, plugin in registry.agents.items():
            if args.mode == "models" and name not in registry.saved_models:
                continue
            print(f"  {name} : {plugin.path}")
            if name in registry.saved_models:
                model = registry.saved_models[name]
                print(f"    {registry.label(name)} | {model['created_at']} | {model['num_players']} joueurs | "
                      f"adversaires : {', '.join(p['label'] for p in model['opponents'] if p['seat'] != model['seat'])}")
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
    if args.mode in ("infer", "arena"):
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
        registry = AgentRegistry(agents_dir, records_dir=records_dir)
        agents = [] if args.mode == "arena" else [registry.load(spec["class"], spec.get("kwargs")) if ":" not in spec["class"]
                  else load_agent(spec["class"], spec.get("kwargs")) for spec in specs]
    except (ValueError, TypeError, ImportError, AttributeError) as exc:
        parser.error(str(exc))
    count = args.episodes if args.episodes is not None else config.get("episodes", 100)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    selections = [(spec["class"], spec.get("kwargs", {})) for spec in specs]
    if args.mode == "train" and any(getattr(agents[seat], "native_training", False) for seat in training_seats):
        timesteps = args.timesteps if args.timesteps is not None else config.get("total_timesteps", args.episodes or 10000)
        session = train_sb3_session(registry, selections, training_seats=training_seats,
            total_timesteps=timesteps, seed=seed, game_kwargs=config.get("environment", {}),
            max_hands_per_match=args.max_hands_per_match, record=not args.no_record, agents=agents,
            validation_hands=validation_hands,
            on_validation_progress=lambda seat, ref, done, total: print(
                f"Validation J{seat + 1} / {ref} : {done}/{total} mains", flush=True),
            on_progress=lambda done, total, stats: print(f"PPO SB3 : {done}/{total} décisions · {stats.hands} mains", flush=True)
                if done % 1024 == 0 or done >= total else None)
        try:
            report = {"hands": session.stats.hands, "completed": session.stats.completed,
                "truncated": session.stats.truncated, "illegal_actions": 0,
                "timesteps": session.sb3_timesteps, "mean_net_bb": session.stats.mean_bb,
                "training": [getattr(agent, "training_stats", {}) for agent in session.agents],
                "validation": compact_report(session.validation_report,
                    hands_file="validation.json" if session.last_archive else None)}
            if session.last_archive:
                report["archive"] = str(session.last_archive)
            print(json.dumps(report, indent=2))
            return report
        finally:
            session.close()
    if args.mode == "arena":
        class ArenaRegistry:
            def __init__(self):
                self.records_dir = registry.records_dir
                self.saved_models = registry.saved_models
            def load(self, name, kwargs):
                return load_agent(name, kwargs) if ":" in name else registry.load(name, kwargs)
            def describe(self, name, kwargs):
                return registry.describe(name, kwargs)
        report = run_arena(ArenaRegistry(), selections, matches=args.matches, seed=seed,
                           records_dir=records_dir, game_kwargs=config.get("environment", {}),
                           max_hands_per_match=args.max_hands_per_match,
                           on_progress=lambda stats: print(f"Arène : {stats.matches}/{args.matches} parties", flush=True))
        print(json.dumps(report, indent=2))
        return report
    print(f"Mode {args.mode} | agents : {specs[0]['class']} / {specs[1]['class']} | "
          f"sièges entraînés : {list(training_seats)}", flush=True)

    def progress(result):
        if (result.episode + 1) % 100 == 0 or result.episode + 1 == count:
            print(f"Main {result.episode + 1}/{count}", flush=True)

    recorder = None
    if not args.no_record and (args.mode == "train" or args.record):
        recorder = SessionRecorder(selections, kind=args.mode, training_seats=training_seats,
                                    seed=seed, game_kwargs=config.get("environment", {}),
                                    root=registry.records_dir, registry=registry)
    results = run_session(agents, episodes=count, training_seats=training_seats,
                          seed=seed, game_kwargs=config.get("environment", {}),
                          render=args.render, delay=args.delay, on_episode_end=progress, recorder=recorder)
    completed = [result for result in results if result.terminated]
    report = {
        "hands": len(results), "completed": len(completed),
        "truncated": sum(result.truncated for result in results),
        "illegal_actions": sum(result.illegal_actions for result in results),
        "mean_net_bb": [sum(r.rewards_bb[seat] for r in completed) / len(completed)
                        if completed else None for seat in (0, 1)],
    }
    if recorder:
        report["archive"] = str(recorder.path)
    if args.mode == "train":
        report["validation"] = evaluate_agents(agents, training_seats,
            game_kwargs=config.get("environment", {}), hands_per_opponent=validation_hands)
        attach_validation(recorder, report["validation"])
        report["validation"] = compact_report(report["validation"], hands_file="validation.json" if recorder else None)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSession interrompue.")
