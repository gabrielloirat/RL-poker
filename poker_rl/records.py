"""Local session archives: configuration, hand histories and agent checkpoints."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from uuid import uuid4

from gym_env.enums import Action


DEFAULT_RECORDS_DIR = Path(__file__).resolve().parents[1] / "runs" / "sessions"


def json_value(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Valeur non sérialisable : {type(value).__name__}")


def replace_file(source, target):
    # Windows/OneDrive and readers can briefly hold an archive file open.
    for attempt in range(7):
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.025 * 2 ** attempt)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value), encoding="utf-8")
    replace_file(temporary, path)


def snapshot(game):
    table = game.table
    return {"stage": table.stage.name, "dealer": table.dealer_pos,
            "current_seat": game.current_seat, "board": list(table.table_cards),
            "hole_cards": [list(player.cards) for player in table.players],
            "stacks": [float(player.stack) for player in table.players],
            "street_bets": list(table.player_pots),
            "contributions": list(table.player_max_win),
            "pot": 0 if game.terminated else float(table.community_pot + table.current_round_pot),
            "legal_actions": [int(i) for i, legal in enumerate(game.action_masks()) if legal],
            "terminated": game.terminated, "truncated": game.truncated}


class SessionRecorder:
    def __init__(self, selections, *, kind, training_seats=(), seed=42, game_kwargs=None,
                 continuous=False, max_hands_per_match=1000, root=DEFAULT_RECORDS_DIR, registry=None):
        now = datetime.now(timezone.utc)
        self.id = now.strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
        self.path = Path(root).resolve() / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        self.history = (self.path / "history.jsonl").open("w", encoding="utf-8")
        self.closed = False
        self.hand = self.match = 0
        self.action_counts = [0] * len(selections)
        agents = []
        for seat, (name, kwargs) in enumerate(selections):
            spec = registry.describe(name, kwargs) if registry and hasattr(registry, "describe") else {"name": name, "kwargs": kwargs}
            agents.append({"seat": seat, "label": name, **spec,
                           "training": seat in training_seats, "checkpoint_status": "pending"})
        self.manifest = {"version": 1, "id": self.id, "kind": kind, "status": "running",
                         "created_at": now.isoformat(), "python_version": platform.python_version(),
                         "seed": seed, "environment": dict(game_kwargs or {}),
                         "num_players": len(selections), "continuous": continuous,
                         "max_hands_per_match": max_hands_per_match, "agents": agents,
                         "history_file": "history.jsonl"}
        self.flush()

    def flush(self):
        write_json(self.path / "manifest.json", self.manifest)
        if not self.closed:
            self.history.flush()

    def event(self, event, **data):
        if not self.closed:
            self.history.write(json.dumps({"event": event, "hand": self.hand, "match": self.match,
                                           **data}, ensure_ascii=False, default=json_value) + "\n")

    def hand_start(self, game, *, hand, match, seed, hand_in_match=1):
        self.hand, self.match = hand, match
        if "observation_fields" not in self.manifest:
            table = game.table
            self.manifest.update(observation_fields=game.observation_fields,
                observation_version=game.observation_version,
                environment={"num_players": game.num_players, "initial_stacks": table.initial_stacks,
                    "small_blind": table.small_blind, "big_blind": table.big_blind,
                    "max_actions": game.max_actions,
                    "max_raises_per_player_round": table.max_raises_per_player_round})
            self.flush()
        self.event("hand_start", seed=seed, hand_in_match=hand_in_match,
                   observation_version=game.observation_version, state=snapshot(game))

    def action(self, before, game, action, *, illegal=False, curriculum_prefix=False):
        seat = before["current_seat"]
        self.action_counts[seat] += 1
        self.event("action", seat=seat, action=int(action), action_name=Action(int(action)).name,
                   illegal=illegal, before=before, after=snapshot(game),
                   training_origin='curriculum_prefix' if curriculum_prefix else 'natural')

    def hand_end(self, game, *, match_finished=False, match_truncated=False, match_winner=None):
        self.event("hand_end", rewards_bb=game.rewards(),
                   winners=list(game.table.winner_indices) if game.terminated else [],
                   state=snapshot(game), match_finished=match_finished,
                   match_truncated=match_truncated, match_winner=match_winner)
        self.history.flush()

    def save_models(self, agents):
        models = self.path / "models"
        models.mkdir(exist_ok=True)
        for seat, agent in enumerate(agents):
            entry = self.manifest["agents"][seat]
            metrics = getattr(agent, "training_stats", None)
            if isinstance(metrics, dict):
                entry["training_metrics"] = metrics
            save = getattr(agent, "save_checkpoint", None)
            if not callable(save) or not callable(getattr(agent, "load_checkpoint", None)):
                entry["checkpoint_status"] = "unsupported"
                continue
            try:
                target = models / f"seat_{seat}.checkpoint"
                temporary = target.with_suffix(".tmp")
                save(temporary)
                if not temporary.is_file():
                    raise ValueError("save_checkpoint(path) doit créer le fichier demandé.")
                replace_file(temporary, target)
                entry.update(checkpoint_status="saved", checkpoint=str(target.relative_to(self.path)),
                             checkpoint_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                             model_id=f"saved_{self.id.rsplit('_', 1)[-1]}_j{seat + 1}")
                entry.pop("checkpoint_error", None)
            except Exception as exc:
                entry.update(checkpoint_status="error", checkpoint_error=f"{type(exc).__name__}: {exc}")
        self.flush()
        return self.path

    def finish(self, stats, *, agents=None, status="completed", error=None):
        if self.closed:
            return self.path
        if agents is not None:
            self.save_models(agents)
        self.manifest.update(status=status, finished_at=datetime.now(timezone.utc).isoformat(),
                             stats=stats, action_counts=self.action_counts)
        if error:
            self.manifest["error"] = str(error)
        self.event("session_end", status=status, stats=stats)
        self.flush()
        self.history.close()
        self.closed = True
        return self.path


def saved_models(root=DEFAULT_RECORDS_DIR):
    """Ignore incomplete archives; surface corrupt archives without breaking discovery."""
    found = {}
    for path in sorted(Path(root).glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                continue
            for entry in manifest.get("agents", []):
                if not isinstance(entry, dict):
                    continue
                if entry.get("checkpoint_status") != "saved":
                    continue
                if (not isinstance(entry.get("name"), str) or not isinstance(entry.get("kwargs"), dict)
                        or not isinstance(entry.get("model_id"), str)
                        or not isinstance(entry.get("checkpoint_sha256"), str)):
                    continue
                checkpoint = (path.parent / entry["checkpoint"]).resolve()
                if not checkpoint.is_relative_to(path.parent.resolve()) or not checkpoint.is_file():
                    continue
                found[entry["model_id"]] = {**entry, "path": checkpoint,
                                           "run_path": path.parent, "run_id": manifest["id"],
                                           "num_players": manifest["num_players"],
                                           "created_at": manifest["created_at"],
                                           "stats": manifest.get("stats", {}),
                                           "kind": manifest.get("kind"),
                                           "status": manifest.get("status"),
                                           "finished_at": manifest.get("finished_at", manifest["created_at"]),
                                           "validation": entry.get("validation"),
                                           "opponents": manifest["agents"]}
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return found


def model_catalog(root=DEFAULT_RECORDS_DIR):
    """One current bot per plugin/table size; extra versions require an explicit save.

    Historical checkpoints remain addressable by their immutable archive IDs.
    Reading the catalog also migrates the presentation of legacy archives without
    modifying or deleting any of them.
    """
    archives = saved_models(root)
    current = {}
    for model in sorted(archives.values(), key=lambda m: (m["finished_at"], m["run_id"], m["seat"])):
        if model["kind"] != "train" or model["status"] != "completed" or not model.get("training"):
            continue
        identity = hashlib.sha256(model["name"].encode()).hexdigest()[:12]
        key = f"current_{identity}_{model['num_players']}p"
        current[key] = {**model, "archive_model_id": model["model_id"], "model_id": key,
                        "catalog_kind": "current"}
    versions_path = Path(root) / "versions.json"
    if versions_path.exists():
        versions = json.loads(versions_path.read_text(encoding="utf-8"))
        for entry in versions:
            model = archives.get(entry["archive_model_id"])
            if model:
                current[entry["model_id"]] = {**model, **entry, "catalog_kind": "version"}
    return current


def save_model_version(root, model_id, *, label=None):
    """Pin an immutable checkpoint; later training only replaces the current bot."""
    model = model_catalog(root).get(model_id) or saved_models(root).get(model_id)
    if model is None:
        raise ValueError("Modèle introuvable dans la bibliothèque.")
    path = Path(root) / "versions.json"
    versions = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    entry = {"model_id": "version_" + uuid4().hex[:12],
             "archive_model_id": model.get("archive_model_id", model["model_id"]),
             "version_label": label or f"Version du {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC"}
    versions.append(entry)
    write_json(path, versions)
    return entry["model_id"]
