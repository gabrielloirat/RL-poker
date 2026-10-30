
import hashlib
from importlib import import_module, invalidate_caches
import inspect
from pathlib import Path
import sys
from types import ModuleType

from poker_rl.records import saved_models, model_catalog


DEFAULT_AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"


class AgentPlugin:
    def __init__(self,
                 name, path, factory,
                 trainable_override=None):  # Enregistre les données
        self.name = name
        self.path = path
        self.factory = factory
        self.trainable_override = trainable_override

    @property
    def trainable(self):
        if self.trainable_override is not None:
            return self.trainable_override
        if inspect.isclass(self.factory):
            return callable(getattr(self.factory, "observe", None))
        return None


def _namespace(name, path):
    if name not in sys.modules:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module


def _factory(module):

    if hasattr(module, "create_agent"):
        factory = module.create_agent
        if not callable(factory):
            raise TypeError("create_agent doit être une fonction ou une classe callable.")
        return factory
    candidates = []
    for value in vars(module).values():
        if not inspect.isclass(value):
            continue
        if value.__module__ != module.__name__ or inspect.isabstract(value):
            continue
        if callable(getattr(value, "act", None)):
            candidates.append(value)

    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise ValueError("Exposez une seule classe avec act(), ou une fonction create_agent(**kwargs).")
    return candidates[0]


class AgentRegistry:


    def __init__(self, agents_dir = DEFAULT_AGENTS_DIR, *, records_dir=None):
        self.directory = Path(agents_dir).resolve()
        if records_dir is None:
            self.records_dir = self.directory.parent / "runs" / "sessions"
        else:
            self.records_dir = Path(records_dir)
        self.agents = {}
        self.errors = {}
        if not self.directory.is_dir():
            raise ValueError(f"Dossier d'agents introuvable : {self.directory}")
        namespace = "_poker_agents_" + hashlib.sha256(str(self.directory).encode()).hexdigest()[:16]
        _namespace(namespace, self.directory)
        invalidate_caches()
        entries = {}
        for path in sorted(self.directory.iterdir()):
            if path.name.startswith(("_", ".")):
                continue
            if path.is_file() and path.suffix == ".py":
                name, entry = path.stem, path
            elif path.is_dir() and (path / "agent.py").is_file():
                name, entry = path.name, path / "agent.py"
            elif path.is_dir() and (path / "__init__.py").is_file():
                name, entry = path.name, path / "__init__.py"
            else:
                continue
            entries.setdefault(name, []).append(entry)
        for name, paths in entries.items():
            if len(paths) > 1:
                self.errors[name] = f"Nom ambigu : plusieurs plugins nommés {name}. Renommez le fichier ou le dossier."
                continue
            if not name.isidentifier():
                self.errors[name] = "Le nom du fichier ou du dossier doit être un identifiant Python (ex. mon_agent)."
                continue
            path = paths[0]
            module_name = f"{namespace}.{name}"
            if path.name == "agent.py" and path.parent != self.directory:


                if not (path.parent / "__init__.py").is_file():
                    _namespace(module_name, path.parent)
                module_name += ".agent"
            try:
                factory = _factory(import_module(module_name))
                self.agents[name] = AgentPlugin(name, path, factory)
            except Exception as exc:
                self.errors[name] = f"{type(exc).__name__}: {exc}"

        self.archived_models = saved_models(self.records_dir)
        self.saved_models = model_catalog(self.records_dir)
        for name, model in self.saved_models.items():
            if name in self.agents or name in self.errors:
                self.errors[name] = "Nom de modèle sauvegardé déjà utilisé par un plugin."
                self.agents.pop(name, None)
                continue
            base = self.agents.get(model["name"])
            trainable = None
            if base is not None:
                trainable = base.trainable
            self.agents[name] = AgentPlugin(name, model["path"], None, trainable)

    def describe(self, name, kwargs=None):
        model = self.resolve_model(name)
        if model:
            options = dict(model.get("kwargs", {}))
            if kwargs is not None:
                options.update(kwargs)
            entry = self.describe(model["name"], options)
            entry["source_model"] = name
            entry["source_run"] = model["run_id"]
            return entry
        entry = {"name": name, "kwargs": dict(kwargs or {})}
        plugin = self.agents.get(name)
        if plugin and plugin.path.is_file():
            entry["source_sha256"] = hashlib.sha256(plugin.path.read_bytes()).hexdigest()
        return entry

    def label(self, name):
        model = self.resolve_model(name)
        if not model:
            return name
        if model.get("catalog_kind") == "current":
            suffix = "courant"
        else:
            suffix = model.get("version_label", name)
        return f"{model['name']} · {suffix} · {model['num_players']}j"

    def resolve_model(self, name):
        return self.saved_models.get(name) or self.archived_models.get(name)

    def choices(self, num_players=2):
        models = {}
        current_bases = set()
        for name, model in self.saved_models.items():
            if model["num_players"] == num_players:
                models[name] = model
                if model["catalog_kind"] == "current":
                    current_bases.add(model["name"])
        choices = []
        for name in self.agents:
            if name not in self.saved_models and name not in current_bases:
                choices.append(name)
        choices.extend(models)
        return choices

    def native_training(self, name):
        model = self.resolve_model(name)
        base = name
        if model:
            base = model["name"]
        plugin = self.agents.get(base)
        return plugin is not None and getattr(plugin.factory, "native_training", False)

    def load(self, name, kwargs = None):
        if name in self.errors:
            raise ValueError(f"Agent '{name}' invalide : {self.errors[name]}")
        model = self.resolve_model(name)
        if name not in self.agents and model is None:
            available = ", ".join(self.agents) or "aucun"
            raise ValueError(f"Agent '{name}' introuvable dans {self.directory}. Disponibles : {available}.")
        if model:
            if hashlib.sha256(model["path"].read_bytes()).hexdigest() != model["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint modifié ou corrompu : {name}")
            options = dict(model.get("kwargs", {}))
            if kwargs is not None:
                options.update(kwargs)
            # The archive supplies the checkpoint after constructing the original plugin.
            options.pop("checkpoint", None)
            if self.native_training(name):
                options["autoload"] = False
            if ":" in model["name"]:
                from poker_rl.interface import load_agent
                agent = load_agent(model["name"], options)
            else:
                agent = self.load(model["name"], options)
            load = getattr(agent, "load_checkpoint", None)
            if not callable(load):
                raise TypeError(f"{model['name']} doit fournir load_checkpoint(path).")
            load(model["path"])
            return agent
        agent = self.agents[name].factory(**(kwargs or {}))
        if not callable(getattr(agent, "act", None)):
            raise TypeError(f"Agent '{name}' : act(observation, action_mask, *, training=False) est requis.")
        return agent
