from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module, invalidate_caches
import inspect
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable


DEFAULT_AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"


@dataclass(frozen=True)
class AgentPlugin:
    name: str
    path: Path
    factory: Callable

    @property
    def trainable(self):

        if inspect.isclass(self.factory):
            return callable(getattr(self.factory, "observe", None))
        return None


def _namespace(name: str, path: Path):
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
    candidates = [value for value in vars(module).values()
                  if inspect.isclass(value) and value.__module__ == module.__name__
                  and not inspect.isabstract(value) and not getattr(value, "_is_protocol", False)
                  and callable(getattr(value, "act", None))]

    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise ValueError("Exposez une seule classe avec act(), ou une fonction create_agent(**kwargs).")
    return candidates[0]


class AgentRegistry:


    def __init__(self, agents_dir: str | Path = DEFAULT_AGENTS_DIR):
        self.directory = Path(agents_dir).resolve()
        self.agents: dict[str, AgentPlugin] = {}
        self.errors: dict[str, str] = {}
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

    def load(self, name: str, kwargs: dict | None = None):
        if name in self.errors:
            raise ValueError(f"Agent '{name}' invalide : {self.errors[name]}")
        if name not in self.agents:
            available = ", ".join(self.agents) or "aucun"
            raise ValueError(f"Agent '{name}' introuvable dans {self.directory}. Disponibles : {available}.")
        agent = self.agents[name].factory(**(kwargs or {}))
        if not callable(getattr(agent, "act", None)):
            raise TypeError(f"Agent '{name}' : act(observation, action_mask, *, training=False) est requis.")
        return agent
