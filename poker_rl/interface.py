from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Transition:


    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    next_action_mask: np.ndarray
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@runtime_checkable
class Agent(Protocol):
    def act(self, observation: np.ndarray, action_mask: np.ndarray,
            *, training: bool = False) -> int:

        ...


@runtime_checkable
class TrainableAgent(Agent, Protocol):
    def observe(self, transition: Transition) -> None:

        ...


def load_agent(spec: str, kwargs: dict[str, Any] | None = None, *,
               agents_dir: str | Path | None = None) -> Agent:

    if ":" not in spec:
        from poker_rl.plugins import AgentRegistry, DEFAULT_AGENTS_DIR
        return AgentRegistry(agents_dir if agents_dir is not None else DEFAULT_AGENTS_DIR).load(spec, kwargs)
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Agent must be specified as python_module:ClassOrFactory.")
    factory = getattr(import_module(module_name), attribute)
    agent = factory(**(kwargs or {}))
    if not callable(getattr(agent, "act", None)):
        raise TypeError(f"{spec} must provide act(observation, action_mask, *, training=False).")
    return agent
