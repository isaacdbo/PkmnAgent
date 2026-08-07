"""Core types shared by the agents in this package.

Deliberately tiny and dependency-free. The competition's own observation/action
format is NOT baked in here, so search and evaluation code stays engine-agnostic
and testable without the engine present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# Opaque aliases rather than concrete classes, so this module never needs to
# import the engine.
Observation = Any
Action = Any


@dataclass(frozen=True, slots=True)
class Budget:
    """Per-move resource limit. Search must respect this or the entry forfeits."""

    seconds: float
    max_simulations: int | None = None

    def with_safety_margin(self, fraction: float = 0.8) -> Budget:
        """Leave headroom for serialization and interpreter overhead."""
        return Budget(self.seconds * fraction, self.max_simulations)


@runtime_checkable
class Agent(Protocol):
    """Everything that plays implements this, from `RandomAgent` to a full search."""

    def act(self, obs: Observation, legal: list[Action], budget: Budget) -> Action:
        """Return one action from `legal`. Must never exceed `budget.seconds`."""
        ...

    def reset(self) -> None:
        """Called at the start of each game. Clear per-game caches here."""
        ...


class BaseAgent:
    """Default `reset` so simple agents only implement `act`."""

    def reset(self) -> None:
        return None
