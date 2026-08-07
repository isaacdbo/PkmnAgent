"""Baselines. Every stronger agent should beat these before it is worth shipping.

`RandomAgent` is the floor. `GreedyAgent` is the handcrafted-heuristic control
group -- the thing a learned method has to actually outperform to earn its cost.
`GreedyAgent` is also a drop-in for a stronger evaluation opponent than uniform
random, which gives a much less noisy win-rate signal (IMPROVEMENT_PLAN 5e).
"""

from __future__ import annotations

import random

from ptcg_ai.api import Action, BaseAgent, Budget, Observation


class RandomAgent(BaseAgent):
    """Uniform over legal actions. The floor; also the default rollout policy."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, obs: Observation, legal: list[Action], budget: Budget) -> Action:
        return self._rng.choice(legal)


class GreedyAgent(BaseAgent):
    """One-ply argmax over a handcrafted score. The control group.

    Pass any `score(obs, action) -> float`. Ties break randomly so that repeated
    matches sample the tied set instead of locking onto list order.
    """

    def __init__(self, score, seed: int | None = None) -> None:
        self._score = score
        self._rng = random.Random(seed)

    def act(self, obs: Observation, legal: list[Action], budget: Budget) -> Action:
        best: list[Action] = []
        best_value = float("-inf")
        for action in legal:
            value = self._score(obs, action)
            if value > best_value:
                best_value, best = value, [action]
            elif value == best_value:
                best.append(action)
        return self._rng.choice(best)
