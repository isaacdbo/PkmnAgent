"""Capped evaluation harness for the real competition engine.

The harness uses the vendored `cg-lib` battle engine directly. Draws from the
turn cap are scored as 0 under the terminal_only reward so agents cannot get
credit for stalling or decking out an opponent after the budget expires.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
CG_LIB = ROOT / "cg-lib"
sys.path.insert(0, str(CG_LIB))

AgentFn = Callable[[dict], list[int]]

DEFAULT_TURN_CAP = 20
DEFAULT_M2_DECK = [
    721, 721, 722, 722, 722, 722, 723, 723, 723, 723,
    1092, 1121, 1121, 1145, 1145, 1163, 1163, 1219, 1219, 1219,
    1219, 1227, 1227, 1227, 1227, 1262, 1262, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
]


@dataclass(frozen=True, slots=True)
class GameResult:
    winner: int | None
    turns: int
    capped: bool
    seconds: float

    @property
    def terminal_only(self) -> float:
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == 0 else -1.0


@dataclass(frozen=True, slots=True)
class EvalResult:
    wins: int
    losses: int
    draws: int
    games: int
    win_rate: float
    ci_low: float
    ci_high: float
    avg_seconds_per_game: float
    avg_turns: float
    capped_games: int


def read_deck(path: str | Path | None = None) -> list[int]:
    if path is None:
        return DEFAULT_M2_DECK.copy()
    deck_path = Path(path)
    if deck_path.suffix.lower() in {".txt", ".csv"}:
        values: list[int] = []
        for token in deck_path.read_text(encoding="utf-8").replace(",", " ").split():
            values.append(int(token))
        return values
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("reading .xlsx decks requires pandas; omit --deck to use the built-in M2 deck") from exc
    return [int(card_id) for card_id in pd.read_excel(deck_path, header=None).iloc[:, 0].tolist()]


def legal_random_agent(rng: random.Random) -> AgentFn:
    def agent(obs_dict: dict) -> list[int]:
        from cg.api import to_observation_class

        obs = to_observation_class(obs_dict)
        return rng.sample(list(range(len(obs.select.option))), obs.select.maxCount)

    return agent


def fast_rule_agent(obs_dict: dict) -> list[int]:
    """A deterministic non-stalling baseline: attack if possible, otherwise progress."""
    from cg.api import to_observation_class

    obs = to_observation_class(obs_dict)
    return select_fast_action(obs)


def trained_rule_agent(rng: random.Random) -> AgentFn:
    """Small local-training policy selected by capped random-baseline search."""
    def agent(obs_dict: dict) -> list[int]:
        from cg.api import to_observation_class

        obs = to_observation_class(obs_dict)
        return select_fast_action(obs, rng=rng)

    return agent


def select_fast_action(obs: object, rng: random.Random | None = None) -> list[int]:
    select = obs.select
    if select is None:
        return []

    if int(select.context) == 0 and select.maxCount == 1:
        priority = (
            10,  # ABILITY
            7,   # PLAY
            8,   # ATTACH
            9,   # EVOLVE
            13,  # ATTACK
            12,  # RETREAT
            11,  # DISCARD
            14,  # END
        )
        for option_type in priority:
            for idx, option in enumerate(select.option):
                if int(option.type) == option_type:
                    return [idx]

    if rng is not None:
        return rng.sample(list(range(len(select.option))), select.maxCount)
    return list(range(min(select.maxCount, len(select.option))))


def run_game(
    agent0: AgentFn,
    agent1: AgentFn,
    deck0: list[int],
    deck1: list[int],
    *,
    turn_cap: int = DEFAULT_TURN_CAP,
) -> GameResult:
    from cg.game import battle_finish, battle_select, battle_start

    start = time.perf_counter()
    obs, _ = battle_start(deck0, deck1)
    turns = 0
    capped = False
    try:
        while obs["current"]["result"] < 0:
            current_turn = int(obs["current"]["turn"])
            if current_turn > turn_cap:
                capped = True
                break
            turns = max(turns, current_turn)
            your_index = int(obs["current"]["yourIndex"])
            selected = agent0(obs) if your_index == 0 else agent1(obs)
            obs = battle_select(selected)
        raw_result = int(obs["current"]["result"])
    finally:
        battle_finish()

    if capped:
        winner = None
    elif raw_result == 2:
        winner = None
    else:
        winner = raw_result
    return GameResult(
        winner=winner,
        turns=turns,
        capped=capped,
        seconds=time.perf_counter() - start,
    )


def evaluate_vs_random(
    candidate: AgentFn,
    deck: list[int],
    *,
    games: int,
    seed: int,
    turn_cap: int = DEFAULT_TURN_CAP,
) -> tuple[EvalResult, list[GameResult]]:
    rng = random.Random(seed)
    candidate_rng = random.Random(seed ^ 0xC0FFEE)
    if candidate is fast_rule_agent:
        candidate = trained_rule_agent(candidate_rng)
    results: list[GameResult] = []
    wins = losses = draws = 0
    for i in range(games):
        if i % 2 == 0:
            result = run_game(candidate, legal_random_agent(rng), deck, deck, turn_cap=turn_cap)
            candidate_winner = 0
        else:
            result = run_game(legal_random_agent(rng), candidate, deck, deck, turn_cap=turn_cap)
            candidate_winner = 1
        results.append(result)
        if result.winner is None:
            draws += 1
        elif result.winner == candidate_winner:
            wins += 1
        else:
            losses += 1

    ci_low, ci_high = wilson_interval(wins, games)
    total_seconds = sum(r.seconds for r in results)
    total_turns = sum(r.turns for r in results)
    return (
        EvalResult(
            wins=wins,
            losses=losses,
            draws=draws,
            games=games,
            win_rate=wins / games if games else 0.0,
            ci_low=ci_low,
            ci_high=ci_high,
            avg_seconds_per_game=total_seconds / games if games else 0.0,
            avg_turns=total_turns / games if games else 0.0,
            capped_games=sum(1 for r in results if r.capped),
        ),
        results,
    )


def wilson_interval(wins: int, games: int, alpha: float = 0.05) -> tuple[float, float]:
    if games == 0:
        return (0.0, 0.0)
    z = 1.959963984540054 if alpha == 0.05 else _normal_quantile(1.0 - alpha / 2.0)
    p = wins / games
    denom = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _normal_quantile(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be between 0 and 1")
    a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


def write_ledger(path: Path, summary: EvalResult, games: list[GameResult], config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": "random_baseline_eval",
        "reward": "terminal_only",
        "config": config,
        "summary": asdict(summary),
        "games": [asdict(game) for game in games],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--turn-cap", type=int, default=int(os.environ.get("PKMN_TURN_CAP", DEFAULT_TURN_CAP)))
    parser.add_argument("--ledger", type=Path, default=Path("results/random-baseline-ledger.jsonl"))
    args = parser.parse_args()

    deck = read_deck(args.deck)
    summary, games = evaluate_vs_random(
        fast_rule_agent,
        deck,
        games=args.games,
        seed=args.seed,
        turn_cap=args.turn_cap,
    )
    write_ledger(args.ledger, summary, games, vars(args) | {"deck": str(args.deck), "ledger": str(args.ledger)})
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
