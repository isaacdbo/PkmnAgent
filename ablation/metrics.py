"""The diagnostic metrics, as values rather than log lines.

Everything downstream of a game — the eval panel, the raw-log ingester, the
ablation driver's comparison table — produces and consumes the same two
shapes:

  GameRecord   one row per finished game
  summarize()  a dict of metrics over a list of those rows

The metrics are the ones the reward ablation is actually trying to move:

  win_rate        with a Wilson 95% interval, because 40-game panels routinely
                  produce differences that a proportion test will not support
  deckout_rate    share of decided games ending in deck-out — the FINDINGS.md
                  pathology, split into wins-by and losses-by
  turns_to_win    mean/median turn count of games the candidate won (and the
                  matching turns_to_loss), so "faster win" is measurable
  attack_rate     share of the candidate's own decisions, among those where an
                  attack was legal, that actually chose the attack. Measured
                  from the trajectory, not from MCTS internals, so it is
                  defined for scripted panel bots too
  win_by_cause    the WIN_BY_CAUSE tally, kept per-cause rather than collapsed

No torch, no engine, no pandas: this module is pure stdlib so it runs on the
host while the engine only runs in the Linux container.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Iterable, Sequence

CAUSES = ("prize", "deckout", "other")


@dataclass
class GameRecord:
    """One finished game, from the candidate's point of view."""

    game_index: int
    opponent: str
    # "win" | "loss" | "draw"
    outcome: str
    # one of CAUSES, or None when the source log did not record a reason
    cause: str | None = None
    final_turn: int | None = None
    # Candidate-side attack decisions, when the producer could observe them.
    attack_legal_decisions: int = 0
    attack_chosen_decisions: int = 0
    # Prizes taken, when available; useful for telling attrition from stalling.
    candidate_prizes: int | None = None
    opponent_prizes: int | None = None
    seconds: float | None = None
    # Free-form provenance: which run/checkpoint/arm this game came from.
    run: str | None = None
    checkpoint: str | None = None
    reward_spec: str | None = None
    source: str | None = None

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None})


def wilson_ci(wins: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Same formula and z as eval_panel.wilson_ci, duplicated here rather than
    imported because eval_panel imports torch and the engine; this module has
    to stay importable on a host that has neither.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = wins / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return float(median(values)) if values else None


def _rate(numer: int, denom: int) -> float | None:
    return numer / denom if denom else None


@dataclass
class Summary:
    """Metrics over one (candidate, opponent) matchup, or over any game set."""

    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    decided: int = 0
    win_rate: float | None = None
    win_rate_ci_low: float | None = None
    win_rate_ci_high: float | None = None
    deckout_rate: float | None = None
    deckout_win_rate: float | None = None
    deckout_loss_rate: float | None = None
    prize_rate: float | None = None
    attack_rate: float | None = None
    attack_legal_decisions: int = 0
    attack_chosen_decisions: int = 0
    turns_to_win_mean: float | None = None
    turns_to_win_median: float | None = None
    turns_to_loss_mean: float | None = None
    turns_to_loss_median: float | None = None
    turns_mean: float | None = None
    win_by_cause: dict = field(default_factory=dict)
    games_with_cause: int = 0
    games_with_turn: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(records: Iterable[GameRecord]) -> Summary:
    records = list(records)
    s = Summary(games=len(records))

    turns_win: list[int] = []
    turns_loss: list[int] = []
    turns_all: list[int] = []
    cause_tally: dict[str, dict[str, int]] = {}

    for r in records:
        if r.outcome == "win":
            s.wins += 1
        elif r.outcome == "loss":
            s.losses += 1
        else:
            s.draws += 1

        s.attack_legal_decisions += r.attack_legal_decisions
        s.attack_chosen_decisions += r.attack_chosen_decisions

        if r.final_turn is not None:
            turns_all.append(r.final_turn)
            if r.outcome == "win":
                turns_win.append(r.final_turn)
            elif r.outcome == "loss":
                turns_loss.append(r.final_turn)

        if r.cause is not None:
            s.games_with_cause += 1
            bucket = cause_tally.setdefault(r.cause, {"W": 0, "L": 0, "D": 0})
            bucket[{"win": "W", "loss": "L", "draw": "D"}[r.outcome]] += 1

    s.games_with_turn = len(turns_all)
    s.decided = s.wins + s.losses
    s.win_rate = _rate(s.wins, s.decided)
    if s.decided:
        s.win_rate_ci_low, s.win_rate_ci_high = wilson_ci(s.wins, s.decided)

    # Cause rates are over games whose cause is known, not over all games: a
    # log that recorded no reason codes should read as "unknown", never as
    # "0% deck-out".
    if s.games_with_cause:
        deckout = cause_tally.get("deckout", {"W": 0, "L": 0, "D": 0})
        prize = cause_tally.get("prize", {"W": 0, "L": 0, "D": 0})
        deckout_total = deckout["W"] + deckout["L"] + deckout["D"]
        prize_total = prize["W"] + prize["L"] + prize["D"]
        s.deckout_rate = _rate(deckout_total, s.games_with_cause)
        s.deckout_win_rate = _rate(deckout["W"], s.games_with_cause)
        s.deckout_loss_rate = _rate(deckout["L"], s.games_with_cause)
        s.prize_rate = _rate(prize_total, s.games_with_cause)

    s.attack_rate = _rate(s.attack_chosen_decisions, s.attack_legal_decisions)
    s.turns_to_win_mean = _mean(turns_win)
    s.turns_to_win_median = _median(turns_win)
    s.turns_to_loss_mean = _mean(turns_loss)
    s.turns_to_loss_median = _median(turns_loss)
    s.turns_mean = _mean(turns_all)
    s.win_by_cause = {c: cause_tally[c] for c in sorted(cause_tally)}
    return s


def summarize_by(records: Iterable[GameRecord], *keys: str) -> dict[tuple, Summary]:
    """Group records by attribute names and summarize each group."""
    groups: dict[tuple, list[GameRecord]] = {}
    for r in records:
        groups.setdefault(tuple(getattr(r, k) for k in keys), []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def _fmt(value, spec: str = ".3f", pct: bool = False) -> str:
    if value is None:
        return "n/a"
    if pct:
        return f"{100 * value:.1f}%"
    return format(value, spec)


# Column order used by both the text table and the CSV, so a reader comparing
# the two is looking at the same thing in the same order.
TABLE_COLUMNS = (
    ("games", lambda s: str(s.games)),
    ("W-L-D", lambda s: f"{s.wins}-{s.losses}-{s.draws}"),
    ("win_rate", lambda s: _fmt(s.win_rate, pct=True)),
    ("wilson95", lambda s: (
        "n/a" if s.win_rate_ci_low is None
        else f"[{100 * s.win_rate_ci_low:.1f},{100 * s.win_rate_ci_high:.1f}]"
    )),
    ("deckout_rate", lambda s: _fmt(s.deckout_rate, pct=True)),
    ("attack_rate", lambda s: _fmt(s.attack_rate, pct=True)),
    ("turns_to_win", lambda s: _fmt(s.turns_to_win_mean, ".1f")),
    ("turns_mean", lambda s: _fmt(s.turns_mean, ".1f")),
    ("win_by_cause", lambda s: (
        "|".join(f"{c}:W={v['W']},L={v['L']},D={v['D']}" for c, v in s.win_by_cause.items())
        or "n/a"
    )),
)


def render_table(rows: dict[str, Summary], label: str = "group") -> str:
    """A fixed-width table of summaries, keyed by an arbitrary label."""
    headers = [label] + [name for name, _ in TABLE_COLUMNS]
    body = [[key] + [fn(s) for _, fn in TABLE_COLUMNS] for key, s in rows.items()]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in body)) if body else len(headers[i])
        for i in range(len(headers))
    ]
    out = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    out.append("-|-".join("-" * w for w in widths))
    for r in body:
        out.append(" | ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def render_markdown_table(rows: dict[str, Summary], label: str = "group") -> str:
    headers = [label] + [name for name, _ in TABLE_COLUMNS]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for key, s in rows.items():
        out.append("| " + " | ".join([key] + [fn(s) for _, fn in TABLE_COLUMNS]) + " |")
    return "\n".join(out)


def write_games_jsonl(records: Iterable[GameRecord], path) -> int:
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(r.to_json() + "\n")
            n += 1
    return n


def read_games_jsonl(path) -> list[GameRecord]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(GameRecord(**json.loads(line)))
    return records


def write_summary_csv(rows: dict[str, Summary], path, label: str = "group") -> None:
    import csv

    fields = [label] + list(Summary().to_dict().keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key, s in rows.items():
            row = s.to_dict()
            row["win_by_cause"] = "|".join(
                f"{c}:W={v['W']},L={v['L']},D={v['D']}" for c, v in s.win_by_cause.items()
            )
            row[label] = key
            w.writerow(row)
