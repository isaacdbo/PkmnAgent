"""Load messy eval outputs into tidy per-run and per-checkpoint metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


WIN_CAUSE = {
    0: "unknown",
    1: "prize",
    2: "deck_out",
    3: "no_active",
    4: "card_effect",
}


@dataclass(frozen=True, slots=True)
class GameMetric:
    source_file: str
    run_id: str
    checkpoint: str
    opponent: str
    game_index: int | None
    result: str
    win_cause: str
    turns: int | None
    attacks_chosen: int | None
    attacks_available: int | None


def load_eval_games(paths: Iterable[Path]) -> list[GameMetric]:
    games: list[GameMetric] = []
    for path in paths:
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
        else:
            files = [path]
        for file_path in files:
            if file_path.suffix.lower() in {".jsonl", ".json", ".txt", ".log", ".csv"}:
                games.extend(_load_file(file_path))
    return games


def summarize_games(games: Iterable[GameMetric]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[GameMetric]] = defaultdict(list)
    for game in games:
        groups[(game.run_id, game.checkpoint, game.opponent)].append(game)

    rows: list[dict[str, Any]] = []
    for (run_id, checkpoint, opponent), group in sorted(groups.items()):
        total = len(group)
        wins = sum(1 for g in group if g.result == "win")
        losses = sum(1 for g in group if g.result == "loss")
        draws = sum(1 for g in group if g.result == "draw")
        deck_outs = sum(1 for g in group if g.win_cause == "deck_out")
        attack_chosen = sum(g.attacks_chosen or 0 for g in group)
        attack_available = sum(g.attacks_available or 0 for g in group)
        turn_values = [g.turns for g in group if g.turns is not None]
        causes = Counter(g.win_cause for g in group)
        rows.append({
            "run_id": run_id,
            "checkpoint": checkpoint,
            "opponent": opponent,
            "games": total,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / total if total else 0.0,
            "deck_out_rate": deck_outs / total if total else 0.0,
            "attack_rate": attack_chosen / attack_available if attack_available else None,
            "attacks_chosen": attack_chosen,
            "attacks_available": attack_available,
            "avg_turns_to_win": (
                sum(g.turns for g in group if g.result == "win" and g.turns is not None)
                / sum(1 for g in group if g.result == "win" and g.turns is not None)
                if any(g.result == "win" and g.turns is not None for g in group)
                else None
            ),
            "avg_turns": sum(turn_values) / len(turn_values) if turn_values else None,
            "WIN_BY_CAUSE": dict(sorted(causes.items())),
            "source_files": sorted({g.source_file for g in group}),
        })
    return rows


def write_outputs(games: list[GameMetric], summary: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "games.jsonl").write_text(
        "".join(json.dumps(asdict(game), sort_keys=True) + "\n" for game in games),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "run_id", "checkpoint", "opponent", "games", "wins", "losses", "draws",
            "win_rate", "deck_out_rate", "attack_rate", "attacks_chosen",
            "attacks_available", "avg_turns_to_win", "avg_turns", "WIN_BY_CAUSE",
            "source_files",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row | {
                "WIN_BY_CAUSE": json.dumps(row["WIN_BY_CAUSE"], sort_keys=True),
                "source_files": json.dumps(row["source_files"], sort_keys=True),
            })


def _load_file(path: Path) -> list[GameMetric]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path, text)
    if path.suffix.lower() == ".json":
        return _load_json(path, text)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    return _load_text(path, text)


def _load_jsonl(path: Path, text: str) -> list[GameMetric]:
    games: list[GameMetric] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        games.extend(_games_from_payload(path, payload))
    return games


def _load_json(path: Path, text: str) -> list[GameMetric]:
    try:
        return _games_from_payload(path, json.loads(text))
    except json.JSONDecodeError:
        return _load_text(path, text)


def _load_csv(path: Path) -> list[GameMetric]:
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f))
    return [_metric_from_mapping(path, row, i) for i, row in enumerate(rows)]


def _load_text(path: Path, text: str) -> list[GameMetric]:
    games: list[GameMetric] = []
    current_run = _infer_run_id(path, {})
    current_checkpoint = _infer_checkpoint(path, {})
    for line in text.splitlines():
        if "checkpoint" in line.lower():
            current_checkpoint = _first_match(line, [r"checkpoint[:= ]+([^,\s]+)", r"(model_[^,\s]+\.pth)"]) or current_checkpoint
        if "run" in line.lower():
            current_run = _first_match(line, [r"run[_ -]?id[:= ]+([^,\s]+)", r"run[:= ]+([^,\s]+)"]) or current_run
        games.extend(_metrics_from_eval_line(path, line, current_run, current_checkpoint))
    return games


def _games_from_payload(path: Path, payload: Any) -> list[GameMetric]:
    if isinstance(payload, list):
        return [game for item in payload for game in _games_from_payload(path, item)]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("games"), list):
        parent = payload
        return [_metric_from_mapping(path, item, i, parent=parent) for i, item in enumerate(payload["games"]) if isinstance(item, dict)]
    if _looks_like_game(payload):
        return [_metric_from_mapping(path, payload, None)]
    return []


def _metrics_from_eval_line(path: Path, line: str, run_id: str, checkpoint: str) -> list[GameMetric]:
    match = re.search(r"vs\s+([A-Za-z0-9_.-]+)\s*:\s*(\d+)%\s*\((\d+)W\s*/\s*(\d+)L\s*/\s*(\d+)D\)", line)
    if not match:
        return []
    opponent, _rate, wins, losses, draws = match.groups()
    games: list[GameMetric] = []
    for result, count in (("win", int(wins)), ("loss", int(losses)), ("draw", int(draws))):
        for _ in range(count):
            games.append(GameMetric(str(path), run_id, checkpoint, opponent, None, result, "unknown", None, None, None))
    return games


def _metric_from_mapping(path: Path, row: dict[str, Any], index: int | None, parent: dict[str, Any] | None = None) -> GameMetric:
    parent = parent or {}
    result = _normalize_result(_get(row, "result", "outcome", "winner", "win"))
    cause_value = _get(row, "win_cause", "WIN_BY_CAUSE", "finish_reason", "reason", "cause")
    return GameMetric(
        source_file=str(path),
        run_id=str(_get(row, "run_id", "run", default=_infer_run_id(path, parent))),
        checkpoint=str(_get(row, "checkpoint", "model", "ckpt", default=_infer_checkpoint(path, parent))),
        opponent=str(_get(row, "opponent", "opponent_name", "vs", default=parent.get("opponent", "unknown"))),
        game_index=_to_int(_get(row, "game_index", "game", "idx", default=index)),
        result=result,
        win_cause=_normalize_cause(cause_value),
        turns=_to_int(_get(row, "turns", "turn", "turns_to_win")),
        attacks_chosen=_to_int(_get(row, "attacks_chosen", "attack_chosen", "chosen_attacks")),
        attacks_available=_to_int(_get(row, "attacks_available", "attack_available", "available_attacks")),
    )


def _looks_like_game(payload: dict[str, Any]) -> bool:
    keys = {k.lower() for k in payload}
    return bool(keys & {"result", "outcome", "winner", "win"}) and bool(keys & {"opponent", "opponent_name", "vs"})


def _get(mapping: dict[str, Any], *names: str, default: Any = None) -> Any:
    lower = {str(k).lower(): v for k, v in mapping.items()}
    for name in names:
        if name in mapping:
            return mapping[name]
        if name.lower() in lower:
            return lower[name.lower()]
    return default


def _infer_run_id(path: Path, payload: dict[str, Any]) -> str:
    return str(payload.get("run_id") or payload.get("run") or path.parent.name or "unknown")


def _infer_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    for key in ("checkpoint", "model", "ckpt"):
        if payload.get(key):
            return str(payload[key])
    match = re.search(r"model[_-][A-Za-z0-9_.-]+\.pth", path.name)
    return match.group(0) if match else path.stem


def _normalize_result(value: Any) -> str:
    if isinstance(value, bool):
        return "win" if value else "loss"
    if isinstance(value, (int, float)):
        if int(value) == 2:
            return "draw"
        return "win" if int(value) == 0 else "loss"
    text = str(value).strip().lower()
    if text in {"w", "win", "won", "true"}:
        return "win"
    if text in {"l", "loss", "lost", "false"}:
        return "loss"
    if text in {"d", "draw", "tie", "capped"}:
        return "draw"
    return "unknown"


def _normalize_cause(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    if isinstance(value, (int, float)):
        return WIN_CAUSE.get(int(value), str(int(value)))
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"prizes", "zero_prize", "0_prize_cards"}:
        return "prize"
    if text in {"deckout", "deck_out", "deck"}:
        return "deck_out"
    return text


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/eval-metrics"))
    args = parser.parse_args()

    games = load_eval_games(args.inputs)
    summary = summarize_games(games)
    write_outputs(games, summary, args.out_dir)
    print(json.dumps({"games": len(games), "groups": len(summary), "out_dir": str(args.out_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
