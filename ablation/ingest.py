"""Turn the repo's accumulated raw eval output into structured metrics.

The repo root holds ~40 files of evaluation output in six different shapes,
written by different scripts on different days: eval_panel logs, head-to-head
logs, training logs with diag dumps interleaved, `sweep_out/*.txt` arm files,
`diagnostic/*.log` dumps, and one already-parsed CSV. They contain real
results, but nothing can query them and nothing says which file is which run.

This module reads all of those shapes and emits one schema:

  games.jsonl       one ablation.metrics.GameRecord per game the logs recorded
  diag_dumps.jsonl  one row per DIAG_DUMP block, with derived rates
  summary.json      per-source and per-opponent metrics
  summary.csv       the same, spreadsheet-shaped
  INDEX.md          an inventory: what each file is, and its headline numbers

Parsers are deliberately tolerant. The EVAL_GAME_DONE line format gained
`winner=`, `turn=` and `cand_prizes=` fields partway through the project's
history, so older logs simply produce records with those fields unset, and
`metrics.summarize` reports the affected metrics as n/a rather than as zero.

Usage:
    python -m ablation.ingest . --out-dir results/ingested
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, field

from ablation import metrics
from ablation.metrics import GameRecord

# `KEY=value` pairs on a GAME_DONE line, where values never contain spaces.
_KV = re.compile(r"(\w+)=([^\s]+)")
_PANEL_HEADER = re.compile(r"^===\s*PANEL:\s*candidate vs (\S+)\s*===")
_GAME_DONE = re.compile(r"^(EVAL|H2H)_GAME_DONE=(\d+)/(\d+)\s+(.*)$")
_OPP_SUMMARY = re.compile(r"^OPP_(\S+)_(CANDIDATE|OPPONENT):\s+(.*)$")

# Header keys worth keeping as run metadata, by log family.
_RUN_META_KEYS = (
    "EVAL_CANDIDATE", "EVAL_CANDIDATE_DECK", "EVAL_SIMULATIONS_PER_MOVE",
    "EVAL_GAMES_PER_OPPONENT", "EVAL_BASE_SEED",
    "H2H_NEW", "H2H_OLD", "H2H_SIMULATIONS_PER_MOVE", "H2H_GAMES", "H2H_BASE_SEED",
    "ARM_SIMULATIONS_PER_MOVE", "ARM_DECK_DIFF_COEF", "ARM_CHECKPOINT", "ARM_GAMES",
    "ARM_BASE_SEED", "ARM_ATTACH_PRIOR_FLOOR", "ARM_POLICY_LABEL_SMOOTHING",
    "REWARD_SPEC",
)

_CAUSE_FROM_DIAG = {
    "PRIZE": "prize",
    "DECK_OUT": "deckout",
    "NO_ACTIVE": "other",
    "CARD_EFFECT_OR_CONCEDE": "other",
    "UNKNOWN": None,
}


@dataclass
class DiagDump:
    """One DIAG_DUMP block, plus the rates that are actually comparable."""

    source: str
    line_number: int
    agent: str | None = None
    games: int | None = None
    win_by_cause: dict = field(default_factory=dict)
    game_length_hist: dict = field(default_factory=dict)
    root_option_stats: dict = field(default_factory=dict)
    deckout_rate: float | None = None
    prize_rate: float | None = None
    attack_rate: float | None = None
    attack_available: int | None = None
    attack_chosen: int | None = None
    mean_game_length_bucket: str | None = None
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "line_number": self.line_number,
            "agent": self.agent,
            "games": self.games,
            "win_by_cause": self.win_by_cause,
            "game_length_hist": self.game_length_hist,
            "root_option_stats": self.root_option_stats,
            "deckout_rate": self.deckout_rate,
            "prize_rate": self.prize_rate,
            "attack_rate": self.attack_rate,
            "attack_available": self.attack_available,
            "attack_chosen": self.attack_chosen,
            "mean_game_length_bucket": self.mean_game_length_bucket,
        }


def _parse_kv_pairs(text: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in _KV.finditer(text)}


def _parse_win_by_cause(value: str) -> dict[str, dict[str, int]]:
    """`DECK_OUT:W=16,L=16,D=0|NO_ACTIVE:W=4,L=4,D=0` -> nested counts.

    diag.record_game_result writes one W *and* one L per decided game, and one
    D per drawn game, so the number of games with a given cause is W + D — not
    W + L + D. FINDINGS.md's "DECK_OUT=16 ... 16/20 = 80%" reads it the same
    way.
    """
    out: dict[str, dict[str, int]] = {}
    if not value or value == "NONE":
        return out
    for part in value.split("|"):
        if ":" not in part:
            continue
        cause, counts = part.split(":", 1)
        rec = {"W": 0, "L": 0, "D": 0}
        for kv in counts.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                if k in rec:
                    rec[k] = int(v)
        out[cause] = rec
    return out


def _parse_hist(value: str) -> dict[str, int]:
    out: dict[str, int] = {}
    if not value or value == "NONE":
        return out
    for part in value.split("|"):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                out[k] = int(v)
            except ValueError:
                continue
    return out


def _parse_root_option_stats(value: str) -> dict[str, dict[str, float]]:
    """`ATTACK:available=97,chosen=6,...|ENERGY_ATTACH:...` -> nested numbers."""
    out: dict[str, dict[str, float]] = {}
    if not value or value == "NONE":
        return out
    for part in value.split("|"):
        if ":" not in part:
            continue
        kind, rest = part.split(":", 1)
        stats: dict[str, float] = {}
        # Values like `zero_visit_rate=8.16%(8/98)` are kept as-is under a
        # separate key rather than coerced; only the plain numbers are needed.
        for kv in rest.split(","):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            try:
                stats[k] = float(v)
            except ValueError:
                stats[k + "_raw"] = v
        out[kind] = stats
    return out


def parse_diag_dumps(path: str) -> list[DiagDump]:
    """Every DIAG_DUMP block in a file, in order."""
    dumps: list[DiagDump] = []
    current: dict[str, str] | None = None
    start_line = 0
    with open(path, errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if "=== DIAG_DUMP_BEGIN ===" in line:
                current, start_line = {}, lineno
                continue
            if "=== DIAG_DUMP_END ===" in line:
                if current is not None:
                    dumps.append(_build_dump(path, start_line, current))
                current = None
                continue
            if current is not None and "=" in line:
                key, value = line.split("=", 1)
                # A training log can interleave progress output into a dump;
                # keys are uppercase identifiers, so anything else is noise.
                if re.fullmatch(r"[A-Z0-9_]+", key):
                    current[key] = value
    return dumps


def _build_dump(source: str, line_number: int, raw: dict[str, str]) -> DiagDump:
    dump = DiagDump(source=source, line_number=line_number, raw=raw)
    dump.agent = raw.get("AGENT")
    if raw.get("DIAG_GAMES", "").isdigit():
        dump.games = int(raw["DIAG_GAMES"])

    dump.win_by_cause = _parse_win_by_cause(raw.get("WIN_BY_CAUSE", ""))
    dump.game_length_hist = _parse_hist(raw.get("GAME_LENGTH_HIST", ""))
    dump.root_option_stats = _parse_root_option_stats(raw.get("ROOT_OPTION_STATS", ""))

    total_games = sum(c["W"] + c["D"] for c in dump.win_by_cause.values())
    if total_games:
        deckout = dump.win_by_cause.get("DECK_OUT", {"W": 0, "D": 0})
        prize = dump.win_by_cause.get("PRIZE", {"W": 0, "D": 0})
        dump.deckout_rate = (deckout["W"] + deckout["D"]) / total_games
        dump.prize_rate = (prize["W"] + prize["D"]) / total_games

    attack = dump.root_option_stats.get("ATTACK")
    if attack and attack.get("available"):
        dump.attack_available = int(attack["available"])
        dump.attack_chosen = int(attack.get("chosen", 0))
        dump.attack_rate = dump.attack_chosen / dump.attack_available

    if dump.game_length_hist:
        dump.mean_game_length_bucket = max(
            dump.game_length_hist, key=lambda k: dump.game_length_hist[k]
        )
    return dump


def parse_game_lines(path: str) -> tuple[list[GameRecord], dict[str, str]]:
    """Per-game records from EVAL_/H2H_ GAME_DONE lines, plus run metadata.

    Outcome is resolved by identity, never by raw seat index: `result` is the
    winning seat, and which seat the candidate occupied flips every game
    (eval_panel alternates who plays first), so `p0=`/`p1=` have to be read to
    know who actually won. FINDINGS.md records this exact misreading as a
    falsified hypothesis, so it is worth being explicit about here.
    """
    records: list[GameRecord] = []
    meta: dict[str, str] = {}
    opponent = "unknown"
    source = os.path.basename(path)

    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()

            header = _PANEL_HEADER.match(line)
            if header:
                opponent = header.group(1)
                continue

            for key in _RUN_META_KEYS:
                if line.startswith(key + "="):
                    meta[key] = line.split("=", 1)[1]

            m = _GAME_DONE.match(line)
            if not m:
                continue

            family, index = m.group(1), int(m.group(2))
            fields = _parse_kv_pairs(m.group(4))

            # H2H logs have no PANEL header and name their sides new/old; the
            # eval panel always prints the candidate's side as "candidate",
            # even in the newer format where the opponent gets a display name.
            if family == "H2H":
                opponent = "old_m2(h2h)"
                candidate_token = "new"
            else:
                candidate_token = "candidate"

            p0, p1 = fields.get("p0"), fields.get("p1")
            result = fields.get("result")
            if p0 is None or p1 is None or result is None:
                continue

            if p0 == candidate_token:
                candidate_seat = 0
            elif p1 == candidate_token:
                candidate_seat = 1
            else:
                # Neither seat is identifiable as the candidate: skip rather
                # than guess, since guessing wrong inverts the win rate.
                continue

            try:
                result_i = int(result)
            except ValueError:
                continue

            if result_i == 2:
                outcome = "draw"
            else:
                outcome = "win" if result_i == candidate_seat else "loss"

            cause = fields.get("cause")
            if cause not in ("prize", "deckout", "other"):
                cause = None

            record = GameRecord(
                game_index=index,
                opponent=opponent,
                outcome=outcome,
                cause=cause,
                final_turn=int(fields["turn"]) if fields.get("turn", "").isdigit() else None,
                candidate_prizes=(
                    int(fields["cand_prizes"]) if fields.get("cand_prizes", "").isdigit() else None
                ),
                opponent_prizes=(
                    int(fields["opp_prizes"]) if fields.get("opp_prizes", "").isdigit() else None
                ),
                seconds=float(fields["sec"]) if _is_float(fields.get("sec")) else None,
                source=source,
                run=meta.get("EVAL_CANDIDATE") or meta.get("H2H_NEW") or meta.get("ARM_CHECKPOINT"),
                checkpoint=meta.get("EVAL_CANDIDATE") or meta.get("H2H_NEW") or meta.get("ARM_CHECKPOINT"),
                reward_spec=meta.get("REWARD_SPEC"),
            )
            records.append(record)

    return records, meta


def _is_float(value) -> bool:
    if value is None:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_per_dump_csv(path: str) -> list[DiagDump]:
    """`task_c_per_dump.csv`: a hand-made per-dump table from an earlier pass.

    Kept rather than discarded because its attack_avail/attack_chosen columns
    are per-training-phase attack rates that no other artifact preserves.
    """
    dumps: list[DiagDump] = []
    with open(path, newline="", errors="replace") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            dump = DiagDump(source=os.path.basename(path), line_number=i)
            dump.agent = row.get("phase")
            if (row.get("games") or "").isdigit():
                dump.games = int(row["games"])
            dump.game_length_hist = _parse_hist(row.get("game_length_hist", ""))
            share = (row.get("deckout_share") or "").rstrip("%")
            if _is_float(share):
                dump.deckout_rate = float(share) / 100.0
            avail = row.get("attack_avail")
            chosen = row.get("attack_chosen")
            if (avail or "").isdigit() and (chosen or "").isdigit() and int(avail) > 0:
                dump.attack_available = int(avail)
                dump.attack_chosen = int(chosen)
                dump.attack_rate = int(chosen) / int(avail)
            dumps.append(dump)
    return dumps


# Files that are large, uninformative, or not eval output at all.
_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "checkpoints",
              ".dockerhome", "cg-lib", "results", "decks", "submissions"}
_TEXT_SUFFIXES = (".log", ".txt", ".err", ".out")


def discover(root: str) -> list[str]:
    """Candidate eval-output files under `root`, deterministically ordered."""
    if os.path.isfile(root):
        return [root]
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if name.endswith(_TEXT_SUFFIXES) or name == "task_c_per_dump.csv":
                found.append(os.path.join(dirpath, name))
    return found


def ingest(paths: list[str]) -> tuple[list[GameRecord], list[DiagDump], dict[str, dict]]:
    all_games: list[GameRecord] = []
    all_dumps: list[DiagDump] = []
    per_source: dict[str, dict] = {}

    for path in paths:
        rel = os.path.relpath(path)
        try:
            if os.path.basename(path) == "task_c_per_dump.csv":
                games, meta, dumps = [], {}, parse_per_dump_csv(path)
            else:
                games, meta = parse_game_lines(path)
                dumps = parse_diag_dumps(path)
        except (OSError, UnicodeDecodeError) as exc:
            per_source[rel] = {"error": str(exc)}
            continue

        if not games and not dumps:
            continue

        for d in dumps:
            d.source = rel
        for g in games:
            g.source = rel

        all_games.extend(games)
        all_dumps.extend(dumps)
        per_source[rel] = {
            "games": len(games),
            "diag_dumps": len(dumps),
            "meta": meta,
            "kind": _classify(rel, games, dumps, meta),
            "summary": metrics.summarize(games).to_dict() if games else None,
            "dump_deckout_rate": _last_not_none(d.deckout_rate for d in dumps),
            "dump_attack_rate": _last_not_none(d.attack_rate for d in dumps),
        }

    return all_games, all_dumps, per_source


def _last_not_none(values):
    out = None
    for v in values:
        if v is not None:
            out = v
    return out


def _classify(rel: str, games, dumps, meta) -> str:
    name = os.path.basename(rel)
    if name.startswith("eval_panel") or "EVAL_CANDIDATE" in meta:
        return "eval_panel run"
    if name.startswith("h2h") or "H2H_NEW" in meta:
        return "head-to-head checkpoint comparison"
    if rel.startswith("sweep_out/") or "ARM_CHECKPOINT" in meta:
        return "parameter-sweep arm"
    if rel.startswith("diagnostic/"):
        return "standalone diag dump"
    if name == "task_c_per_dump.csv":
        return "pre-parsed per-dump table"
    if dumps and not games:
        return "training log with diag dumps"
    return "other eval output"


def write_outputs(out_dir: str, games, dumps, per_source) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    games_path = os.path.join(out_dir, "games.jsonl")
    metrics.write_games_jsonl(games, games_path)

    dumps_path = os.path.join(out_dir, "diag_dumps.jsonl")
    with open(dumps_path, "w") as f:
        for d in dumps:
            f.write(json.dumps(d.to_dict()) + "\n")

    by_opponent = metrics.summarize_by(games, "opponent")
    by_source = metrics.summarize_by(games, "source")

    summary = {
        "totals": {
            "files_with_content": len(per_source),
            "games": len(games),
            "diag_dumps": len(dumps),
        },
        "by_opponent": {str(k[0]): v.to_dict() for k, v in by_opponent.items()},
        "by_source": {str(k[0]): v.to_dict() for k, v in by_source.items()},
        "sources": per_source,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    metrics.write_summary_csv(
        {str(k[0]): v for k, v in by_source.items()},
        os.path.join(out_dir, "summary.csv"),
        label="source",
    )

    with open(os.path.join(out_dir, "INDEX.md"), "w") as f:
        f.write(_render_index(games, dumps, per_source, by_opponent))

    return summary


def _render_index(games, dumps, per_source, by_opponent) -> str:
    lines = [
        "# Ingested eval output",
        "",
        "Generated by `python -m ablation.ingest`. One row per source file that",
        "contained parseable eval output; regenerate rather than hand-editing.",
        "",
        f"- source files with content: {len(per_source)}",
        f"- per-game records: {len(games)}",
        f"- diag dumps: {len(dumps)}",
        "",
        "## By opponent (per-game records only)",
        "",
        metrics.render_markdown_table(
            {str(k[0]): v for k, v in by_opponent.items()}, label="opponent"
        ),
        "",
        "## Source files",
        "",
        "| file | kind | games | dumps | deckout_rate (last dump) | attack_rate (last dump) |",
        "|---|---|---|---|---|---|",
    ]
    for rel in sorted(per_source):
        info = per_source[rel]
        if "error" in info:
            lines.append(f"| `{rel}` | unreadable: {info['error']} | - | - | - | - |")
            continue
        deckout = info["dump_deckout_rate"]
        attack = info["dump_attack_rate"]
        lines.append(
            f"| `{rel}` | {info['kind']} | {info['games']} | {info['diag_dumps']} | "
            f"{'n/a' if deckout is None else f'{100 * deckout:.1f}%'} | "
            f"{'n/a' if attack is None else f'{100 * attack:.1f}%'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", default=["."],
                    help="files or directories of raw eval output (default: .)")
    ap.add_argument("--out-dir", default=os.path.join("results", "ingested"))
    args = ap.parse_args()

    files: list[str] = []
    for p in args.paths or ["."]:
        files.extend(discover(p))

    games, dumps, per_source = ingest(files)
    summary = write_outputs(args.out_dir, games, dumps, per_source)

    print(f"scanned {len(files)} files, {len(per_source)} contained eval output")
    print(f"games={summary['totals']['games']} diag_dumps={summary['totals']['diag_dumps']}")
    print(f"wrote {args.out_dir}/games.jsonl, diag_dumps.jsonl, summary.json, summary.csv, INDEX.md")
    by_opponent = metrics.summarize_by(games, "opponent")
    if by_opponent:
        print()
        print(metrics.render_table({str(k[0]): v for k, v in by_opponent.items()}, label="opponent"))


if __name__ == "__main__":
    main()
