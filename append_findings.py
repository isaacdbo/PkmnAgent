"""Parse an eval_panel.py log and append a FINDINGS.md entry in the
established protocol format. Used by run_cycle.sh; also runnable standalone.

Usage:
    python append_findings.py --eval-log eval_panel_cycle.log \\
        --label "NEW (cycle, 8 epochs, Task 2 diff)" \\
        --epochs 8 --sims-train 20 --findings FINDINGS.md --pass-threshold 0.60

Prints the final line as:
    CYCLE_RESULT=PASS win_rate=XX.X%(W/N)
or
    CYCLE_RESULT=FAIL win_rate=XX.X%(W/N)
"""
from __future__ import annotations

import argparse
import datetime
import re


def parse_eval_log(path: str) -> dict:
    text = open(path, encoding="utf-8", errors="replace").read()
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line and re.match(r"^[A-Z][A-Za-z0-9_]*=", line):
            k, v = line.split("=", 1)
            fields[k] = v
    return fields


def _games_from_win_by_cause(wbc: str) -> dict:
    # WIN_BY_CAUSE format: CAUSE:W=x,L=y,D=z|CAUSE2:...  W and L tag the same
    # games — report W alone as the game count for that cause, per protocol.
    out = {}
    if not wbc or wbc == "NONE":
        return out
    for part in wbc.split("|"):
        cause, rest = part.split(":", 1)
        w = 0
        for kv in rest.split(","):
            k, v = kv.split("=")
            if k == "W":
                w = int(v)
        out[cause] = w
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-log", required=True, help="stdout log from eval_panel.py (--panel sample_bot)")
    ap.add_argument("--label", required=True, help="human label for the candidate, e.g. 'NEW (cycle, 8 epochs)'")
    ap.add_argument("--epochs", type=int, required=True, help="MAIN_EPOCHS trained this cycle")
    ap.add_argument("--sims-train", type=int, required=True, help="sims used during self-play training")
    ap.add_argument("--findings", default="FINDINGS.md")
    ap.add_argument("--pass-threshold", type=float, default=0.60,
                    help="candidate win rate vs sample_bot must exceed this to PASS")
    args = ap.parse_args()

    f = parse_eval_log(args.eval_log)
    raw_text = open(args.eval_log, encoding="utf-8", errors="replace").read()

    # OPP_<opponent>_CANDIDATE: wins=.. losses=.. win_rate=X%(W/N) ... — colon
    # + space-separated key=val, unlike the diag NAME=value fields above.
    m = re.search(r"OPP_\w+_CANDIDATE:.*?win_rate=([\d.]+)%\((\d+)/(\d+)\)", raw_text)
    if not m:
        raise ValueError("Could not find an OPP_<opponent>_CANDIDATE line with win_rate=... in the eval log")
    win_pct, wins, games = float(m.group(1)), int(m.group(2)), int(m.group(3))

    root = f.get("ROOT_OPTION_STATS", "")
    root_kinds = {}
    for part in root.split("|"):
        if ":" not in part:
            continue
        kind, rest = part.split(":", 1)
        d = {}
        for kv in rest.split(","):
            if "=" not in kv:
                continue
            k, v = kv.split("=")
            d[k] = v
        root_kinds[kind] = d
    attack = root_kinds.get("ATTACK", {})
    attach = root_kinds.get("ENERGY_ATTACH", {})

    cause_games = _games_from_win_by_cause(f.get("WIN_BY_CAUSE", ""))
    deckout_games = cause_games.get("DECK_OUT", 0)
    total_cause_games = sum(cause_games.values())

    prize_reward = f.get("PRIZE_REWARD_REACHED", "n/a")
    search_depth = f.get("SEARCH_MEAN_DEPTH", "n/a")
    mean_sec = f.get("OPP_MEAN_SEC_PER_GAME", "n/a")
    eval_sims = f.get("EVAL_SIMULATIONS_PER_MOVE", "n/a")
    candidate_ckpt = f.get("EVAL_CANDIDATE", "n/a")

    passed = win_pct > args.pass_threshold * 100.0
    verdict = "PASS" if passed else "FAIL"

    date = datetime.date.today().isoformat()
    entry = f"""
## {date} — Cycle: {args.label}

**Run**: checkpoint=`{candidate_ckpt}`, trained {args.epochs} main epochs (sims={args.sims_train}), opponent=sample_bot, eval sims={eval_sims}, games={games}. Source: `{args.eval_log}`.

- Candidate win rate (harness role-mapped): {win_pct:.1f}% ({wins}/{games}). {wins}+{games - wins}={games}.
- `ROOT_OPTION_STATS.ATTACK`: available={attack.get('available', 'n/a')}, chosen={attack.get('chosen', 'n/a')}.
- `ROOT_OPTION_STATS.ENERGY_ATTACH`: available={attach.get('available', 'n/a')}, chosen={attach.get('chosen', 'n/a')}.
- `WIN_BY_CAUSE` (games): {", ".join(f"{k}={v}" for k, v in cause_games.items())}. Sum={total_cause_games}.
- deck-out share: {deckout_games}/{total_cause_games} = {100 * deckout_games / total_cause_games if total_cause_games else 0:.1f}%.
- `PRIZE_REWARD_REACHED`: {prize_reward}.
- `SEARCH_MEAN_DEPTH`: {search_depth}.
- mean sec/game: {mean_sec}.

**Conclusion**: {verdict} — candidate win rate vs sample_bot is {win_pct:.1f}%, threshold {args.pass_threshold * 100:.0f}%.

---
"""
    with open(args.findings, "a", encoding="utf-8") as fh:
        fh.write(entry)

    print(f"CYCLE_RESULT={verdict} win_rate={win_pct:.1f}%({wins}/{games})", flush=True)


if __name__ == "__main__":
    main()
