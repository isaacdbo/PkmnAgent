"""Aggregate diag.py DIAG_DUMP blocks from a run log into summary stats.

Handles both plain diag logs (diagnostic/diag_*.log) and timestamp-prefixed
logs (e.g. `python -u RLTRM2.py 2>&1 | while read line; do printf '%s %s\\n'
"$(date +%s.%N)" "$line"; done > run.log`) — a leading numeric token followed
by a space is stripped before parsing.

Usage:
    python parse_diag_log.py <path-to-log> [<path-to-log> ...]
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

_PRIOR_HIST_LABELS = ["<0.001", "0.001-0.005", "0.005-0.01", "0.01-0.03", "0.03-0.1", "0.1-0.3", "0.3-1.0", "1.0"]


def _strip_timestamp(line: str) -> str:
    if line and line[0].isdigit():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            return parts[1]
    return line


def parse_blocks(path: str) -> list[dict]:
    blocks: list[dict] = []
    cur: dict = {}
    in_block = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = _strip_timestamp(raw).strip()
            if line == "=== DIAG_DUMP_BEGIN ===":
                in_block = True
                cur = {}
                continue
            if line == "=== DIAG_DUMP_END ===":
                in_block = False
                blocks.append(cur)
                continue
            if in_block and "=" in line:
                k, v = line.split("=", 1)
                cur[k] = v
    return blocks


def aggregate(blocks: list[dict]) -> dict:
    prize_num = prize_den = 0
    cause_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"W": 0, "L": 0, "D": 0})
    p0 = p1 = draws = 0
    root: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    game_len: dict[str, int] = defaultdict(int)
    attach_hist: dict[str, dict[str, int]] = {"special": defaultdict(int), "plain": defaultdict(int)}

    for b in blocks:
        m = re.match(r"([\d.]+)%\((\d+)/(\d+)\)", b.get("PRIZE_REWARD_REACHED", ""))
        if m:
            prize_num += int(m.group(2))
            prize_den += int(m.group(3))

        wbc = b.get("WIN_BY_CAUSE", "")
        if wbc and wbc != "NONE":
            for part in wbc.split("|"):
                cause, rest = part.split(":")
                for kv in rest.split(","):
                    k, v = kv.split("=")
                    cause_totals[cause][k] += int(v)

        tr = b.get("TRUE_RESULT_TALLY", "")
        if tr:
            for kv in tr.split(","):
                k, v = kv.split("=")
                if k == "P0_WINS":
                    p0 += int(v)
                elif k == "P1_WINS":
                    p1 += int(v)
                elif k == "DRAWS":
                    draws += int(v)

        ros = b.get("ROOT_OPTION_STATS", "")
        if ros and ros != "NONE":
            for part in ros.split("|"):
                kind, rest = part.split(":", 1)
                for kv in rest.split(","):
                    k, v = kv.split("=")
                    if k in ("available", "chosen", "not_chosen"):
                        root[kind][k] += int(v)
                    elif k == "zero_visit_rate":
                        num, _den = re.search(r"\((\d+)/(\d+)\)", v).groups()
                        root[kind]["zero_visit"] += int(num)

        glh = b.get("GAME_LENGTH_HIST", "")
        if glh:
            for kv in glh.split("|"):
                k, v = kv.split(":")
                game_len[k] += int(v)

        apd = b.get("ATTACH_PRIOR_DIST", "")
        if apd:
            for group in apd.split("|"):
                gname = group.split(":")[0]
                m2 = re.search(r"attach_prior\[.*?hist=\(([^)]*)\)\]", group)
                if m2 and gname in attach_hist:
                    for kv in m2.group(1).split(","):
                        label, cnt = kv.rsplit(":", 1)
                        attach_hist[gname][label] += int(cnt)

    return {
        "n_games": len(blocks),
        "prize_num": prize_num,
        "prize_den": prize_den,
        "cause_totals": dict(cause_totals),
        "true_result": {"P0_WINS": p0, "P1_WINS": p1, "DRAWS": draws},
        "root_option_stats": {k: dict(v) for k, v in root.items()},
        "game_length_hist": dict(game_len),
        "attach_prior_hist": {k: dict(v) for k, v in attach_hist.items()},
    }


def report(agg: dict) -> None:
    print(f"Diag-tracked games: {agg['n_games']}")

    if agg["prize_den"]:
        pct = 100 * agg["prize_num"] / agg["prize_den"]
        print(f"\nPRIZE_REWARD_REACHED: {agg['prize_num']}/{agg['prize_den']} = {pct:.2f}%")

    cause = agg["cause_totals"]
    if cause:
        total_wl = sum(c["W"] for c in cause.values())
        print(f"\nWIN_BY_CAUSE: {cause}")
        print(f"  sum(W)={total_wl}, sum(L)={sum(c['L'] for c in cause.values())}, "
              f"sum(D)={sum(c['D'] for c in cause.values())}")
        deckout_w = cause.get("DECK_OUT", {}).get("W", 0)
        if total_wl:
            print(f"  deck-out share = {deckout_w}/{total_wl} = {100 * deckout_w / total_wl:.1f}%")

    tr = agg["true_result"]
    print(f"\nTRUE_RESULT_TALLY: P0_WINS={tr['P0_WINS']}, P1_WINS={tr['P1_WINS']}, DRAWS={tr['DRAWS']}")

    root = agg["root_option_stats"]
    if root:
        print("\nROOT_OPTION_STATS:")
        for kind, d in root.items():
            avail = d.get("available", 0)
            zv = d.get("zero_visit", 0)
            if avail:
                print(f"  {kind}: available={avail}, chosen={d.get('chosen', 0)}, "
                      f"not_chosen={d.get('not_chosen', 0)}, zero_visit={zv} ({100 * zv / avail:.1f}%)")
            else:
                print(f"  {kind}: available=0")

    if agg["game_length_hist"]:
        print(f"\nGAME_LENGTH_HIST: {agg['game_length_hist']}")

    ah = agg["attach_prior_hist"]
    if any(ah.get(g) for g in ("special", "plain")):
        print("\nATTACH_PRIOR_DIST histogram (proves ATTACH_PRIOR_FLOOR active if low buckets are empty):")
        for gname in ("special", "plain"):
            row = ah.get(gname, {})
            print(f"  {gname}: " + str({label: row.get(label, 0) for label in _PRIOR_HIST_LABELS}))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        print(f"=== {path} ===")
        blocks = parse_blocks(path)
        agg = aggregate(blocks)
        report(agg)
        print()


if __name__ == "__main__":
    main()
