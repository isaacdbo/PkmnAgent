"""Turn a pair of eval_panel.py run directories into a BEFORE/AFTER report.

Reads the `OPP_<name>_CANDIDATE:` summary lines that eval_panel.py prints, pairs
them by opponent, and emits per-opponent win rates with Wilson 95% intervals
plus a two-sided Fisher exact test on the BEFORE-vs-AFTER 2x2 table.

The Wilson interval says how precisely each arm is measured. It does NOT answer
"did this change anything" — two overlapping intervals can still come from
significantly different rates. That question is the Fisher test's job, so both
are reported and the verdict keys off the p-value.

Usage:
    python compare_before_after.py --before LOG [LOG ...] --after LOG [LOG ...]
"""

import argparse
import math
import re
from typing import NamedTuple

SUMMARY_RE = re.compile(
    r"^OPP_(?P<opp>\S+?)_CANDIDATE:\s+wins=(?P<wins>\d+)\s+losses=(?P<losses>\d+)\s+"
    r"draws=(?P<draws>\d+)\s+win_rate=(?P<rate>[\d.]+)%\((?P<w>\d+)/(?P<n>\d+)\)",
    re.M,
)


class Arm(NamedTuple):
    opponent: str
    wins: int
    games: int
    source: str


def parse(paths: list[str]) -> dict[str, Arm]:
    arms: dict[str, Arm] = {}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for m in SUMMARY_RE.finditer(text):
            opp = m.group("opp")
            arm = Arm(opp, int(m.group("w")), int(m.group("n")), path)
            if opp in arms:
                raise SystemExit(
                    f"opponent {opp!r} appears twice ({arms[opp].source}, {path}); "
                    "refusing to guess which run is authoritative"
                )
            arms[opp] = arm
    return arms


def wilson(wins: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    p = wins / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]].

    Sums the hypergeometric probability of every table with the same margins
    that is no more likely than the observed one.
    """
    row1, row2 = a + b, c + d
    col1, total = a + c, a + b + c + d
    if total == 0:
        return 1.0

    def prob(x: int) -> float:
        return (
            math.comb(row1, x)
            * math.comb(row2, col1 - x)
            / math.comb(total, col1)
        )

    observed = prob(a)
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    # 1e-9 slack: tables that are equally likely in exact arithmetic can differ
    # in the last float bits, and dropping them would understate the p-value.
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", nargs="+", required=True)
    ap.add_argument("--after", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    before, after = parse(args.before), parse(args.after)
    shared = [o for o in before if o in after]
    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))

    print("| opponent | before | wilson95 | after | wilson95 | delta | fisher p |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    verdicts = []
    for opp in sorted(shared):
        b, a = before[opp], after[opp]
        bl, bh = wilson(b.wins, b.games)
        al, ah = wilson(a.wins, a.games)
        br, ar = b.wins / b.games, a.wins / a.games
        p = fisher_exact_two_sided(
            a.wins, a.games - a.wins, b.wins, b.games - b.wins
        )
        print(
            f"| {opp} "
            f"| {br:.1%} ({b.wins}/{b.games}) | [{bl:.1%}, {bh:.1%}] "
            f"| {ar:.1%} ({a.wins}/{a.games}) | [{al:.1%}, {ah:.1%}] "
            f"| {(ar - br) * 100:+.1f}pp | {p:.4f} |"
        )
        verdicts.append((opp, (ar - br) * 100, p))

    print()
    for opp, delta, p in verdicts:
        if p < args.alpha and delta > 0:
            note = f"SIGNIFICANT IMPROVEMENT (p={p:.4f} < {args.alpha})"
        elif p < args.alpha and delta < 0:
            note = f"SIGNIFICANT REGRESSION (p={p:.4f} < {args.alpha})"
        else:
            note = f"not significant at alpha={args.alpha} (p={p:.4f})"
        print(f"VERDICT {opp}: {delta:+.1f}pp, {note}")

    for opp in only_before:
        print(f"WARNING: {opp} has a BEFORE arm but no AFTER arm; not compared")
    for opp in only_after:
        print(f"WARNING: {opp} has an AFTER arm but no BEFORE arm; not compared")
    if not shared:
        raise SystemExit("no opponent appears in both BEFORE and AFTER")


if __name__ == "__main__":
    main()
