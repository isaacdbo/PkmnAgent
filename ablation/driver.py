"""Autoresearch driver: train one arm per reward spec, score them on one panel.

This drives the project's real training entrypoint — `python RLTRM2.py` with
the environment knobs that already exist there (FAST_TEST, M2_ONLY,
SIMULATIONS_PER_MOVE, WARMUP_*, MAIN_*, SELF_PLAY_WORKERS, SKIP_CHECKPOINT_LOAD,
STOP_AFTER_MAIN_EPOCH) — rather than reimplementing a training loop. The only
thing it adds to that command line is `REWARD_SPEC=<arm>`, which
`ablation/rewards.py` reads.

Per arm:

  1. train      subprocess RLTRM2.py in a per-arm CHECKPOINT_ROOT, so arms
                cannot load each other's checkpoints or race on the same path
  2. evaluate   the newest m2 checkpoint against the pinned panel, via
                ablation.eval_runner
  3. record     games.jsonl + summary.json per arm

Then one comparison table across arms, on the metrics the ablation exists to
move: win rate (with Wilson intervals), deck-out rate, attack rate, turns to
win, WIN_BY_CAUSE.

Every arm is trained with the same seed, the same game counts and the same
panel, and a fresh network (SKIP_CHECKPOINT_LOAD=1); the reward spec is the
only difference between arms. Runs are resumable: an arm whose summary.json
already exists is skipped unless --force is given, so a long sweep can be
restarted without redoing finished work.

Usage:
    python -m ablation.driver --arms baseline,terminal_only,deckout_penalty \\
        --out-dir results/ablation --games 20
    python -m ablation.driver --dry-run          # print the plan, run nothing
    python -m ablation.driver --eval-only ...    # score existing checkpoints
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

from ablation import metrics, rewards
from ablation.metrics import GameRecord, Summary

DEFAULT_ARMS = "baseline,terminal_only,deckout_penalty,turns_to_win_mild"

# Training configuration applied to every arm. These are RLTRM2's own env
# knobs; the defaults here are a deliberately small "does the whole loop run
# end to end" size, meant to be raised for a real sweep.
DEFAULT_TRAIN_ENV = {
    "FAST_TEST": "0",
    "M2_ONLY": "1",
    "SKIP_CHECKPOINT_LOAD": "1",
    "SIMULATIONS_PER_MOVE": "20",
    "WARMUP_EPOCHS": "1",
    "MAIN_EPOCHS": "1",
    "SELF_PLAY_WORKERS": "1",
    "SELF_PLAY_BASE_SEED": "20260810",
    "STOP_AFTER_MAIN_EPOCH": "1",
}


def _env_for_arm(arm: str, checkpoint_root: str, overrides: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(DEFAULT_TRAIN_ENV)
    env.update(overrides)
    env["REWARD_SPEC"] = arm
    env["CHECKPOINT_ROOT"] = checkpoint_root
    # Keep worker processes single-threaded: the arms are run one after
    # another and BLAS threads fighting over the same cores makes wall-clock
    # comparisons between arms meaningless.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def latest_checkpoint(checkpoint_root: str, agent: str = "m2") -> str | None:
    pattern = os.path.join(checkpoint_root, agent, "model_*.pth")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def train_arm(arm: str, arm_dir: str, overrides: dict[str, str], timeout: int | None) -> dict:
    """Run RLTRM2.py for one arm. Returns a receipt of what actually ran."""
    checkpoint_root = os.path.join(arm_dir, "checkpoints")
    os.makedirs(checkpoint_root, exist_ok=True)
    env = _env_for_arm(arm, checkpoint_root, overrides)
    log_path = os.path.join(arm_dir, "train.log")

    cmd = [sys.executable, "-u", "RLTRM2.py"]
    started = time.time()
    print(f"[{arm}] training: {' '.join(cmd)} (log: {log_path})", flush=True)

    with open(log_path, "w") as log:
        log.write(f"# REWARD_SPEC={arm}\n")
        log.write(f"# {rewards.get(arm).describe()}\n")
        for key in sorted(set(DEFAULT_TRAIN_ENV) | set(overrides) | {"CHECKPOINT_ROOT"}):
            log.write(f"# {key}={env.get(key)}\n")
        # Flush before handing the fd to the child, or the header lands after
        # the training output it is supposed to describe.
        log.flush()
        try:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False)
            returncode, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            returncode, timed_out = None, True

    receipt = {
        "arm": arm,
        "cmd": cmd,
        "env": {k: env.get(k) for k in sorted(set(DEFAULT_TRAIN_ENV) | set(overrides)
                                              | {"REWARD_SPEC", "CHECKPOINT_ROOT"})},
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": round(time.time() - started, 1),
        "log": os.path.relpath(log_path),
        "checkpoint": latest_checkpoint(checkpoint_root),
    }
    with open(os.path.join(arm_dir, "train_receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)

    status = "timeout" if timed_out else f"rc={returncode}"
    print(f"[{arm}] training finished ({status}, {receipt['wall_seconds']}s), "
          f"checkpoint={receipt['checkpoint']}", flush=True)
    return receipt


def evaluate_arm(arm: str, arm_dir: str, checkpoint: str, panel: str,
                 games: int, base_seed: int, sims: str | None) -> list[GameRecord]:
    """Score one arm's checkpoint on the pinned panel, in-process."""
    if sims is not None:
        # Set before the first import of RLTRM2, which reads it once into
        # SEARCH_COUNT at import time. It is the same value for every arm, so
        # a later arm re-reading the already-imported module is not a problem
        # — but a per-arm value would silently apply only to the first arm.
        os.environ["SIMULATIONS_PER_MOVE"] = sims
    # Imported lazily: these pull in torch and the compiled engine, which is
    # not importable on the macOS host where --dry-run is useful.
    from ablation import eval_runner

    records = eval_runner.run_panel(
        candidate_path=checkpoint,
        candidate_random_init=False,
        candidate_deck="M2Deck.xlsx",
        panel=panel,
        games=games,
        base_seed=base_seed,
        context={"reward_spec": arm, "run": arm},
    )
    eval_runner.write_results(records, os.path.join(arm_dir, "eval"))
    return records


def load_arm_records(arm_dir: str) -> list[GameRecord] | None:
    path = os.path.join(arm_dir, "eval", "games.jsonl")
    if not os.path.exists(path):
        return None
    return metrics.read_games_jsonl(path)


def compare(arm_summaries: dict[str, Summary],
            per_arm_opponent: dict[str, dict[str, Summary]]) -> str:
    out = ["# Reward ablation comparison", ""]
    out.append("## Overall, across the whole panel")
    out.append("")
    out.append(metrics.render_markdown_table(arm_summaries, label="arm"))
    out.append("")
    out.append("## Per panel opponent")
    out.append("")
    for opponent in sorted({o for arms in per_arm_opponent.values() for o in arms}):
        rows = {
            arm: by_opp[opponent]
            for arm, by_opp in per_arm_opponent.items()
            if opponent in by_opp
        }
        out.append(f"### vs {opponent}")
        out.append("")
        out.append(metrics.render_markdown_table(rows, label="arm"))
        out.append("")
    out.append("## Reward specs")
    out.append("")
    for arm in arm_summaries:
        spec = rewards.get(arm)
        out.append(f"- **{arm}** — {spec.description} (`{spec.describe()}`)")
    out.append("")
    out.append(
        "Win rates are shown with Wilson 95% intervals. At the game counts a "
        "local run can afford, overlapping intervals are the normal outcome: "
        "the deck-out rate and attack rate move first and are what these arms "
        "are actually steering."
    )
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default=DEFAULT_ARMS,
                    help=f"comma list of reward specs (choices: {', '.join(sorted(rewards.REGISTRY))})")
    ap.add_argument("--out-dir", default=os.path.join("results", "ablation"))
    ap.add_argument("--panel", default=None,
                    help="panel members to score against (default: ablation.eval_runner.DEFAULT_PANEL)")
    ap.add_argument("--games", type=int, default=20, help="games per panel opponent")
    ap.add_argument("--base-seed", type=int, default=20260809)
    ap.add_argument("--eval-sims", default=None,
                    help="SIMULATIONS_PER_MOVE for evaluation (default: leave as-is)")
    ap.add_argument("--train-env", action="append", default=[], metavar="KEY=VALUE",
                    help="override a training env knob; repeatable")
    ap.add_argument("--train-timeout", type=int, default=None,
                    help="seconds before a training arm is abandoned (default: no limit)")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; evaluate the checkpoint already in each arm directory")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--force", action="store_true",
                    help="re-run arms that already have results")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        rewards.get(arm)  # fail fast on a typo, before any training starts

    overrides = {}
    for item in args.train_env:
        if "=" not in item:
            ap.error(f"--train-env expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k] = v

    panel = args.panel
    if panel is None:
        # Read the default without importing torch, so --dry-run works on a
        # host that cannot load the engine.
        panel = "random,first,iono_rule"

    print(f"arms: {', '.join(arms)}")
    print(f"panel: {panel} ({args.games} games per opponent, base seed {args.base_seed})")
    print(f"out-dir: {args.out_dir}")
    print("training env (per arm, plus REWARD_SPEC and CHECKPOINT_ROOT):")
    for k, v in sorted({**DEFAULT_TRAIN_ENV, **overrides}.items()):
        print(f"  {k}={v}")
    for arm in arms:
        print(f"  - {arm}: {rewards.get(arm).describe()}")
    if args.dry_run:
        print("\ndry run: nothing executed")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    arm_records: dict[str, list[GameRecord]] = {}

    for arm in arms:
        arm_dir = os.path.join(args.out_dir, arm)
        os.makedirs(arm_dir, exist_ok=True)

        existing = load_arm_records(arm_dir)
        if existing and not args.force:
            print(f"[{arm}] reusing existing results ({len(existing)} games); --force to redo")
            arm_records[arm] = existing
            continue

        if args.eval_only:
            checkpoint = latest_checkpoint(os.path.join(arm_dir, "checkpoints"))
            if checkpoint is None:
                print(f"[{arm}] no checkpoint under {arm_dir}/checkpoints; skipping")
                continue
        else:
            receipt = train_arm(arm, arm_dir, overrides, args.train_timeout)
            checkpoint = receipt["checkpoint"]
            if checkpoint is None:
                print(f"[{arm}] training produced no checkpoint; skipping evaluation")
                continue

        arm_records[arm] = evaluate_arm(
            arm, arm_dir, checkpoint, panel, args.games, args.base_seed, args.eval_sims
        )

    if not arm_records:
        print("no arms produced results")
        return

    arm_summaries = {arm: metrics.summarize(r) for arm, r in arm_records.items()}
    per_arm_opponent = {
        arm: {str(k[0]): v for k, v in metrics.summarize_by(r, "opponent").items()}
        for arm, r in arm_records.items()
    }

    report = compare(arm_summaries, per_arm_opponent)
    report_path = os.path.join(args.out_dir, "COMPARISON.md")
    with open(report_path, "w") as f:
        f.write(report)

    with open(os.path.join(args.out_dir, "comparison.json"), "w") as f:
        json.dump(
            {
                "arms": {a: s.to_dict() for a, s in arm_summaries.items()},
                "by_opponent": {
                    a: {o: s.to_dict() for o, s in by_opp.items()}
                    for a, by_opp in per_arm_opponent.items()
                },
                "specs": {a: rewards.get(a).to_dict() for a in arm_summaries},
            },
            f, indent=2, sort_keys=True,
        )

    metrics.write_summary_csv(arm_summaries, os.path.join(args.out_dir, "comparison.csv"), label="arm")

    print()
    print(metrics.render_table(arm_summaries, label="arm"))
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
