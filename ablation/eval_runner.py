"""Run the pinned eval panel and emit structured metrics, not just log lines.

Wraps `eval_panel`'s panel definitions — the same Side objects, the same
builders, the same alternating-seat rule — but plays the games through a loop
that records one `metrics.GameRecord` per game and writes them as JSONL
alongside a summary. That gives the ablation driver something to compare arms
on without re-parsing its own stdout.

The loop is written out here rather than calling `eval_panel.run_matchup`
because run_matchup only prints; it keeps no per-game structure to return. The
two loops must stay behaviourally identical in the parts that decide a result:
alternating seats by game parity, per-game seeding from `base_seed + g`, and
attributing the win by identity rather than by raw seat index.

Attack-rate is measured here and nowhere else: at every candidate decision
where at least one ATTACK option is legal, the loop records whether the
candidate actually chose one. That is a property of the trajectory, so it is
defined for scripted panel bots too — unlike the MCTS-internal ATTACK counters
in `diag.ROOT_OPTION_STATS`, which only exist when a search ran.

Usage:
    SIMULATIONS_PER_MOVE=20 python -m ablation.eval_runner \\
        --candidate checkpoints/m2/model_2026-08-11_00-16.pth \\
        --panel random,first,iono_rule --games 20 \\
        --out-dir results/ablation/baseline/eval
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

import eval_panel as EP
import RLTRM2 as R
from ablation import metrics
from ablation.metrics import GameRecord

# The panel the ablation is scored against unless told otherwise: a random
# baseline, a degenerate deterministic baseline that exposes "wins only
# because the opponent never does anything", and a real leaderboard rule bot.
DEFAULT_PANEL = "random,first,iono_rule"


def build_opponent(name: str, device) -> "EP.Side":
    if name == "old_m2":
        return EP.build_old_m2(device)
    if name == "iono_rule":
        return EP.build_iono_rule()
    return EP.build_scripted_panel(name)


def _attack_options(obs_obj) -> set[int]:
    """Indices of legal ATTACK options at this decision point."""
    if obs_obj.select is None:
        return set()
    return {
        i for i, option in enumerate(obs_obj.select.option)
        if R._option_type(option) == R.OptionType.ATTACK
    }


def run_matchup(
    candidate: "EP.Side",
    opponent: "EP.Side",
    opponent_name: str,
    n_games: int,
    base_seed: int,
    *,
    verbose: bool = True,
    context: dict | None = None,
) -> list[GameRecord]:
    records: list[GameRecord] = []
    context = context or {}

    with torch.inference_mode():
        for g in range(n_games):
            random.seed(base_seed + g)
            torch.manual_seed(base_seed + g)

            # Alternate who moves first, so a first-player advantage cannot be
            # mistaken for a reward-arm effect.
            if g % 2 == 0:
                by_player = [("candidate", candidate), ("opponent", opponent)]
            else:
                by_player = [("opponent", opponent), ("candidate", candidate)]

            t0 = time.perf_counter()
            obs, _ = R.battle_start(by_player[0][1].deck, by_player[1][1].deck)

            attack_legal = 0
            attack_chosen = 0
            prize_events = [0, 0]

            while obs["current"]["result"] < 0:
                yi = obs["current"]["yourIndex"]
                role, side = by_player[yi]
                obs_obj = R.to_observation_class(obs)

                attack_indices: set[int] = set()
                if role == "candidate":
                    attack_indices = _attack_options(obs_obj)
                    if attack_indices:
                        attack_legal += 1

                selected = side.act(obs)

                if attack_indices and any(i in attack_indices for i in selected):
                    attack_chosen += 1

                obs = R.battle_select(selected)

                for log in R.to_observation_class(obs).logs:
                    if (log.type == R.LogType.MOVE_CARD and
                            log.fromArea == R.AreaType.PRIZE and
                            log.toArea == R.AreaType.HAND):
                        prize_events[log.playerIndex] += 1

            R.battle_finish()
            final_obs = R.to_observation_class(obs)
            reason = R._extract_result_reason(final_obs)
            result = obs["current"]["result"]
            candidate_seat = 0 if by_player[0][0] == "candidate" else 1

            if result == 2:
                outcome = "draw"
            else:
                outcome = "win" if result == candidate_seat else "loss"

            record = GameRecord(
                game_index=g + 1,
                opponent=opponent_name,
                outcome=outcome,
                cause=EP._CAUSE_LABEL.get(reason, "other") if reason is not None else None,
                final_turn=final_obs.current.turn,
                attack_legal_decisions=attack_legal,
                attack_chosen_decisions=attack_chosen,
                candidate_prizes=prize_events[candidate_seat],
                opponent_prizes=prize_events[1 - candidate_seat],
                seconds=time.perf_counter() - t0,
                **context,
            )
            records.append(record)

            if verbose:
                print(
                    f"EVAL_GAME_DONE={g + 1}/{n_games} opponent={opponent_name} "
                    f"outcome={outcome} cause={record.cause} turn={record.final_turn} "
                    f"attack_rate={_pct(attack_chosen, attack_legal)} "
                    f"cand_prizes={record.candidate_prizes} opp_prizes={record.opponent_prizes} "
                    f"sec={record.seconds:.2f}",
                    flush=True,
                )

    return records


def _pct(numer: int, denom: int) -> str:
    return "n/a" if not denom else f"{100 * numer / denom:.1f}%"


def run_panel(
    *,
    candidate_path: str | None,
    candidate_random_init: bool,
    candidate_deck: str,
    panel: str,
    games: int,
    base_seed: int,
    context: dict | None = None,
    verbose: bool = True,
) -> list[GameRecord]:
    device = torch.device("cpu")
    if candidate_random_init:
        torch.manual_seed(base_seed)
    model = R.MyModel(128, 2, 256, 3, 1).to(device)
    if not candidate_random_init:
        model.load_state_dict(torch.load(candidate_path, map_location=device))
    model.eval()
    deck = R.pd.read_excel(candidate_deck, header=None).iloc[:, 0].tolist()
    candidate = EP.Side("candidate", deck, model)

    context = dict(context or {})
    context.setdefault("checkpoint", "RANDOM_INIT" if candidate_random_init else candidate_path)

    print(f"EVAL_CANDIDATE={context['checkpoint']}", flush=True)
    print(f"EVAL_CANDIDATE_DECK={candidate_deck}", flush=True)
    print(f"EVAL_SIMULATIONS_PER_MOVE={R.SEARCH_COUNT}", flush=True)
    print(f"EVAL_GAMES_PER_OPPONENT={games}", flush=True)
    print(f"EVAL_BASE_SEED={base_seed}", flush=True)
    print(f"EVAL_PANEL={panel}", flush=True)
    if context.get("reward_spec"):
        print(f"REWARD_SPEC={context['reward_spec']}", flush=True)

    all_records: list[GameRecord] = []
    for name in [p.strip() for p in panel.split(",") if p.strip()]:
        if name not in EP._PANEL_CHOICES:
            raise ValueError(f"Unknown panel member: {name} (choices: {EP._PANEL_CHOICES})")
        opponent = build_opponent(name, device)
        print(f"\n=== PANEL: candidate vs {name} ===", flush=True)
        all_records.extend(
            run_matchup(candidate, opponent, name, games, base_seed,
                        verbose=verbose, context=context)
        )
    return all_records


def write_results(records: list[GameRecord], out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    metrics.write_games_jsonl(records, os.path.join(out_dir, "games.jsonl"))

    by_opponent = {str(k[0]): v for k, v in metrics.summarize_by(records, "opponent").items()}
    overall = metrics.summarize(records)

    payload = {
        "overall": overall.to_dict(),
        "by_opponent": {k: v.to_dict() for k, v in by_opponent.items()},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    metrics.write_summary_csv(by_opponent, os.path.join(out_dir, "summary.csv"), label="opponent")

    table = metrics.render_table(by_opponent, label="opponent")
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(table + "\n")
    print()
    print(table, flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidate", help="checkpoint .pth to evaluate")
    ap.add_argument("--candidate-random-init", action="store_true",
                    help="evaluate a frozen randomly-initialised network instead")
    ap.add_argument("--candidate-deck", default="M2Deck.xlsx")
    ap.add_argument("--panel", default=DEFAULT_PANEL,
                    help=f"comma list from {EP._PANEL_CHOICES} (default: {DEFAULT_PANEL})")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=20260809)
    ap.add_argument("--out-dir", default=os.path.join("results", "eval"))
    ap.add_argument("--reward-spec", default=os.environ.get("REWARD_SPEC"),
                    help="label only: records which reward arm produced the checkpoint")
    args = ap.parse_args()

    if not args.candidate_random_init and not args.candidate:
        ap.error("--candidate is required unless --candidate-random-init is set")

    records = run_panel(
        candidate_path=args.candidate,
        candidate_random_init=args.candidate_random_init,
        candidate_deck=args.candidate_deck,
        panel=args.panel,
        games=args.games,
        base_seed=args.base_seed,
        context={"reward_spec": args.reward_spec} if args.reward_spec else None,
    )
    write_results(records, args.out_dir)


if __name__ == "__main__":
    main()
