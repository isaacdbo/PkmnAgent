"""Verifies diag.py's windowed (non-verbose) dump path.

FAST_TEST=True always forces DIAG_VERBOSE=True (dump after every game), so
the DIAG_DUMP_EVERY_GAMES=25 windowed path (used by the real, FAST_TEST=False
run) has never been exercised. This driver reproduces that path directly:
verbose=False, a fixed dump interval decoupled from total game count, and
enough games in one process to cross the interval multiple times — so we can
confirm dumps fire on schedule and per-window counters reset rather than
accumulate.

Inference-only, no training, no checkpoint writes. Random-init frozen net by
default (pass --checkpoint <stamp> to pin checkpoints/m2/model_<stamp>.pth
instead).

Usage:
    SIMULATIONS_PER_MOVE=20 python probe_dump_window.py --games 50 --dump-every 25
"""
from __future__ import annotations

import argparse
import os
import random
import time

# SIMULATIONS_PER_MOVE / ATTACH_PRIOR_FLOOR are read at RLTRM2 import time, so
# the environment must already be set by the caller before this import runs.
import torch

import diag
import RLTRM2 as R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--dump-every", type=int, default=25)
    ap.add_argument("--base-seed", type=int, default=20260808)
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoints/m2/model_<stamp>.pth to pin; default random-init frozen net")
    args = ap.parse_args()

    diag.configure(enabled=True, verbose=False, dump_every_games=args.dump_every)

    device = torch.device("cpu")
    if args.checkpoint is None:
        torch.manual_seed(args.base_seed)
    model = R.MyModel(128, 2, 256, 3, 1).to(device)
    if args.checkpoint is not None:
        path = os.path.join("checkpoints", "m2", f"model_{args.checkpoint}.pth")
        model.load_state_dict(torch.load(path, map_location=device))
        ckpt = path
    else:
        ckpt = "RANDOM_INIT (no checkpoint loaded)"
    model.eval()

    my_deck = R.pd.read_excel("M2Deck.xlsx", header=None).iloc[:, 0].tolist()

    print(f"PROBE_SIMULATIONS_PER_MOVE={R.SEARCH_COUNT}", flush=True)
    print(f"PROBE_ATTACH_PRIOR_FLOOR={R.ATTACH_PRIOR_FLOOR}", flush=True)
    print(f"PROBE_CHECKPOINT={ckpt}", flush=True)
    print(f"PROBE_GAMES={args.games}", flush=True)
    print(f"PROBE_DUMP_EVERY={args.dump_every}", flush=True)
    print(f"PROBE_BASE_SEED={args.base_seed}", flush=True)

    arm_start = time.perf_counter()

    with torch.inference_mode():
        for g in range(args.games):
            # Same seed-per-index scheme as sweep_deck_diff.py, for reproducibility.
            random.seed(args.base_seed + g)
            torch.manual_seed(args.base_seed + g)

            game_t0 = time.perf_counter()
            obs, _ = R.battle_start(my_deck, my_deck)
            diag.start_game()
            while obs["current"]["result"] < 0:
                prev_obs = obs
                yi = obs["current"]["yourIndex"]
                selected, _ = R.mcts_agent(obs, my_deck, model)
                obs = R.battle_select(selected)

                prev_obs_obj = R.to_observation_class(prev_obs)
                next_obs_obj = R.to_observation_class(obs)
                attach_legal, attach_made, board_affecting, shuffle_with_resources = \
                    R._diag_step_features(prev_obs_obj, selected, next_obs_obj)
                diag.record_turn_step(
                    turn=prev_obs_obj.current.turn,
                    player_index=yi,
                    attach_legal=attach_legal,
                    attach_made=attach_made,
                    board_affecting=board_affecting,
                    shuffle_with_resources=shuffle_with_resources,
                )
            R.battle_finish()
            final_obs_obj = R.to_observation_class(obs)
            diag.record_game_result(
                final_turn=final_obs_obj.current.turn,
                reason_code=R._extract_result_reason(final_obs_obj),
                did_draw=(obs["current"]["result"] == 2),
            )
            diag.record_true_result(result=obs["current"]["result"])
            diag.end_game()
            print(f"PROBE_GAME_DONE={g + 1}/{args.games} sec={time.perf_counter() - game_t0:.2f}", flush=True)

    elapsed = time.perf_counter() - arm_start
    print(f"PROBE_WALL_SEC={elapsed:.2f}", flush=True)
    print(f"PROBE_SEC_PER_GAME={elapsed / args.games:.2f}", flush=True)


if __name__ == "__main__":
    main()
