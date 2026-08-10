"""Measurement-only driver for the DECK_DIFF_COEF sweep.

Runs N games of m2 self-play against a pinned frozen checkpoint. No training,
no checkpoint writes, no replay buffer. Seeds are fixed per game index so every
arm plays the same sequence of determinisations.

Usage:
    SIMULATIONS_PER_MOVE=20 DECK_DIFF_COEF=0.0 python sweep_deck_diff.py --games 20
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import time

# SIMULATIONS_PER_MOVE / DECK_DIFF_COEF are read at RLTRM2 import time, so the
# environment must already be set by the caller before this import runs.
import torch

import diag
import RLTRM2 as R


def pinned_checkpoint(name: str, stamp: str | None) -> str:
    folder = os.path.join("checkpoints", name)
    files = sorted(glob.glob(os.path.join(folder, "model_*.pth")))
    if not files:
        raise FileNotFoundError(f"No checkpoint in {folder}")
    if stamp is None:
        return files[-1]
    want = os.path.join(folder, f"model_{stamp}.pth")
    if want not in files:
        raise FileNotFoundError(f"{want} not found; have {[os.path.basename(f) for f in files]}")
    return want


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--checkpoint", default="2026-08-08_09-48",
                    help="checkpoint stamp to pin; default is the newest")
    ap.add_argument("--base-seed", type=int, default=20260808)
    ap.add_argument("--random-init", action="store_true",
                    help="skip checkpoint loading; use a freshly random-initialised, frozen network")
    args = ap.parse_args()

    # One dump covering the whole arm; verbose off so per-game noise stays out.
    diag.configure(enabled=True, verbose=False, dump_every_games=args.games)

    device = torch.device("cpu")
    if args.random_init:
        # Fixed, dedicated seed for the init draw only — independent of the
        # per-game seeding loop below.
        torch.manual_seed(args.base_seed)
    model = R.MyModel(128, 2, 256, 3, 1).to(device)
    if args.random_init:
        ckpt = "RANDOM_INIT (no checkpoint loaded)"
    else:
        ckpt = pinned_checkpoint("m2", args.checkpoint)
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    file_path = "M2Deck.xlsx"
    my_deck = R.pd.read_excel(file_path, header=None).iloc[:, 0].tolist()

    print(f"ARM_SIMULATIONS_PER_MOVE={R.SEARCH_COUNT}", flush=True)
    print(f"ARM_DECK_DIFF_COEF={R.DECK_DIFF_COEF}", flush=True)
    print(f"ARM_ATTACH_PRIOR_FLOOR={R.ATTACH_PRIOR_FLOOR}", flush=True)
    print(f"ARM_CHECKPOINT={ckpt}", flush=True)
    print(f"ARM_RANDOM_INIT={args.random_init}", flush=True)
    print(f"ARM_GAMES={args.games}", flush=True)
    print(f"ARM_BASE_SEED={args.base_seed}", flush=True)

    decisions = 0
    arm_start = time.perf_counter()

    with torch.inference_mode():
        for g in range(args.games):
            # Same seed per game index across every arm: identical deck shuffles
            # and identical determinisation draws, so arms differ only by coef.
            random.seed(args.base_seed + g)
            torch.manual_seed(args.base_seed + g)

            obs, _ = R.battle_start(my_deck, my_deck)
            diag.start_game()
            while obs["current"]["result"] < 0:
                prev_obs = obs
                yi = obs["current"]["yourIndex"]
                selected, _ = R.mcts_agent(obs, my_deck, model)
                decisions += 1
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
            diag.end_game()

    elapsed = time.perf_counter() - arm_start
    # Timing excludes model construction and checkpoint load, unlike the
    # mtime-derived figures recovered for the sims sweep.
    print(f"ARM_WALL_SEC={elapsed:.2f}", flush=True)
    print(f"ARM_DECISIONS={decisions}", flush=True)
    print(f"ARM_SEC_PER_MOVE={elapsed / decisions:.4f}" if decisions else "ARM_SEC_PER_MOVE=0",
          flush=True)
    print(f"ARM_SEC_PER_GAME={elapsed / args.games:.2f}", flush=True)


if __name__ == "__main__":
    main()
