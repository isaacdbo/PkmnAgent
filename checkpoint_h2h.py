"""Head-to-head evaluation between two m2 checkpoints on the same deck.

Inference-only, no training. Alternates who plays first each game so neither
checkpoint gets a first-player edge. Win/loss is attributed by checkpoint
identity (not raw player index), since which side is P0/P1 flips every game.

Usage:
    SIMULATIONS_PER_MOVE=20 python checkpoint_h2h.py \\
        --new checkpoints/m2/model_2026-08-09_03-38.pth \\
        --old checkpoints/m2/model_2026-08-08_09-48.pth \\
        --games 40
"""
from __future__ import annotations

import argparse
import random
import time

import torch

import diag
import RLTRM2 as R

_CAUSE_LABEL = {1: "prize", 2: "deckout", 3: "other", 4: "other"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="checkpoint .pth for the 'new' side")
    ap.add_argument("--old", required=True, help="checkpoint .pth for the 'old' side")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=20260809)
    args = ap.parse_args()

    diag.configure(enabled=True, verbose=False, dump_every_games=args.games)

    device = torch.device("cpu")
    model_new = R.MyModel(128, 2, 256, 3, 1).to(device)
    model_new.load_state_dict(torch.load(args.new, map_location=device))
    model_new.eval()

    model_old = R.MyModel(128, 2, 256, 3, 1).to(device)
    model_old.load_state_dict(torch.load(args.old, map_location=device))
    model_old.eval()

    deck = R.pd.read_excel("M2Deck.xlsx", header=None).iloc[:, 0].tolist()

    print(f"H2H_SIMULATIONS_PER_MOVE={R.SEARCH_COUNT}", flush=True)
    print(f"H2H_NEW={args.new}", flush=True)
    print(f"H2H_OLD={args.old}", flush=True)
    print(f"H2H_GAMES={args.games}", flush=True)
    print(f"H2H_BASE_SEED={args.base_seed}", flush=True)

    stats = {
        side: {"wins": 0, "losses": 0, "draws": 0,
               "prize_win": 0, "prize_loss": 0,
               "deckout_win": 0, "deckout_loss": 0,
               "other_win": 0, "other_loss": 0}
        for side in ("new", "old")
    }

    t0 = time.perf_counter()
    with torch.inference_mode():
        for g in range(args.games):
            random.seed(args.base_seed + g)
            torch.manual_seed(args.base_seed + g)

            if g % 2 == 0:
                by_player = [("new", model_new), ("old", model_old)]
            else:
                by_player = [("old", model_old), ("new", model_new)]

            game_t0 = time.perf_counter()
            obs, _ = R.battle_start(deck, deck)
            diag.start_game()
            while obs["current"]["result"] < 0:
                prev_obs = obs
                yi = obs["current"]["yourIndex"]
                _name, model = by_player[yi]
                selected, _ = R.mcts_agent(obs, deck, model)
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
            reason = R._extract_result_reason(final_obs_obj)
            result = obs["current"]["result"]
            did_draw = (result == 2)

            diag.record_game_result(final_turn=final_obs_obj.current.turn, reason_code=reason, did_draw=did_draw)
            diag.record_true_result(result=result)
            diag.end_game()

            cause = _CAUSE_LABEL.get(reason, "other")
            game_sec = time.perf_counter() - game_t0

            if did_draw:
                stats["new"]["draws"] += 1
                stats["old"]["draws"] += 1
            else:
                winner_side = by_player[result][0]
                loser_side = by_player[1 - result][0]
                stats[winner_side]["wins"] += 1
                stats[loser_side]["losses"] += 1
                stats[winner_side][f"{cause}_win"] += 1
                stats[loser_side][f"{cause}_loss"] += 1

            print(
                f"H2H_GAME_DONE={g + 1}/{args.games} p0={by_player[0][0]} p1={by_player[1][0]} "
                f"result={result} reason={reason} cause={cause} sec={game_sec:.2f}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    print(f"H2H_WALL_SEC={elapsed:.2f}", flush=True)
    print(f"H2H_SEC_PER_GAME={elapsed / args.games:.2f}", flush=True)

    for side in ("new", "old"):
        s = stats[side]
        decided = s["wins"] + s["losses"]
        wr = 100 * s["wins"] / decided if decided else 0.0
        print(
            f"H2H_{side.upper()}: wins={s['wins']} losses={s['losses']} draws={s['draws']} "
            f"win_rate={wr:.1f}%({s['wins']}/{decided}) "
            f"prize_win={s['prize_win']} prize_loss={s['prize_loss']} "
            f"deckout_win={s['deckout_win']} deckout_loss={s['deckout_loss']} "
            f"other_win={s['other_win']} other_loss={s['other_loss']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
