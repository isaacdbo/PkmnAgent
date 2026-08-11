"""Evaluate a candidate m2 checkpoint against a fixed panel of opponents.

Panel members (fixed, reusable across candidates):
  random       - uniform-random legal-action selection with the official
                 sample_submission deck.csv.
  first        - deterministic first-legal-action degeneracy check with the
                 official sample_submission deck.csv.
  starter_rule - small deterministic starter rule-bot with the official
                 sample_submission deck.csv.
  sample_bot   - alias for random; retained for comparability with FINDINGS.md.
  old_m2       - checkpoints/m2/model_2026-08-08_09-48.pth on M2Deck.xlsx, the
                 checkpoint that won last night's 40-game head-to-head 75/25.

Inference-only, no training. Alternates who plays first each game. Win/loss is
attributed by identity (candidate vs opponent), not raw player index, since
which side is P0/P1 flips every game.

Usage:
    SIMULATIONS_PER_MOVE=20 python eval_panel.py \\
        --candidate checkpoints/m2/model_2026-08-08_09-48.pth --candidate-deck M2Deck.xlsx \\
        --panel sample_bot --games 40
"""
from __future__ import annotations

import argparse
import os
import random
import tarfile
import time

import torch

import diag
import RLTRM2 as R

_CAUSE_LABEL = {1: "prize", 2: "deckout", 3: "other", 4: "other"}

_SAMPLE_BOT_DECK_CSV = os.path.join(
    "pokemon-tcg-ai-battle", "sample_submission", "sample_submission", "deck.csv"
)
_REMOTE_TARBALL = "pkmnagent_remote.tar.gz"

_PANEL_CHOICES = ("random", "first", "starter_rule", "sample_bot", "old_m2", "iono_rule")


def _read_deck_csv(path: str) -> list[int]:
    with open(path, "r") as f:
        lines = f.read().split("\n")
    return [int(lines[i]) for i in range(60)]


def _read_sample_submission_deck() -> list[int]:
    if os.path.exists(_SAMPLE_BOT_DECK_CSV):
        return _read_deck_csv(_SAMPLE_BOT_DECK_CSV)
    if os.path.exists(_REMOTE_TARBALL):
        with tarfile.open(_REMOTE_TARBALL, "r:gz") as tar:
            member = tar.extractfile(_SAMPLE_BOT_DECK_CSV)
            if member is not None:
                lines = member.read().decode("utf-8").split("\n")
                return [int(lines[i]) for i in range(60)]
    raise FileNotFoundError(
        f"Could not find sample_submission deck at {_SAMPLE_BOT_DECK_CSV} "
        f"or inside {_REMOTE_TARBALL}"
    )


def _random_bot_move(obs_dict: dict) -> list[int]:
    # Identical logic to pokemon-tcg-ai-battle/sample_submission/sample_submission/main.py's agent()
    obs = R.to_observation_class(obs_dict)
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)


def _first_bot_move(obs_dict: dict) -> list[int]:
    obs = R.to_observation_class(obs_dict)
    return list(range(min(obs.select.maxCount, len(obs.select.option))))


def _starter_rule_move(obs_dict: dict) -> list[int]:
    obs = R.to_observation_class(obs_dict)
    select = obs.select
    if select is None:
        return []
    if select.context == R.SelectContext.MAIN and select.maxCount == 1:
        priority = (
            R.OptionType.ATTACK,
            R.OptionType.ABILITY,
            R.OptionType.ATTACH,
            R.OptionType.EVOLVE,
            R.OptionType.PLAY,
            R.OptionType.RETREAT,
            R.OptionType.END,
        )
        for wanted in priority:
            for idx, option in enumerate(select.option):
                if option.type == wanted:
                    return [idx]
    return _first_bot_move(obs_dict)


class Side:
    """One participant: either an MCTS checkpoint or the random sample bot."""

    def __init__(self, name: str, deck: list[int], model=None, display: str | None = None, policy=None):
        self.name = name
        self.deck = deck
        self.model = model  # None => random sample-bot logic
        self.policy = policy
        # display: what gets printed in place of the bare role string
        # ("candidate"/"opponent") in per-game log lines, so a reader doesn't
        # have to cross-reference the "=== PANEL: ... ===" header to know who
        # "opponent" actually is on any given line. Defaults to name.
        self.display = display if display is not None else name

    def act(self, obs_dict: dict) -> list[int]:
        if self.policy is not None:
            return self.policy(obs_dict)
        if self.model is None:
            return _random_bot_move(obs_dict)
        selected, _ = R.mcts_agent(obs_dict, self.deck, self.model)
        return selected


def build_old_m2(device) -> Side:
    path = os.path.join("checkpoints", "m2", "model_2026-08-08_09-48.pth")
    model = R.MyModel(128, 2, 256, 3, 1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    deck = R.pd.read_excel("M2Deck.xlsx", header=None).iloc[:, 0].tolist()
    return Side("old_m2", deck, model, display="old_m2(checkpoint)")


def build_scripted_panel(name: str) -> Side:
    deck = _read_sample_submission_deck()
    if name in {"random", "sample_bot"}:
        # "random", not "staller": the competition's sample_submission picks
        # uniform-random legal actions every decision (_random_bot_move above,
        # identical logic to the real sample_submission's agent()). The
        # "stalling" behavior discussed elsewhere in this project belongs to the
        # OLD m2 checkpoint (model_2026-08-08_09-48.pth) — a different, trained
        # agent that passively ran games to deck-out — not to sample_bot, which
        # has no policy to stall with in the first place.
        return Side(name, deck, model=None, display=f"{name}(random)", policy=_random_bot_move)
    if name == "first":
        return Side(name, deck, model=None, display="first(first-legal)", policy=_first_bot_move)
    if name == "starter_rule":
        return Side(name, deck, model=None, display="starter_rule(priority)", policy=_starter_rule_move)
    raise ValueError(f"Unknown scripted panel member: {name}")


def build_iono_rule() -> Side:
    """Real Kaggle rule-based starter agent (see panel_bots/iono_rule/PROVENANCE.md).

    main.py is the verbatim `%%writefile main.py` cell from the public notebook
    "A Sample Rule-Based Agent Iono's Deck" (Kiyota + The Pokémon Company
    collaborators, Apache 2.0). It reads deck.csv from the cwd at import time,
    so import it with the bot directory temporarily as cwd.
    """
    import importlib.util

    bot_dir = os.path.join("panel_bots", "iono_rule")
    deck = _read_deck_csv(os.path.join(bot_dir, "deck.csv"))
    missing = [cid for cid in deck if cid not in R.card_table]
    if missing:
        raise ValueError(f"iono_rule deck has card ids unknown to the engine: {missing}")

    prev_cwd = os.getcwd()
    os.chdir(bot_dir)
    try:
        spec = importlib.util.spec_from_file_location("iono_rule_main", "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(prev_cwd)
    return Side("iono_rule", deck, model=None, display="iono_rule(kaggle)", policy=mod.agent)


def wilson_ci(wins: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = wins / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * ((p * (1.0 - p) + z * z / (4.0 * total)) / total) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def run_matchup(candidate: Side, opponent: Side, n_games: int, base_seed: int) -> dict:
    diag.configure(enabled=True, verbose=False, dump_every_games=n_games)

    stats = {
        side: {"wins": 0, "losses": 0, "draws": 0,
               "prize_win": 0, "prize_loss": 0,
               "deckout_win": 0, "deckout_loss": 0,
               "other_win": 0, "other_loss": 0}
        for side in ("candidate", "opponent")
    }
    # role ("candidate"/"opponent") -> display string, for per-game print
    # lines. Stats stay keyed by role (fixed, needed for aggregation); only
    # what gets printed changes.
    display = {"candidate": candidate.display, "opponent": opponent.display}
    game_secs: list[float] = []

    with torch.inference_mode():
        for g in range(n_games):
            random.seed(base_seed + g)
            torch.manual_seed(base_seed + g)

            if g % 2 == 0:
                by_player = [("candidate", candidate), ("opponent", opponent)]
            else:
                by_player = [("opponent", opponent), ("candidate", candidate)]

            game_t0 = time.perf_counter()
            obs, _ = R.battle_start(by_player[0][1].deck, by_player[1][1].deck)
            diag.start_game()
            # Seat-indexed prize-take (== KO landed) count, same detection rule
            # as shaped_reward_terms' KO_BONUS (RLTRM2.py:322-328): a card
            # moving PRIZE->HAND for a seat means that seat just knocked out an
            # opposing Pokemon. Lets a NO_ACTIVE-ending game be read as
            # "sustained attrition" (prizes_taken > 0, candidate landed KOs) vs
            # "opponent self-stranded" (0 prizes taken by either side) without
            # a separate rerun or new diag plumbing.
            prize_events = [0, 0]
            while obs["current"]["result"] < 0:
                prev_obs = obs
                yi = obs["current"]["yourIndex"]
                _role, side = by_player[yi]
                selected = side.act(obs)
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
                for log in next_obs_obj.logs:
                    if (log.type == R.LogType.MOVE_CARD and
                            log.fromArea == R.AreaType.PRIZE and log.toArea == R.AreaType.HAND):
                        prize_events[log.playerIndex] += 1
            R.battle_finish()
            final_obs_obj = R.to_observation_class(obs)
            reason = R._extract_result_reason(final_obs_obj)
            result = obs["current"]["result"]
            did_draw = (result == 2)
            final_turn = final_obs_obj.current.turn
            cand_seat = 0 if by_player[0][0] == "candidate" else 1
            cand_prizes, opp_prizes = prize_events[cand_seat], prize_events[1 - cand_seat]

            diag.record_game_result(final_turn=final_obs_obj.current.turn, reason_code=reason, did_draw=did_draw)
            diag.record_true_result(result=result)
            diag.end_game()

            cause = _CAUSE_LABEL.get(reason, "other")
            game_sec = time.perf_counter() - game_t0
            game_secs.append(game_sec)

            if did_draw:
                winner_role = "draw"
                stats["candidate"]["draws"] += 1
                stats["opponent"]["draws"] += 1
            else:
                winner_role = by_player[result][0]
                loser_role = by_player[1 - result][0]
                stats[winner_role]["wins"] += 1
                stats[loser_role]["losses"] += 1
                stats[winner_role][f"{cause}_win"] += 1
                stats[loser_role][f"{cause}_loss"] += 1

            # winner= resolves identity directly via display name (e.g.
            # "candidate" or "sample_bot(random)"), not the bare role string
            # — a reader shouldn't need to cross-reference the "=== PANEL:
            # ... ===" header to know who "opponent" is on any given line.
            # result= is kept alongside it (not replaced) because it's a raw
            # engine seat index whose meaning flips with p0/p1 every game —
            # useful for cross-checking against engine-level logs, but
            # misleading read alone as if it meant "player 1 won" (see
            # FINDINGS.md).
            winner_display = "draw" if did_draw else display[winner_role]
            print(
                f"EVAL_GAME_DONE={g + 1}/{n_games} winner={winner_display} "
                f"p0={display[by_player[0][0]]} p1={display[by_player[1][0]]} "
                f"result={result} reason={reason} cause={cause} turn={final_turn} "
                f"cand_prizes={cand_prizes} opp_prizes={opp_prizes} sec={game_sec:.2f}",
                flush=True,
            )

    return {"stats": stats, "game_secs": game_secs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", help="checkpoint .pth for the candidate")
    ap.add_argument("--candidate-random-init", action="store_true",
                    help="skip checkpoint loading; use a freshly random-initialised, frozen network")
    ap.add_argument("--candidate-deck", default="M2Deck.xlsx")
    ap.add_argument("--panel", default="sample_bot,old_m2",
                    help=f"comma list from {_PANEL_CHOICES}")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=20260809)
    args = ap.parse_args()
    if not args.candidate_random_init and not args.candidate:
        ap.error("--candidate is required unless --candidate-random-init is set")

    device = torch.device("cpu")
    if args.candidate_random_init:
        # Fixed, dedicated seed for the init draw only — independent of the
        # per-game seeding loop in run_matchup.
        torch.manual_seed(args.base_seed)
    cand_model = R.MyModel(128, 2, 256, 3, 1).to(device)
    if not args.candidate_random_init:
        cand_model.load_state_dict(torch.load(args.candidate, map_location=device))
    cand_model.eval()
    cand_deck = R.pd.read_excel(args.candidate_deck, header=None).iloc[:, 0].tolist()
    candidate = Side("candidate", cand_deck, cand_model)

    print(f"EVAL_CANDIDATE={'RANDOM_INIT' if args.candidate_random_init else args.candidate}", flush=True)
    print(f"EVAL_CANDIDATE_DECK={args.candidate_deck}", flush=True)
    print(f"EVAL_SIMULATIONS_PER_MOVE={R.SEARCH_COUNT}", flush=True)
    print(f"EVAL_GAMES_PER_OPPONENT={args.games}", flush=True)
    print(f"EVAL_BASE_SEED={args.base_seed}", flush=True)

    for opp_name in args.panel.split(","):
        opp_name = opp_name.strip()
        if opp_name not in _PANEL_CHOICES:
            raise ValueError(f"Unknown panel member: {opp_name} (choices: {_PANEL_CHOICES})")

        if opp_name == "old_m2":
            opponent = build_old_m2(device)
        elif opp_name == "iono_rule":
            opponent = build_iono_rule()
        else:
            opponent = build_scripted_panel(opp_name)

        print(f"\n=== PANEL: candidate vs {opp_name} ===", flush=True)
        t0 = time.perf_counter()
        result = run_matchup(candidate, opponent, args.games, args.base_seed)
        elapsed = time.perf_counter() - t0

        stats = result["stats"]
        secs = result["game_secs"]
        mean_sec = sum(secs) / len(secs)

        print(f"OPP_WALL_SEC={elapsed:.2f}", flush=True)
        print(f"OPP_MEAN_SEC_PER_GAME={mean_sec:.2f}", flush=True)
        for side in ("candidate", "opponent"):
            s = stats[side]
            decided = s["wins"] + s["losses"]
            wr = 100 * s["wins"] / decided if decided else 0.0
            ci_low, ci_high = wilson_ci(s["wins"], decided)
            total_decided_causes = s["prize_win"] + s["prize_loss"] + s["deckout_win"] + s["deckout_loss"] + \
                s["other_win"] + s["other_loss"]
            prize_decided = s["prize_win"] + s["prize_loss"]
            deckout_decided = s["deckout_win"] + s["deckout_loss"]
            print(
                f"OPP_{opp_name}_{side.upper()}: wins={s['wins']} losses={s['losses']} draws={s['draws']} "
                f"win_rate={wr:.1f}%({s['wins']}/{decided}) "
                f"wilson95=[{100 * ci_low:.1f}%,{100 * ci_high:.1f}%] "
                f"prize_decided={prize_decided}/{total_decided_causes} "
                f"deckout_share={100 * deckout_decided / total_decided_causes if total_decided_causes else 0:.1f}%"
                f"({deckout_decided}/{total_decided_causes})",
                flush=True,
            )


if __name__ == "__main__":
    main()
