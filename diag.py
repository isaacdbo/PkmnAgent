from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import os
from typing import Dict, Iterable, Set

DIAG_ENABLED = False
DIAG_VERBOSE = False

_DUMP_EVERY_GAMES = 25
_LOG_DIR = "diagnostic"
_LOG_PATH: str | None = None

_GAME_LEN_BUCKETS = (
    (1, 15, "1-15"),
    (16, 25, "16-25"),
    (26, 40, "26-40"),
    (41, 60, "41-60"),
)

_WIN_CAUSE_LABELS = {
    1: "PRIZE",
    2: "DECK_OUT",
    3: "NO_ACTIVE",
    4: "CARD_EFFECT_OR_CONCEDE",
}

_HIST_BUCKETS = (
    (1, 8, "1-8"),
    (9, 16, "9-16"),
    (17, 32, "17-32"),
    (33, 64, "33-64"),
    (65, 128, "65-128"),
    (129, 256, "129-256"),
)

_CRITICAL_TYPES = {"ATTACH", "EVOLVE", "ABILITY"}

_PRIOR_BUCKETS = (
    (0.0, 0.001, "<0.001"),
    (0.001, 0.005, "0.001-0.005"),
    (0.005, 0.01, "0.005-0.01"),
    (0.01, 0.03, "0.01-0.03"),
    (0.03, 0.1, "0.03-0.1"),
    (0.1, 0.3, "0.1-0.3"),
    (0.3, 1.0, "0.3-1.0"),
)


def _prior_bucket(p: float) -> str:
    for low, high, label in _PRIOR_BUCKETS:
        if low <= p < high:
            return label
    return "1.0"


# Named (not lambda) default factories: defaultdict(lambda: ...) can't be
# pickled, which breaks sending a window snapshot across a process boundary
# (parallel self-play workers). Named module-level functions pickle fine.
def _new_multiselect_slot() -> dict:
    return {"nodes": 0, "shorter_legal": 0, "shorter_generated": 0, "shorter_missed": 0}


def _new_cause_bucket() -> dict:
    return {"W": 0, "L": 0, "D": 0}


def _new_root_option_slot() -> dict:
    return {
        "available": 0,
        "chosen": 0,
        "not_chosen": 0,
        "zero_visit": 0,
        "opt_visit_sum": 0,
        "opt_q_sum": 0.0,
        "opt_q_count": 0,
        "chosen_visit_sum": 0,
        "chosen_q_sum": 0.0,
        "chosen_q_count": 0,
    }


def _new_prior_group() -> dict:
    return {
        "count": 0,
        "attach_sum": 0.0,
        "attach_min": None,
        "attach_max": 0.0,
        "attach_hist": {label: 0 for _, _, label in _PRIOR_BUCKETS} | {"1.0": 0},
        "chosen_sum": 0.0,
        "chosen_min": None,
        "chosen_max": 0.0,
        "chosen_hist": {label: 0 for _, _, label in _PRIOR_BUCKETS} | {"1.0": 0},
    }


def _new_state() -> dict:
    return {
        "TRUNC_TOTAL_NODES": 0,
        "TRUNC_FIRED": 0,
        "TRUNC_OPTCOUNT_HIST": {label: 0 for _, _, label in _HIST_BUCKETS} | {"257+": 0},
        "TRUNC_TURN_TOTAL": {"1-2": 0, "3-5": 0, "6+": 0},
        "TRUNC_TURN_FIRED": {"1-2": 0, "3-5": 0, "6+": 0},
        "TRUNC_DROPPED_BY_TYPE": defaultdict(int),
        "TRUNC_DROPPED_CRITICAL": 0,
        "MULTISELECT_SHORTER_MISSED": 0,
        "MULTISELECT_CTX_HIST": defaultdict(int),
        "MULTISELECT_FIXED_LEN": defaultdict(_new_multiselect_slot),
        "GAME_LENGTH_HIST": {label: 0 for _, _, label in _GAME_LEN_BUCKETS} | {"61+": 0},
        "WIN_BY_CAUSE": defaultdict(_new_cause_bucket),
        "NODE_TURN_TOTAL": {"1-2": 0, "3-5": 0, "6+": 0},
        "SEARCH_DECISIONS": 0,
        "SEARCH_MAX_DEPTH": 0,
        "SEARCH_MEAN_DEPTH_SUM": 0.0,
        "SEARCH_PRIZE_REWARD_REACHED": 0,
        "SEARCH_SIMS_CONTRIBUTED": 0,
        "SEARCH_SIMS_CONFIGURED": 0,
        "STALL_DRAW_PASS_TURNS": 0,
        "STALL_SHUFFLE_WITH_RESOURCES": 0,
        "MISSED_ATTACH": 0,
        "SHAPING_SUM": defaultdict(float),
        "SHAPING_COUNT": 0,
        "ROOT_OPTION_STATS": defaultdict(_new_root_option_slot),
        "ATTACH_PRIOR": {"special": _new_prior_group(), "plain": _new_prior_group()},
        "SIM_KO_TOTAL": 0,
        "SIM_KO_REACHED": 0,
        "SIM_KO_DEPTH_SUM": 0,
        "SIM_KO_DEPTH_MIN": None,
        "SIM_KO_DEPTH_MAX": 0,
        # Independent, unaggregated per-game result tally — unlike WIN_BY_CAUSE
        # (which increments W and L together per cause and cannot show who won),
        # this counts genuine player-0/-1/draw outcomes straight off `result`.
        "TRUE_P0_WINS": 0,
        "TRUE_P1_WINS": 0,
        "TRUE_DRAWS": 0,
        # Full root_sample.policy vector (the actual cross-entropy training
        # target), not the ATTACK/ENERGY_ATTACH-only proxy ROOT_OPTION_STATS
        # gives. Own counter (POLICY_TARGET_COUNT), not reusing
        # SEARCH_DECISIONS, so this stays correct even if the two call sites
        # ever fire at different cadences.
        "POLICY_TARGET_ZERO_FRACTION_SUM": 0.0,
        "POLICY_TARGET_ENTROPY_SUM": 0.0,
        "POLICY_TARGET_COUNT": 0,
        # Root-only branching measurements (distinct from TRUNC_OPTCOUNT_HIST /
        # NODE_TURN_TOTAL, which mix root and internal expansion nodes). Raw
        # per-decision lists, not running sums, so exact mean/median can be
        # computed at dump time.
        "ROOT_CHILD_COUNTS": [],
        "ROOT_CHILD_COUNTS_BY_TURN": {"1-2": [], "3-5": [], "6+": []},
        "ROOT_VISIT_HIST": {"0": 0, "1": 0, "2-4": 0, "5+": 0},
        # Combined ENERGY_ATTACH-or-Spidops-ABILITY view: a low ENERGY_ATTACH
        # rate alone can't distinguish "not developing energy" from "routing
        # through the ability instead" — this counts decisions where either
        # path was legal/chosen, so the two ROOT_OPTION_STATS entries don't
        # have to be added by hand (and to avoid double-counting decisions
        # where both were legal).
        "ENERGY_DEV_LEGAL": 0,
        "ENERGY_DEV_CHOSEN": 0,
        # Root-decision rate of "any Basic {G} Energy already in your own
        # discard pile" — Spidops' ability only needs *a* Basic Energy (any
        # type) in discard, not grass specifically, so this measures the
        # grass-specific setup precondition, not general ability legality.
        "GRASS_IN_DISCARD_PRESENT": 0,
        "GRASS_IN_DISCARD_TOTAL": 0,
    }


_current = _new_state()
_window = _new_state()
_games_in_window = 0
_current_agent: str | None = None
_turn_key: tuple[int, int] | None = None
_turn_attach_legal = False
_turn_attach_made = False
_turn_board_affecting = False


def configure(enabled: bool, verbose: bool, dump_every_games: int = 25) -> None:
    global DIAG_ENABLED, DIAG_VERBOSE, _DUMP_EVERY_GAMES, _LOG_PATH
    DIAG_ENABLED = enabled
    DIAG_VERBOSE = verbose
    _DUMP_EVERY_GAMES = max(1, int(dump_every_games))

    os.makedirs(_LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _LOG_PATH = os.path.join(_LOG_DIR, f"diag_{ts}.log")


def _emit(line: str) -> None:
    print(line, flush=True)
    if _LOG_PATH is None:
        return
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _turn_bucket(turn: int) -> str:
    if turn <= 2:
        return "1-2"
    if turn <= 5:
        return "3-5"
    return "6+"


def _game_len_bucket(turn_count: int) -> str:
    for low, high, label in _GAME_LEN_BUCKETS:
        if low <= turn_count <= high:
            return label
    return "61+"


def _cause_label(reason_code: int | None) -> str:
    if reason_code in _WIN_CAUSE_LABELS:
        return _WIN_CAUSE_LABELS[reason_code]
    _emit(f"[WARN] Unknown result reason code: {reason_code}")
    return "UNKNOWN"


def _optcount_bucket(count: int) -> str:
    for low, high, label in _HIST_BUCKETS:
        if low <= count <= high:
            return label
    return "257+"


def _mean_median(values: list) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    median = float(s[mid]) if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0
    return sum(s) / n, median


def start_game() -> None:
    global _current, _turn_key, _turn_attach_legal, _turn_attach_made, _turn_board_affecting
    if not DIAG_ENABLED:
        return
    _current = _new_state()
    _turn_key = None
    _turn_attach_legal = False
    _turn_attach_made = False
    _turn_board_affecting = False


def record_node_turn(*, turn: int) -> None:
    if not DIAG_ENABLED:
        return
    _current["NODE_TURN_TOTAL"][_turn_bucket(turn)] += 1


def record_search_decision(
    *,
    max_depth: int,
    mean_depth: float,
    prize_reward_reached: bool,
    sims_contributed: int,
    sims_configured: int,
) -> None:
    if not DIAG_ENABLED:
        return
    _current["SEARCH_DECISIONS"] += 1
    _current["SEARCH_MAX_DEPTH"] = max(_current["SEARCH_MAX_DEPTH"], int(max_depth))
    _current["SEARCH_MEAN_DEPTH_SUM"] += float(mean_depth)
    _current["SEARCH_SIMS_CONTRIBUTED"] += int(sims_contributed)
    _current["SEARCH_SIMS_CONFIGURED"] += int(sims_configured)
    if prize_reward_reached:
        _current["SEARCH_PRIZE_REWARD_REACHED"] += 1


def record_root_option_stats(
    *,
    kind: str,
    legal: bool,
    chosen: bool,
    option_visit: int | None,
    option_q: float | None,
    chosen_visit: int | None,
    chosen_q: float | None,
) -> None:
    if not DIAG_ENABLED or not legal:
        return
    slot = _current["ROOT_OPTION_STATS"][kind]
    slot["available"] += 1
    if option_visit is not None and option_visit == 0:
        slot["zero_visit"] += 1
    if chosen:
        slot["chosen"] += 1
        return
    slot["not_chosen"] += 1
    if option_visit is not None:
        slot["opt_visit_sum"] += option_visit
    if option_q is not None:
        slot["opt_q_sum"] += option_q
        slot["opt_q_count"] += 1
    if chosen_visit is not None:
        slot["chosen_visit_sum"] += chosen_visit
    if chosen_q is not None:
        slot["chosen_q_sum"] += chosen_q
        slot["chosen_q_count"] += 1


def record_attach_prior(*, is_special: bool, attach_prior: float, chosen_prior: float) -> None:
    if not DIAG_ENABLED:
        return
    group = _current["ATTACH_PRIOR"]["special" if is_special else "plain"]
    group["count"] += 1
    group["attach_sum"] += attach_prior
    group["attach_min"] = attach_prior if group["attach_min"] is None else min(group["attach_min"], attach_prior)
    group["attach_max"] = max(group["attach_max"], attach_prior)
    group["attach_hist"][_prior_bucket(attach_prior)] += 1
    group["chosen_sum"] += chosen_prior
    group["chosen_min"] = chosen_prior if group["chosen_min"] is None else min(group["chosen_min"], chosen_prior)
    group["chosen_max"] = max(group["chosen_max"], chosen_prior)
    group["chosen_hist"][_prior_bucket(chosen_prior)] += 1


def record_true_result(*, result: int) -> None:
    if not DIAG_ENABLED:
        return
    if result == 0:
        _current["TRUE_P0_WINS"] += 1
    elif result == 1:
        _current["TRUE_P1_WINS"] += 1
    elif result == 2:
        _current["TRUE_DRAWS"] += 1


def record_policy_target(*, zero_fraction: float, entropy: float) -> None:
    if not DIAG_ENABLED:
        return
    _current["POLICY_TARGET_ZERO_FRACTION_SUM"] += zero_fraction
    _current["POLICY_TARGET_ENTROPY_SUM"] += entropy
    _current["POLICY_TARGET_COUNT"] += 1


def record_root_branching(*, turn: int, child_count: int) -> None:
    if not DIAG_ENABLED:
        return
    _current["ROOT_CHILD_COUNTS"].append(child_count)
    _current["ROOT_CHILD_COUNTS_BY_TURN"][_turn_bucket(turn)].append(child_count)


def _root_visit_bucket(visit: int) -> str:
    if visit <= 0:
        return "0"
    if visit == 1:
        return "1"
    if visit <= 4:
        return "2-4"
    return "5+"


def record_root_visit_dist(visit_counts: Iterable[int]) -> None:
    if not DIAG_ENABLED:
        return
    hist = _current["ROOT_VISIT_HIST"]
    for v in visit_counts:
        hist[_root_visit_bucket(v)] += 1


def record_energy_dev_stats(*, legal: bool, chosen: bool) -> None:
    if not DIAG_ENABLED:
        return
    if legal:
        _current["ENERGY_DEV_LEGAL"] += 1
        if chosen:
            _current["ENERGY_DEV_CHOSEN"] += 1


def record_grass_in_discard(*, present: bool) -> None:
    if not DIAG_ENABLED:
        return
    _current["GRASS_IN_DISCARD_TOTAL"] += 1
    if present:
        _current["GRASS_IN_DISCARD_PRESENT"] += 1


def record_sim_ko(*, reached: bool, depth: int | None) -> None:
    if not DIAG_ENABLED:
        return
    _current["SIM_KO_TOTAL"] += 1
    if not reached:
        return
    _current["SIM_KO_REACHED"] += 1
    _current["SIM_KO_DEPTH_SUM"] += depth
    _current["SIM_KO_DEPTH_MAX"] = max(_current["SIM_KO_DEPTH_MAX"], depth)
    cur_min = _current["SIM_KO_DEPTH_MIN"]
    _current["SIM_KO_DEPTH_MIN"] = depth if cur_min is None else min(cur_min, depth)


def _flush_turn_flags() -> None:
    global _turn_attach_legal, _turn_attach_made, _turn_board_affecting
    if _turn_key is None:
        return
    if _turn_attach_legal and not _turn_attach_made:
        _current["MISSED_ATTACH"] += 1
    if not _turn_board_affecting:
        _current["STALL_DRAW_PASS_TURNS"] += 1
    _turn_attach_legal = False
    _turn_attach_made = False
    _turn_board_affecting = False


def record_turn_step(
    *,
    turn: int,
    player_index: int,
    attach_legal: bool,
    attach_made: bool,
    board_affecting: bool,
    shuffle_with_resources: bool,
) -> None:
    global _turn_key, _turn_attach_legal, _turn_attach_made, _turn_board_affecting
    if not DIAG_ENABLED:
        return

    key = (turn, player_index)
    if _turn_key is None:
        _turn_key = key
    elif key != _turn_key:
        _flush_turn_flags()
        _turn_key = key

    _turn_attach_legal = _turn_attach_legal or attach_legal
    _turn_attach_made = _turn_attach_made or attach_made
    _turn_board_affecting = _turn_board_affecting or board_affecting
    if shuffle_with_resources:
        _current["STALL_SHUFFLE_WITH_RESOURCES"] += 1


def record_game_result(*, final_turn: int, reason_code: int | None, did_draw: bool) -> None:
    if not DIAG_ENABLED:
        return
    _current["GAME_LENGTH_HIST"][_game_len_bucket(final_turn)] += 1
    cause = _cause_label(reason_code)
    if did_draw:
        _current["WIN_BY_CAUSE"][cause]["D"] += 1
    else:
        _current["WIN_BY_CAUSE"][cause]["W"] += 1
        _current["WIN_BY_CAUSE"][cause]["L"] += 1


def record_shaping_terms(terms: Dict[str, float]) -> None:
    if not DIAG_ENABLED:
        return
    _current["SHAPING_COUNT"] += 1
    for k, v in terms.items():
        _current["SHAPING_SUM"][k] += float(v)


def record_truncation(
    *,
    turn: int,
    context: str,
    option_type_names: Iterable[str],
    kept_option_indices: Set[int],
    cap: int,
) -> None:
    if not DIAG_ENABLED:
        return

    option_types = list(option_type_names)
    opt_count = len(option_types)

    _current["TRUNC_TOTAL_NODES"] += 1
    _current["TRUNC_OPTCOUNT_HIST"][_optcount_bucket(opt_count)] += 1
    t_bucket = _turn_bucket(turn)
    _current["TRUNC_TURN_TOTAL"][t_bucket] += 1

    if opt_count <= cap:
        return

    _current["TRUNC_FIRED"] += 1
    _current["TRUNC_TURN_FIRED"][t_bucket] += 1

    dropped_types: Dict[str, int] = defaultdict(int)
    critical_exists = False
    critical_kept = False

    for idx, type_name in enumerate(option_types):
        if type_name in _CRITICAL_TYPES:
            critical_exists = True
            if idx in kept_option_indices:
                critical_kept = True
        if idx not in kept_option_indices:
            dropped_types[type_name] += 1
            _current["TRUNC_DROPPED_BY_TYPE"][type_name] += 1

    if critical_exists and not critical_kept:
        _current["TRUNC_DROPPED_CRITICAL"] += 1

    if DIAG_VERBOSE:
        # Keep key order stable for easy grepping/diffing.
        ordered = ",".join(f"{k}:{dropped_types[k]}" for k in sorted(dropped_types))
        _emit(
            f"[TRUNC] turn={turn} ctx={context} opts={opt_count} kept={len(kept_option_indices)} "
            f"dropped_types={{{ordered}}}"
        )


def record_multiselect(
    *,
    context: str,
    min_count: int,
    max_count: int,
    generated_lengths: Set[int],
) -> None:
    if not DIAG_ENABLED:
        return

    _current["MULTISELECT_CTX_HIST"][context] += 1

    shorter_legal_exists = min_count < max_count
    shorter_generated = any(length < max_count for length in generated_lengths)
    shorter_missed = shorter_legal_exists and not shorter_generated

    key = f"min={min_count},max={max_count}"
    slot = _current["MULTISELECT_FIXED_LEN"][key]
    slot["nodes"] += 1
    if shorter_legal_exists:
        slot["shorter_legal"] += 1
    if shorter_generated:
        slot["shorter_generated"] += 1
    if shorter_missed:
        slot["shorter_missed"] += 1
        _current["MULTISELECT_SHORTER_MISSED"] += 1

    if DIAG_VERBOSE:
        lens_text = ",".join(str(x) for x in sorted(generated_lengths))
        _emit(
            f"[MULTISELECT] ctx={context} min={min_count} max={max_count} "
            f"generated_lens=[{lens_text}] shorter_legal_exists={shorter_legal_exists} "
            f"shorter_missed={shorter_missed}"
        )


def _merge_state_into(source: dict, target: dict) -> None:
    """Merge `source` (a state dict from _new_state()) into `target` in place.
    Used both for the per-game _current -> _window merge (single process) and
    for merging a parallel worker's returned window snapshot into the parent's
    _window (see merge_worker_window) — same shape, same merge rules either way.
    """
    target["TRUNC_TOTAL_NODES"] += source["TRUNC_TOTAL_NODES"]
    target["TRUNC_FIRED"] += source["TRUNC_FIRED"]
    target["TRUNC_DROPPED_CRITICAL"] += source["TRUNC_DROPPED_CRITICAL"]
    target["MULTISELECT_SHORTER_MISSED"] += source["MULTISELECT_SHORTER_MISSED"]
    target["SEARCH_DECISIONS"] += source["SEARCH_DECISIONS"]
    target["SEARCH_MAX_DEPTH"] = max(target["SEARCH_MAX_DEPTH"], source["SEARCH_MAX_DEPTH"])
    target["SEARCH_MEAN_DEPTH_SUM"] += source["SEARCH_MEAN_DEPTH_SUM"]
    target["SEARCH_PRIZE_REWARD_REACHED"] += source["SEARCH_PRIZE_REWARD_REACHED"]
    target["SEARCH_SIMS_CONTRIBUTED"] += source["SEARCH_SIMS_CONTRIBUTED"]
    target["SEARCH_SIMS_CONFIGURED"] += source["SEARCH_SIMS_CONFIGURED"]
    target["STALL_DRAW_PASS_TURNS"] += source["STALL_DRAW_PASS_TURNS"]
    target["STALL_SHUFFLE_WITH_RESOURCES"] += source["STALL_SHUFFLE_WITH_RESOURCES"]
    target["MISSED_ATTACH"] += source["MISSED_ATTACH"]
    target["SHAPING_COUNT"] += source["SHAPING_COUNT"]

    for k, v in source["TRUNC_OPTCOUNT_HIST"].items():
        target["TRUNC_OPTCOUNT_HIST"][k] += v
    for k, v in source["TRUNC_TURN_TOTAL"].items():
        target["TRUNC_TURN_TOTAL"][k] += v
    for k, v in source["TRUNC_TURN_FIRED"].items():
        target["TRUNC_TURN_FIRED"][k] += v
    for k, v in source["TRUNC_DROPPED_BY_TYPE"].items():
        target["TRUNC_DROPPED_BY_TYPE"][k] += v
    for k, v in source["MULTISELECT_CTX_HIST"].items():
        target["MULTISELECT_CTX_HIST"][k] += v
    for k, v in source["GAME_LENGTH_HIST"].items():
        target["GAME_LENGTH_HIST"][k] += v
    for k, v in source["NODE_TURN_TOTAL"].items():
        target["NODE_TURN_TOTAL"][k] += v
    for cause, rec in source["WIN_BY_CAUSE"].items():
        target["WIN_BY_CAUSE"][cause]["W"] += rec["W"]
        target["WIN_BY_CAUSE"][cause]["L"] += rec["L"]
        target["WIN_BY_CAUSE"][cause]["D"] += rec["D"]
    for k, v in source["SHAPING_SUM"].items():
        target["SHAPING_SUM"][k] += v

    target["TRUE_P0_WINS"] += source["TRUE_P0_WINS"]
    target["TRUE_P1_WINS"] += source["TRUE_P1_WINS"]
    target["TRUE_DRAWS"] += source["TRUE_DRAWS"]

    target["POLICY_TARGET_ZERO_FRACTION_SUM"] += source["POLICY_TARGET_ZERO_FRACTION_SUM"]
    target["POLICY_TARGET_ENTROPY_SUM"] += source["POLICY_TARGET_ENTROPY_SUM"]
    target["POLICY_TARGET_COUNT"] += source["POLICY_TARGET_COUNT"]

    target["ROOT_CHILD_COUNTS"].extend(source["ROOT_CHILD_COUNTS"])
    for k, v in source["ROOT_CHILD_COUNTS_BY_TURN"].items():
        target["ROOT_CHILD_COUNTS_BY_TURN"][k].extend(v)
    for k, v in source["ROOT_VISIT_HIST"].items():
        target["ROOT_VISIT_HIST"][k] += v

    target["ENERGY_DEV_LEGAL"] += source["ENERGY_DEV_LEGAL"]
    target["ENERGY_DEV_CHOSEN"] += source["ENERGY_DEV_CHOSEN"]
    target["GRASS_IN_DISCARD_PRESENT"] += source["GRASS_IN_DISCARD_PRESENT"]
    target["GRASS_IN_DISCARD_TOTAL"] += source["GRASS_IN_DISCARD_TOTAL"]

    target["SIM_KO_TOTAL"] += source["SIM_KO_TOTAL"]
    target["SIM_KO_REACHED"] += source["SIM_KO_REACHED"]
    target["SIM_KO_DEPTH_SUM"] += source["SIM_KO_DEPTH_SUM"]
    target["SIM_KO_DEPTH_MAX"] = max(target["SIM_KO_DEPTH_MAX"], source["SIM_KO_DEPTH_MAX"])
    if source["SIM_KO_DEPTH_MIN"] is not None:
        cur_min = target["SIM_KO_DEPTH_MIN"]
        target["SIM_KO_DEPTH_MIN"] = (
            source["SIM_KO_DEPTH_MIN"] if cur_min is None else min(cur_min, source["SIM_KO_DEPTH_MIN"])
        )

    for kind, slot in source["ROOT_OPTION_STATS"].items():
        dst = target["ROOT_OPTION_STATS"][kind]
        for k, v in slot.items():
            dst[k] += v

    for group_name, src in source["ATTACH_PRIOR"].items():
        dst = target["ATTACH_PRIOR"][group_name]
        dst["count"] += src["count"]
        dst["attach_sum"] += src["attach_sum"]
        dst["chosen_sum"] += src["chosen_sum"]
        if src["attach_min"] is not None:
            dst["attach_min"] = src["attach_min"] if dst["attach_min"] is None else min(dst["attach_min"], src["attach_min"])
        dst["attach_max"] = max(dst["attach_max"], src["attach_max"])
        if src["chosen_min"] is not None:
            dst["chosen_min"] = src["chosen_min"] if dst["chosen_min"] is None else min(dst["chosen_min"], src["chosen_min"])
        dst["chosen_max"] = max(dst["chosen_max"], src["chosen_max"])
        for label, cnt in src["attach_hist"].items():
            dst["attach_hist"][label] += cnt
        for label, cnt in src["chosen_hist"].items():
            dst["chosen_hist"][label] += cnt

    for k, slot in source["MULTISELECT_FIXED_LEN"].items():
        dst = target["MULTISELECT_FIXED_LEN"][k]
        dst["nodes"] += slot["nodes"]
        dst["shorter_legal"] += slot["shorter_legal"]
        dst["shorter_generated"] += slot["shorter_generated"]
        dst["shorter_missed"] += slot["shorter_missed"]


def _merge_current_into_window() -> None:
    _merge_state_into(_current, _window)


def enable_for_worker() -> None:
    """Lightweight enable for a parallel self-play worker process: turns on
    recording without configure()'s side effects (no log directory/file
    creation — the worker never dumps or writes a log itself; it hands its
    accumulated window state back to the parent via take_window_snapshot()
    for the parent to merge and dump). Verbose stays off; a worker producing
    per-game dumps would defeat the point of batching games across workers.

    Also pins _DUMP_EVERY_GAMES effectively to infinity. Without this, a
    worker process that inherited FAST_TEST=True's import-time
    configure(dump_every_games=1) would self-dump-and-reset its _window after
    every single game (end_game()'s threshold branch, `_games_in_window >=
    _DUMP_EVERY_GAMES` with the threshold stuck at 1) — leaving nothing for
    take_window_snapshot() to return. The parent alone decides when to dump,
    via merge_worker_window(); the worker only ever accumulates.
    """
    global DIAG_ENABLED, DIAG_VERBOSE, _DUMP_EVERY_GAMES
    DIAG_ENABLED = True
    DIAG_VERBOSE = False
    _DUMP_EVERY_GAMES = 2**31


def take_window_snapshot() -> tuple[dict, int]:
    """Return (window_state, games_in_window) and reset the window to fresh.
    Called by a parallel worker at the end of its assigned games, so it can
    hand its accumulated stats back to the parent process for merging.
    """
    global _window, _games_in_window
    snapshot = _window
    games = _games_in_window
    _window = _new_state()
    _games_in_window = 0
    return snapshot, games


def set_agent(name: str) -> None:
    """Record which agent is currently generating games. If this differs from
    whichever agent the window was already accumulating for, flush (dump +
    reset) the window first — so a single dump block never mixes games from
    two different agents, regardless of where the DIAG_DUMP_EVERY_GAMES
    threshold would otherwise have landed. Call this in the training loop
    every time it switches from one agent (or cross-play pairing) to another,
    before dispatching that agent's games.
    """
    global _current_agent, _window, _games_in_window
    if not DIAG_ENABLED:
        _current_agent = name
        return
    if _current_agent is not None and _current_agent != name and _games_in_window > 0:
        _dump_from_state(_window, games=_games_in_window, agent=_current_agent)
        _window = _new_state()
        _games_in_window = 0
    _current_agent = name


def merge_worker_window(state: dict, games: int) -> None:
    """Parent-side counterpart to take_window_snapshot(): merge one worker's
    returned window state into our own _window, dumping if the combined count
    crosses the configured threshold — same trigger _merge_current_into_window
    ordinarily fires from end_game(), just batched across `games` at once
    instead of one game at a time.
    """
    global _window, _games_in_window
    if not DIAG_ENABLED:
        return
    _merge_state_into(state, _window)
    _games_in_window += games
    if not DIAG_VERBOSE and _games_in_window >= _DUMP_EVERY_GAMES:
        _dump_from_state(_window, games=_games_in_window, agent=_current_agent)
        _window = _new_state()
        _games_in_window = 0


def _fmt_pct(numer: int, denom: int) -> str:
    if denom <= 0:
        return "0.00%"
    return f"{(100.0 * numer / denom):.2f}%"


def _dump_from_state(state: dict, games: int, agent: str | None = None) -> None:
    total = state["TRUNC_TOTAL_NODES"]
    fired = state["TRUNC_FIRED"]

    hist_labels = [label for _, _, label in _HIST_BUCKETS] + ["257+"]
    hist_text = "|".join(f"{label}:{state['TRUNC_OPTCOUNT_HIST'][label]}" for label in hist_labels)

    by_turn_parts = []
    for bucket in ("1-2", "3-5", "6+"):
        t_total = state["TRUNC_TURN_TOTAL"][bucket]
        t_fired = state["TRUNC_TURN_FIRED"][bucket]
        by_turn_parts.append(f"{bucket}:{_fmt_pct(t_fired, t_total)}({t_fired}/{t_total})")

    dropped = state["TRUNC_DROPPED_BY_TYPE"]
    dropped_text = "NONE"
    if dropped:
        dropped_text = ",".join(f"{k}:{dropped[k]}" for k in sorted(dropped))

    critical = state["TRUNC_DROPPED_CRITICAL"]
    critical_status = "PASS" if critical == 0 else "FAIL"

    multiselect_fixed = state["MULTISELECT_FIXED_LEN"]
    multiselect_fixed_text = "NONE"
    if multiselect_fixed:
        parts = []
        for k in sorted(multiselect_fixed):
            slot = multiselect_fixed[k]
            parts.append(
                f"{k}:nodes={slot['nodes']},shorter_legal={slot['shorter_legal']},"
                f"shorter_generated={slot['shorter_generated']},shorter_missed={slot['shorter_missed']}"
            )
        multiselect_fixed_text = "|".join(parts)

    multiselect_ctx = state["MULTISELECT_CTX_HIST"]
    multiselect_ctx_text = "NONE"
    if multiselect_ctx:
        multiselect_ctx_text = ",".join(f"{k}:{multiselect_ctx[k]}" for k in sorted(multiselect_ctx))

    game_hist_labels = [label for _, _, label in _GAME_LEN_BUCKETS] + ["61+"]
    game_hist_text = "|".join(f"{label}:{state['GAME_LENGTH_HIST'][label]}" for label in game_hist_labels)

    win_by_cause = state["WIN_BY_CAUSE"]
    win_by_cause_text = "NONE"
    if win_by_cause:
        parts = []
        for cause in sorted(win_by_cause):
            rec = win_by_cause[cause]
            parts.append(f"{cause}:W={rec['W']},L={rec['L']},D={rec['D']}")
        win_by_cause_text = "|".join(parts)

    node_total = sum(state["NODE_TURN_TOTAL"].values())
    turn_node_dist_parts = []
    for bucket in ("1-2", "3-5", "6+"):
        b_count = state["NODE_TURN_TOTAL"][bucket]
        turn_node_dist_parts.append(f"{bucket}:{_fmt_pct(b_count, node_total)}({b_count}/{node_total})")

    search_decisions = state["SEARCH_DECISIONS"]
    search_mean_depth = 0.0
    if search_decisions > 0:
        search_mean_depth = state["SEARCH_MEAN_DEPTH_SUM"] / search_decisions
    sims_contributed_per_decision = state["SEARCH_SIMS_CONTRIBUTED"] / search_decisions if search_decisions > 0 else 0.0
    sims_configured_per_decision = state["SEARCH_SIMS_CONFIGURED"] / search_decisions if search_decisions > 0 else 0.0

    shaping_text = "NONE"
    shaping_count = state["SHAPING_COUNT"]
    if shaping_count > 0 and state["SHAPING_SUM"]:
        shaping_text = "|".join(
            f"{k}:{(state['SHAPING_SUM'][k] / shaping_count):.6f}"
            for k in sorted(state["SHAPING_SUM"])
        )

    root_opt_stats = state["ROOT_OPTION_STATS"]
    root_opt_text = "NONE"
    if root_opt_stats:
        parts = []
        for kind in sorted(root_opt_stats):
            s = root_opt_stats[kind]
            mean_opt_visit = s["opt_visit_sum"] / s["not_chosen"] if s["not_chosen"] else 0.0
            mean_opt_q = s["opt_q_sum"] / s["opt_q_count"] if s["opt_q_count"] else float("nan")
            mean_chosen_visit = s["chosen_visit_sum"] / s["not_chosen"] if s["not_chosen"] else 0.0
            mean_chosen_q = s["chosen_q_sum"] / s["chosen_q_count"] if s["chosen_q_count"] else float("nan")
            zero_visit_rate = _fmt_pct(s["zero_visit"], s["available"])
            parts.append(
                f"{kind}:available={s['available']},chosen={s['chosen']},not_chosen={s['not_chosen']},"
                f"zero_visit_rate={zero_visit_rate}({s['zero_visit']}/{s['available']}),"
                f"mean_opt_visit={mean_opt_visit:.2f},mean_opt_q={mean_opt_q:.4f},"
                f"mean_chosen_visit={mean_chosen_visit:.2f},mean_chosen_q={mean_chosen_q:.4f}"
            )
        root_opt_text = "|".join(parts)

    _prior_hist_labels = [label for _, _, label in _PRIOR_BUCKETS] + ["1.0"]

    attach_prior_parts = []
    for group_name in ("special", "plain"):
        g = state["ATTACH_PRIOR"][group_name]
        n = g["count"]
        mean_attach = g["attach_sum"] / n if n else float("nan")
        mean_chosen = g["chosen_sum"] / n if n else float("nan")
        attach_hist_text = ",".join(f"{label}:{g['attach_hist'][label]}" for label in _prior_hist_labels)
        chosen_hist_text = ",".join(f"{label}:{g['chosen_hist'][label]}" for label in _prior_hist_labels)
        attach_prior_parts.append(
            f"{group_name}:n={n},"
            f"attach_prior[mean={mean_attach:.4f},min={(g['attach_min'] if g['attach_min'] is not None else 0.0):.4f},max={g['attach_max']:.4f},hist=({attach_hist_text})],"
            f"chosen_prior[mean={mean_chosen:.4f},min={(g['chosen_min'] if g['chosen_min'] is not None else 0.0):.4f},max={g['chosen_max']:.4f},hist=({chosen_hist_text})]"
        )
    attach_prior_text = "|".join(attach_prior_parts)

    sim_ko_total = state["SIM_KO_TOTAL"]
    sim_ko_reached = state["SIM_KO_REACHED"]
    sim_ko_depth_mean = (state["SIM_KO_DEPTH_SUM"] / sim_ko_reached) if sim_ko_reached else 0.0
    sim_ko_depth_min = state["SIM_KO_DEPTH_MIN"] if state["SIM_KO_DEPTH_MIN"] is not None else 0

    _emit("=== DIAG_DUMP_BEGIN ===")
    _emit(f"DIAG_GAMES={games}")
    _emit(f"AGENT={agent if agent is not None else 'UNKNOWN'}")
    _emit(f"TRUNC_TOTAL_NODES={total}")
    _emit(f"TRUNC_FIRED={fired}")
    _emit(f"TRUNC_FIRE_RATE={_fmt_pct(fired, total)}")
    _emit(f"TRUNC_OPTCOUNT_HIST={hist_text}")
    _emit(f"TRUNC_FIRE_RATE_BY_TURN={'|'.join(by_turn_parts)}")
    _emit(f"TRUNC_DROPPED_BY_TYPE={dropped_text}")
    _emit(f"TRUNC_DROPPED_CRITICAL={critical}")
    _emit(f"TRUNC_DROPPED_CRITICAL_STATUS={critical_status}")
    _emit(f"MULTISELECT_FIXED_LEN={multiselect_fixed_text}")
    _emit(f"MULTISELECT_SHORTER_MISSED={state['MULTISELECT_SHORTER_MISSED']}")
    _emit(f"MULTISELECT_CTX_HIST={multiselect_ctx_text}")
    _emit(f"GAME_LENGTH_HIST={game_hist_text}")
    _emit(f"WIN_BY_CAUSE={win_by_cause_text}")
    _emit(f"TURN_NODE_DIST={'|'.join(turn_node_dist_parts)}")
    _emit(f"SEARCH_MAX_DEPTH={state['SEARCH_MAX_DEPTH']}")
    _emit(f"SEARCH_MEAN_DEPTH={search_mean_depth:.4f}")
    _emit(f"MEAN_DEPTH={search_mean_depth:.4f}")
    _emit(f"SIMS_CONTRIBUTED={state['SEARCH_SIMS_CONTRIBUTED']}")
    _emit(f"SIMS_CONFIGURED={state['SEARCH_SIMS_CONFIGURED']}")
    _emit(f"SIMS_CONTRIBUTED_PER_DECISION={sims_contributed_per_decision:.4f}")
    _emit(f"SIMS_CONFIGURED_PER_DECISION={sims_configured_per_decision:.4f}")
    _emit(f"PRIZE_REWARD_REACHED={_fmt_pct(state['SEARCH_PRIZE_REWARD_REACHED'], search_decisions)}({state['SEARCH_PRIZE_REWARD_REACHED']}/{search_decisions})")
    _emit(f"STALL_DRAW_PASS_TURNS={state['STALL_DRAW_PASS_TURNS']}")
    _emit(f"STALL_SHUFFLE_WITH_RESOURCES={state['STALL_SHUFFLE_WITH_RESOURCES']}")
    _emit(f"MISSED_ATTACH={state['MISSED_ATTACH']}")
    _emit(f"SHAPING_BREAKDOWN={shaping_text}")
    _emit(f"ROOT_OPTION_STATS={root_opt_text}")
    _emit(f"ATTACH_PRIOR_DIST={attach_prior_text}")
    _emit(f"SIM_KO_REACHED={_fmt_pct(sim_ko_reached, sim_ko_total)}({sim_ko_reached}/{sim_ko_total})")
    _emit(f"SIM_KO_DEPTH_MEAN={sim_ko_depth_mean:.4f}")
    _emit(f"SIM_KO_DEPTH_MIN={sim_ko_depth_min}")
    _emit(f"SIM_KO_DEPTH_MAX={state['SIM_KO_DEPTH_MAX']}")
    _emit(
        f"TRUE_RESULT_TALLY=P0_WINS={state['TRUE_P0_WINS']},"
        f"P1_WINS={state['TRUE_P1_WINS']},DRAWS={state['TRUE_DRAWS']}"
    )
    policy_target_count = state["POLICY_TARGET_COUNT"]
    policy_target_zero_fraction_mean = (
        state["POLICY_TARGET_ZERO_FRACTION_SUM"] / policy_target_count if policy_target_count else 0.0
    )
    policy_target_entropy_mean = (
        state["POLICY_TARGET_ENTROPY_SUM"] / policy_target_count if policy_target_count else 0.0
    )
    _emit(f"POLICY_TARGET_ZERO_FRACTION_MEAN={policy_target_zero_fraction_mean:.4f}")
    _emit(f"POLICY_TARGET_ENTROPY_MEAN={policy_target_entropy_mean:.4f}")
    _emit(f"POLICY_TARGET_COUNT={policy_target_count}")

    root_child_counts = state["ROOT_CHILD_COUNTS"]
    root_mean, root_median = _mean_median(root_child_counts)
    by_turn_child_parts = []
    for bucket in ("1-2", "3-5", "6+"):
        vals = state["ROOT_CHILD_COUNTS_BY_TURN"][bucket]
        b_mean, b_median = _mean_median(vals)
        by_turn_child_parts.append(f"{bucket}:mean={b_mean:.2f},median={b_median:.1f},n={len(vals)}")
    _emit(
        f"ROOT_CHILD_COUNT=mean={root_mean:.2f},median={root_median:.1f},n={len(root_child_counts)},"
        f"by_turn=({'|'.join(by_turn_child_parts)})"
    )

    visit_hist = state["ROOT_VISIT_HIST"]
    visit_hist_total = sum(visit_hist.values())
    visit_hist_parts = [
        f"{label}:{_fmt_pct(visit_hist[label], visit_hist_total)}({visit_hist[label]}/{visit_hist_total})"
        for label in ("0", "1", "2-4", "5+")
    ]
    _emit(f"ROOT_VISIT_HIST={'|'.join(visit_hist_parts)}")

    edev_legal = state["ENERGY_DEV_LEGAL"]
    edev_chosen = state["ENERGY_DEV_CHOSEN"]
    _emit(f"ENERGY_DEV_RATE={_fmt_pct(edev_chosen, edev_legal)}({edev_chosen}/{edev_legal})")

    grass_present = state["GRASS_IN_DISCARD_PRESENT"]
    grass_total = state["GRASS_IN_DISCARD_TOTAL"]
    _emit(f"GRASS_IN_DISCARD_RATE={_fmt_pct(grass_present, grass_total)}({grass_present}/{grass_total})")
    _emit("=== DIAG_DUMP_END ===")


def dump() -> None:
    if not DIAG_ENABLED:
        return
    _dump_from_state(_current, games=1, agent=_current_agent)


def end_game() -> None:
    global _games_in_window, _window
    if not DIAG_ENABLED:
        return

    _flush_turn_flags()

    _merge_current_into_window()
    _games_in_window += 1

    if DIAG_VERBOSE:
        _dump_from_state(_current, games=1, agent=_current_agent)
    elif _games_in_window >= _DUMP_EVERY_GAMES:
        _dump_from_state(_window, games=_games_in_window, agent=_current_agent)
        _window = _new_state()
        _games_in_window = 0