"""Named reward specifications for the ablation, selected by REWARD_SPEC.

Why this exists
---------------
FINDINGS.md records that deck-out share sits at 80-90% of self-play games and
rises monotonically with search depth (up to 100% at 800 sims), while
PRIZE_REWARD_REACHED stays flat under 1%. That is the signature of a reward
function under which running the opponent out of cards is a better-valued
outcome than attacking, so the search finds it and the value target confirms
it. Testing that claim means changing one reward term at a time and re-scoring
the result against a fixed panel — which is what this registry, plus
`ablation/driver.py`, exist to do.

Two knobs are exposed, matching the two places reward actually enters training:

  shaping_weights   per-term multipliers on `RLTRM2.shaped_reward_terms`, which
                    is added to the MCTS *leaf value* (RLTRM2.py:822-827). It
                    steers search, never the training target.
  terminal_value()  the value target z written onto every LearnSample of a
                    finished game (`_backup_and_store`). It is what the value
                    head actually regresses on.

`board_diff_coef` overrides DECK_DIFF_COEF, the existing anti-mill term in
`board_reward`, so an arm can turn it off rather than inheriting the ambient
env value.

The default spec is `baseline`, which is byte-for-byte the behaviour of the
repo before this harness existed: all shaping weights 1.0, terminal target
+1/0/-1, DECK_DIFF_COEF untouched. `tests/test_rewards.py` pins that.

Terminal values are clamped to [-1, 1] because the value head is a tanh
(RLTRM2.MyModel.forward) and cannot represent targets outside that range;
asking it to would just saturate the head and distort the gradient.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

# Reason codes as the engine reports them, mirrored from diag._WIN_CAUSE_LABELS.
CAUSE_PRIZE = "prize"
CAUSE_DECKOUT = "deckout"
CAUSE_OTHER = "other"

REASON_TO_CAUSE = {
    1: CAUSE_PRIZE,
    2: CAUSE_DECKOUT,
    3: CAUSE_OTHER,  # NO_ACTIVE
    4: CAUSE_OTHER,  # CARD_EFFECT_OR_CONCEDE
}

# Every term name produced by RLTRM2.shaped_reward_terms. Listed explicitly so
# a spec that means "no shaping at all" cannot silently miss a term that gets
# added to RLTRM2 later — `validate_terms` fails loudly instead.
SHAPING_TERMS = (
    "PRIZE_DIFF",
    "KO_BONUS",
    "ENERGY_DENIAL",
    "DISRUPTION",
    "DAMAGE_PRESSURE",
    "PARALYSIS",
    "SLEEP",
    "ITEM_LOCK_PENALTY",
)

# Turn count treated as "a full-length game" when scaling a turns-to-win
# penalty. 40 is the top of diag's 26-40 game-length bucket and roughly where
# the observed length histograms stop being dense (eval_panel_arm_B.log:
# 1-15:22|16-25:7|26-40:3|41-60:7|61+:1).
DEFAULT_TURN_REFERENCE = 40


@dataclass(frozen=True)
class RewardSpec:
    """One ablation arm's reward definition."""

    name: str
    description: str
    # Per-term multiplier applied to shaped_reward_terms. Missing term => 1.0.
    shaping_weights: Mapping[str, float] = field(default_factory=dict)
    # Override for DECK_DIFF_COEF in board_reward. None => leave RLTRM2 alone.
    board_diff_coef: float | None = None
    # Terminal value when the game ended by deck-out. None => the ordinary
    # +1 / -1 (i.e. a deck-out win is worth exactly as much as a prize win).
    deckout_win_value: float | None = None
    deckout_loss_value: float | None = None
    # Turns-to-win shaping of the terminal target: a win late in the game is
    # worth less, a loss late in the game costs less. 0.0 => off.
    turns_coef: float = 0.0
    turn_reference: int = DEFAULT_TURN_REFERENCE

    def weight_for(self, term: str) -> float:
        return float(self.shaping_weights.get(term, 1.0))

    def apply_shaping(self, terms: dict[str, float]) -> dict[str, float]:
        """Scale a shaped_reward_terms dict in place-compatible fashion."""
        if not self.shaping_weights:
            return terms
        return {k: v * self.weight_for(k) for k, v in terms.items()}

    def describe(self) -> str:
        bits = [f"name={self.name}"]
        if self.shaping_weights:
            zeroed = sorted(k for k, v in self.shaping_weights.items() if v == 0.0)
            scaled = sorted(f"{k}x{v}" for k, v in self.shaping_weights.items() if v != 0.0)
            if zeroed:
                bits.append("shaping_off=" + ",".join(zeroed))
            if scaled:
                bits.append("shaping=" + ",".join(scaled))
        if self.board_diff_coef is not None:
            bits.append(f"board_diff_coef={self.board_diff_coef}")
        if self.deckout_win_value is not None:
            bits.append(f"deckout_win={self.deckout_win_value}")
        if self.deckout_loss_value is not None:
            bits.append(f"deckout_loss={self.deckout_loss_value}")
        if self.turns_coef:
            bits.append(f"turns_coef={self.turns_coef}@ref{self.turn_reference}")
        return " ".join(bits)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "shaping_weights": dict(self.shaping_weights),
            "board_diff_coef": self.board_diff_coef,
            "deckout_win_value": self.deckout_win_value,
            "deckout_loss_value": self.deckout_loss_value,
            "turns_coef": self.turns_coef,
            "turn_reference": self.turn_reference,
        }


def _all_terms_off() -> dict[str, float]:
    return {term: 0.0 for term in SHAPING_TERMS}


REGISTRY: dict[str, RewardSpec] = {
    "baseline": RewardSpec(
        name="baseline",
        description=(
            "The repo's reward as-is: full shaping into the MCTS leaf value, "
            "terminal target +1/0/-1, DECK_DIFF_COEF from the environment. The "
            "control arm — must reproduce pre-harness behaviour exactly."
        ),
    ),
    "terminal_only": RewardSpec(
        name="terminal_only",
        description=(
            "No shaping anywhere: every shaped_reward_terms component is zeroed "
            "and board_reward's deck-difference term is off, so the only signal "
            "is who won. The floor the other arms have to beat — if shaping is "
            "not earning its keep, this arm says so."
        ),
        shaping_weights=_all_terms_off(),
        board_diff_coef=0.0,
    ),
    "deckout_penalty": RewardSpec(
        name="deckout_penalty",
        description=(
            "Baseline shaping, but a win by deck-out is worth 0.25 instead of "
            "1.0 at the training target. Directly attacks the FINDINGS.md "
            "pathology: milling still wins the game, it just stops being worth "
            "as much as taking prizes."
        ),
        deckout_win_value=0.25,
    ),
    "deckout_penalty_hard": RewardSpec(
        name="deckout_penalty_hard",
        description=(
            "As deckout_penalty but a deck-out win scores -0.25: winning by "
            "mill is treated as a failure to find the prize line. Deliberately "
            "over-corrected, to bracket the effect size."
        ),
        deckout_win_value=-0.25,
    ),
    "turns_to_win_mild": RewardSpec(
        name="turns_to_win_mild",
        description=(
            "Baseline shaping plus a turn-count term on the terminal target: a "
            "win at turn 40 is worth 0.75 instead of 1.0, and a loss at turn 40 "
            "costs -0.75 instead of -1.0. Prefers short decisive games without "
            "saying how to get them."
        ),
        turns_coef=0.25,
    ),
    "turns_to_win_strong": RewardSpec(
        name="turns_to_win_strong",
        description=(
            "Same shape as turns_to_win_mild at twice the strength (a turn-40 "
            "win is worth 0.5). Present so the driver can tell a real effect "
            "from a coefficient that was simply too small to matter."
        ),
        turns_coef=0.5,
    ),
    "deckout_penalty_turns": RewardSpec(
        name="deckout_penalty_turns",
        description=(
            "Both levers at once: deck-out wins devalued and long games "
            "discounted. Tells whether the two effects compose or overlap."
        ),
        deckout_win_value=0.25,
        turns_coef=0.25,
    ),
}

DEFAULT_SPEC_NAME = "baseline"


def validate_terms(term_names) -> None:
    """Fail loudly if RLTRM2 grew a shaping term this registry does not know.

    A spec that means 'no shaping' expresses that as an explicit zero per term,
    so an unknown term would silently keep firing at full strength and quietly
    invalidate the terminal_only arm.
    """
    unknown = sorted(set(term_names) - set(SHAPING_TERMS))
    if unknown:
        raise ValueError(
            f"shaped_reward_terms produced term(s) unknown to ablation.rewards: {unknown}. "
            f"Add them to SHAPING_TERMS (and to any spec that zeroes shaping) before "
            f"trusting an ablation result."
        )


def get(name: str) -> RewardSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown REWARD_SPEC {name!r}. Choices: {', '.join(sorted(REGISTRY))}"
        ) from None


def active_spec() -> RewardSpec:
    """The spec selected by the REWARD_SPEC environment variable.

    Read from the environment (not passed around) because training fans out
    through multiprocessing 'spawn' workers that re-import RLTRM2 in a fresh
    interpreter; the environment is what reliably crosses that boundary.
    """
    return get(os.environ.get("REWARD_SPEC", DEFAULT_SPEC_NAME))


def clamp_value(v: float) -> float:
    return max(-1.0, min(1.0, v))


def terminal_value(
    spec: RewardSpec,
    *,
    result: int,
    player_index: int,
    cause: str | None = None,
    final_turn: int | None = None,
) -> float:
    """The value target z for one player of a finished game.

    `result` is the engine's raw result field: 0/1 for the winning seat index,
    2 for a draw. `cause` is one of CAUSE_*; None means "unknown", which is
    treated as the ordinary +1/-1 so a missing reason code can never silently
    turn into a reward change.
    """
    if result == 2:
        return 0.0

    won = result == player_index
    if cause == CAUSE_DECKOUT:
        if won and spec.deckout_win_value is not None:
            z = spec.deckout_win_value
        elif not won and spec.deckout_loss_value is not None:
            z = spec.deckout_loss_value
        else:
            z = 1.0 if won else -1.0
    else:
        z = 1.0 if won else -1.0

    if spec.turns_coef and final_turn is not None and spec.turn_reference > 0:
        # Longer games move z toward 0 from whichever side it started on: a
        # slow win is worth less, a slow loss costs less. Never crosses zero,
        # so the sign of the target still says who won.
        lateness = min(1.0, max(0.0, final_turn / float(spec.turn_reference)))
        z *= 1.0 - spec.turns_coef * lateness

    return clamp_value(z)


def cause_from_reason(reason_code: int | None) -> str | None:
    if reason_code is None:
        return None
    return REASON_TO_CAUSE.get(reason_code, CAUSE_OTHER)
