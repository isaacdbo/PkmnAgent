"""The reward registry, with the baseline arm pinned to pre-harness behaviour.

These run without torch, the engine, or a checkpoint: `ablation.rewards` is
pure stdlib precisely so the contract it defines can be checked on any host.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation import rewards  # noqa: E402


class TestBaselineIsUnchanged:
    """`baseline` must reproduce the reward this repo had before the harness.

    If any of these fail, every ablation result is uninterpretable: the
    control arm would no longer be the control.
    """

    @pytest.mark.parametrize(
        "result,player_index,expected",
        [
            (0, 0, 1.0),    # seat 0 won, scoring seat 0
            (0, 1, -1.0),   # seat 0 won, scoring seat 1
            (1, 1, 1.0),
            (1, 0, -1.0),
            (2, 0, 0.0),    # draw
            (2, 1, 0.0),
        ],
    )
    def test_terminal_target_is_plus_one_zero_minus_one(self, result, player_index, expected):
        spec = rewards.get("baseline")
        assert rewards.terminal_value(spec, result=result, player_index=player_index) == expected

    @pytest.mark.parametrize("cause", [rewards.CAUSE_PRIZE, rewards.CAUSE_DECKOUT,
                                       rewards.CAUSE_OTHER, None])
    @pytest.mark.parametrize("final_turn", [1, 20, 40, 99, None])
    def test_cause_and_turn_do_not_matter(self, cause, final_turn):
        spec = rewards.get("baseline")
        z = rewards.terminal_value(spec, result=0, player_index=0,
                                   cause=cause, final_turn=final_turn)
        assert z == 1.0

    def test_shaping_passes_through_untouched(self):
        spec = rewards.get("baseline")
        terms = {"PRIZE_DIFF": 0.15, "KO_BONUS": 0.2, "DAMAGE_PRESSURE": -0.03}
        assert spec.apply_shaping(terms) == terms

    def test_board_diff_coef_not_overridden(self):
        assert rewards.get("baseline").board_diff_coef is None

    def test_is_the_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("REWARD_SPEC", raising=False)
        assert rewards.active_spec().name == "baseline"


class TestTerminalOnly:
    def test_every_shaping_term_is_zeroed(self):
        spec = rewards.get("terminal_only")
        terms = {name: 0.5 for name in rewards.SHAPING_TERMS}
        assert set(spec.apply_shaping(terms).values()) == {0.0}

    def test_covers_every_known_term(self):
        spec = rewards.get("terminal_only")
        assert set(spec.shaping_weights) == set(rewards.SHAPING_TERMS)

    def test_board_reward_is_off(self):
        assert rewards.get("terminal_only").board_diff_coef == 0.0

    def test_terminal_target_is_still_plus_minus_one(self):
        spec = rewards.get("terminal_only")
        assert rewards.terminal_value(spec, result=0, player_index=0) == 1.0
        assert rewards.terminal_value(spec, result=0, player_index=1) == -1.0


class TestDeckoutPenalty:
    def test_deckout_win_is_worth_less_than_a_prize_win(self):
        spec = rewards.get("deckout_penalty")
        prize_win = rewards.terminal_value(spec, result=0, player_index=0,
                                           cause=rewards.CAUSE_PRIZE)
        deckout_win = rewards.terminal_value(spec, result=0, player_index=0,
                                             cause=rewards.CAUSE_DECKOUT)
        assert prize_win == 1.0
        assert deckout_win == 0.25
        assert deckout_win < prize_win

    def test_hard_variant_makes_a_deckout_win_negative(self):
        spec = rewards.get("deckout_penalty_hard")
        assert rewards.terminal_value(spec, result=0, player_index=0,
                                      cause=rewards.CAUSE_DECKOUT) == -0.25

    def test_losses_are_untouched(self):
        spec = rewards.get("deckout_penalty")
        assert rewards.terminal_value(spec, result=1, player_index=0,
                                      cause=rewards.CAUSE_DECKOUT) == -1.0

    def test_unknown_cause_falls_back_to_plain_outcome(self):
        """A missing reason code must never be read as a deck-out."""
        spec = rewards.get("deckout_penalty")
        assert rewards.terminal_value(spec, result=0, player_index=0, cause=None) == 1.0

    def test_shaping_is_still_on(self):
        spec = rewards.get("deckout_penalty")
        assert spec.apply_shaping({"KO_BONUS": 0.2}) == {"KO_BONUS": 0.2}


class TestTurnsToWin:
    def test_a_faster_win_is_worth_more(self):
        spec = rewards.get("turns_to_win_mild")
        fast = rewards.terminal_value(spec, result=0, player_index=0, final_turn=5)
        slow = rewards.terminal_value(spec, result=0, player_index=0, final_turn=40)
        assert fast > slow
        assert slow == pytest.approx(0.75)

    def test_a_slower_loss_costs_less(self):
        spec = rewards.get("turns_to_win_mild")
        fast = rewards.terminal_value(spec, result=1, player_index=0, final_turn=5)
        slow = rewards.terminal_value(spec, result=1, player_index=0, final_turn=40)
        assert slow > fast
        assert slow == pytest.approx(-0.75)

    def test_strong_variant_is_twice_the_discount(self):
        mild = rewards.get("turns_to_win_mild")
        strong = rewards.get("turns_to_win_strong")
        assert rewards.terminal_value(strong, result=0, player_index=0,
                                      final_turn=40) == pytest.approx(0.5)
        assert rewards.terminal_value(mild, result=0, player_index=0,
                                      final_turn=40) == pytest.approx(0.75)

    def test_past_the_reference_turn_the_discount_stops_growing(self):
        spec = rewards.get("turns_to_win_mild")
        at_ref = rewards.terminal_value(spec, result=0, player_index=0, final_turn=40)
        way_past = rewards.terminal_value(spec, result=0, player_index=0, final_turn=400)
        assert at_ref == way_past

    def test_missing_turn_count_disables_the_term(self):
        spec = rewards.get("turns_to_win_mild")
        assert rewards.terminal_value(spec, result=0, player_index=0, final_turn=None) == 1.0

    def test_a_draw_stays_zero(self):
        spec = rewards.get("turns_to_win_strong")
        assert rewards.terminal_value(spec, result=2, player_index=0, final_turn=40) == 0.0

    def test_sign_still_says_who_won(self):
        """The discount must never push a win to <= 0 or a loss to >= 0."""
        for name in rewards.REGISTRY:
            spec = rewards.get(name)
            if name == "deckout_penalty_hard":
                continue  # deliberately scores a deck-out win negative
            for turn in (1, 10, 40, 200):
                for cause in (rewards.CAUSE_PRIZE, rewards.CAUSE_DECKOUT, None):
                    win = rewards.terminal_value(spec, result=0, player_index=0,
                                                 cause=cause, final_turn=turn)
                    loss = rewards.terminal_value(spec, result=0, player_index=1,
                                                  cause=cause, final_turn=turn)
                    assert win > 0, (name, turn, cause)
                    assert loss < 0, (name, turn, cause)


class TestCombinedArm:
    def test_both_levers_apply(self):
        spec = rewards.get("deckout_penalty_turns")
        # 0.25 for the deck-out win, then discounted 25% for a turn-40 game.
        assert rewards.terminal_value(
            spec, result=0, player_index=0,
            cause=rewards.CAUSE_DECKOUT, final_turn=40,
        ) == pytest.approx(0.1875)


class TestRegistryContract:
    def test_every_value_stays_in_the_tanh_range(self):
        """The value head is a tanh; targets outside [-1, 1] are unreachable."""
        for name in rewards.REGISTRY:
            spec = rewards.get(name)
            for result in (0, 1, 2):
                for player_index in (0, 1):
                    for cause in (rewards.CAUSE_PRIZE, rewards.CAUSE_DECKOUT,
                                  rewards.CAUSE_OTHER, None):
                        for turn in (0, 1, 40, 500, None):
                            z = rewards.terminal_value(
                                spec, result=result, player_index=player_index,
                                cause=cause, final_turn=turn,
                            )
                            assert -1.0 <= z <= 1.0, (name, result, cause, turn)

    def test_unknown_spec_name_fails_with_the_choices(self):
        with pytest.raises(ValueError, match="terminal_only"):
            rewards.get("no_such_arm")

    def test_active_spec_reads_the_environment(self, monkeypatch):
        monkeypatch.setenv("REWARD_SPEC", "deckout_penalty")
        assert rewards.active_spec().name == "deckout_penalty"

    def test_active_spec_rejects_a_typo_rather_than_falling_back(self, monkeypatch):
        monkeypatch.setenv("REWARD_SPEC", "deckout-penalty")
        with pytest.raises(ValueError):
            rewards.active_spec()

    def test_registry_keys_match_spec_names(self):
        for name, spec in rewards.REGISTRY.items():
            assert spec.name == name

    def test_every_spec_has_a_description(self):
        for spec in rewards.REGISTRY.values():
            assert len(spec.description) > 40

    def test_validate_terms_accepts_the_known_set(self):
        rewards.validate_terms(rewards.SHAPING_TERMS)

    def test_validate_terms_rejects_an_unregistered_term(self):
        with pytest.raises(ValueError, match="NEW_TERM"):
            rewards.validate_terms(list(rewards.SHAPING_TERMS) + ["NEW_TERM"])

    def test_cause_mapping_matches_the_engine_reason_codes(self):
        assert rewards.cause_from_reason(1) == rewards.CAUSE_PRIZE
        assert rewards.cause_from_reason(2) == rewards.CAUSE_DECKOUT
        assert rewards.cause_from_reason(3) == rewards.CAUSE_OTHER
        assert rewards.cause_from_reason(4) == rewards.CAUSE_OTHER
        assert rewards.cause_from_reason(None) is None

    def test_describe_is_stable_and_names_the_arm(self):
        for name in rewards.REGISTRY:
            assert rewards.get(name).describe().startswith(f"name={name}")
