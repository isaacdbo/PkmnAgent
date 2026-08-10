from types import SimpleNamespace

import pytest

import reward_harness


def test_terminal_only_draw_is_zero():
    assert reward_harness.GameResult(winner=None, turns=20, capped=True, seconds=1.0).terminal_only == 0.0


def test_wilson_interval_bounds_perfect_record():
    lo, hi = reward_harness.wilson_interval(40, 40)
    assert 0.91 < lo < 0.92
    assert hi == pytest.approx(1.0)


def test_fast_rule_agent_plays_before_attacking_or_ending():
    obs = SimpleNamespace(
        select=SimpleNamespace(
            context=0,
            maxCount=1,
            option=[
                SimpleNamespace(type=14),
                SimpleNamespace(type=7),
                SimpleNamespace(type=13),
            ],
        )
    )

    assert reward_harness.select_fast_action(obs) == [1]
