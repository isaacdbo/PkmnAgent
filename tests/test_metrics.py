"""The diagnostic metrics: rates, intervals, and what "unknown" must mean."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation import metrics  # noqa: E402
from ablation.metrics import GameRecord  # noqa: E402


def game(outcome, cause=None, turn=None, legal=0, chosen=0, opponent="random", **kw):
    return GameRecord(
        game_index=kw.pop("game_index", 1),
        opponent=opponent,
        outcome=outcome,
        cause=cause,
        final_turn=turn,
        attack_legal_decisions=legal,
        attack_chosen_decisions=chosen,
        **kw,
    )


class TestWinRate:
    def test_counts_and_rate(self):
        s = metrics.summarize([game("win"), game("win"), game("loss")])
        assert (s.wins, s.losses, s.draws, s.decided) == (2, 1, 0, 3)
        assert s.win_rate == pytest.approx(2 / 3)

    def test_draws_are_excluded_from_the_denominator(self):
        s = metrics.summarize([game("win"), game("loss"), game("draw")])
        assert s.decided == 2
        assert s.win_rate == pytest.approx(0.5)
        assert s.games == 3

    def test_no_games_reports_none_not_zero(self):
        s = metrics.summarize([])
        assert s.win_rate is None
        assert s.win_rate_ci_low is None

    def test_all_draws_leaves_win_rate_undefined(self):
        s = metrics.summarize([game("draw"), game("draw")])
        assert s.win_rate is None


class TestWilsonInterval:
    def test_brackets_the_point_estimate(self):
        low, high = metrics.wilson_ci(10, 20)
        assert low < 0.5 < high

    def test_is_wider_for_fewer_games(self):
        narrow = metrics.wilson_ci(200, 400)
        wide = metrics.wilson_ci(10, 20)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_stays_inside_zero_to_one_at_the_extremes(self):
        assert metrics.wilson_ci(0, 5)[0] == 0.0
        assert metrics.wilson_ci(5, 5)[1] == 1.0

    def test_zero_games_is_not_a_division_error(self):
        assert metrics.wilson_ci(0, 0) == (0.0, 0.0)

    def test_matches_an_independent_implementation(self):
        """eval_panel carries its own copy of this formula; drift would be a bug.

        Checked against the textbook Wilson expression written a different way
        rather than against constants copied out of this implementation, so the
        test can actually fail if the implementation is wrong.
        """
        import math

        def reference(wins, total, z=1.959963984540054):
            p = wins / total
            z2 = z * z
            center = (p + z2 / (2 * total)) / (1 + z2 / total)
            half = (z / (1 + z2 / total)) * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
            return center - half, center + half

        for wins, total in [(0, 40), (1, 40), (11, 20), (20, 40), (39, 40), (40, 40)]:
            low, high = metrics.wilson_ci(wins, total)
            ref_low, ref_high = reference(wins, total)
            assert low == pytest.approx(max(0.0, ref_low), abs=1e-12)
            assert high == pytest.approx(min(1.0, ref_high), abs=1e-12)


class TestDeckoutRate:
    def test_share_of_games_with_a_known_cause(self):
        s = metrics.summarize([
            game("win", cause="deckout"),
            game("win", cause="deckout"),
            game("loss", cause="prize"),
            game("loss", cause="other"),
        ])
        assert s.deckout_rate == pytest.approx(0.5)
        assert s.deckout_win_rate == pytest.approx(0.5)
        assert s.deckout_loss_rate == 0.0
        assert s.prize_rate == pytest.approx(0.25)

    def test_unknown_causes_are_excluded_not_counted_as_zero(self):
        """A log with no reason codes must read as n/a, never as 0% deck-out."""
        s = metrics.summarize([game("win"), game("loss")])
        assert s.games_with_cause == 0
        assert s.deckout_rate is None

    def test_partial_cause_coverage_uses_only_the_known_games(self):
        s = metrics.summarize([
            game("win", cause="deckout"),
            game("loss"),  # cause unknown
        ])
        assert s.games_with_cause == 1
        assert s.deckout_rate == 1.0


class TestTurns:
    def test_separates_wins_from_losses(self):
        s = metrics.summarize([
            game("win", turn=10),
            game("win", turn=20),
            game("loss", turn=50),
        ])
        assert s.turns_to_win_mean == pytest.approx(15.0)
        assert s.turns_to_win_median == pytest.approx(15.0)
        assert s.turns_to_loss_mean == pytest.approx(50.0)
        assert s.turns_mean == pytest.approx(80 / 3)

    def test_missing_turn_counts_are_dropped_from_the_mean(self):
        s = metrics.summarize([game("win", turn=10), game("win")])
        assert s.games_with_turn == 1
        assert s.turns_to_win_mean == pytest.approx(10.0)

    def test_no_wins_leaves_turns_to_win_undefined(self):
        s = metrics.summarize([game("loss", turn=10)])
        assert s.turns_to_win_mean is None


class TestAttackRate:
    def test_is_chosen_over_legal_not_over_all_decisions(self):
        s = metrics.summarize([game("win", legal=10, chosen=3), game("loss", legal=10, chosen=7)])
        assert s.attack_rate == pytest.approx(0.5)
        assert s.attack_legal_decisions == 20

    def test_never_legal_is_undefined_rather_than_zero(self):
        s = metrics.summarize([game("win", legal=0, chosen=0)])
        assert s.attack_rate is None


class TestWinByCause:
    def test_tallies_wins_losses_and_draws_per_cause(self):
        s = metrics.summarize([
            game("win", cause="deckout"),
            game("loss", cause="deckout"),
            game("draw", cause="other"),
        ])
        assert s.win_by_cause["deckout"] == {"W": 1, "L": 1, "D": 0}
        assert s.win_by_cause["other"] == {"W": 0, "L": 0, "D": 1}


class TestGrouping:
    def test_summarize_by_opponent(self):
        rows = metrics.summarize_by([
            game("win", opponent="random"),
            game("loss", opponent="random"),
            game("win", opponent="iono_rule"),
        ], "opponent")
        assert rows[("random",)].win_rate == pytest.approx(0.5)
        assert rows[("iono_rule",)].win_rate == 1.0


class TestRoundTrip:
    def test_jsonl_survives_a_write_read_cycle(self, tmp_path):
        records = [
            game("win", cause="prize", turn=12, legal=5, chosen=2, game_index=1),
            game("loss", cause="deckout", turn=44, game_index=2),
        ]
        path = tmp_path / "games.jsonl"
        assert metrics.write_games_jsonl(records, path) == 2
        back = metrics.read_games_jsonl(path)
        assert [r.outcome for r in back] == ["win", "loss"]
        assert back[0].final_turn == 12
        assert back[1].cause == "deckout"
        assert metrics.summarize(back).to_dict() == metrics.summarize(records).to_dict()

    def test_csv_has_a_row_per_group(self, tmp_path):
        rows = {"baseline": metrics.summarize([game("win", cause="prize", turn=9)])}
        path = tmp_path / "summary.csv"
        metrics.write_summary_csv(rows, path, label="arm")
        text = path.read_text()
        assert "arm" in text.splitlines()[0]
        assert "baseline" in text


class TestRendering:
    def test_table_has_a_row_per_group_and_the_metric_columns(self):
        rows = {
            "baseline": metrics.summarize([game("win", cause="deckout", turn=30, legal=4, chosen=1)]),
            "terminal_only": metrics.summarize([game("loss", cause="prize", turn=12)]),
        }
        table = metrics.render_table(rows, label="arm")
        assert "arm" in table
        assert "deckout_rate" in table
        assert "attack_rate" in table
        assert "baseline" in table and "terminal_only" in table

    def test_markdown_table_is_pipe_delimited(self):
        rows = {"baseline": metrics.summarize([game("win")])}
        md = metrics.render_markdown_table(rows, label="arm")
        assert md.startswith("| arm |")
        assert md.splitlines()[1].startswith("|---")

    def test_undefined_metrics_render_as_na(self):
        rows = {"x": metrics.summarize([game("win")])}
        table = metrics.render_table(rows)
        assert "n/a" in table
