"""Ingestion of the repo's raw eval output, including its older log formats."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation import ingest  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Verbatim from eval_panel_arm_B.log — the format before winner=/turn= existed.
OLD_EVAL_LOG = """\
EVAL_CANDIDATE=RANDOM_INIT
EVAL_CANDIDATE_DECK=M2Deck.xlsx
EVAL_SIMULATIONS_PER_MOVE=20
EVAL_GAMES_PER_OPPONENT=4
EVAL_BASE_SEED=20260809

=== PANEL: candidate vs sample_bot ===
EVAL_GAME_DONE=1/4 p0=candidate p1=opponent result=1 reason=3 cause=other sec=1.12
EVAL_GAME_DONE=2/4 p0=opponent p1=candidate result=0 reason=3 cause=other sec=0.58
EVAL_GAME_DONE=3/4 p0=candidate p1=opponent result=0 reason=2 cause=deckout sec=1.24
EVAL_GAME_DONE=4/4 p0=opponent p1=candidate result=1 reason=1 cause=prize sec=1.30
OPP_WALL_SEC=4.24
OPP_sample_bot_CANDIDATE: wins=2 losses=2 draws=0 win_rate=50.0%(2/4)
"""

# Verbatim shape from overnight_results/before_eval.log — the current format.
NEW_EVAL_LOG = """\
EVAL_CANDIDATE=checkpoints/m2/model_2026-08-11_00-16.pth
EVAL_GAMES_PER_OPPONENT=2

=== PANEL: candidate vs random ===
EVAL_GAME_DONE=1/2 winner=random(random) p0=candidate p1=random(random) result=1 reason=1 cause=prize turn=19 cand_prizes=0 opp_prizes=6 sec=15.97
EVAL_GAME_DONE=2/2 winner=random(random) p0=random(random) p1=candidate result=0 reason=1 cause=prize turn=15 cand_prizes=0 opp_prizes=6 sec=12.64
"""

H2H_LOG = """\
H2H_SIMULATIONS_PER_MOVE=20
H2H_NEW=checkpoints/m2/model_2026-08-09_03-38.pth
H2H_OLD=checkpoints/m2/model_2026-08-08_09-48.pth
H2H_GAME_DONE=1/2 p0=new p1=old result=1 reason=2 cause=deckout sec=6.80
H2H_GAME_DONE=2/2 p0=old p1=new result=1 reason=2 cause=deckout sec=5.88
"""

DIAG_LOG = """\
ARM_DECK_DIFF_COEF=0.0
ARM_CHECKPOINT=checkpoints/m2/model_2026-08-08_09-48.pth
=== DIAG_DUMP_BEGIN ===
DIAG_GAMES=20
AGENT=m2
GAME_LENGTH_HIST=1-15:2|16-25:7|26-40:10|41-60:1|61+:0
WIN_BY_CAUSE=DECK_OUT:W=16,L=16,D=0|NO_ACTIVE:W=4,L=4,D=0
ROOT_OPTION_STATS=ATTACK:available=100,chosen=25,not_chosen=75,mean_opt_visit=1.12|ENERGY_ATTACH:available=2161,chosen=66,not_chosen=2095
=== DIAG_DUMP_END ===
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


class TestOldEvalFormat:
    def test_attributes_wins_by_seat_not_by_raw_result(self, tmp_path):
        """result= is the winning seat, and the candidate's seat alternates.

        Reading result= as "the candidate won" is the exact misreading
        FINDINGS.md records as falsified, so it is worth a test.
        """
        games, meta = ingest.parse_game_lines(write(tmp_path, "eval.log", OLD_EVAL_LOG))
        assert [g.outcome for g in games] == ["loss", "loss", "win", "win"]
        assert meta["EVAL_CANDIDATE"] == "RANDOM_INIT"

    def test_reads_the_opponent_from_the_panel_header(self, tmp_path):
        games, _ = ingest.parse_game_lines(write(tmp_path, "eval.log", OLD_EVAL_LOG))
        assert {g.opponent for g in games} == {"sample_bot"}

    def test_missing_turn_and_prize_fields_stay_unset(self, tmp_path):
        games, _ = ingest.parse_game_lines(write(tmp_path, "eval.log", OLD_EVAL_LOG))
        assert all(g.final_turn is None for g in games)
        assert all(g.candidate_prizes is None for g in games)

    def test_causes_are_normalised(self, tmp_path):
        games, _ = ingest.parse_game_lines(write(tmp_path, "eval.log", OLD_EVAL_LOG))
        assert [g.cause for g in games] == ["other", "other", "deckout", "prize"]


class TestNewEvalFormat:
    def test_reads_turn_and_prize_counts(self, tmp_path):
        games, _ = ingest.parse_game_lines(write(tmp_path, "eval.log", NEW_EVAL_LOG))
        assert [g.final_turn for g in games] == [19, 15]
        assert [g.opponent_prizes for g in games] == [6, 6]

    def test_display_names_do_not_break_seat_attribution(self, tmp_path):
        games, _ = ingest.parse_game_lines(write(tmp_path, "eval.log", NEW_EVAL_LOG))
        assert [g.outcome for g in games] == ["loss", "loss"]

    def test_checkpoint_is_carried_onto_every_record(self, tmp_path):
        games, _ = ingest.parse_game_lines(write(tmp_path, "eval.log", NEW_EVAL_LOG))
        assert all(g.checkpoint.endswith("model_2026-08-11_00-16.pth") for g in games)


class TestHeadToHead:
    def test_new_side_is_the_candidate(self, tmp_path):
        """Both games have result=1; the sides swap seats, so the outcomes differ."""
        games, _ = ingest.parse_game_lines(write(tmp_path, "h2h.log", H2H_LOG))
        assert [g.outcome for g in games] == ["loss", "win"]
        assert {g.opponent for g in games} == {"old_m2(h2h)"}


class TestDiagDumps:
    def test_parses_win_by_cause_into_a_deckout_rate(self, tmp_path):
        dumps = ingest.parse_diag_dumps(write(tmp_path, "arm.txt", DIAG_LOG))
        assert len(dumps) == 1
        # 16 deck-out games + 4 no-active games = 20; W + D counts games,
        # because record_game_result writes one W and one L per decided game.
        assert dumps[0].deckout_rate == pytest.approx(0.8)
        assert dumps[0].games == 20

    def test_derives_attack_rate_from_root_option_stats(self, tmp_path):
        dumps = ingest.parse_diag_dumps(write(tmp_path, "arm.txt", DIAG_LOG))
        assert dumps[0].attack_rate == pytest.approx(0.25)
        assert dumps[0].attack_available == 100

    def test_keeps_the_game_length_histogram(self, tmp_path):
        dumps = ingest.parse_diag_dumps(write(tmp_path, "arm.txt", DIAG_LOG))
        assert dumps[0].game_length_hist["26-40"] == 10
        assert dumps[0].mean_game_length_bucket == "26-40"

    def test_a_dump_without_attack_options_reports_no_rate(self, tmp_path):
        text = DIAG_LOG.replace(
            "ROOT_OPTION_STATS=ATTACK:available=100,chosen=25,not_chosen=75,mean_opt_visit=1.12|",
            "ROOT_OPTION_STATS=",
        )
        dumps = ingest.parse_diag_dumps(write(tmp_path, "arm.txt", text))
        assert dumps[0].attack_rate is None

    def test_unterminated_dump_is_dropped_rather_than_half_parsed(self, tmp_path):
        text = DIAG_LOG.replace("=== DIAG_DUMP_END ===", "")
        assert ingest.parse_diag_dumps(write(tmp_path, "arm.txt", text)) == []


class TestPerDumpCsv:
    def test_reads_attack_columns_no_other_artifact_preserves(self, tmp_path):
        csv_text = (
            "idx,phase,games,prize_decided,deckout_share,true_result,"
            "attach_zero_visit,attack_zero_visit,attack_avail,attack_chosen,game_length_hist\n"
            '1,Warm-up 1/4,25,0/25,100.0%,"P0_WINS=15,P1_WINS=10,DRAWS=0",'
            "11.40%(87/763),55.88%(19/34),34,7,1-15:0|16-25:3|26-40:21|41-60:1|61+:0\n"
        )
        dumps = ingest.parse_per_dump_csv(write(tmp_path, "task_c_per_dump.csv", csv_text))
        assert len(dumps) == 1
        assert dumps[0].attack_rate == pytest.approx(7 / 34)
        assert dumps[0].deckout_rate == pytest.approx(1.0)
        assert dumps[0].games == 25


class TestDiscoveryAndOutputs:
    def test_skips_directories_that_are_not_eval_output(self, tmp_path):
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "junk.log").write_text("x")
        (tmp_path / "real.log").write_text(OLD_EVAL_LOG)
        found = [os.path.basename(p) for p in ingest.discover(str(tmp_path))]
        assert "real.log" in found
        assert "junk.log" not in found

    def test_writes_every_output_artifact(self, tmp_path):
        write(tmp_path, "eval.log", OLD_EVAL_LOG)
        write(tmp_path, "arm.txt", DIAG_LOG)
        out_dir = tmp_path / "out"
        games, dumps, per_source = ingest.ingest(ingest.discover(str(tmp_path)))
        summary = ingest.write_outputs(str(out_dir), games, dumps, per_source)

        for name in ("games.jsonl", "diag_dumps.jsonl", "summary.json",
                     "summary.csv", "INDEX.md"):
            assert (out_dir / name).exists(), name

        assert summary["totals"]["games"] == 4
        assert summary["totals"]["diag_dumps"] == 1

        rows = [json.loads(line) for line in (out_dir / "games.jsonl").read_text().splitlines()]
        assert len(rows) == 4

        index = (out_dir / "INDEX.md").read_text()
        assert "eval.log" in index
        assert "arm.txt" in index

    def test_classifies_each_source_kind(self, tmp_path):
        write(tmp_path, "eval_panel_arm_X.log", OLD_EVAL_LOG)
        write(tmp_path, "h2h_out.log", H2H_LOG)
        _, _, per_source = ingest.ingest(ingest.discover(str(tmp_path)))
        kinds = {os.path.basename(k): v["kind"] for k, v in per_source.items()}
        assert kinds["eval_panel_arm_X.log"] == "eval_panel run"
        assert kinds["h2h_out.log"] == "head-to-head checkpoint comparison"

    def test_a_file_with_no_eval_content_is_omitted(self, tmp_path):
        write(tmp_path, "noise.log", "installing torch...\nDone.\n")
        _, _, per_source = ingest.ingest(ingest.discover(str(tmp_path)))
        assert per_source == {}


class TestAgainstTheRealRepo:
    """The point of the module is these files, so exercise them directly."""

    @pytest.mark.parametrize("name", ["eval_panel_arm_B.log", "h2h_out.log"])
    def test_known_repo_logs_still_parse(self, name):
        path = os.path.join(REPO_ROOT, name)
        if not os.path.exists(path):
            pytest.skip(f"{name} not present")
        games, _ = ingest.parse_game_lines(path)
        assert games, f"{name} produced no records"
        assert all(g.outcome in ("win", "loss", "draw") for g in games)

    def test_sweep_arm_files_yield_diag_dumps(self):
        path = os.path.join(REPO_ROOT, "sweep_out", "coef_0.0.txt")
        if not os.path.exists(path):
            pytest.skip("sweep_out not present")
        dumps = ingest.parse_diag_dumps(path)
        assert dumps
        assert dumps[0].deckout_rate is not None
