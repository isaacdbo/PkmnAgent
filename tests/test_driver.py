"""Driver behaviour that can be checked without the engine.

Training and evaluation need the compiled simulator, so they are not exercised
here. Everything around them is: the per-arm environment the driver hands to
RLTRM2.py, checkpoint discovery, resumability, and the comparison report.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation import driver, metrics, rewards  # noqa: E402
from ablation.metrics import GameRecord  # noqa: E402


def game(outcome, opponent="random", cause=None, turn=None):
    return GameRecord(game_index=1, opponent=opponent, outcome=outcome,
                      cause=cause, final_turn=turn)


class TestPerArmEnvironment:
    def test_sets_the_reward_spec(self):
        env = driver._env_for_arm("deckout_penalty", "/tmp/ckpt", {})
        assert env["REWARD_SPEC"] == "deckout_penalty"

    def test_gives_each_arm_its_own_checkpoint_root(self):
        """Shared checkpoints would let arms warm-start from each other."""
        a = driver._env_for_arm("baseline", "/runs/baseline/checkpoints", {})
        b = driver._env_for_arm("terminal_only", "/runs/terminal_only/checkpoints", {})
        assert a["CHECKPOINT_ROOT"] != b["CHECKPOINT_ROOT"]

    def test_starts_every_arm_from_a_fresh_network(self):
        env = driver._env_for_arm("baseline", "/tmp/ckpt", {})
        assert env["SKIP_CHECKPOINT_LOAD"] == "1"

    def test_holds_seed_and_training_size_fixed_across_arms(self):
        a = driver._env_for_arm("baseline", "/tmp/a", {})
        b = driver._env_for_arm("turns_to_win_mild", "/tmp/b", {})
        for key in ("SELF_PLAY_BASE_SEED", "SIMULATIONS_PER_MOVE",
                    "WARMUP_EPOCHS", "MAIN_EPOCHS", "M2_ONLY"):
            assert a[key] == b[key], key

    def test_overrides_win_over_the_defaults(self):
        env = driver._env_for_arm("baseline", "/tmp/ckpt", {"MAIN_EPOCHS": "7"})
        assert env["MAIN_EPOCHS"] == "7"

    def test_every_default_knob_is_one_rltrm2_actually_reads(self):
        """A knob RLTRM2 ignores would be a setting that silently does nothing."""
        source = open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "RLTRM2.py")).read()
        for key in driver.DEFAULT_TRAIN_ENV:
            assert f'"{key}"' in source, f"RLTRM2.py never reads {key}"

    def test_pins_blas_threads_so_arm_wall_times_are_comparable(self):
        env = driver._env_for_arm("baseline", "/tmp/ckpt", {})
        assert env["OMP_NUM_THREADS"] == "1"


class TestCheckpointDiscovery:
    def test_picks_the_newest_by_name(self, tmp_path):
        folder = tmp_path / "m2"
        folder.mkdir()
        for name in ("model_2026-08-01_10-00.pth", "model_2026-08-11_09-30.pth",
                     "model_2026-08-09_23-00.pth"):
            (folder / name).write_text("x")
        found = driver.latest_checkpoint(str(tmp_path))
        assert found.endswith("model_2026-08-11_09-30.pth")

    def test_no_checkpoints_is_none_not_an_error(self, tmp_path):
        assert driver.latest_checkpoint(str(tmp_path)) is None


class TestResume:
    def test_existing_results_are_reused(self, tmp_path):
        arm_dir = tmp_path / "baseline"
        (arm_dir / "eval").mkdir(parents=True)
        metrics.write_games_jsonl([game("win"), game("loss")],
                                  arm_dir / "eval" / "games.jsonl")
        loaded = driver.load_arm_records(str(arm_dir))
        assert loaded is not None and len(loaded) == 2

    def test_an_unstarted_arm_loads_as_none(self, tmp_path):
        assert driver.load_arm_records(str(tmp_path / "nothing")) is None


class TestComparisonReport:
    def _report(self):
        arm_summaries = {
            "baseline": metrics.summarize([
                game("win", cause="deckout", turn=40),
                game("loss", cause="deckout", turn=38),
            ]),
            "deckout_penalty": metrics.summarize([
                game("win", cause="prize", turn=20),
                game("win", cause="prize", turn=18),
            ]),
        }
        per_arm_opponent = {
            "baseline": {"random": arm_summaries["baseline"]},
            "deckout_penalty": {"random": arm_summaries["deckout_penalty"]},
        }
        return driver.compare(arm_summaries, per_arm_opponent)

    def test_names_every_arm(self):
        report = self._report()
        assert "baseline" in report and "deckout_penalty" in report

    def test_breaks_results_out_per_panel_opponent(self):
        assert "### vs random" in self._report()

    def test_states_what_each_arm_changed(self):
        """A table of numbers without the specs is not reviewable."""
        report = self._report()
        assert rewards.get("deckout_penalty").description[:40] in report
        assert "deckout_win=0.25" in report

    def test_says_that_overlapping_intervals_are_expected(self):
        assert "Wilson" in self._report()

    def test_carries_the_metrics_the_ablation_targets(self):
        report = self._report()
        for column in ("win_rate", "deckout_rate", "attack_rate", "turns_to_win"):
            assert column in report


class TestCliContract:
    def test_rejects_an_unknown_arm_before_training_anything(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["driver", "--arms", "baseline,typo_arm"])
        with pytest.raises((SystemExit, ValueError)):
            driver.main()

    def test_dry_run_executes_nothing_and_prints_the_plan(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(sys, "argv", [
            "driver", "--dry-run", "--arms", "baseline,terminal_only",
            "--out-dir", str(tmp_path / "out"),
        ])

        def explode(*a, **k):
            raise AssertionError("dry run must not train")

        monkeypatch.setattr(driver, "train_arm", explode)
        monkeypatch.setattr(driver, "evaluate_arm", explode)
        driver.main()

        out = capsys.readouterr().out
        assert "dry run: nothing executed" in out
        assert "terminal_only" in out
        assert "REWARD_SPEC" in out
        assert not (tmp_path / "out").exists()

    def test_train_env_needs_key_value_form(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["driver", "--train-env", "MAIN_EPOCHS"])
        with pytest.raises(SystemExit):
            driver.main()

    def test_writes_comparison_artifacts_from_arm_records(self, monkeypatch, tmp_path):
        """End-to-end through main(), with train/evaluate stubbed out."""
        out_dir = tmp_path / "out"
        monkeypatch.setattr(sys, "argv", [
            "driver", "--arms", "baseline,deckout_penalty",
            "--out-dir", str(out_dir), "--eval-only",
        ])
        monkeypatch.setattr(driver, "latest_checkpoint", lambda *a, **k: "fake.pth")
        monkeypatch.setattr(
            driver, "evaluate_arm",
            lambda arm, arm_dir, *a, **k: [game("win" if arm == "baseline" else "loss")],
        )
        driver.main()

        assert (out_dir / "COMPARISON.md").exists()
        assert (out_dir / "comparison.csv").exists()
        payload = json.loads((out_dir / "comparison.json").read_text())
        assert set(payload["arms"]) == {"baseline", "deckout_penalty"}
        # The specs travel with the numbers, so a stored result can be read
        # later without going back to the registry to find out what it meant.
        assert payload["specs"]["deckout_penalty"]["deckout_win_value"] == 0.25
