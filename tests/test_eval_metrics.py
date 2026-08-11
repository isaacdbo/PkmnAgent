import json

import eval_metrics


def test_loads_jsonl_games_and_summarizes_metrics(tmp_path):
    raw = tmp_path / "eval.jsonl"
    raw.write_text(
        json.dumps({
            "run_id": "run-a",
            "checkpoint": "checkpoints/m2/model_001.pth",
            "games": [
                {
                    "opponent": "sample_bot",
                    "result": "win",
                    "win_cause": "prize",
                    "turns": 12,
                    "attacks_chosen": 3,
                    "attacks_available": 4,
                },
                {
                    "opponent": "sample_bot",
                    "result": "loss",
                    "win_cause": "deck_out",
                    "turns": 19,
                    "attacks_chosen": 1,
                    "attacks_available": 2,
                },
            ],
        })
        + "\n",
        encoding="utf-8",
    )

    games = eval_metrics.load_eval_games([raw])
    summary = eval_metrics.summarize_games(games)

    assert len(games) == 2
    assert summary[0]["opponent"] == "sample_bot"
    assert summary[0]["win_rate"] == 0.5
    assert summary[0]["deck_out_rate"] == 0.5
    assert summary[0]["attack_rate"] == 4 / 6
    assert summary[0]["avg_turns_to_win"] == 12
    assert summary[0]["WIN_BY_CAUSE"] == {"deck_out": 1, "prize": 1}


def test_loads_notebook_style_eval_summary_lines(tmp_path):
    raw = tmp_path / "train.log"
    raw.write_text(
        "checkpoint: model_2026-08-10_12-00.pth\n"
        "=== Evaluation ===\n"
        "  vs sample_bot    :  55%  (11W / 9L / 0D)\n",
        encoding="utf-8",
    )

    games = eval_metrics.load_eval_games([raw])
    summary = eval_metrics.summarize_games(games)

    assert len(games) == 20
    assert summary[0]["checkpoint"] == "model_2026-08-10_12-00.pth"
    assert summary[0]["opponent"] == "sample_bot"
    assert summary[0]["wins"] == 11
    assert summary[0]["losses"] == 9
    assert summary[0]["win_rate"] == 0.55


def test_writes_queryable_outputs(tmp_path):
    raw = tmp_path / "eval.csv"
    raw.write_text(
        "run_id,checkpoint,opponent,result,win_cause,turns,attacks_chosen,attacks_available\n"
        "run-b,ckpt-1,sample_bot,win,prize,10,2,2\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "organized"

    games = eval_metrics.load_eval_games([raw])
    summary = eval_metrics.summarize_games(games)
    eval_metrics.write_outputs(games, summary, out_dir)

    assert (out_dir / "games.jsonl").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.csv").exists()
    assert json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))[0]["win_rate"] == 1.0
