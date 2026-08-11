"""Reward-ablation harness for PkmnAgent.

Four pieces, deliberately kept importable without the engine or torch so the
non-simulation parts (reward registry, metrics, ingestion) stay unit-testable
on a plain macOS host:

  rewards.py      named reward specs, selected with REWARD_SPEC=<name>
  metrics.py      the diagnostic metrics as first-class values
  ingest.py       raw eval logs/CSVs -> the same structured metrics
  eval_runner.py  runs the pinned panel and emits those metrics   (needs engine)
  driver.py       trains one arm per reward spec, evaluates, compares (needs engine)
"""
