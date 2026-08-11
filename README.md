# PkmnAgent reward and eval harness

This repository uses the vendored real competition engine in `cg-lib` and its
Search API surfaces from `RLTRM2.py`; the core random-baseline harness does not
route games through the public `kaggle-environments` proxy.

Run a capped verification against the random actor:

```bash
python -m reward_harness --games 40 --turn-cap 20 --seed 7
```

The harness writes per-game diagnostics in the ledger: `win_cause`, turns,
attack choices, and available attack actions. The summary includes win-rate,
deck-out-rate, attack-rate, average turns, and capped games.

Organize raw eval output files into queryable metrics:

```bash
python -m eval_metrics path/to/eval-files-or-dir --out-dir results/eval-metrics
```

Outputs:

- `results/eval-metrics/games.jsonl`: one normalized row per game
- `results/eval-metrics/summary.json`: per-run/per-checkpoint/per-opponent metrics
- `results/eval-metrics/summary.csv`: spreadsheet-friendly summary

The loader accepts JSONL/JSON/CSV game rows and notebook-style lines like
`vs sample_bot: 55% (11W / 9L / 0D)`.
