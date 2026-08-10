# PkmnAgent reward harness

This repository uses the vendored real competition engine in `cg-lib` and its
Search API surfaces from `RLTRM2.py`; the core random-baseline harness does not
route games through the public `kaggle-environments` proxy.

Run a capped verification against the random actor:

```bash
python -m reward_harness --games 40 --turn-cap 20 --seed 7
```

The default turn cap is 20 turns. If a game exceeds the cap, the harness ends it
with no winner and `terminal_only` scores the result as `0`. This prevents an
agent from getting credit by stalling or by winning only after the runtime budget
has been exceeded.

The cap is a local-training and verification guardrail. It should be lifted
later during self-play once strong opponents implicitly force fast wins.

