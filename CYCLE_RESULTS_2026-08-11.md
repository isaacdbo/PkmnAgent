# BEFORE/AFTER training cycle, 2026-08-11

**Headline: this cycle did not produce a win-rate improvement.** No arm of the
panel moved significantly. One arm moved *down* by 15pp. What did change, and
change overwhelmingly, is *how* the agent plays: it now contests the prize race
instead of drifting to deck-out. It has not converted that into wins.

Everything below is reproducible from the committed logs in
`local_cycle_results/`.

## Setup

| | |
|---|---|
| BEFORE checkpoint | `checkpoints/m2/model_2026-08-08_09-48.pth` (the repo's only pre-existing checkpoint, and the "OLD checkpoint" FINDINGS.md already uses as its reference) |
| AFTER checkpoint | `checkpoints/m2/model_2026-08-11_22-44.pth` (final main-epoch-8 save of this run) |
| Training | fresh init (`SKIP_CHECKPOINT_LOAD=1`), `M2_ONLY=1`, 2 warm-up + 8 main epochs, 8 self-play games/epoch, `SIMULATIONS_PER_MOVE=100`, `POLICY_LABEL_SMOOTHING=0`, 8 workers. 536 min wall, 80 games, halted cleanly on `STOP_AFTER_MAIN_EPOCH=8`. |
| Eval | `eval_panel.py`, 40 games per opponent, `--base-seed 20260811`, `SIMULATIONS_PER_MOVE=100`, seats alternating each game |
| Panel | `random`, `first`, `iono_rule` (the real Kaggle rule-based agent) |

Both arms of every matchup use identical seeds and sims; only the checkpoint
differs.

## Win rate

Wilson 95% intervals say how precisely each arm is measured; they do not answer
"did anything change". That is the two-sided Fisher exact test on the
BEFORE-vs-AFTER 2x2 table. Produced by `compare_before_after.py`.

| opponent | before | wilson95 | after | wilson95 | delta | fisher p |
| --- | --- | --- | --- | --- | --- | --- |
| random | 12.5% (5/40) | [5.5%, 26.1%] | 15.0% (6/40) | [7.1%, 29.1%] | +2.5pp | 1.0000 |
| first | 32.5% (13/40) | [20.1%, 48.0%] | 17.5% (7/40) | [8.7%, 31.9%] | -15.0pp | 0.1961 |
| iono_rule | 0.0% (0/40) | [0.0%, 8.8%] | 0.0% (0/40) | [0.0%, 8.8%] | +0.0pp | 1.0000 |

Nothing here is significant at alpha=0.05. The `first` arm is the largest move
and it is a regression; at p=0.20 it is not distinguishable from noise either,
so the honest reading is "no evidence of improvement, weak hint of harm on one
arm", not "improved on two of three".

**Against the real Kaggle bot the agent is 0 for 80 across both checkpoints.**
That is the number that matters for the competition and it has not moved.

## What did change: the agent now contests prizes

`prize_decided` counts games that reached a prize decision rather than ending by
deck-out or a stranded board.

| opponent | prize_decided before | after | fisher p | deckout before | after | fisher p |
| --- | --- | --- | --- | --- | --- | --- |
| random | 9/40 | 19/40 | 0.034 | 6/40 | 12/40 | 0.18 |
| first | 4/40 | 19/40 | 0.000394 | 12/40 | 7/40 | 0.293 |
| iono_rule | 0/40 | 37/40 | 2.3e-19 | 0/40 | 0/40 | 1 |

This is significant on all three arms and, against `iono_rule`, is about as far
outside noise as a 40-game sample can put anything. The old checkpoint never
once reached a prize decision against the Kaggle bot; the new one does so in 37
of 40 games — and still loses all 40.

So the training run produced a real, measurable change in play style that has
not yet paid off in results. Contesting the prize race and losing it is not
obviously better than stalling, but it is a different failure mode, and a more
promising base to iterate on than an agent that never engages.

## Confounds — what this comparison cannot support

1. **This is not an ablation of the exploration knobs.** BEFORE and AFTER differ
   in checkpoint *and* training length *and* label smoothing *and* run identity.
   Dirichlet noise and the temperature schedule were already present and already
   active in the code before this branch existed (see below). Isolating their
   effect needs a same-config pair with `SELF_PLAY_DIRICHLET_EPSILON=0` vs `0.25`
   and nothing else varied.
2. **n=40 per cell is small.** It cannot resolve anything under roughly 20pp.
   Every "flat" verdict here is consistent with a real effect of ±15pp.
3. **Single seed per arm.** No replication across seed families, so run-to-run
   variance is unmeasured.

## Provenance of the panel

`iono_rule` is the real Kaggle agent, re-verified this session rather than
trusted: the `%%writefile main.py` cell of the live notebook is byte-identical
to the committed bot (MD5 `eb1aac45bd00f968430f01319edfcaa0`). Full method and
the credential-free reproduction steps are in
`panel_bots/iono_rule/PROVENANCE.md`.

## Attribution

The masking and exploration machinery evaluated here was **already in the
codebase** at commit `17d8092` (Isaac Debono) before this branch's first commit:
`shaped_reward_terms`/`PRIZE_DIFF`/`KO_BONUS`, `SELF_PLAY_DIRICHLET_ALPHA`,
`SELF_PLAY_TEMP_HIGH_TURN`, `_sample_dirichlet`, and the correct
`out_dec + (mask - 1.0) * 1e9` masking before `log_softmax`.

This branch's contribution is harness and verification, not algorithm:
`test_masking.py` (18 checks confirming the existing masking is correct — a
verification, not a fix), the `iono_rule` panel member, `compare_before_after.py`,
the cycle drivers, and the committed logs.

## Reproduce

```sh
# BEFORE arm (per opponent)
SIMULATIONS_PER_MOVE=100 python -u eval_panel.py \
  --candidate checkpoints/m2/model_2026-08-08_09-48.pth --candidate-deck M2Deck.xlsx \
  --panel random --games 40 --base-seed 20260811

# training
FAST_TEST=0 SIMULATIONS_PER_MOVE=100 M2_ONLY=1 \
  WARMUP_EPOCHS=2 WARMUP_SELF_PLAY_GAMES=8 \
  MAIN_EPOCHS=8 MAIN_SELF_PLAY_M2_GAMES=8 POLICY_LABEL_SMOOTHING=0 \
  SELF_PLAY_WORKERS=8 SKIP_CHECKPOINT_LOAD=1 STOP_AFTER_MAIN_EPOCH=8 \
  python -u RLTRM2.py

# AFTER arm, then the comparison
python3 compare_before_after.py \
  --before local_cycle_results/before_{random,first,iono}.log \
  --after  local_cycle_results/after_{random,first,iono}.log
```

## Run-log note

The `docker run` client for the training stage was killed by a host session
restart while the container kept running. Training completed cleanly inside the
container; `local_cycle_results/train.log` is the stdout captured before the
pipe broke and stops mid-epoch-8, and `local_cycle_results/train_full.log` is
the complete log recovered afterwards from the Docker log driver, ending in
`[STOP] Reached STOP_AFTER_MAIN_EPOCH=8; halting cleanly.` Both are committed;
prefer `train_full.log`.

## Next

The cheapest experiment that would actually answer something: the
epsilon=0 vs epsilon=0.25 ablation at fixed everything-else, at 100+ games per
cell so the interval is narrow enough to resolve a 10pp effect.
