# Overnight Local Repro Receipt

Date: 2026-08-10/2026-08-11 local overnight run.
Branch: `exploration-masking-eval`.
Base commit: `17d809296fdc1cfae4f014427193e4110e497427`.

## Host Check

The run used the existing amd64 Docker/Colima environment because `cg-lib/cg/libcg.so`
is a Linux x86-64 shared library.

```sh
uptime
colima list
docker ps --format '{{.ID}} {{.Image}} {{.Status}} {{.Names}}'
```

Observed before the run:

```text
20:48  up 19 days, 52 mins, 4 users, load averages: 2.33 3.36 3.72
pkmn-amd64  Running  x86_64  2  4GiB  docker
```

No containers were left running after the measured run.

## Before Eval

Command:

```sh
docker run --rm --dns 8.8.8.8 -u 501:20 -e HOME=/tmp \
  -v /Users/vbonnet/worktrees/PkmnAgent/exploration-masking-eval:/work \
  -w /work python:3.11-slim sh -lc '
    set -eu
    mkdir -p overnight_results
    python -m pip install --user --no-cache-dir pandas openpyxl >/work/overnight_results/pip_before.log 2>&1
    python -m pip install --user --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch >>/work/overnight_results/pip_before.log 2>&1
    SIMULATIONS_PER_MOVE=5 python eval_panel.py \
      --candidate-random-init \
      --panel random,first,starter_rule \
      --games 6 \
      --base-seed 424200 \
      > overnight_results/before_eval.log 2>&1
  '
```

Summary:

```text
OPP_random_CANDIDATE: wins=0 losses=6 draws=0 win_rate=0.0%(0/6) wilson95=[0.0%,39.0%] prize_decided=3/6 deckout_share=50.0%(3/6)
OPP_first_CANDIDATE: wins=1 losses=5 draws=0 win_rate=16.7%(1/6) wilson95=[3.0%,56.4%] prize_decided=4/6 deckout_share=33.3%(2/6)
OPP_starter_rule_CANDIDATE: wins=3 losses=3 draws=0 win_rate=50.0%(3/6) wilson95=[18.8%,81.2%] prize_decided=0/6 deckout_share=50.0%(3/6)
```

## Local Training Run

Command:

```sh
docker run --rm --dns 8.8.8.8 -u 501:20 -e HOME=/tmp \
  -v /Users/vbonnet/worktrees/PkmnAgent/exploration-masking-eval:/work \
  -w /work python:3.11-slim sh -lc '
    set -eu
    mkdir -p overnight_results
    python -m pip install --user --no-cache-dir pandas openpyxl >/work/overnight_results/pip_train4.log 2>&1
    python -m pip install --user --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch >>/work/overnight_results/pip_train4.log 2>&1
    FAST_TEST=0 SIMULATIONS_PER_MOVE=2 M2_ONLY=1 \
      WARMUP_EPOCHS=1 WARMUP_SELF_PLAY_GAMES=2 \
      MAIN_EPOCHS=1 MAIN_SELF_PLAY_M2_GAMES=2 \
      BATCH_SIZE=16 MAX_TRAIN_BATCHES=4 POLICY_LABEL_SMOOTHING=0 \
      SELF_PLAY_WORKERS=0 SKIP_CHECKPOINT_LOAD=1 STOP_AFTER_MAIN_EPOCH=1 \
      python -u RLTRM2.py > overnight_results/train4.log 2>&1
    CANDIDATE=$(ls -t checkpoints/m2/model_*.pth | head -1)
    echo CANDIDATE=$CANDIDATE | tee overnight_results/candidate4.txt
    SIMULATIONS_PER_MOVE=5 python eval_panel.py \
      --candidate "$CANDIDATE" \
      --candidate-deck M2Deck.xlsx \
      --panel random,first,starter_rule \
      --games 6 \
      --base-seed 424200 \
      > overnight_results/after4_eval.log 2>&1
  '
```

Training receipt:

```text
=== Warm-up Phase (1 epochs) ===
--- Warm-up Epoch 1/1 ---
[m2] Training (336 samples, 4 batches)...
[m2] Done (4 batches).
[m2] Saved: checkpoints/m2/model_2026-08-11_04-49.pth
=== Main Phase (1 epochs) ===
--- Main Epoch 1/1 ---
[m2] Training (670 samples, 4 batches)...
[m2] Done (4 batches).
[m2] Saved: checkpoints/m2/model_2026-08-11_04-50.pth
Games: 4 | Epoch: 1.3m | Elapsed: 2.6m | ETA: 0.0m
[STOP] Reached STOP_AFTER_MAIN_EPOCH=1; halting cleanly.
```

Produced checkpoint:

```text
checkpoints/m2/model_2026-08-11_04-50.pth
```

## After Eval

Summary:

```text
OPP_random_CANDIDATE: wins=0 losses=6 draws=0 win_rate=0.0%(0/6) wilson95=[0.0%,39.0%] prize_decided=0/6 deckout_share=0.0%(0/6)
OPP_first_CANDIDATE: wins=2 losses=4 draws=0 win_rate=33.3%(2/6) wilson95=[9.7%,70.0%] prize_decided=0/6 deckout_share=33.3%(2/6)
OPP_starter_rule_CANDIDATE: wins=3 losses=3 draws=0 win_rate=50.0%(3/6) wilson95=[18.8%,81.2%] prize_decided=0/6 deckout_share=100.0%(6/6)
```

## Interpretation

Verified:

- Local training runs end-to-end in the amd64 Docker setup.
- The run generated self-play data, performed optimizer updates, saved checkpoints, and stopped cleanly.
- The pinned eval panel runs against `random`, `first`, and `starter_rule` with Wilson 95% intervals.

Not established by this tiny run:

- A useful measured improvement. The only positive movement was `first` from 1/6 to 2/6, which is too small and too noisy to claim as meaningful.
- Progress toward the 70% sample-bot target.
- A reward-ablation keep/drop cycle.

