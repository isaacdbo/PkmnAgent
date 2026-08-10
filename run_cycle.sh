#!/usr/bin/env bash
# Unattended cycle: train N epochs (Task 2 diff: Dirichlet root noise + temperature
# move sampling, self-play only) -> evaluate the resulting m2 checkpoint vs
# sample_bot at fixed inference sims -> append FINDINGS.md -> exit.
#
# Fresh random init each cycle (SKIP_CHECKPOINT_LOAD=1) so results aren't
# contaminated by checkpoints trained under the old no-noise/no-temperature
# regime. Pass --resume to disable this and continue from whatever checkpoint
# is already in checkpoints/m2/.
#
# Usage:
#   ./run_cycle.sh [--epochs N] [--train-sims N] [--eval-sims N] [--eval-games N] [--pass-threshold F] [--resume]
#
# Exit code: 0 always (the PASS/FAIL verdict is a line in the output and in
# FINDINGS.md, not the process exit code — this runs unattended and a FAIL is
# an expected, valid outcome, not a script error).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

EPOCHS=8
TRAIN_SIMS=20
# 200, not 20: measured 22.5% win rate vs sample_bot at sims=20 vs 42.5% at
# sims=200 on the same checkpoint (FINDINGS.md) — evaluating at 20 understates
# every candidate's real strength.
EVAL_SIMS=200
EVAL_GAMES=40
# 60%, not 51%: NEW already reached 42.5% at sims=200 with no training changes
# at all (FINDINGS.md), so a bare-majority pass threshold would be uninformative.
PASS_THRESHOLD=0.60
RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --train-sims) TRAIN_SIMS="$2"; shift 2 ;;
    --eval-sims) EVAL_SIMS="$2"; shift 2 ;;
    --eval-games) EVAL_GAMES="$2"; shift 2 ;;
    --pass-threshold) PASS_THRESHOLD="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# NOTE on --eval-games: 40 matches every prior FINDINGS.md entry for
# continuity, but Task 1's power calculation put n~389/arm as the games count
# needed to reliably resolve a 10-point win-rate difference at worst-case
# variance. At n=40 this cycle's PASS/FAIL line is a directional signal, not a
# statistically powered one — bump --eval-games if that matters more than
# unattended runtime.

TS=$(date +%Y-%m-%d_%H-%M-%S)
TRAIN_LOG="cycle_train_${TS}.log"
EVAL_LOG="cycle_eval_${TS}.log"

echo "=== Training: ${EPOCHS} main epochs, sims=${TRAIN_SIMS}, resume=${RESUME} ==="
SKIP_LOAD=1
if [[ "$RESUME" == "1" ]]; then SKIP_LOAD=0; fi

FAST_TEST=0 \
SIMULATIONS_PER_MOVE="$TRAIN_SIMS" \
ATTACH_PRIOR_FLOOR=0 \
DECK_DIFF_COEF=0.01 \
MAIN_EPOCHS="$EPOCHS" \
SKIP_CHECKPOINT_LOAD="$SKIP_LOAD" \
STOP_AFTER_MAIN_EPOCH="$EPOCHS" \
python -u RLTRM2.py > "$TRAIN_LOG" 2>&1

if ! grep -q "STOP_AFTER_MAIN_EPOCH" "$TRAIN_LOG"; then
  echo "CYCLE_RESULT=FAIL reason=training_did_not_reach_epoch_${EPOCHS} see=${TRAIN_LOG}"
  exit 0
fi

CANDIDATE=$(ls -t checkpoints/m2/model_*.pth | head -1)
echo "=== Trained checkpoint: ${CANDIDATE} ==="

echo "=== Evaluating vs sample_bot: sims=${EVAL_SIMS}, games=${EVAL_GAMES} ==="
SIMULATIONS_PER_MOVE="$EVAL_SIMS" \
python eval_panel.py --candidate "$CANDIDATE" --candidate-deck M2Deck.xlsx \
  --panel sample_bot --games "$EVAL_GAMES" > "$EVAL_LOG" 2>&1

echo "=== Appending FINDINGS.md ==="
python append_findings.py \
  --eval-log "$EVAL_LOG" \
  --label "Task 2 diff (Dirichlet+temperature), fresh init, ${EPOCHS} epochs" \
  --epochs "$EPOCHS" \
  --sims-train "$TRAIN_SIMS" \
  --findings FINDINGS.md \
  --pass-threshold "$PASS_THRESHOLD"
