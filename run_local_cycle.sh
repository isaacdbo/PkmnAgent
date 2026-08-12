#!/bin/sh
# Full local cycle: BEFORE eval -> training with exploration fix -> AFTER eval.
# Run inside the amd64 container from /work. All knobs via env with defaults
# filled in from the 2026-08-11 Rosetta benchmark.
#
#   TRAIN_SIMS       self-play search sims during training
#   TRAIN_WARMUP     warm-up epochs
#   TRAIN_MAIN       main epochs
#   TRAIN_GAMES      self-play games per epoch (warm-up and main)
#   EVAL_SIMS        inference sims for both panel evals
#   EVAL_GAMES       games per opponent per eval
#   PANEL            comma list of opponents
set -eu

TRAIN_SIMS="${TRAIN_SIMS:-100}"
TRAIN_WARMUP="${TRAIN_WARMUP:-2}"
TRAIN_MAIN="${TRAIN_MAIN:-8}"
TRAIN_GAMES="${TRAIN_GAMES:-16}"
EVAL_SIMS="${EVAL_SIMS:-100}"
EVAL_GAMES="${EVAL_GAMES:-40}"
PANEL="${PANEL:-random,iono_rule}"
BASE_SEED="${BASE_SEED:-20260811}"
OUT="${OUT:-local_cycle_results}"

mkdir -p "$OUT"
echo "=== CONFIG ===" | tee "$OUT/config.txt"
env | grep -E "^(TRAIN_|EVAL_|PANEL|BASE_SEED)" | tee -a "$OUT/config.txt" || true
echo "TRAIN_SIMS=$TRAIN_SIMS TRAIN_WARMUP=$TRAIN_WARMUP TRAIN_MAIN=$TRAIN_MAIN TRAIN_GAMES=$TRAIN_GAMES EVAL_SIMS=$EVAL_SIMS EVAL_GAMES=$EVAL_GAMES PANEL=$PANEL BASE_SEED=$BASE_SEED" | tee -a "$OUT/config.txt"

echo "=== [1/3] BEFORE eval: old_m2 collapsed checkpoint vs panel ==="
SIMULATIONS_PER_MOVE="$EVAL_SIMS" python -u eval_panel.py \
  --candidate checkpoints/m2/model_2026-08-08_09-48.pth \
  --candidate-deck M2Deck.xlsx \
  --panel "$PANEL" --games "$EVAL_GAMES" --base-seed "$BASE_SEED" \
  > "$OUT/before_eval.log" 2>&1
grep -E "OPP_.*_CANDIDATE|OPP_MEAN" "$OUT/before_eval.log" || true

echo "=== [2/3] training: fresh init, exploration fix (dirichlet+temp+mask, smoothing=0) ==="
FAST_TEST=0 SIMULATIONS_PER_MOVE="$TRAIN_SIMS" M2_ONLY=1 \
  WARMUP_EPOCHS="$TRAIN_WARMUP" WARMUP_SELF_PLAY_GAMES="$TRAIN_GAMES" \
  MAIN_EPOCHS="$TRAIN_MAIN" MAIN_SELF_PLAY_M2_GAMES="$TRAIN_GAMES" \
  POLICY_LABEL_SMOOTHING=0 \
  SELF_PLAY_WORKERS=8 SKIP_CHECKPOINT_LOAD=1 STOP_AFTER_MAIN_EPOCH="$TRAIN_MAIN" \
  python -u RLTRM2.py > "$OUT/train.log" 2>&1
CANDIDATE=$(ls -t checkpoints/m2/model_*.pth | head -1)
echo "CANDIDATE=$CANDIDATE" | tee "$OUT/candidate.txt"

echo "=== [3/3] AFTER eval: new checkpoint vs same panel, same sims, same seed ==="
SIMULATIONS_PER_MOVE="$EVAL_SIMS" python -u eval_panel.py \
  --candidate "$CANDIDATE" \
  --candidate-deck M2Deck.xlsx \
  --panel "$PANEL" --games "$EVAL_GAMES" --base-seed "$BASE_SEED" \
  > "$OUT/after_eval.log" 2>&1
grep -E "OPP_.*_CANDIDATE|OPP_MEAN" "$OUT/after_eval.log" || true

echo "=== DONE ==="
