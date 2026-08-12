#!/bin/bash
# Host-side driver: BEFORE eval (2 parallel panels) -> training -> AFTER eval
# (2 parallel panels), each stage a docker run on the pkmn-rosetta colima VM.
# Receipts land in local_cycle_results/.
set -eu
cd "$(dirname "$0")"
OUT=local_cycle_results
mkdir -p "$OUT"

DRUN="docker run --rm --platform linux/amd64 --dns 8.8.8.8 -u 501:20 -e HOME=/work/.dockerhome -v $PWD:/work -w /work python:3.11-slim"

EVAL_SIMS=100
EVAL_GAMES=40
BASE_SEED=20260811

echo "STAGE1_BEGIN before-eval $(date -u +%FT%TZ)"
$DRUN sh -lc "SIMULATIONS_PER_MOVE=$EVAL_SIMS python -u eval_panel.py \
    --candidate checkpoints/m2/model_2026-08-08_09-48.pth --candidate-deck M2Deck.xlsx \
    --panel random --games $EVAL_GAMES --base-seed $BASE_SEED" \
  > "$OUT/before_random.log" 2>&1 &
P1=$!
$DRUN sh -lc "SIMULATIONS_PER_MOVE=$EVAL_SIMS python -u eval_panel.py \
    --candidate checkpoints/m2/model_2026-08-08_09-48.pth --candidate-deck M2Deck.xlsx \
    --panel iono_rule --games $EVAL_GAMES --base-seed $BASE_SEED" \
  > "$OUT/before_iono.log" 2>&1 &
P2=$!
wait $P1; wait $P2
echo "STAGE1_DONE $(date -u +%FT%TZ)"
grep -h "OPP_.*_CANDIDATE" "$OUT/before_random.log" "$OUT/before_iono.log" || true

echo "STAGE2_BEGIN training $(date -u +%FT%TZ)"
$DRUN sh -lc "FAST_TEST=0 SIMULATIONS_PER_MOVE=100 M2_ONLY=1 \
    WARMUP_EPOCHS=2 WARMUP_SELF_PLAY_GAMES=8 \
    MAIN_EPOCHS=8 MAIN_SELF_PLAY_M2_GAMES=8 \
    POLICY_LABEL_SMOOTHING=0 \
    SELF_PLAY_WORKERS=8 SKIP_CHECKPOINT_LOAD=1 STOP_AFTER_MAIN_EPOCH=8 \
    python -u RLTRM2.py" > "$OUT/train.log" 2>&1
echo "STAGE2_DONE $(date -u +%FT%TZ)"

CANDIDATE=$(ls -t checkpoints/m2/model_*.pth | head -1)
echo "CANDIDATE=$CANDIDATE" | tee "$OUT/candidate.txt"

echo "STAGE3_BEGIN after-eval $(date -u +%FT%TZ)"
$DRUN sh -lc "SIMULATIONS_PER_MOVE=$EVAL_SIMS python -u eval_panel.py \
    --candidate $CANDIDATE --candidate-deck M2Deck.xlsx \
    --panel random --games $EVAL_GAMES --base-seed $BASE_SEED" \
  > "$OUT/after_random.log" 2>&1 &
P3=$!
$DRUN sh -lc "SIMULATIONS_PER_MOVE=$EVAL_SIMS python -u eval_panel.py \
    --candidate $CANDIDATE --candidate-deck M2Deck.xlsx \
    --panel iono_rule --games $EVAL_GAMES --base-seed $BASE_SEED" \
  > "$OUT/after_iono.log" 2>&1 &
P4=$!
wait $P3; wait $P4
echo "STAGE3_DONE $(date -u +%FT%TZ)"
grep -h "OPP_.*_CANDIDATE" "$OUT/after_random.log" "$OUT/after_iono.log" || true
echo "CYCLE_COMPLETE $(date -u +%FT%TZ)"
