#!/bin/bash
# The host driver (host_drive_cycle.sh) only runs the random and iono_rule AFTER
# arms. This adds the third panel member, `first`, on the same checkpoint, seed,
# sims, and game count, so the BEFORE/AFTER table covers the full panel.
#
# Waits for the driver to write local_cycle_results/candidate.txt (which it does
# immediately after training stops), then launches the arm.
set -eu
cd "$(dirname "$0")"
ROOT=$(pwd -P)
OUT="$ROOT/local_cycle_results"

EVAL_SIMS=100
EVAL_GAMES=40
BASE_SEED=20260811

echo "WAIT_FOR_CANDIDATE $(date -u +%FT%TZ)"
while [ ! -s "$OUT/candidate.txt" ]; do sleep 30; done
CANDIDATE=$(sed 's/^CANDIDATE=//' "$OUT/candidate.txt")
echo "AFTER_FIRST_BEGIN candidate=$CANDIDATE $(date -u +%FT%TZ)"

docker run --rm --platform linux/amd64 --dns 8.8.8.8 -u 501:20 \
  -e HOME=/work/.dockerhome -e OMP_NUM_THREADS=2 \
  -v "$ROOT:/work" -w /work python:3.11-slim \
  sh -lc "SIMULATIONS_PER_MOVE=$EVAL_SIMS python -u eval_panel.py \
    --candidate $CANDIDATE --candidate-deck M2Deck.xlsx \
    --panel first --games $EVAL_GAMES --base-seed $BASE_SEED" \
  > "$OUT/after_first.log" 2>&1

echo "AFTER_FIRST_DONE $(date -u +%FT%TZ)"
grep -h "OPP_.*_CANDIDATE" "$OUT/after_first.log" || true
