#!/usr/bin/env bash
set -euo pipefail
cd "/mnt/c/Users/isaas/OneDrive - Vrije Universiteit Amsterdam/Desktop/Important Docs/Me/My Games/Pokemon Agent"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pkmnagent
for floor in 0.05 0.10 0.15 0.25; do
  echo "=== ARM floor=$floor ==="
  SIMULATIONS_PER_MOVE=20 ATTACH_PRIOR_FLOOR=$floor python sweep_deck_diff.py --games 20 --random-init \
    > "sweep_out/floor_${floor}.txt" 2> "sweep_out/floor_${floor}.err"
  echo "=== DONE floor=$floor ==="
done
