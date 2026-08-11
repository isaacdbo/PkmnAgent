#!/bin/sh
# Smoke test + speed benchmark after the RLTRM2 refactor and iono_rule addition.
# Run inside the amd64 container from /work.
set -eu

echo "=== py_compile ==="
python -m py_compile RLTRM2.py eval_panel.py test_masking.py panel_bots/iono_rule/main.py
echo PY_COMPILE_OK

echo "=== iono_rule deck id -> engine card name receipt ==="
python - <<'EOF'
import sys, os
sys.path.append(os.path.join(os.getcwd(), "cg-lib"))
from cg.api import all_card_data
table = {c.cardId: c for c in all_card_data()}
deck = [int(x) for x in open("panel_bots/iono_rule/deck.csv").read().split()]
assert len(deck) == 60, len(deck)
from collections import Counter
for cid, n in sorted(Counter(deck).items()):
    print(f"  id={cid} x{n} -> {table[cid].name}")
EOF

echo "=== eval_panel smoke: random-init candidate vs iono_rule, 2 games, sims=3 ==="
SIMULATIONS_PER_MOVE=3 python -u eval_panel.py \
  --candidate-random-init --panel iono_rule --games 2 --base-seed 990001

echo "=== training smoke + benchmark: sims=50, 8+8 games, 8 workers ==="
t0=$(date +%s)
FAST_TEST=0 SIMULATIONS_PER_MOVE=50 M2_ONLY=1 \
  WARMUP_EPOCHS=1 WARMUP_SELF_PLAY_GAMES=8 \
  MAIN_EPOCHS=1 MAIN_SELF_PLAY_M2_GAMES=8 \
  POLICY_LABEL_SMOOTHING=0 \
  SELF_PLAY_WORKERS=8 SKIP_CHECKPOINT_LOAD=1 STOP_AFTER_MAIN_EPOCH=1 \
  python -u RLTRM2.py 2>&1 | tail -25
t1=$(date +%s)
echo "BENCH_TRAIN_SMOKE_WALL_SEC=$((t1 - t0))"
