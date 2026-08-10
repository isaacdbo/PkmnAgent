#!/usr/bin/env bash
# One-time environment setup for a fresh Ubuntu + CUDA GPU box.
# Tested-locally baseline: Python 3.11.15, torch==2.13.0+cu130, pandas==3.0.5,
# numpy==2.4.6, openpyxl==3.1.5. The cg engine (cg-lib/cg/libcg.so) is a
# prebuilt x86_64 Linux shared library loaded via ctypes — no compilation step,
# just needs to be present next to cg-lib/cg/sim.py and linked against
# libstdc++6/libgcc_s1/libc6/libm (all present on any current Ubuntu LTS).
#
# Usage: run from the repo root after transferring files (see TRANSFER.md / report).
set -euo pipefail

echo "=== System packages ==="
sudo apt-get update
# python3.11 ships by default on Ubuntu 23.10+/24.04; on 22.04 add deadsnakes first:
#   sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip libstdc++6 rsync

echo "=== Python venv ==="
python3.11 -m venv "$HOME/pkmnagent-venv"
source "$HOME/pkmnagent-venv/bin/activate"
pip install --upgrade pip

echo "=== torch (CUDA build) ==="
# Check the box's driver first: nvidia-smi | grep "CUDA Version"
# cu130 matches the locally-pinned version; fall back to the closest cuXXX build
# the driver supports (e.g. cu121/cu124) if cu130 isn't available for the driver.
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130

echo "=== Remaining Python deps ==="
pip install pandas==3.0.5 numpy==2.4.6 openpyxl==3.1.5

echo "=== Verify cg engine loads (prebuilt .so, no build step) ==="
python3 -c "
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'cg-lib'))
from cg.api import all_card_data
print('cg engine OK —', len(all_card_data()), 'cards loaded')
"

echo "=== Verify GPU visible to torch ==="
python3 -c "
import torch
ok = torch.cuda.is_available()
print('CUDA available:', ok)
print('device:', torch.cuda.get_device_name(0) if ok else 'NONE — training will run on CPU')
"

echo "=== Verify M2Deck.xlsx readable (openpyxl backend) ==="
python3 -c "
import pandas as pd
deck = pd.read_excel('M2Deck.xlsx', header=None).iloc[:, 0].tolist()
assert len(deck) == 60, f'expected 60 cards, got {len(deck)}'
print('M2Deck.xlsx OK — 60 cards')
"

echo "Setup complete. Activate with: source \$HOME/pkmnagent-venv/bin/activate"
