#!/usr/bin/env bash
# Reproducible local run of the reward-ablation harness.
#
# The engine (cg-lib/cg/libcg.so) is a Linux x86-64 ELF shared object and
# cg-lib/cg/sim.py looks for libcg.dylib on Darwin, so nothing that touches the
# simulator runs natively on an Apple-silicon Mac. This script runs the harness
# in a linux/amd64 container instead — on Apple silicon that means Rosetta
# emulation, which works and is what every local training receipt in this repo
# was produced under.
#
#   ./ablation/run_local.sh dry        plan only; no container, no training
#   ./ablation/run_local.sh smoke      2 arms x 1 short training + 4-game panel
#   ./ablation/run_local.sh ablation   the full arm list at the default size
#   ./ablation/run_local.sh eval CKPT  panel-only, against an existing .pth
#   ./ablation/run_local.sh tests      the host-side unit tests (no container)
#   ./ablation/run_local.sh ingest     organise raw eval output (no container)
#
# Environment:
#   PKMN_IMAGE       container image (default python:3.11-slim)
#   PKMN_ARMS        comma list of reward specs
#   PKMN_GAMES       games per panel opponent
#   PKMN_PANEL       panel members
#   DOCKER_CONTEXT   docker context to use (default: whatever is current)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-smoke}"
IMAGE="${PKMN_IMAGE:-python:3.11-slim}"
GAMES="${PKMN_GAMES:-20}"
PANEL="${PKMN_PANEL:-random,first,iono_rule}"
ARMS="${PKMN_ARMS:-baseline,terminal_only,deckout_penalty,turns_to_win_mild}"

# pip --user installs land here and persist between runs, so only the first
# run pays for downloading torch. Gitignored.
DOCKER_HOME="$REPO_ROOT/.dockerhome"

run_in_container() {
  mkdir -p "$DOCKER_HOME"
  docker run --rm --platform linux/amd64 \
    -u "$(id -u):$(id -g)" \
    -e HOME=/work/.dockerhome \
    -e PYTHONPATH=/work \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$REPO_ROOT:/work" -w /work \
    "$IMAGE" bash -lc "
      set -euo pipefail
      python -m pip install --user --quiet --no-warn-script-location \
        pandas openpyxl >/tmp/pip.log 2>&1
      python -m pip install --user --quiet --no-warn-script-location \
        --index-url https://download.pytorch.org/whl/cpu torch >>/tmp/pip.log 2>&1
      $1
    "
}

case "$MODE" in
  dry)
    python3 -m ablation.driver --dry-run --arms "$ARMS" --panel "$PANEL" --games "$GAMES"
    ;;

  tests)
    python3 -m pytest tests/ -q
    ;;

  ingest)
    python3 -m ablation.ingest . --out-dir results/ingested
    ;;

  smoke)
    # Smallest configuration that still exercises every stage: reward spec ->
    # training -> checkpoint -> pinned panel -> per-arm metrics -> comparison.
    run_in_container "python -m ablation.driver \
      --arms '${PKMN_ARMS:-baseline,deckout_penalty}' \
      --panel '${PKMN_PANEL:-random,first}' \
      --games '${PKMN_GAMES:-4}' \
      --eval-sims 5 \
      --out-dir results/ablation-smoke \
      --train-env SIMULATIONS_PER_MOVE=5 \
      --train-env WARMUP_SELF_PLAY_GAMES=2 \
      --train-env MAIN_SELF_PLAY_M2_GAMES=2 \
      --train-env EVAL_GAMES_PER_MATCHUP=1 \
      --train-timeout 3600"
    ;;

  ablation)
    run_in_container "python -m ablation.driver \
      --arms '$ARMS' --panel '$PANEL' --games '$GAMES' \
      --out-dir results/ablation"
    ;;

  eval)
    CKPT="${2:?usage: run_local.sh eval <checkpoint.pth>}"
    run_in_container "python -m ablation.eval_runner \
      --candidate '$CKPT' --panel '$PANEL' --games '$GAMES' \
      --out-dir results/eval"
    ;;

  *)
    echo "unknown mode: $MODE" >&2
    sed -n '3,25p' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac
