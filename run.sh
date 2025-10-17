#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
PY=python3
VENV=".venv"

# 1) Create venv
if [ ! -d "$VENV" ]; then
  $PY -m venv "$VENV"
fi
source "$VENV/bin/activate"

# 2) Upgrade pip
python -m pip install --upgrade pip wheel

# 3) Install deps with correct PyTorch index (CUDA if GPU present; else CPU)
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_MINOR=$(nvidia-smi | awk '/CUDA Version/ {print $9}' | cut -d'.' -f1-2)
  case "$CUDA_MINOR" in
    12.6*) IDX="https://download.pytorch.org/whl/cu126" ;;
    12.4*) IDX="https://download.pytorch.org/whl/cu124" ;;
    12.1*) IDX="https://download.pytorch.org/whl/cu121" ;;
    *)     IDX="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}" ;;
  esac
  python -m pip install --upgrade pip wheel
  python -m pip install -r requirements.txt --extra-index-url "$IDX"
else
  python -m pip install --upgrade pip wheel
  python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
fi


# 4) Core deps
python -m pip install -r requirements.txt

# 5) Pass-through run script
#./run.sh --image images/bike_remove.jpg --prompt "remove the man on the bike in the center"
#   ./run.sh --image images --prompt "remove people" --save-debug
python remove.py "$@"