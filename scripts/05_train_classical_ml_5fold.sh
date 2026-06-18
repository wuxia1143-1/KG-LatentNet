#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

if [ -f "/root/kg_latentnet_env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "/root/kg_latentnet_env/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate kg_latentnet
fi

mkdir -p results/logs results/tables results/predictions data/processed/tabular

python -m src.training.train_classical_baselines \
  --mode full \
  --baseline all \
  --build-tabular
