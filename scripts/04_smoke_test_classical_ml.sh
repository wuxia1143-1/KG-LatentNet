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

python - <<'PY'
import importlib.util
import subprocess
import sys
from pathlib import Path

log_path = Path("results/logs/install_xgboost_check.log")
if importlib.util.find_spec("xgboost") is None:
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("xgboost not found; installing xgboost package.\n")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"], stdout=handle, stderr=subprocess.STDOUT)
else:
    log_path.write_text("xgboost already installed.\n", encoding="utf-8")
PY

python -m src.training.train_classical_baselines \
  --mode smoke \
  --baseline all \
  --fold 0 \
  --build-tabular \
  --reset-smoke-outputs
