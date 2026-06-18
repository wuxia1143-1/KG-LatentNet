#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f /root/kg_latentnet_env/bin/activate ]; then
  # shellcheck disable=SC1091
  source /root/kg_latentnet_env/bin/activate
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate kg_latentnet
fi

python -m src.training.train_official_baseline --project-root "$PROJECT_ROOT" --baseline hyperimts --fold 0 --epochs 3 --reset-outputs || true
python -m src.training.train_official_baseline --project-root "$PROJECT_ROOT" --baseline trans --fold 0 --epochs 3 || true
python -m src.training.train_official_baseline --project-root "$PROJECT_ROOT" --baseline tgnn4i --fold 0 --epochs 3 || true
python -m src.training.train_official_baseline --project-root "$PROJECT_ROOT" --baseline dhgas --fold 0 --epochs 3 || true
python -m src.training.train_official_baseline --project-root "$PROJECT_ROOT" --baseline kedgn --fold 0 --epochs 3 || true
python -m src.training.train_official_baseline --project-root "$PROJECT_ROOT" --baseline graphcare --fold 0 --epochs 3 || true
