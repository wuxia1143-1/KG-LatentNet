#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/KG_LatentNet_Project}"
ENV_DIR="${ENV_DIR:-/root/kg_latentnet_env}"
PYTHON="${ENV_DIR}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python3)"
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" scripts/full5_posthoc_analysis.py --project-root "${PROJECT_ROOT}" relation_heatmap
