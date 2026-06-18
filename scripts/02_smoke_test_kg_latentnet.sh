#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/root/KG_LatentNet_Project}"
ENV_DIR="${2:-/root/kg_latentnet_env}"

cd "${PROJECT_DIR}"
mkdir -p results/logs results/tables results/predictions results/checkpoints
"${ENV_DIR}/bin/python" -m src.training.train --project-root "${PROJECT_DIR}" --fold 0 --epochs 5
