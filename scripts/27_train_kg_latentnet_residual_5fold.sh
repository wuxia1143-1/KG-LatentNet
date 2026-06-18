#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/KG_LatentNet_Project}"
ENV_DIR="${ENV_DIR:-/root/kg_latentnet_env}"
CONFIG="configs/kg_latentnet_residual.yaml"
MODE="smoke"
FOLD=0
EPOCHS=5

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)  CONFIG="${2:-}"; shift 2 ;;
    --mode)    MODE="${2:-}"; shift 2 ;;
    --fold)    FOLD="${2:-}"; shift 2 ;;
    --epochs)  EPOCHS="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "${PROJECT_ROOT}"
mkdir -p results/logs/full_5fold results/tables/full_5fold results/predictions/full_5fold results/checkpoints/full_5fold results/latent/full_5fold

PYTHON="${ENV_DIR}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python3)"
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Config: ${CONFIG}"
echo "Mode: ${MODE}"
echo "Fold: ${FOLD}"
echo "Python: ${PYTHON}"

if [ "${MODE}" = "smoke" ]; then
  echo "Running fold_${FOLD} smoke test..."
  "${PYTHON}" -m src.training.train_kg_latentnet_residual \
    --project-root "${PROJECT_ROOT}" \
    --fold "${FOLD}" \
    --mode smoke \
    --epochs "${EPOCHS}" \
    --config "${CONFIG}"
elif [ "${MODE}" = "train" ]; then
  echo "Running fold_${FOLD} full training..."
  "${PYTHON}" -m src.training.train_kg_latentnet_residual \
    --project-root "${PROJECT_ROOT}" \
    --fold "${FOLD}" \
    --mode train \
    --config "${CONFIG}"
elif [ "${MODE}" = "test_eval" ]; then
  echo "Running all 5 folds test evaluation..."
  for f in 0 1 2 3 4; do
    echo "=== Fold ${f} ==="
    "${PYTHON}" -m src.training.train_kg_latentnet_residual \
      --project-root "${PROJECT_ROOT}" \
      --fold "${f}" \
      --mode train \
      --config "${CONFIG}"
  done
else
  echo "Unknown mode: ${MODE}" >&2
  exit 3
fi

echo "Done."
