#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/KG_LatentNet_Project}"
ENV_DIR="${ENV_DIR:-/root/kg_latentnet_env}"
CONFIG="${CONFIG:-configs/validation_tuning.yaml}"
MODELS="${MODELS:-all}"
FOLDS="${FOLDS:-}"

cd "${PROJECT_ROOT}"
mkdir -p results/logs/tuning results/tables/tuning results/predictions/tuning results/checkpoints/tuning

echo "$$" > results/logs/tuning/validation_tuning.pid
cat > results/logs/tuning/current_status.json <<JSON
{
  "stage": "validation_only_tuning",
  "status": "starting",
  "models": "${MODELS}",
  "folds": "${FOLDS:-from_config}",
  "test_set_used_for_selection": false
}
JSON

PYTHON="${ENV_DIR}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python3)"
fi

CMD=("${PYTHON}" -m src.training.validation_tuning --project-root "${PROJECT_ROOT}" --config "${CONFIG}" --models "${MODELS}" --reset)
if [ -n "${FOLDS}" ]; then
  CMD+=("--folds" "${FOLDS}")
fi

echo "Validation-only tuning command: ${CMD[*]}"
echo "Test evaluation is not run by this script."
"${CMD[@]}"
