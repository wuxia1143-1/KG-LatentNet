#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/KG_LatentNet_Project}"
ENV_DIR="${ENV_DIR:-/root/kg_latentnet_env}"
CONFIG="configs/locked_full_5fold_config.yaml"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "${PROJECT_ROOT}"
mkdir -p results/logs/full_5fold results/tables/full_5fold results/predictions/full_5fold results/checkpoints/full_5fold results/latent/full_5fold

if [ ! -f "${CONFIG}" ]; then
  echo "Refusing full evaluation: locked config not found at ${CONFIG}."
  exit 3
fi

for required in \
  results/tables/tuning/locked_config_summary.csv \
  results/tables/tuning/test_set_not_used_check.csv \
  results/tables/tuning/pre_full_eval_leakage_check.csv
do
  if [ ! -f "${required}" ]; then
    echo "Refusing full evaluation: missing ${required}."
    exit 4
  fi
done

if grep -qi "blocked" results/tables/tuning/test_set_not_used_check.csv; then
  echo "Refusing full evaluation: test_set_not_used_check.csv contains blocked rows."
  exit 5
fi

if grep -qi "blocked" results/tables/tuning/pre_full_eval_leakage_check.csv; then
  echo "Refusing full evaluation: pre_full_eval_leakage_check.csv contains blocked rows."
  exit 6
fi

PYTHON="${ENV_DIR}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python3)"
fi

echo "$$" > results/logs/full_5fold/full_5fold.pid
cat > results/logs/full_5fold/current_status.json <<JSON
{
  "stage": "full_5fold_test_evaluation",
  "status": "starting",
  "config": "${CONFIG}"
}
JSON

"${PYTHON}" -m src.training.full_5fold_evaluation \
  --project-root "${PROJECT_ROOT}" \
  --config "${CONFIG}" \
  --models all \
  --reset
