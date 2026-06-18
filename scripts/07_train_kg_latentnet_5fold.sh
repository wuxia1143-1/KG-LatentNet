#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/KG_LatentNet_Project}"
ENV_DIR="${ENV_DIR:-/root/kg_latentnet_env}"
CONFIG="configs/locked_full_5fold_config.yaml"
MODE="test_eval"
RESUME_OR_RERUN_KG_ONLY=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --resume_or_rerun_kg_only)
      RESUME_OR_RERUN_KG_ONLY=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ "${MODE}" != "test_eval" ]; then
  echo "Only --mode test_eval is supported by this locked KG-only runner." >&2
  exit 2
fi

if [ "${RESUME_OR_RERUN_KG_ONLY}" != "true" ]; then
  echo "Refusing to run without --resume_or_rerun_kg_only." >&2
  exit 2
fi

PYTHON="${ENV_DIR}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python3)"
fi

cd "${PROJECT_ROOT}"
mkdir -p results/logs/full_5fold
"${PYTHON}" -m src.training.full_5fold_evaluation \
  --project-root "${PROJECT_ROOT}" \
  --config "${CONFIG}" \
  --models kg_latentnet
