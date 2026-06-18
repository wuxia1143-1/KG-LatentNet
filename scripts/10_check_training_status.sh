#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/KG_LatentNet_Project}"
cd "${PROJECT_ROOT}"

PID_FILE="results/logs/tuning/validation_tuning.pid"
STATUS_FILE="results/logs/tuning/current_status.json"
TUNING_LOG="results/logs/tuning/main_tuning.log"
LOCKED_CONFIG="configs/locked_full_5fold_config.yaml"

echo "project_root=${PROJECT_ROOT}"
echo "run_method=nohup"

if [ -f "${PID_FILE}" ]; then
  PID="$(cat "${PID_FILE}")"
  echo "pid=${PID}"
  if ps -p "${PID}" >/dev/null 2>&1; then
    echo "process_status=running"
  else
    echo "process_status=not_running"
  fi
else
  echo "pid_file=missing"
fi

if [ -f "${STATUS_FILE}" ]; then
  echo "current_status:"
  cat "${STATUS_FILE}"
  echo
else
  echo "current_status=missing"
fi

if [ -f "${LOCKED_CONFIG}" ]; then
  echo "locked_config=${LOCKED_CONFIG}"
else
  echo "locked_config=not_yet_generated"
fi

if [ -f "results/tables/tuning/validation_tuning_results.csv" ]; then
  echo "completed_or_attempted_candidates=$(($(wc -l < results/tables/tuning/validation_tuning_results.csv)-1))"
fi

if [ -f "${TUNING_LOG}" ]; then
  echo "tuning_log=${TUNING_LOG}"
  echo "tail:"
  tail -40 "${TUNING_LOG}"
else
  echo "tuning_log=missing"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu_status:"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
else
  echo "gpu_status=nvidia-smi_not_available"
fi

echo "next_step=wait_for_validation_tuning_or_inspect_failed_candidates"
