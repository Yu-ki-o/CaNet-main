#!/usr/bin/env bash

set -euo pipefail

# Poll GPU 1 until it has at least 20 GiB free memory, then launch the
# PubMed GCN Graph-CFAM+NeGo sweep on that device.
#
# Usage:
#   bash wait_for_gpu1_20g_and_run_pubmed_sweep.sh
#   CHECK_INTERVAL=120 bash wait_for_gpu1_20g_and_run_pubmed_sweep.sh
#   REQUIRED_FREE_MIB=24576 DEVICE=1 bash wait_for_gpu1_20g_and_run_pubmed_sweep.sh
#   DRY_RUN=1 bash wait_for_gpu1_20g_and_run_pubmed_sweep.sh

GPU_INDEX="${DEVICE:-1}"
REQUIRED_FREE_MIB="${REQUIRED_FREE_MIB:-20480}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
PYTHON="${PYTHON:-python}"
RUN_SCRIPT="${RUN_SCRIPT:-./run_pubmed_gcn_graph_cfam_nego_sweep.sh}"
LOG_DIR="${LOG_DIR:-./search_logs/pubmed/gcn/auto_wait}"
DRY_RUN="${DRY_RUN:-0}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[$(timestamp)] [ERROR] Missing required command: $1" >&2
    exit 1
  fi
}

query_gpu_free_mib() {
  local line
  line="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F',' -v gpu="${GPU_INDEX}" '$1 ~ ("^" gpu "$") {gsub(/ /, "", $2); print $2}')"
  if [[ -z "${line}" ]]; then
    return 1
  fi
  printf '%s\n' "${line}"
}

require_cmd nvidia-smi

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "[$(timestamp)] [ERROR] Sweep script not found: ${RUN_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
launch_log="${LOG_DIR}/launch_$(date +%Y%m%d_%H%M%S).log"

echo "[$(timestamp)] [INFO] Waiting for GPU ${GPU_INDEX} to have at least ${REQUIRED_FREE_MIB} MiB free."
echo "[$(timestamp)] [INFO] Check interval: ${CHECK_INTERVAL}s"
echo "[$(timestamp)] [INFO] Sweep script: ${RUN_SCRIPT}"
echo "[$(timestamp)] [INFO] Launch log: ${launch_log}"

while true; do
  if free_mib="$(query_gpu_free_mib)"; then
    if [[ "${free_mib}" =~ ^[0-9]+$ ]]; then
      echo "[$(timestamp)] [INFO] GPU ${GPU_INDEX} free memory: ${free_mib} MiB"
      if (( free_mib >= REQUIRED_FREE_MIB )); then
        echo "[$(timestamp)] [INFO] Threshold reached. Launching sweep."
        cmd=(
          bash "${RUN_SCRIPT}"
        )
        if [[ "${DRY_RUN}" == "1" ]]; then
          printf '  '
          printf '%q ' DEVICE="${GPU_INDEX}" PYTHON="${PYTHON}" "${cmd[@]}"
          printf '\n'
          exit 0
        fi

        DEVICE="${GPU_INDEX}" PYTHON="${PYTHON}" "${cmd[@]}" 2>&1 | tee "${launch_log}"
        exit "${PIPESTATUS[0]}"
      fi
    else
      echo "[$(timestamp)] [WARN] Unexpected GPU free-memory value: ${free_mib}" >&2
    fi
  else
    echo "[$(timestamp)] [WARN] Could not query GPU ${GPU_INDEX} free memory." >&2
  fi

  sleep "${CHECK_INTERVAL}"
done
