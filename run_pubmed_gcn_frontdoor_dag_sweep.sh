#!/usr/bin/env bash

set -euo pipefail

# Focused hyper-parameter sweep for DAG-Core on PubMed with a GCN backbone.
# This intentionally scans only the knobs most likely to move PubMed:
#   1) lr/dropout: PubMed stage1 is sensitive to these two.
#   2) lambda_dag/lambda_dag_label/dag_latent_dim: strength and capacity of the DAG split.
#   3) fd_blend/gmm_sample_k/edge_blend: stability of the front-door context path.
#
# Usage:
#   DEVICE=1 bash run_pubmed_gcn_frontdoor_dag_sweep.sh
#   EPOCHS=300 RUNS=2 bash run_pubmed_gcn_frontdoor_dag_sweep.sh

DATASET="pubmed"
BACKBONE="gcn"
ENV_TYPE="graph"

EPOCHS="${EPOCHS:-500}"
RUNS="${RUNS:-1}"
DEVICE="${DEVICE:-0}"
PYTHON="${PYTHON:-python}"
DISPLAY_STEP="${DISPLAY_STEP:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-80}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"

WEIGHT_DECAY="${WEIGHT_DECAY:-5e-5}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-64}"
K="${K:-3}"
TAU="${TAU:-2}"

# name|lr|dropout|lambda_dag|lambda_dag_label|lambda_spu|lambda_env|fd_blend|dag_latent_dim|edge_blend|gmm_sample_k|extra_args
CONFIGS=(
  "weak_lr005_dp03|0.005|0.3|0.005|0.01|0.02|0.0|0.3|8|0.05|1|"
  "weak_fd05|0.005|0.3|0.005|0.01|0.02|0.0|0.5|8|0.05|1|"
  "dag01|0.005|0.3|0.01|0.01|0.02|0.0|0.3|8|0.05|1|"
  "daglabel05|0.005|0.3|0.005|0.05|0.02|0.0|0.3|8|0.05|1|"
  "spu05_env02|0.005|0.3|0.005|0.01|0.05|0.02|0.3|8|0.05|1|"
  "dim16_edge02|0.005|0.3|0.005|0.01|0.02|0.0|0.3|16|0.2|1|"
  "nogmm|0.005|0.3|0.005|0.01|0.02|0.0|0.3|8|0.05|0|--disable_spu_gmm"
  "nomixer|0.005|0.3|0.005|0.01|0.02|0.0|0.3|8|0.05|1|--disable_dag_mixer"
  "lr01_dp03|0.01|0.3|0.005|0.01|0.02|0.0|0.3|8|0.05|1|"
  "lr01_dp02|0.01|0.2|0.005|0.01|0.02|0.0|0.3|8|0.05|1|"
  "lr005_dp02|0.005|0.2|0.005|0.01|0.02|0.0|0.3|8|0.05|1|"
  "balanced|0.005|0.3|0.01|0.05|0.05|0.02|0.5|16|0.1|3|"
)

echo "[INFO] PubMed GCN DAG-Core focused sweep"
echo "[INFO] epochs=${EPOCHS}, runs=${RUNS}, configs=${#CONFIGS[@]}, device=${DEVICE}, python=${PYTHON}"
echo "[INFO] fixed: wd=${WEIGHT_DECAY}, hidden=${HIDDEN_CHANNELS}, K=${K}, tau=${TAU}"

for config in "${CONFIGS[@]}"; do
  IFS='|' read -r tag lr dropout lambda_dag lambda_dag_label lambda_spu lambda_env fd_blend dag_latent_dim edge_blend gmm_sample_k extra_args <<< "${config}"

  result_name="pubmed_gcn_dagcore_${tag}_lr${lr}_dp${dropout}_d${lambda_dag}_dl${lambda_dag_label}_spu${lambda_spu}_env${lambda_env}_fd${fd_blend}_dim${dag_latent_dim}_edge${edge_blend}_gmm${gmm_sample_k}"
  echo "[RUN] ${result_name}"

  # shellcheck disable=SC2206
  extra_args_array=(${extra_args})

  "${PYTHON}" main_frontdoor_dag_core.py \
    --dataset "${DATASET}" \
    --backbone "${BACKBONE}" \
    --lr "${lr}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --dropout "${dropout}" \
    --tau "${TAU}" \
    --hidden_channels "${HIDDEN_CHANNELS}" \
    --K "${K}" \
    --lambda_dag "${lambda_dag}" \
    --lambda_dag_label "${lambda_dag_label}" \
    --lambda_spu "${lambda_spu}" \
    --lambda_env "${lambda_env}" \
    --lambda_fd 0.5 \
    --fd_blend "${fd_blend}" \
    --dag_latent_dim "${dag_latent_dim}" \
    --edge_blend "${edge_blend}" \
    --gmm_sample_k "${gmm_sample_k}" \
    --env_type "${ENV_TYPE}" \
    --epochs "${EPOCHS}" \
    --runs "${RUNS}" \
    --device "${DEVICE}" \
    --display_step "${DISPLAY_STEP}" \
    --early_stop_patience "${EARLY_STOP_PATIENCE}" \
    --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}" \
    --combine_result \
    --store \
    --result_name "${result_name}" \
    "${extra_args_array[@]}"
done
