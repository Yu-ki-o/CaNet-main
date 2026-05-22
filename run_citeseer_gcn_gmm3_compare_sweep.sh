#!/usr/bin/env bash

set -euo pipefail

# Comparison sweep for model_gmm3.py on Citeseer with a GCN backbone.
#
# Citeseer GCN older results are strongest around:
#   lr=0.005/0.01, wd=5e-5, dropout=0.1, hidden=32, K=2
# with node environments. This sweep starts from those clean gated dual-head
# baselines, then adds GMM, light DAG, and edge context components.
#
# Usage:
#   bash run_citeseer_gcn_gmm3_compare_sweep.sh
#   DEVICE=1 bash run_citeseer_gcn_gmm3_compare_sweep.sh
#   RUNS=3 EPOCHS=500 bash run_citeseer_gcn_gmm3_compare_sweep.sh
#   DRY_RUN=1 bash run_citeseer_gcn_gmm3_compare_sweep.sh
#   MAX_CONFIGS=4 bash run_citeseer_gcn_gmm3_compare_sweep.sh

DATASET="citeseer"
BACKBONE="gcn"
ENTRYPOINT="main_gmm3.py"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-500}"
RUNS="${RUNS:-1}"
DISPLAY_STEP="${DISPLAY_STEP:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-80}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
MAX_CONFIGS="${MAX_CONFIGS:-0}"
DRY_RUN="${DRY_RUN:-0}"

# tag|env_type|lr|weight_decay|dropout|hidden_channels|K|tau|lambda_dag|lambda_dag_label|lambda_spu|lambda_env|lambda_fd|fd_blend|dag_latent_dim|edge_blend|edge_score_temp|gmm_sample_k|extra_args
CONFIGS=(
  "front_lr005_k2_node_clean|node|0.005|5e-5|0.1|32|2|1|0.0|0.0|0.10|0.0|0.5|0.5|8|0.0|5.0|0|--disable_spu_gmm --disable_dag_mixer"
  "front_lr005_k2_node_gmm1|node|0.005|5e-5|0.1|32|2|1|0.0|0.0|0.10|0.0|0.5|0.5|8|0.0|5.0|1|--disable_dag_mixer"
  "front_lr005_k2_node_gmm2|node|0.005|5e-5|0.1|32|2|1|0.0|0.0|0.10|0.0|0.5|0.5|8|0.0|5.0|2|--disable_dag_mixer"
  "front_lr005_k2_node_edge005|node|0.005|5e-5|0.1|32|2|1|0.0|0.0|0.10|0.0|0.5|0.5|8|0.05|5.0|1|--disable_dag_mixer"
  "front_lr005_k2_node_edge010|node|0.005|5e-5|0.1|32|2|1|0.0|0.0|0.10|0.0|0.5|0.5|8|0.10|5.0|1|--disable_dag_mixer"
  "front_lr005_k2_node_lightdag|node|0.005|5e-5|0.1|32|2|1|0.005|0.01|0.10|0.0|0.5|0.5|8|0.0|5.0|1|"
  "front_lr005_k2_node_lightdag_edge005|node|0.005|5e-5|0.1|32|2|1|0.005|0.01|0.10|0.0|0.5|0.5|8|0.05|5.0|1|"
  "front_lr005_k2_node_fd03_lightdag|node|0.005|5e-5|0.1|32|2|1|0.005|0.01|0.10|0.0|0.5|0.3|8|0.0|5.0|1|"
  "front_lr01_k2_node_clean|node|0.01|5e-5|0.1|32|2|1|0.0|0.0|0.10|0.0|0.5|0.5|8|0.0|5.0|0|--disable_spu_gmm --disable_dag_mixer"
  "front_lr01_k2_node_gmm1|node|0.01|5e-5|0.1|32|2|1|0.0|0.0|0.10|0.0|0.5|0.5|8|0.0|5.0|1|--disable_dag_mixer"
  "front_lr01_k2_node_lightdag|node|0.01|5e-5|0.1|32|2|1|0.005|0.01|0.10|0.0|0.5|0.5|8|0.0|5.0|1|"
  "front_lr01_k2_node_edge005|node|0.01|5e-5|0.1|32|2|1|0.0|0.0|0.10|0.0|0.5|0.5|8|0.05|5.0|1|--disable_dag_mixer"
  "graph_k3_lr01_dp01_clean|graph|0.01|5e-5|0.1|32|3|1|0.0|0.0|0.10|0.0|0.5|0.5|8|0.0|5.0|0|--disable_spu_gmm --disable_dag_mixer"
  "graph_k3_lr01_dp01_gmm1|graph|0.01|5e-5|0.1|32|3|1|0.0|0.0|0.10|0.0|0.5|0.5|8|0.0|5.0|1|--disable_dag_mixer"
  "graph_k3_lr01_dp01_lightdag|graph|0.01|5e-5|0.1|32|3|1|0.005|0.01|0.10|0.0|0.5|0.5|8|0.0|5.0|1|"
  "graph_k3_lr01_dp02_lightdag_edge010|graph|0.01|5e-5|0.2|32|3|1|0.005|0.01|0.10|0.0|0.5|0.5|8|0.10|5.0|1|"
)

echo "[INFO] Citeseer GCN model_gmm3 comparison sweep"
echo "[INFO] entrypoint=${ENTRYPOINT}"
echo "[INFO] epochs=${EPOCHS}, runs=${RUNS}, configs=${#CONFIGS[@]}, device=${DEVICE}, python=${PYTHON}"
echo "[INFO] early_stop_patience=${EARLY_STOP_PATIENCE}, dry_run=${DRY_RUN}, max_configs=${MAX_CONFIGS}"

config_idx=0
for config in "${CONFIGS[@]}"; do
  config_idx=$((config_idx + 1))
  if [[ "${MAX_CONFIGS}" -gt 0 && "${config_idx}" -gt "${MAX_CONFIGS}" ]]; then
    break
  fi

  IFS='|' read -r tag env_type lr weight_decay dropout hidden_channels k tau lambda_dag lambda_dag_label lambda_spu lambda_env lambda_fd fd_blend dag_latent_dim edge_blend edge_score_temp gmm_sample_k extra_args <<< "${config}"

  result_name="citeseer_gcn_gmm3_compare_${tag}_lr${lr}_wd${weight_decay}_dp${dropout}_h${hidden_channels}_K${k}_tau${tau}_d${lambda_dag}_dl${lambda_dag_label}_spu${lambda_spu}_env${lambda_env}_lfd${lambda_fd}_fd${fd_blend}_dim${dag_latent_dim}_edge${edge_blend}_score${edge_score_temp}_gmm${gmm_sample_k}"
  echo "[RUN ${config_idx}/${#CONFIGS[@]}] ${result_name}"

  extra_args_array=()
  if [[ -n "${extra_args}" ]]; then
    read -r -a extra_args_array <<< "${extra_args}"
  fi

  cmd=(
    "${PYTHON}" "${ENTRYPOINT}"
    --dataset "${DATASET}"
    --backbone "${BACKBONE}"
    --lr "${lr}"
    --weight_decay "${weight_decay}"
    --dropout "${dropout}"
    --tau "${tau}"
    --hidden_channels "${hidden_channels}"
    --num_layers 2
    --K "${k}"
    --lambda_dag "${lambda_dag}"
    --lambda_dag_label "${lambda_dag_label}"
    --lambda_spu "${lambda_spu}"
    --lambda_env "${lambda_env}"
    --lambda_fd "${lambda_fd}"
    --fd_blend "${fd_blend}"
    --dag_latent_dim "${dag_latent_dim}"
    --edge_feat_mode mul
    --edge_blend "${edge_blend}"
    --edge_score_temp "${edge_score_temp}"
    --gmm_sample_k "${gmm_sample_k}"
    --env_type "${env_type}"
    --epochs "${EPOCHS}"
    --runs "${RUNS}"
    --device "${DEVICE}"
    --display_step "${DISPLAY_STEP}"
    --early_stop_patience "${EARLY_STOP_PATIENCE}"
    --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}"
    --combine_result
    --store
    --result_name "${result_name}"
    "${extra_args_array[@]}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
done
