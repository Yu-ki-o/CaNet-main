#!/usr/bin/env bash

set -euo pipefail

# Hyper-parameter sweep for model_gmm3.py on Twitch with a GCN backbone.
#
# Usage:
#   bash run_twitch_gcn_gmm3_sweep.sh
#   DEVICE=1 bash run_twitch_gcn_gmm3_sweep.sh
#   PYTHON=python3 EPOCHS=300 RUNS=2 bash run_twitch_gcn_gmm3_sweep.sh
#   DRY_RUN=1 bash run_twitch_gcn_gmm3_sweep.sh
#   MAX_CONFIGS=3 bash run_twitch_gcn_gmm3_sweep.sh

DATASET="twitch"
BACKBONE="gcn"
ENV_TYPE="graph"
ENTRYPOINT="main_gmm3.py"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-300}"
RUNS="${RUNS:-2}"
DISPLAY_STEP="${DISPLAY_STEP:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-80}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
MAX_CONFIGS="${MAX_CONFIGS:-0}"
DRY_RUN="${DRY_RUN:-0}"

# tag|lr|weight_decay|dropout|hidden_channels|K|tau|lambda_dag|lambda_dag_label|lambda_spu|lambda_env|lambda_fd|fd_blend|dag_latent_dim|edge_feat_mode|edge_blend|edge_score_temp|gmm_sample_k|noise_alpha|noise_temp|pos_weight|extra_args
CONFIGS=(
  "base_h64_k3|0.01|5e-5|0.0|64|3|3|0.05|0.05|0.05|0.05|0.5|0.5|16|mul|0.2|5.0|3|0.1|1.0|5.0|"
  "base_h32_k3_wd1e4|0.01|0.0001|0.0|32|3|3|0.05|0.05|0.05|0.05|0.5|0.5|16|mul|0.2|5.0|3|0.1|1.0|5.0|"
  "pos7_h32_k3|0.01|0.0001|0.0|32|3|3|0.05|0.05|0.05|0.05|0.5|0.5|16|mul|0.2|5.0|3|0.1|1.0|7.0|"
  "pos3_h32_k3|0.01|0.0001|0.0|32|3|3|0.05|0.05|0.05|0.05|0.5|0.5|16|mul|0.2|5.0|3|0.1|1.0|3.0|"
  "light_dag_k4_dim8|0.005|5e-5|0.0|32|4|3|0.005|0.01|0.02|0.0|0.5|0.3|8|mul|0.05|2.0|1|0.1|1.0|5.0|"
  "light_dag_k3_dim8|0.005|5e-5|0.0|32|3|3|0.005|0.01|0.02|0.0|0.5|0.3|8|mul|0.05|2.0|1|0.1|1.0|5.0|"
  "balanced_dim16_gmm3|0.005|5e-5|0.0|32|4|3|0.01|0.05|0.05|0.02|0.5|0.5|16|mul|0.1|2.0|3|0.1|1.0|5.0|"
  "edge_score5_dim16|0.005|5e-5|0.0|32|4|3|0.01|0.05|0.05|0.02|0.5|0.5|16|mul|0.1|5.0|3|0.1|1.0|5.0|"
  "denoise_a005|0.005|5e-5|0.0|32|4|3|0.005|0.01|0.02|0.0|0.5|0.3|8|mul|0.05|2.0|1|0.05|1.0|5.0|--use_neighbor_denoise"
  "denoise_a01|0.005|5e-5|0.0|32|4|3|0.005|0.01|0.02|0.0|0.5|0.3|8|mul|0.05|2.0|1|0.1|1.0|5.0|--use_neighbor_denoise"
  "denoise_edge01|0.005|5e-5|0.0|32|4|3|0.005|0.01|0.02|0.0|0.5|0.3|8|mul|0.1|2.0|1|0.1|1.0|5.0|--use_neighbor_denoise"
  "denoise_temp05|0.005|5e-5|0.0|32|4|3|0.005|0.01|0.02|0.0|0.5|0.3|8|mul|0.05|2.0|1|0.1|0.5|5.0|--use_neighbor_denoise"
)

echo "[INFO] Twitch GCN model_gmm3 sweep"
echo "[INFO] entrypoint=${ENTRYPOINT}"
echo "[INFO] epochs=${EPOCHS}, runs=${RUNS}, configs=${#CONFIGS[@]}, device=${DEVICE}, python=${PYTHON}"
echo "[INFO] early_stop_patience=${EARLY_STOP_PATIENCE}, dry_run=${DRY_RUN}, max_configs=${MAX_CONFIGS}"

config_idx=0
for config in "${CONFIGS[@]}"; do
  config_idx=$((config_idx + 1))
  if [[ "${MAX_CONFIGS}" -gt 0 && "${config_idx}" -gt "${MAX_CONFIGS}" ]]; then
    break
  fi

  IFS='|' read -r tag lr weight_decay dropout hidden_channels k tau lambda_dag lambda_dag_label lambda_spu lambda_env lambda_fd fd_blend dag_latent_dim edge_feat_mode edge_blend edge_score_temp gmm_sample_k noise_alpha noise_temp pos_weight extra_args <<< "${config}"

  result_name="twitch_gcn_gmm3_${tag}_lr${lr}_wd${weight_decay}_dp${dropout}_h${hidden_channels}_K${k}_tau${tau}_d${lambda_dag}_dl${lambda_dag_label}_spu${lambda_spu}_env${lambda_env}_lfd${lambda_fd}_fd${fd_blend}_dim${dag_latent_dim}_${edge_feat_mode}_edge${edge_blend}_score${edge_score_temp}_gmm${gmm_sample_k}_na${noise_alpha}_nt${noise_temp}_pos${pos_weight}"
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
    --edge_feat_mode "${edge_feat_mode}"
    --edge_blend "${edge_blend}"
    --edge_score_temp "${edge_score_temp}"
    --gmm_sample_k "${gmm_sample_k}"
    --noise_subtract_alpha "${noise_alpha}"
    --noise_gate_temp "${noise_temp}"
    --pos_weight "${pos_weight}"
    --env_type "${ENV_TYPE}"
    --epochs "${EPOCHS}"
    --runs "${RUNS}"
    --device "${DEVICE}"
    --display_step "${DISPLAY_STEP}"
    --early_stop_patience "${EARLY_STOP_PATIENCE}"
    --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}"
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
