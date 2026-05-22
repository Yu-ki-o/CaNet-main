#!/usr/bin/env bash

set -euo pipefail

# Conservative rescue sweep for model_gmm3.py on PubMed with a GAT backbone.
#
# Existing results show PubMed GAT can be strong with the older front-door
# recipe, while the default model_gmm3 GAT setting is much weaker. This sweep
# starts near the known GAT-good region and adds DAG/GMM/edge components
# gradually.
#
# Usage:
#   bash run_pubmed_gat_gmm3_rescue_sweep.sh
#   DEVICE=1 bash run_pubmed_gat_gmm3_rescue_sweep.sh
#   PYTHON=python3 EPOCHS=500 RUNS=3 bash run_pubmed_gat_gmm3_rescue_sweep.sh
#   DRY_RUN=1 bash run_pubmed_gat_gmm3_rescue_sweep.sh
#   MAX_CONFIGS=4 bash run_pubmed_gat_gmm3_rescue_sweep.sh

DATASET="pubmed"
BACKBONE="gat"
ENV_TYPE="graph"
ENTRYPOINT="main_gmm3.py"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-500}"
RUNS="${RUNS:-3}"
DISPLAY_STEP="${DISPLAY_STEP:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-80}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
MAX_CONFIGS="${MAX_CONFIGS:-0}"
DRY_RUN="${DRY_RUN:-0}"

# tag|lr|weight_decay|dropout|hidden_channels|K|tau|lambda_dag|lambda_dag_label|lambda_spu|lambda_env|lambda_fd|fd_blend|dag_latent_dim|edge_blend|edge_score_temp|gmm_sample_k|extra_args
CONFIGS=(
  "front_best_like_k2_noedge_nogmm|0.005|5e-5|0.2|64|2|1|0.0|0.0|0.05|0.0|0.5|0.5|8|0.0|5.0|0|--disable_spu_gmm --disable_dag_mixer"
  "front_best_like_k2_gmm1|0.005|5e-5|0.2|64|2|1|0.0|0.0|0.05|0.0|0.5|0.5|8|0.0|5.0|1|--disable_dag_mixer"
  "front_best_like_k2_gmm2|0.005|5e-5|0.2|64|2|1|0.0|0.0|0.05|0.0|0.5|0.5|8|0.0|5.0|2|--disable_dag_mixer"
  "front_best_like_k2_edge005|0.005|5e-5|0.2|64|2|1|0.0|0.0|0.05|0.0|0.5|0.5|8|0.05|5.0|1|--disable_dag_mixer"
  "light_dag_k2|0.005|5e-5|0.2|64|2|1|0.005|0.01|0.05|0.0|0.5|0.5|8|0.0|5.0|1|"
  "light_dag_k2_edge005|0.005|5e-5|0.2|64|2|1|0.005|0.01|0.05|0.0|0.5|0.5|8|0.05|5.0|1|"
  "old_high_ood_k2_lr01|0.01|5e-5|0.2|64|2|1|0.0|0.0|0.05|0.0|0.5|0.5|8|0.0|5.0|1|--disable_dag_mixer"
  "old_high_ood_k2_lr01_lightdag|0.01|5e-5|0.2|64|2|1|0.005|0.01|0.05|0.0|0.5|0.5|8|0.0|5.0|1|"
  "stable_k3_dp01_noedge|0.005|5e-5|0.1|64|3|1|0.0|0.0|0.05|0.0|0.5|0.5|8|0.0|5.0|1|--disable_dag_mixer"
  "stable_k3_dp01_lightdag|0.005|5e-5|0.1|64|3|1|0.005|0.01|0.05|0.0|0.5|0.5|8|0.0|5.0|1|"
  "fd03_k2_lightdag|0.005|5e-5|0.2|64|2|1|0.005|0.01|0.05|0.0|0.5|0.3|8|0.0|5.0|1|"
  "fd07_k2_lightdag|0.005|5e-5|0.2|64|2|1|0.005|0.01|0.05|0.0|0.5|0.7|8|0.0|5.0|1|"
)

echo "[INFO] PubMed GAT model_gmm3 rescue sweep"
echo "[INFO] entrypoint=${ENTRYPOINT}"
echo "[INFO] epochs=${EPOCHS}, runs=${RUNS}, configs=${#CONFIGS[@]}, device=${DEVICE}, python=${PYTHON}"
echo "[INFO] early_stop_patience=${EARLY_STOP_PATIENCE}, dry_run=${DRY_RUN}, max_configs=${MAX_CONFIGS}"

config_idx=0
for config in "${CONFIGS[@]}"; do
  config_idx=$((config_idx + 1))
  if [[ "${MAX_CONFIGS}" -gt 0 && "${config_idx}" -gt "${MAX_CONFIGS}" ]]; then
    break
  fi

  IFS='|' read -r tag lr weight_decay dropout hidden_channels k tau lambda_dag lambda_dag_label lambda_spu lambda_env lambda_fd fd_blend dag_latent_dim edge_blend edge_score_temp gmm_sample_k extra_args <<< "${config}"

  result_name="pubmed_gat_gmm3_rescue_${tag}_lr${lr}_wd${weight_decay}_dp${dropout}_h${hidden_channels}_K${k}_tau${tau}_d${lambda_dag}_dl${lambda_dag_label}_spu${lambda_spu}_env${lambda_env}_lfd${lambda_fd}_fd${fd_blend}_dim${dag_latent_dim}_edge${edge_blend}_score${edge_score_temp}_gmm${gmm_sample_k}"
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
    --env_type "${ENV_TYPE}"
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
