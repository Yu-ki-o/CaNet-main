#!/usr/bin/env bash

set -euo pipefail

# Focused hyper-parameter sweep for model_gmm3_reviewed1_graph_cfam_nego.py
# on PubMed with a GCN backbone.
#
# Default protocol for this sweep:
# - each configuration runs 2 independent runs
# - each run trains for 500 epochs
# - Graph-CFAM and NeGo are enabled in every configuration
#
# Usage:
#   bash run_pubmed_gcn_graph_cfam_nego_sweep.sh
#   DEVICE=1 bash run_pubmed_gcn_graph_cfam_nego_sweep.sh
#   DRY_RUN=1 bash run_pubmed_gcn_graph_cfam_nego_sweep.sh
#   MAX_CONFIGS=4 bash run_pubmed_gcn_graph_cfam_nego_sweep.sh

DATASET="pubmed"
BACKBONE="gcn"
ENV_TYPE="graph"
ENTRYPOINT="main_gmm3_reviewed1_graph_cfam_nego.py"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-1}"
EPOCHS="${EPOCHS:-500}"
RUNS="${RUNS:-2}"
DISPLAY_STEP="${DISPLAY_STEP:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-80}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
MAX_CONFIGS="${MAX_CONFIGS:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Fixed base command inherited from the current strong PubMed setup.
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-5}"
TAU="${TAU:-2}"
NUM_LAYERS="${NUM_LAYERS:-2}"
EDGE_SCORE_TEMP="${EDGE_SCORE_TEMP:-5.0}"
LAMBDA_DAG="${LAMBDA_DAG:-0.05}"
LAMBDA_DAG_LABEL="${LAMBDA_DAG_LABEL:-0.05}"
LAMBDA_SPU="${LAMBDA_SPU:-0.05}"
LAMBDA_ENV="${LAMBDA_ENV:-0.05}"
LAMBDA_FD="${LAMBDA_FD:-0.5}"
NEGO_TEMP="${NEGO_TEMP:-0.2}"

# This sweep is intentionally restricted to fd_context_source=nego_only.
# gmm_sample_k is kept fixed because it does not affect the final front-door
# contexts when fd_context_source=nego_only.
# tag|lr|dropout|hidden|K|gmm_k|fd_blend|edge_blend|dag_dim|lambda_nego|nego_ctx_w|edge_feat_mode|fd_context_source|nego_source|extra_args
CONFIGS=(
  "base_spurious|0.005|0.3|64|2|2|0.5|0.2|16|0.01|0.5|mul|nego_only|spurious|"
  "base_z|0.005|0.3|64|2|2|0.5|0.2|16|0.01|0.5|mul|nego_only|z|"
  "base_mediator|0.005|0.3|64|2|2|0.5|0.2|16|0.01|0.5|mul|nego_only|mediator|"
  "dp02_z|0.005|0.2|64|2|2|0.5|0.2|16|0.01|0.5|mul|nego_only|z|"
  "fd03_z|0.005|0.3|64|2|2|0.3|0.2|16|0.01|0.5|mul|nego_only|z|"
  "fd07_z|0.005|0.3|64|2|2|0.7|0.2|16|0.01|0.5|mul|nego_only|z|"
  "nego005_ctx03_z|0.005|0.3|64|2|2|0.5|0.2|16|0.005|0.3|mul|nego_only|z|"
  "nego02_ctx07_z|0.005|0.3|64|2|2|0.5|0.2|16|0.02|0.7|mul|nego_only|z|"
  "dim08_z|0.005|0.3|64|2|2|0.5|0.2|8|0.01|0.5|mul|nego_only|z|"
  "edge01_z|0.005|0.3|64|2|2|0.5|0.1|16|0.01|0.5|mul|nego_only|z|"
  "signeddeg_z|0.005|0.3|64|2|2|0.5|0.2|16|0.01|0.5|mul_signed_diff_degree|nego_only|z|"
  "signeddeg_spurious|0.005|0.3|64|2|2|0.5|0.2|16|0.01|0.5|mul_signed_diff_degree|nego_only|spurious|"
  "signeddeg_mediator|0.005|0.3|64|2|2|0.5|0.2|16|0.01|0.5|mul_signed_diff_degree|nego_only|mediator|"
  "k3_signeddeg_z|0.005|0.3|64|3|2|0.5|0.2|16|0.01|0.5|mul_signed_diff_degree|nego_only|z|"
)

echo "[INFO] PubMed GCN Graph-CFAM+NeGo sweep"
echo "[INFO] entrypoint=${ENTRYPOINT}"
echo "[INFO] epochs=${EPOCHS}, runs=${RUNS}, configs=${#CONFIGS[@]}, device=${DEVICE}, python=${PYTHON}"
echo "[INFO] dry_run=${DRY_RUN}, max_configs=${MAX_CONFIGS}, early_stop_patience=${EARLY_STOP_PATIENCE}"

config_idx=0
for config in "${CONFIGS[@]}"; do
  config_idx=$((config_idx + 1))
  if [[ "${MAX_CONFIGS}" -gt 0 && "${config_idx}" -gt "${MAX_CONFIGS}" ]]; then
    break
  fi

  IFS='|' read -r tag lr dropout hidden_channels k gmm_sample_k fd_blend edge_blend dag_latent_dim lambda_nego nego_context_weight edge_feat_mode fd_context_source nego_source extra_args <<< "${config}"

  result_name="pubmed_gcn_graphcfam_nego_${tag}_lr${lr}_dp${dropout}_h${hidden_channels}_K${k}_gmm${gmm_sample_k}_fd${fd_blend}_edge${edge_blend}_dim${dag_latent_dim}_nlg${lambda_nego}_nctx${nego_context_weight}_${edge_feat_mode}_${fd_context_source}_${nego_source}"
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
    --weight_decay "${WEIGHT_DECAY}"
    --tau "${TAU}"
    --K "${k}"
    --dropout "${dropout}"
    --hidden_channels "${hidden_channels}"
    --num_layers "${NUM_LAYERS}"
    --env_type "${ENV_TYPE}"
    --edge_feat_mode "${edge_feat_mode}"
    --gmm_sample_k "${gmm_sample_k}"
    --edge_blend "${edge_blend}"
    --edge_score_temp "${EDGE_SCORE_TEMP}"
    --dag_latent_dim "${dag_latent_dim}"
    --lambda_dag "${LAMBDA_DAG}"
    --lambda_dag_label "${LAMBDA_DAG_LABEL}"
    --lambda_spu "${LAMBDA_SPU}"
    --lambda_env "${LAMBDA_ENV}"
    --lambda_fd "${LAMBDA_FD}"
    --fd_blend "${fd_blend}"
    --runs "${RUNS}"
    --epochs "${EPOCHS}"
    --device "${DEVICE}"
    --display_step "${DISPLAY_STEP}"
    --early_stop_patience "${EARLY_STOP_PATIENCE}"
    --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}"
    --lambda_nego "${lambda_nego}"
    --nego_source "${nego_source}"
    --nego_temp "${NEGO_TEMP}"
    --nego_context_weight "${nego_context_weight}"
    --fd_context_source "${fd_context_source}"
    --combine_result
    --store
    --use_graph_cfam
    --use_nego_prompt
    --use_nego_context
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
