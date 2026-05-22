#!/usr/bin/env bash

set -euo pipefail

# Focused hyper-parameter sweep for model_gmm3_reviewed1_graph_cfam_nego.py
# on Twitch with a GCN backbone.
#
# Twitch is a cross-language OOD benchmark here, so this sweep focuses on
# a small set of knobs that are most likely to matter:
# - class imbalance handling (pos_weight)
# - edge relation encoding / gating
# - NeGo source and context source
# - lightweight Graph-CFAM strength
#
# Usage:
#   bash run_twitch_gcn_graph_cfam_nego_sweep.sh
#   DEVICE=1 bash run_twitch_gcn_graph_cfam_nego_sweep.sh
#   DRY_RUN=1 bash run_twitch_gcn_graph_cfam_nego_sweep.sh
#   MAX_CONFIGS=4 bash run_twitch_gcn_graph_cfam_nego_sweep.sh

DATASET="twitch"
BACKBONE="gcn"
ENV_TYPE="graph"
ENTRYPOINT="main_gmm3_reviewed1_graph_cfam_nego.py"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-1}"
EPOCHS="${EPOCHS:-300}"
RUNS="${RUNS:-2}"
DISPLAY_STEP="${DISPLAY_STEP:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-80}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
MAX_CONFIGS="${MAX_CONFIGS:-0}"
DRY_RUN="${DRY_RUN:-0}"

WEIGHT_DECAY="${WEIGHT_DECAY:-5e-5}"
NUM_LAYERS="${NUM_LAYERS:-2}"
TAU="${TAU:-3}"
LAMBDA_DAG="${LAMBDA_DAG:-0.05}"
LAMBDA_DAG_LABEL="${LAMBDA_DAG_LABEL:-0.05}"
LAMBDA_SPU="${LAMBDA_SPU:-0.05}"
LAMBDA_ENV="${LAMBDA_ENV:-0.05}"
LAMBDA_FD="${LAMBDA_FD:-0.5}"
EDGE_SCORE_TEMP="${EDGE_SCORE_TEMP:-5.0}"
NEGO_TEMP="${NEGO_TEMP:-0.2}"

# tag|lr|dropout|hidden|K|pos_weight|edge_feat_mode|edge_gate_mode|edge_blend|fd_blend|dag_dim|gmm_k|lambda_nego|nego_ctx_w|fd_context_source|nego_source|graph_residual|graph_gate_temp|extra_args
CONFIGS=(
  "base_mixed_spu|0.01|0.0|64|3|5.0|mul|vector|0.2|0.5|16|3|0.01|0.5|mixed|spurious|0.1|1.0|"
  "base_negoonly_spu|0.01|0.0|64|3|5.0|mul|vector|0.2|0.5|16|3|0.01|0.5|nego_only|spurious|0.1|1.0|"
  "base_negoonly_z|0.01|0.0|64|3|5.0|mul|vector|0.2|0.5|16|3|0.01|0.5|nego_only|z|0.1|1.0|"
  "dp01_negoonly_z|0.01|0.1|64|3|5.0|mul|vector|0.2|0.5|16|3|0.01|0.5|nego_only|z|0.1|1.0|"
  "h32_negoonly_z|0.01|0.0|32|3|5.0|mul|vector|0.2|0.5|16|3|0.01|0.5|nego_only|z|0.1|1.0|"
  "pos3_negoonly_z|0.01|0.0|64|3|3.0|mul|vector|0.2|0.5|16|3|0.01|0.5|nego_only|z|0.1|1.0|"
  "pos7_negoonly_z|0.01|0.0|64|3|7.0|mul|vector|0.2|0.5|16|3|0.01|0.5|nego_only|z|0.1|1.0|"
  "signeddeg_negoonly_z|0.01|0.0|64|3|5.0|mul_signed_diff_degree|vector|0.2|0.5|16|3|0.01|0.5|nego_only|z|0.1|1.0|"
  "scalar_signeddeg_z|0.01|0.0|64|3|5.0|mul_signed_diff_degree|scalar|0.2|0.5|16|3|0.01|0.5|nego_only|z|0.1|1.0|"
  "fd03_negoonly_z|0.01|0.0|64|3|5.0|mul|vector|0.2|0.3|16|3|0.01|0.5|nego_only|z|0.1|1.0|"
  "fd07_negoonly_z|0.01|0.0|64|3|5.0|mul|vector|0.2|0.7|16|3|0.01|0.5|nego_only|z|0.1|1.0|"
  "nego02_ctx07_z|0.01|0.0|64|3|5.0|mul|vector|0.2|0.5|16|3|0.02|0.7|nego_only|z|0.1|1.0|"
  "graphlite_z|0.01|0.0|64|3|5.0|mul|vector|0.1|0.5|8|1|0.01|0.5|nego_only|z|0.05|2.0|"
  "global_z|0.01|0.0|64|3|5.0|mul|vector|0.2|0.5|16|3|0.01|0.5|nego_only|z|0.1|1.0|--use_global_info"
)

echo "[INFO] Twitch GCN Graph-CFAM+NeGo sweep"
echo "[INFO] entrypoint=${ENTRYPOINT}"
echo "[INFO] epochs=${EPOCHS}, runs=${RUNS}, configs=${#CONFIGS[@]}, device=${DEVICE}, python=${PYTHON}"
echo "[INFO] dry_run=${DRY_RUN}, max_configs=${MAX_CONFIGS}, early_stop_patience=${EARLY_STOP_PATIENCE}"

config_idx=0
for config in "${CONFIGS[@]}"; do
  config_idx=$((config_idx + 1))
  if [[ "${MAX_CONFIGS}" -gt 0 && "${config_idx}" -gt "${MAX_CONFIGS}" ]]; then
    break
  fi

  IFS='|' read -r tag lr dropout hidden_channels k pos_weight edge_feat_mode edge_gate_mode edge_blend fd_blend dag_latent_dim gmm_sample_k lambda_nego nego_context_weight fd_context_source nego_source graph_residual graph_gate_temp extra_args <<< "${config}"

  result_name="twitch_gcn_graphcfam_nego_${tag}_lr${lr}_dp${dropout}_h${hidden_channels}_K${k}_pos${pos_weight}_${edge_feat_mode}_${edge_gate_mode}_fd${fd_blend}_edge${edge_blend}_dim${dag_latent_dim}_gmm${gmm_sample_k}_nlg${lambda_nego}_nctx${nego_context_weight}_${fd_context_source}_${nego_source}_gr${graph_residual}_gt${graph_gate_temp}"
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
    --dropout "${dropout}"
    --tau "${TAU}"
    --hidden_channels "${hidden_channels}"
    --num_layers "${NUM_LAYERS}"
    --K "${k}"
    --pos_weight "${pos_weight}"
    --env_type "${ENV_TYPE}"
    --edge_feat_mode "${edge_feat_mode}"
    --edge_gate_mode "${edge_gate_mode}"
    --edge_blend "${edge_blend}"
    --edge_score_temp "${EDGE_SCORE_TEMP}"
    --dag_latent_dim "${dag_latent_dim}"
    --gmm_sample_k "${gmm_sample_k}"
    --lambda_dag "${LAMBDA_DAG}"
    --lambda_dag_label "${LAMBDA_DAG_LABEL}"
    --lambda_spu "${LAMBDA_SPU}"
    --lambda_env "${LAMBDA_ENV}"
    --lambda_fd "${LAMBDA_FD}"
    --fd_blend "${fd_blend}"
    --lambda_nego "${lambda_nego}"
    --nego_temp "${NEGO_TEMP}"
    --nego_context_weight "${nego_context_weight}"
    --fd_context_source "${fd_context_source}"
    --nego_source "${nego_source}"
    --graph_cfam_residual_blend "${graph_residual}"
    --graph_cfam_gate_temp "${graph_gate_temp}"
    --epochs "${EPOCHS}"
    --runs "${RUNS}"
    --device "${DEVICE}"
    --display_step "${DISPLAY_STEP}"
    --early_stop_patience "${EARLY_STOP_PATIENCE}"
    --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}"
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
