#!/usr/bin/env bash

set -euo pipefail

# Non-front-door ablation sweep for model_gmm3_reviewed1_graph_cfam_nego.py on Arxiv.
#
# The default configs force --model_variant frontdoor_nego and use
# --direct_z_spurious_mode none, so evaluation and training are driven by the
# mediator/enhanced node representation instead of the front-door context path.
# Use the first two configs to verify the known observation that FD is not the
# main source, then compare the representation-only configs.
#
# Usage:
#   bash run_arxiv_gcn_nonfrontdoor_ablation_sweep.sh
#   DEVICE=0 MAX_CONFIGS=3 bash run_arxiv_gcn_nonfrontdoor_ablation_sweep.sh
#   DRY_RUN=1 bash run_arxiv_gcn_nonfrontdoor_ablation_sweep.sh
#   EXTRA_BASE_ARGS="--lambda_entropy_dro 0.1" bash run_arxiv_gcn_nonfrontdoor_ablation_sweep.sh

DATASET="arxiv"
BACKBONE="gcn"
ENV_TYPE="node"
ENTRYPOINT="main_gmm3_reviewed1_graph_cfam_nego.py"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-1}"
EPOCHS="${EPOCHS:-500}"
RUNS="${RUNS:-1}"
DISPLAY_STEP="${DISPLAY_STEP:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-80}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
MAX_CONFIGS="${MAX_CONFIGS:-0}"
DRY_RUN="${DRY_RUN:-0}"

LR="${LR:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
DROPOUT="${DROPOUT:-0.2}"
TAU="${TAU:-1}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
K="${K:-3}"
EDGE_FEAT_MODE="${EDGE_FEAT_MODE:-mul}"
EDGE_GATE_MODE="${EDGE_GATE_MODE:-vector}"
EDGE_SCORE_TEMP="${EDGE_SCORE_TEMP:-5.0}"
EDGE_BLEND="${EDGE_BLEND:-0.2}"
GRAPH_CFAM_RESIDUAL_BLEND="${GRAPH_CFAM_RESIDUAL_BLEND:-0.1}"
GRAPH_CFAM_GATE_TEMP="${GRAPH_CFAM_GATE_TEMP:-1.0}"

# tag|edge_blend|extra_args
CONFIGS=(
  "full_recipe_fd_train_eval_mediator|${EDGE_BLEND}|--use_graph_cfam --use_nego_prompt --use_nego_context --lambda_nego 0.01 --lambda_fd 0.5 --fd_blend 0.5"
  "repr_only_graphcfam|${EDGE_BLEND}|--use_graph_cfam --direct_z_spurious_mode none --lambda_fd 0 --fd_blend 0 --lambda_nego 0 --disable_spu_gmm"
  "plain_backbone|0.0|--direct_z_spurious_mode none --lambda_fd 0 --fd_blend 0 --lambda_nego 0 --disable_spu_gmm --disable_final_edge_enhance --disable_node_edge_norm"
  "final_edge_only|${EDGE_BLEND}|--direct_z_spurious_mode none --lambda_fd 0 --fd_blend 0 --lambda_nego 0 --disable_spu_gmm"
  "final_edge_no_norm|${EDGE_BLEND}|--direct_z_spurious_mode none --lambda_fd 0 --fd_blend 0 --lambda_nego 0 --disable_spu_gmm --disable_node_edge_norm"
  "layerwise_local_igm|${EDGE_BLEND}|--direct_z_spurious_mode none --lambda_fd 0 --fd_blend 0 --lambda_nego 0 --disable_spu_gmm --use_layerwise_local_igm --layerwise_local_igm_include_last"
  "graphcfam_full|${EDGE_BLEND}|--use_graph_cfam --direct_z_spurious_mode none --lambda_fd 0 --fd_blend 0 --lambda_nego 0 --disable_spu_gmm"
  "graphcfam_final_only|${EDGE_BLEND}|--use_graph_cfam --disable_layerwise_graph_cfam --direct_z_spurious_mode none --lambda_fd 0 --fd_blend 0 --lambda_nego 0 --disable_spu_gmm"
  "graphcfam_layerwise_only|${EDGE_BLEND}|--use_graph_cfam --disable_final_graph_cfam --direct_z_spurious_mode none --lambda_fd 0 --fd_blend 0 --lambda_nego 0 --disable_spu_gmm"
  "graphcfam_no_residual|${EDGE_BLEND}|--use_graph_cfam --graph_cfam_residual_blend 0.0 --direct_z_spurious_mode none --lambda_fd 0 --fd_blend 0 --lambda_nego 0 --disable_spu_gmm"
  "graphcfam_soft_gate|${EDGE_BLEND}|--use_graph_cfam --graph_cfam_gate_temp 1000000.0 --direct_z_spurious_mode none --lambda_fd 0 --fd_blend 0 --lambda_nego 0 --disable_spu_gmm"
  "no_cipt_no_cosine|${EDGE_BLEND}|--use_graph_cfam --direct_z_spurious_mode none --lambda_fd 0 --fd_blend 0 --lambda_nego 0 --disable_spu_gmm --disable_cipt_schedule --disable_cosine_lr --grad_clip 0"
)

extra_base_args_array=()
if [[ -n "${EXTRA_BASE_ARGS:-}" ]]; then
  read -r -a extra_base_args_array <<< "${EXTRA_BASE_ARGS}"
fi

echo "[INFO] Arxiv GCN non-front-door ablation sweep"
echo "[INFO] entrypoint=${ENTRYPOINT}"
echo "[INFO] epochs=${EPOCHS}, runs=${RUNS}, configs=${#CONFIGS[@]}, device=${DEVICE}, python=${PYTHON}"
echo "[INFO] dry_run=${DRY_RUN}, max_configs=${MAX_CONFIGS}"

config_idx=0
for config in "${CONFIGS[@]}"; do
  config_idx=$((config_idx + 1))
  if [[ "${MAX_CONFIGS}" -gt 0 && "${config_idx}" -gt "${MAX_CONFIGS}" ]]; then
    break
  fi

  IFS='|' read -r tag edge_blend extra_args <<< "${config}"
  result_name="arxiv_gcn_nonfd_${tag}_lr${LR}_wd${WEIGHT_DECAY}_dp${DROPOUT}_h${HIDDEN_CHANNELS}_L${NUM_LAYERS}_K${K}_${EDGE_FEAT_MODE}_${EDGE_GATE_MODE}_edge${edge_blend}_score${EDGE_SCORE_TEMP}_gr${GRAPH_CFAM_RESIDUAL_BLEND}_gt${GRAPH_CFAM_GATE_TEMP}"
  echo "[RUN ${config_idx}/${#CONFIGS[@]}] ${result_name}"

  extra_args_array=()
  if [[ -n "${extra_args}" ]]; then
    read -r -a extra_args_array <<< "${extra_args}"
  fi

  cmd=(
    "${PYTHON}" "${ENTRYPOINT}"
    --model_variant frontdoor_nego
    --dataset "${DATASET}"
    --backbone "${BACKBONE}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --dropout "${DROPOUT}"
    --tau "${TAU}"
    --hidden_channels "${HIDDEN_CHANNELS}"
    --num_layers "${NUM_LAYERS}"
    --K "${K}"
    --env_type "${ENV_TYPE}"
    --variant
    --edge_feat_mode "${EDGE_FEAT_MODE}"
    --edge_gate_mode "${EDGE_GATE_MODE}"
    --edge_blend "${edge_blend}"
    --edge_score_temp "${EDGE_SCORE_TEMP}"
    --graph_cfam_residual_blend "${GRAPH_CFAM_RESIDUAL_BLEND}"
    --graph_cfam_gate_temp "${GRAPH_CFAM_GATE_TEMP}"
    --eval_pred_mode mediator
    --epochs "${EPOCHS}"
    --runs "${RUNS}"
    --device "${DEVICE}"
    --display_step "${DISPLAY_STEP}"
    --early_stop_patience "${EARLY_STOP_PATIENCE}"
    --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}"
    --store
    --result_name "${result_name}"
    "${extra_base_args_array[@]}"
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
