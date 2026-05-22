#!/usr/bin/env bash

set -euo pipefail

# DAG-module ablation sweep for model_gmm3_reviewed1_graph_cfam_nego.py on Arxiv.
#
# Graph-CFAM and NeGo are enabled for every config.  The sweep keeps the Arxiv
# GCN + Graph-CFAM + NeGo recipe fixed and removes one DAG submodule/signal at
# a time, so the delta from full_dag points to the likely source of the gain.
#
# Usage:
#   bash run_arxiv_gcn_dag_ablation_sweep.sh
#   DEVICE=1 bash run_arxiv_gcn_dag_ablation_sweep.sh
#   DRY_RUN=1 bash run_arxiv_gcn_dag_ablation_sweep.sh
#   MAX_CONFIGS=4 bash run_arxiv_gcn_dag_ablation_sweep.sh
#   EXTRA_BASE_ARGS="--use_global_info" bash run_arxiv_gcn_dag_ablation_sweep.sh

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
EDGE_BLEND="${EDGE_BLEND:-0.2}"
EDGE_SCORE_TEMP="${EDGE_SCORE_TEMP:-5.0}"
DAG_LATENT_DIM="${DAG_LATENT_DIM:-16}"
GMM_SAMPLE_K="${GMM_SAMPLE_K:-3}"
LAMBDA_SPU="${LAMBDA_SPU:-0.05}"
LAMBDA_ENV="${LAMBDA_ENV:-0.05}"
LAMBDA_FD="${LAMBDA_FD:-0.5}"
FD_BLEND="${FD_BLEND:-0.5}"
LAMBDA_NEGO="${LAMBDA_NEGO:-0.01}"
NEGO_TEMP="${NEGO_TEMP:-0.2}"
NEGO_CONTEXT_WEIGHT="${NEGO_CONTEXT_WEIGHT:-0.5}"
NEGO_SOURCE="${NEGO_SOURCE:-z}"
FD_CONTEXT_SOURCE="${FD_CONTEXT_SOURCE:-nego_only}"
GRAPH_CFAM_RESIDUAL_BLEND="${GRAPH_CFAM_RESIDUAL_BLEND:-0.1}"
GRAPH_CFAM_GATE_TEMP="${GRAPH_CFAM_GATE_TEMP:-1.0}"

# tag|lambda_dag|lambda_dag_label|lambda_l1|edge_pollution_coeff|extra_args
CONFIGS=(
  "full_dag|0.05|0.05|1e-5|0.5|"
  "no_dag_losses|0.0|0.0|0.0|0.5|"
  "no_acyclic|0.05|0.05|1e-5|0.5|--dag_ablate_acyclic_loss"
  "no_l1|0.05|0.05|0.0|0.5|"
  "no_flow_consistency|0.05|0.05|1e-5|0.5|--dag_ablate_flow_consistency"
  "no_dag_label|0.05|0.0|1e-5|0.5|"
  "no_label_effect_gate|0.05|0.05|1e-5|0.5|--dag_ablate_label_effect"
  "no_causal_support_gate|0.05|0.05|1e-5|0.5|--dag_ablate_causal_support"
  "no_pollution_gate|0.05|0.05|1e-5|0.5|--dag_ablate_pollution"
  "no_edge_pollution|0.05|0.05|1e-5|0.0|"
  "no_dag_mixer|0.05|0.05|1e-5|0.5|--disable_dag_mixer"
  "ica_no_dag_split|0.05|0.05|1e-5|0.5|--use_ica_split"
)

extra_base_args_array=()
if [[ -n "${EXTRA_BASE_ARGS:-}" ]]; then
  read -r -a extra_base_args_array <<< "${EXTRA_BASE_ARGS}"
fi

echo "[INFO] Arxiv GCN DAG ablation sweep"
echo "[INFO] entrypoint=${ENTRYPOINT}"
echo "[INFO] epochs=${EPOCHS}, runs=${RUNS}, configs=${#CONFIGS[@]}, device=${DEVICE}, python=${PYTHON}"
echo "[INFO] dry_run=${DRY_RUN}, max_configs=${MAX_CONFIGS}"

config_idx=0
for config in "${CONFIGS[@]}"; do
  config_idx=$((config_idx + 1))
  if [[ "${MAX_CONFIGS}" -gt 0 && "${config_idx}" -gt "${MAX_CONFIGS}" ]]; then
    break
  fi

  IFS='|' read -r tag lambda_dag lambda_dag_label lambda_l1 edge_pollution_coeff extra_args <<< "${config}"

  result_name="arxiv_gcn_graphcfam_nego_dagabl_${tag}_lr${LR}_wd${WEIGHT_DECAY}_dp${DROPOUT}_h${HIDDEN_CHANNELS}_K${K}_d${lambda_dag}_dl${lambda_dag_label}_l1${lambda_l1}_ep${edge_pollution_coeff}_dim${DAG_LATENT_DIM}_nlg${LAMBDA_NEGO}_nctx${NEGO_CONTEXT_WEIGHT}_${FD_CONTEXT_SOURCE}_${NEGO_SOURCE}_gr${GRAPH_CFAM_RESIDUAL_BLEND}_gt${GRAPH_CFAM_GATE_TEMP}"
  echo "[RUN ${config_idx}/${#CONFIGS[@]}] ${result_name}"

  extra_args_array=()
  if [[ -n "${extra_args}" ]]; then
    read -r -a extra_args_array <<< "${extra_args}"
  fi

  cmd=(
    "${PYTHON}" "${ENTRYPOINT}"
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
    --edge_blend "${EDGE_BLEND}"
    --edge_score_temp "${EDGE_SCORE_TEMP}"
    --dag_latent_dim "${DAG_LATENT_DIM}"
    --lambda_dag "${lambda_dag}"
    --lambda_dag_label "${lambda_dag_label}"
    --lambda_l1 "${lambda_l1}"
    --edge_pollution_coeff "${edge_pollution_coeff}"
    --lambda_spu "${LAMBDA_SPU}"
    --lambda_env "${LAMBDA_ENV}"
    --lambda_fd "${LAMBDA_FD}"
    --fd_blend "${FD_BLEND}"
    --gmm_sample_k "${GMM_SAMPLE_K}"
    --lambda_nego "${LAMBDA_NEGO}"
    --nego_temp "${NEGO_TEMP}"
    --nego_context_weight "${NEGO_CONTEXT_WEIGHT}"
    --fd_context_source "${FD_CONTEXT_SOURCE}"
    --nego_source "${NEGO_SOURCE}"
    --graph_cfam_residual_blend "${GRAPH_CFAM_RESIDUAL_BLEND}"
    --graph_cfam_gate_temp "${GRAPH_CFAM_GATE_TEMP}"
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
