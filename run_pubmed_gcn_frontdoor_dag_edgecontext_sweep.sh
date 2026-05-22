#!/usr/bin/env bash

set -euo pipefail

# Hyper-parameter sweep for the PubMed edge-context DAG-Core front-door model
# with a GCN backbone.
#
# Usage:
#   DEVICE=1 bash run_pubmed_gcn_frontdoor_dag_edgecontext_sweep.sh
#   EPOCHS=300 RUNS=1 bash run_pubmed_gcn_frontdoor_dag_edgecontext_sweep.sh

DATASET="pubmed"
BACKBONE="gcn"
ENV_TYPE="graph"
ENTRYPOINT="main_frontdoor_dag_core_edgefeat_pubmed_edgecontext.py"

EPOCHS="${EPOCHS:-500}"
RUNS="${RUNS:-1}"
DEVICE="${DEVICE:-0}"
PYTHON="${PYTHON:-python}"
DISPLAY_STEP="${DISPLAY_STEP:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-80}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"

WEIGHT_DECAY="${WEIGHT_DECAY:-5e-5}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
K="${K:-2}"
TAU="${TAU:-2}"
EDGE_FEAT_MODE="mul"

# name|lr|dropout|lambda_dag|lambda_dag_label|lambda_spu|lambda_env|lambda_fd|fd_blend|dag_latent_dim|edge_blend|edge_score_temp|edge_spu_threshold|edge_spu_temp|edge_spu_alpha|edge_spu_msg_mode|gmm_sample_k|extra_args
CONFIGS=(
  "base_edgectx|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.5|16|0.2|5.0|0.35|8.0|0.3|residual|2|"
  "edge_blend01|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.5|16|0.1|5.0|0.35|8.0|0.3|residual|2|"
  "score_temp2|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.5|16|0.2|2.0|0.35|8.0|0.3|residual|2|"
  "score_temp8|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.5|16|0.2|8.0|0.35|8.0|0.3|residual|2|"
  "spu_thr025|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.5|16|0.2|5.0|0.25|8.0|0.3|residual|2|"
  "spu_thr045|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.5|16|0.2|5.0|0.45|8.0|0.3|residual|2|"
  "spu_temp4|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.5|16|0.2|5.0|0.35|4.0|0.3|residual|2|"
  "spu_temp12|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.5|16|0.2|5.0|0.35|12.0|0.3|residual|2|"
  "spu_alpha05|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.5|16|0.2|5.0|0.35|8.0|0.5|residual|2|"
  "spu_neighbor|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.5|16|0.2|5.0|0.35|8.0|0.3|neighbor|2|"
  "fd03|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.3|16|0.2|5.0|0.35|8.0|0.3|residual|2|"
  "fd07|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.7|16|0.2|5.0|0.35|8.0|0.3|residual|2|"
  "dim8|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.5|8|0.2|5.0|0.35|8.0|0.3|residual|2|"
  "lr01|0.01|0.2|0.05|0.05|0.05|0.05|0.5|0.5|16|0.2|5.0|0.35|8.0|0.3|residual|2|"
  "dp03|0.005|0.3|0.05|0.05|0.05|0.05|0.5|0.5|16|0.2|5.0|0.35|8.0|0.3|residual|2|"
  "node_edge|0.005|0.2|0.05|0.05|0.05|0.05|0.5|0.5|16|0.2|5.0|0.35|8.0|0.3|residual|2|--dag_input_mode node_edge"
)

echo "[INFO] PubMed GCN edge-context DAG-Core sweep"
echo "[INFO] entrypoint=${ENTRYPOINT}"
echo "[INFO] epochs=${EPOCHS}, runs=${RUNS}, configs=${#CONFIGS[@]}, device=${DEVICE}, python=${PYTHON}"
echo "[INFO] fixed: wd=${WEIGHT_DECAY}, hidden=${HIDDEN_CHANNELS}, layers=${NUM_LAYERS}, K=${K}, tau=${TAU}, edge_feat_mode=${EDGE_FEAT_MODE}"

for config in "${CONFIGS[@]}"; do
  IFS='|' read -r tag lr dropout lambda_dag lambda_dag_label lambda_spu lambda_env lambda_fd fd_blend dag_latent_dim edge_blend edge_score_temp edge_spu_threshold edge_spu_temp edge_spu_alpha edge_spu_msg_mode gmm_sample_k extra_args <<< "${config}"

  result_name="pubmed_gcn_edgectx_${tag}_lr${lr}_dp${dropout}_dag${lambda_dag}_dl${lambda_dag_label}_spu${lambda_spu}_env${lambda_env}_lfd${lambda_fd}_fd${fd_blend}_dim${dag_latent_dim}_edge${edge_blend}_score${edge_score_temp}_thr${edge_spu_threshold}_stemp${edge_spu_temp}_alpha${edge_spu_alpha}_${edge_spu_msg_mode}_gmm${gmm_sample_k}"
  echo "[RUN] ${result_name}"

  # shellcheck disable=SC2206
  extra_args_array=(${extra_args})

  "${PYTHON}" "${ENTRYPOINT}" \
    --dataset "${DATASET}" \
    --backbone "${BACKBONE}" \
    --lr "${lr}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --dropout "${dropout}" \
    --tau "${TAU}" \
    --hidden_channels "${HIDDEN_CHANNELS}" \
    --num_layers "${NUM_LAYERS}" \
    --K "${K}" \
    --lambda_dag "${lambda_dag}" \
    --lambda_dag_label "${lambda_dag_label}" \
    --lambda_spu "${lambda_spu}" \
    --lambda_env "${lambda_env}" \
    --lambda_fd "${lambda_fd}" \
    --fd_blend "${fd_blend}" \
    --dag_latent_dim "${dag_latent_dim}" \
    --edge_feat_mode "${EDGE_FEAT_MODE}" \
    --edge_blend "${edge_blend}" \
    --edge_score_temp "${edge_score_temp}" \
    --use_edge_spu_context \
    --edge_spu_threshold "${edge_spu_threshold}" \
    --edge_spu_temp "${edge_spu_temp}" \
    --edge_spu_context_alpha "${edge_spu_alpha}" \
    --edge_spu_msg_mode "${edge_spu_msg_mode}" \
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
