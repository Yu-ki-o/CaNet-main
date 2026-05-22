#!/usr/bin/env bash

set -euo pipefail

# Hyper-parameter sweep for model_frontdoor_dag_core_edgefeat_pubmed_denoise.py
# on Twitch with a GCN backbone.
#
# Each config runs once for 300 epochs by default.
#
# Usage:
#   bash run_twitch_gcn_frontdoor_dag_edgefeat_denoise_sweep.sh
#   DEVICE=1 bash run_twitch_gcn_frontdoor_dag_edgefeat_denoise_sweep.sh
#   PYTHON=python3 EPOCHS=300 RUNS=1 bash run_twitch_gcn_frontdoor_dag_edgefeat_denoise_sweep.sh

DATASET="twitch"
BACKBONE="gcn"
ENV_TYPE="graph"
ENTRYPOINT="main_frontdoor_dag_core_edgefeat_pubmed_denoise.py"

EPOCHS="${EPOCHS:-300}"
RUNS="${RUNS:-1}"
DEVICE="${DEVICE:-0}"
PYTHON="${PYTHON:-python}"
DISPLAY_STEP="${DISPLAY_STEP:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"

WEIGHT_DECAY="${WEIGHT_DECAY:-5e-5}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-32}"
NUM_LAYERS="${NUM_LAYERS:-2}"
K="${K:-4}"
TAU="${TAU:-3}"
POS_WEIGHT="${POS_WEIGHT:-5.0}"

# name|lr|dropout|lambda_dag|lambda_dag_label|lambda_spu|lambda_env|lambda_fd|fd_blend|dag_latent_dim|edge_feat_mode|edge_blend|edge_score_temp|gmm_sample_k|noise_alpha|noise_temp|extra_args
CONFIGS=(
  "edge_only|0.005|0.2|0.005|0.01|0.02|0.0|0.5|0.3|8|mul_diff|0.05|2.0|1|0.0|1.0|"
  "denoise_a005|0.005|0.2|0.005|0.01|0.02|0.0|0.5|0.3|8|mul_diff|0.05|2.0|1|0.05|1.0|--use_neighbor_denoise"
  "denoise_a01|0.005|0.2|0.005|0.01|0.02|0.0|0.5|0.3|8|mul_diff|0.05|2.0|1|0.1|1.0|--use_neighbor_denoise"
  "denoise_a02|0.005|0.2|0.005|0.01|0.02|0.0|0.5|0.3|8|mul_diff|0.05|2.0|1|0.2|1.0|--use_neighbor_denoise"
  "edge_blend01|0.005|0.2|0.005|0.01|0.02|0.0|0.5|0.3|8|mul_diff|0.1|2.0|1|0.1|1.0|--use_neighbor_denoise"
  "score_temp5|0.005|0.2|0.005|0.01|0.02|0.0|0.5|0.3|8|mul_diff|0.05|5.0|1|0.1|1.0|--use_neighbor_denoise"
  "noise_temp05|0.005|0.2|0.005|0.01|0.02|0.0|0.5|0.3|8|mul_diff|0.05|2.0|1|0.1|0.5|--use_neighbor_denoise"
  "noise_temp2|0.005|0.2|0.005|0.01|0.02|0.0|0.5|0.3|8|mul_diff|0.05|2.0|1|0.1|2.0|--use_neighbor_denoise"
  "diff_degree|0.005|0.2|0.005|0.01|0.02|0.0|0.5|0.3|8|diff_degree|0.05|2.0|1|0.1|1.0|--use_neighbor_denoise"
  "mul_diff_degree|0.005|0.2|0.005|0.01|0.02|0.0|0.5|0.3|8|mul_diff_degree|0.05|2.0|1|0.1|1.0|--use_neighbor_denoise"
  "dim16_gmm3|0.005|0.2|0.005|0.01|0.02|0.0|0.5|0.3|16|mul_diff|0.1|2.0|3|0.1|1.0|--use_neighbor_denoise"
  "balanced|0.005|0.2|0.01|0.05|0.05|0.02|0.5|0.5|16|mul_diff|0.1|2.0|3|0.1|1.0|--use_neighbor_denoise"
)

echo "[INFO] Twitch GCN DAG edgefeat-denoise sweep"
echo "[INFO] entrypoint=${ENTRYPOINT}"
echo "[INFO] epochs=${EPOCHS}, runs=${RUNS}, configs=${#CONFIGS[@]}, device=${DEVICE}, python=${PYTHON}"
echo "[INFO] fixed: wd=${WEIGHT_DECAY}, hidden=${HIDDEN_CHANNELS}, layers=${NUM_LAYERS}, K=${K}, tau=${TAU}, pos_weight=${POS_WEIGHT}"

for config in "${CONFIGS[@]}"; do
  IFS='|' read -r tag lr dropout lambda_dag lambda_dag_label lambda_spu lambda_env lambda_fd fd_blend dag_latent_dim edge_feat_mode edge_blend edge_score_temp gmm_sample_k noise_alpha noise_temp extra_args <<< "${config}"

  result_name="twitch_gcn_dag_edgefeat_denoise_${tag}_lr${lr}_dp${dropout}_d${lambda_dag}_dl${lambda_dag_label}_spu${lambda_spu}_env${lambda_env}_lfd${lambda_fd}_fd${fd_blend}_dim${dag_latent_dim}_${edge_feat_mode}_edge${edge_blend}_score${edge_score_temp}_gmm${gmm_sample_k}_na${noise_alpha}_nt${noise_temp}"
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
    --edge_feat_mode "${edge_feat_mode}" \
    --edge_blend "${edge_blend}" \
    --edge_score_temp "${edge_score_temp}" \
    --gmm_sample_k "${gmm_sample_k}" \
    --noise_subtract_alpha "${noise_alpha}" \
    --noise_gate_temp "${noise_temp}" \
    --pos_weight "${POS_WEIGHT}" \
    --env_type "${ENV_TYPE}" \
    --epochs "${EPOCHS}" \
    --runs "${RUNS}" \
    --device "${DEVICE}" \
    --display_step "${DISPLAY_STEP}" \
    --early_stop_patience "${EARLY_STOP_PATIENCE}" \
    --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}" \
    --store \
    --result_name "${result_name}" \
    "${extra_args_array[@]}"
done
