#!/usr/bin/env bash

set -euo pipefail

# Fine sweep around the best coarse PubMed GCN edge-context run:
# pubmed_gcn_edgectx_dp03_lr0.005_dp0.3_dag0.05_dl0.05_spu0.05_env0.05_lfd0.5_fd0.5_dim16_edge0.2_score5.0_thr0.35_stemp8.0_alpha0.3_residual_gmm2
#
# Usage:
#   DEVICE=1 bash run_pubmed_gcn_frontdoor_dag_edgecontext_finetune_dp03.sh
#   EPOCHS=500 RUNS=2 DEVICE=1 bash run_pubmed_gcn_frontdoor_dag_edgecontext_finetune_dp03.sh

DATASET="pubmed"
BACKBONE="gcn"
ENV_TYPE="graph"
ENTRYPOINT="main_frontdoor_dag_core_edgefeat_pubmed_edgecontext.py"

EPOCHS="${EPOCHS:-500}"
RUNS="${RUNS:-2}"
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

# Center point from the best coarse run.
BASE_LR="0.005"
BASE_DROPOUT="0.3"
BASE_LAMBDA_DAG="0.05"
BASE_LAMBDA_DAG_LABEL="0.05"
BASE_LAMBDA_SPU="0.05"
BASE_LAMBDA_ENV="0.05"
BASE_LAMBDA_FD="0.5"
BASE_FD_BLEND="0.5"
BASE_DAG_LATENT_DIM="16"
BASE_EDGE_BLEND="0.2"
BASE_EDGE_SCORE_TEMP="5.0"
BASE_EDGE_SPU_THRESHOLD="0.35"
BASE_EDGE_SPU_TEMP="8.0"
BASE_EDGE_SPU_ALPHA="0.3"
BASE_EDGE_SPU_MSG_MODE="residual"
BASE_GMM_SAMPLE_K="2"

run_one() {
  local stage="$1"
  local tag="$2"
  local lr="$3"
  local dropout="$4"
  local lambda_dag="$5"
  local lambda_dag_label="$6"
  local lambda_spu="$7"
  local lambda_env="$8"
  local lambda_fd="$9"
  local fd_blend="${10}"
  local dag_latent_dim="${11}"
  local edge_blend="${12}"
  local edge_score_temp="${13}"
  local edge_spu_threshold="${14}"
  local edge_spu_temp="${15}"
  local edge_spu_alpha="${16}"
  local edge_spu_msg_mode="${17}"
  local gmm_sample_k="${18}"

  local result_name
  result_name="pubmed_gcn_edgectx_fine_${stage}_${tag}_lr${lr}_dp${dropout}_dag${lambda_dag}_dl${lambda_dag_label}_spu${lambda_spu}_env${lambda_env}_lfd${lambda_fd}_fd${fd_blend}_dim${dag_latent_dim}_edge${edge_blend}_score${edge_score_temp}_thr${edge_spu_threshold}_stemp${edge_spu_temp}_alpha${edge_spu_alpha}_${edge_spu_msg_mode}_gmm${gmm_sample_k}"

  echo "[RUN] ${result_name}"

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
    --result_name "${result_name}"
}

echo "[INFO] PubMed GCN edge-context fine sweep around dp03"
echo "[INFO] entrypoint=${ENTRYPOINT}"
echo "[INFO] epochs=${EPOCHS}, runs=${RUNS}, device=${DEVICE}, python=${PYTHON}"
echo "[INFO] fixed: wd=${WEIGHT_DECAY}, hidden=${HIDDEN_CHANNELS}, layers=${NUM_LAYERS}, K=${K}, tau=${TAU}, edge_feat_mode=${EDGE_FEAT_MODE}"
echo "[INFO] planned runs: 54"

# Stage A: refine optimizer regularization around lr=0.005, dropout=0.3.
for lr in 0.003 0.005 0.007; do
  for dropout in 0.25 0.30 0.35; do
    run_one "opt" "lr${lr}_dp${dropout}" \
      "${lr}" "${dropout}" \
      "${BASE_LAMBDA_DAG}" "${BASE_LAMBDA_DAG_LABEL}" "${BASE_LAMBDA_SPU}" "${BASE_LAMBDA_ENV}" \
      "${BASE_LAMBDA_FD}" "${BASE_FD_BLEND}" "${BASE_DAG_LATENT_DIM}" \
      "${BASE_EDGE_BLEND}" "${BASE_EDGE_SCORE_TEMP}" "${BASE_EDGE_SPU_THRESHOLD}" "${BASE_EDGE_SPU_TEMP}" \
      "${BASE_EDGE_SPU_ALPHA}" "${BASE_EDGE_SPU_MSG_MODE}" "${BASE_GMM_SAMPLE_K}"
  done
done

# Stage B: refine edge-aware aggregation and edge gate smoothness.
for edge_blend in 0.15 0.20 0.25; do
  for edge_score_temp in 4.0 5.0 6.0; do
    for edge_spu_alpha in 0.20 0.30 0.40; do
      run_one "edge" "eb${edge_blend}_score${edge_score_temp}_alpha${edge_spu_alpha}" \
        "${BASE_LR}" "${BASE_DROPOUT}" \
        "${BASE_LAMBDA_DAG}" "${BASE_LAMBDA_DAG_LABEL}" "${BASE_LAMBDA_SPU}" "${BASE_LAMBDA_ENV}" \
        "${BASE_LAMBDA_FD}" "${BASE_FD_BLEND}" "${BASE_DAG_LATENT_DIM}" \
        "${edge_blend}" "${edge_score_temp}" "${BASE_EDGE_SPU_THRESHOLD}" "${BASE_EDGE_SPU_TEMP}" \
        "${edge_spu_alpha}" "${BASE_EDGE_SPU_MSG_MODE}" "${BASE_GMM_SAMPLE_K}"
    done
  done
done

# Stage C: refine the low-score structural-spurious edge selector.
for edge_spu_threshold in 0.30 0.35 0.40; do
  for edge_spu_temp in 6.0 8.0 10.0; do
    run_one "spugate" "thr${edge_spu_threshold}_temp${edge_spu_temp}" \
      "${BASE_LR}" "${BASE_DROPOUT}" \
      "${BASE_LAMBDA_DAG}" "${BASE_LAMBDA_DAG_LABEL}" "${BASE_LAMBDA_SPU}" "${BASE_LAMBDA_ENV}" \
      "${BASE_LAMBDA_FD}" "${BASE_FD_BLEND}" "${BASE_DAG_LATENT_DIM}" \
      "${BASE_EDGE_BLEND}" "${BASE_EDGE_SCORE_TEMP}" "${edge_spu_threshold}" "${edge_spu_temp}" \
      "${BASE_EDGE_SPU_ALPHA}" "${BASE_EDGE_SPU_MSG_MODE}" "${BASE_GMM_SAMPLE_K}"
  done
done

# Stage D: refine front-door blend and DAG label supervision strength.
for fd_blend in 0.40 0.50 0.60; do
  for lambda_dag_label in 0.03 0.05 0.07; do
    run_one "obj" "fd${fd_blend}_dl${lambda_dag_label}" \
      "${BASE_LR}" "${BASE_DROPOUT}" \
      "${BASE_LAMBDA_DAG}" "${lambda_dag_label}" "${BASE_LAMBDA_SPU}" "${BASE_LAMBDA_ENV}" \
      "${BASE_LAMBDA_FD}" "${fd_blend}" "${BASE_DAG_LATENT_DIM}" \
      "${BASE_EDGE_BLEND}" "${BASE_EDGE_SCORE_TEMP}" "${BASE_EDGE_SPU_THRESHOLD}" "${BASE_EDGE_SPU_TEMP}" \
      "${BASE_EDGE_SPU_ALPHA}" "${BASE_EDGE_SPU_MSG_MODE}" "${BASE_GMM_SAMPLE_K}"
  done
done
