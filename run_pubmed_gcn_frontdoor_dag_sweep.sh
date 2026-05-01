#!/usr/bin/env bash

set -euo pipefail

# Hyper-parameter sweep for the current front-door DAG model on PubMed with GCN.
# Override these from the shell when needed, e.g.:
#   DEVICE=1 bash run_pubmed_gcn_frontdoor_dag_sweep.sh

DATASET="pubmed"
BACKBONE="gcn"
ENV_TYPE="graph"

EPOCHS=500
RUNS=1
DEVICE="${DEVICE:-0}"
PYTHON="${PYTHON:-python}"
DISPLAY_STEP="${DISPLAY_STEP:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-50}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"

LRS=(0.01)
WEIGHT_DECAYS=(5e-5 0.0005)
DROPOUTS=(0.1 0.2 0.3)
TAUS=(1 2)
HIDDEN_CHANNELS=(64)
KS=(3)
DAG_LATENT_DIMS=(32 64 16)

TOTAL_CONFIGS=$((
  ${#LRS[@]} *
  ${#WEIGHT_DECAYS[@]} *
  ${#DROPOUTS[@]} *
  ${#TAUS[@]} *
  ${#HIDDEN_CHANNELS[@]} *
  ${#KS[@]} *
  ${#DAG_LATENT_DIMS[@]}
))

echo "[INFO] PubMed GCN front-door DAG sweep"
echo "[INFO] epochs=${EPOCHS}, runs=${RUNS}, total_configs=${TOTAL_CONFIGS}, device=${DEVICE}, python=${PYTHON}"

for lr in "${LRS[@]}"; do
  for weight_decay in "${WEIGHT_DECAYS[@]}"; do
    for dropout in "${DROPOUTS[@]}"; do
      for tau in "${TAUS[@]}"; do
        for hidden_channels in "${HIDDEN_CHANNELS[@]}"; do
          for k in "${KS[@]}"; do
            for dag_latent_dim in "${DAG_LATENT_DIMS[@]}"; do
              result_name="pubmed_gcn_fd_dag_lr${lr}_wd${weight_decay}_dp${dropout}_tau${tau}_h${hidden_channels}_K${k}_dagdim${dag_latent_dim}"
              echo "[RUN] ${result_name}"

              "${PYTHON}" main_frontdoor_dag.py \
                --dataset "${DATASET}" \
                --backbone "${BACKBONE}" \
                --lr "${lr}" \
                --weight_decay "${weight_decay}" \
                --dropout "${dropout}" \
                --tau "${tau}" \
                --hidden_channels "${hidden_channels}" \
                --K "${k}" \
                --dag_latent_dim "${dag_latent_dim}" \
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
            done
          done
        done
      done
    done
  done
done
