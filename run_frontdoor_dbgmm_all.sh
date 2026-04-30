#!/usr/bin/env bash
set -euo pipefail

# Dirichlet-Barycentric GMM Front-Door OOD batch runner
#
# 用法：
#   bash run_frontdoor_dbgmm_all.sh
#
# 可覆盖变量示例：
#   DEVICE=1 RUNS=5 EPOCHS=300 bash run_frontdoor_dbgmm_all.sh
#
# 说明：
#   1) K 按“训练环境数”设置：Cora/Citeseer/PubMed/Arxiv/Twitch = 3，Elliptic = 5。
#   2) gmm_sample_k 是从训练环境基础分布中扩展出的 virtual spurious contexts 数量。
#   3) 默认使用 backbonefix 版本，确保 --backbone gat/gcn 能真正传到模型中。

SCRIPT=${SCRIPT:-main_frontdoor2_dirichlet_barycentric_backbonefix.py}
PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-./data}
DEVICE=${DEVICE:-0}
RUNS=${RUNS:-10}
EPOCHS=${EPOCHS:-500}

BASE_ARGS="--data_dir ${DATA_DIR} --device ${DEVICE} --runs ${RUNS} --epochs ${EPOCHS}"

DBGMM_ARGS="\
  --use_spu_gmm \
  --gmm_alpha 0.1 \
  --gmm_sample_k 16 \
  --virtual_dir_alpha 0.5 \
  --virtual_between_scale 0.15 \
  --virtual_sample_temp 0.35 \
  --virtual_maha_max 4.0"

# 当前不加 --gmm_cap_by_fd_k，因为目标是从少数训练环境分布扩展出更多虚拟分布。
# 如果你想把 GMM context 数量限制回 K，可以额外加 --gmm_cap_by_fd_k。

echo "Running Dirichlet-Barycentric GMM Front-Door experiments"
echo "SCRIPT=${SCRIPT}"
echo "DATA_DIR=${DATA_DIR}, DEVICE=${DEVICE}, RUNS=${RUNS}, EPOCHS=${EPOCHS}"

# =========================
# GCN backbone
# =========================

${PYTHON} -u ${SCRIPT} ${BASE_ARGS} \
  --dataset cora \
  --backbone gcn \
  --K 3 \
  --weight_decay 5e-5 \
  --tau 1 \
  --dropout 0.2 \
  --env_type graph \
  --combine_result \
  ${DBGMM_ARGS} \
  --result_name frontdoor_dbgmm_cora_gcn \
  --store

${PYTHON} -u ${SCRIPT} ${BASE_ARGS} \
  --dataset citeseer \
  --backbone gcn \
  --K 3 \
  --weight_decay 5e-5 \
  --tau 1 \
  --dropout 0.1 \
  --env_type graph \
  --combine_result \
  ${DBGMM_ARGS} \
  --result_name frontdoor_dbgmm_citeseer_gcn \
  --store

${PYTHON} -u ${SCRIPT} ${BASE_ARGS} \
  --dataset pubmed \
  --backbone gcn \
  --K 3 \
  --weight_decay 5e-5 \
  --tau 2 \
  --dropout 0.2 \
  --env_type graph \
  --combine_result \
  ${DBGMM_ARGS} \
  --result_name frontdoor_dbgmm_pubmed_gcn \
  --store

${PYTHON} -u ${SCRIPT} ${BASE_ARGS} \
  --dataset arxiv \
  --backbone gcn \
  --K 3 \
  --weight_decay 0.0005 \
  --tau 1 \
  --dropout 0.2 \
  --env_type node \
  --variant \
  ${DBGMM_ARGS} \
  --result_name frontdoor_dbgmm_arxiv_gcn \
  --store

${PYTHON} -u ${SCRIPT} ${BASE_ARGS} \
  --dataset twitch \
  --backbone gcn \
  --K 3 \
  --weight_decay 5e-5 \
  --tau 3 \
  --dropout 0 \
  --env_type graph \
  ${DBGMM_ARGS} \
  --result_name frontdoor_dbgmm_twitch_gcn \
  --store

${PYTHON} -u ${SCRIPT} ${BASE_ARGS} \
  --dataset elliptic \
  --backbone gcn \
  --K 5 \
  --weight_decay 0.001 \
  --tau 1 \
  --dropout 0.2 \
  --env_type node \
  --variant \
  --num_layers 3 \
  --hidden_channels 32 \
  ${DBGMM_ARGS} \
  --result_name frontdoor_dbgmm_elliptic_gcn \
  --store


# =========================
# GAT backbone
# =========================

${PYTHON} -u ${SCRIPT} ${BASE_ARGS} \
  --dataset cora \
  --backbone gat \
  --K 3 \
  --weight_decay 0 \
  --tau 3 \
  --dropout 0.2 \
  --env_type graph \
  --combine_result \
  ${DBGMM_ARGS} \
  --result_name frontdoor_dbgmm_cora_gat \
  --store

${PYTHON} -u ${SCRIPT} ${BASE_ARGS} \
  --dataset citeseer \
  --backbone gat \
  --K 3 \
  --weight_decay 0 \
  --tau 3 \
  --dropout 0.2 \
  --env_type graph \
  --combine_result \
  ${DBGMM_ARGS} \
  --result_name frontdoor_dbgmm_citeseer_gat \
  --store

${PYTHON} -u ${SCRIPT} ${BASE_ARGS} \
  --dataset pubmed \
  --backbone gat \
  --K 3 \
  --weight_decay 5e-5 \
  --tau 1 \
  --dropout 0.2 \
  --env_type graph \
  --combine_result \
  ${DBGMM_ARGS} \
  --result_name frontdoor_dbgmm_pubmed_gat \
  --store

${PYTHON} -u ${SCRIPT} ${BASE_ARGS} \
  --dataset arxiv \
  --backbone gat \
  --K 3 \
  --weight_decay 5e-5 \
  --tau 2 \
  --dropout 0.2 \
  --env_type graph \
  ${DBGMM_ARGS} \
  --result_name frontdoor_dbgmm_arxiv_gat \
  --store

${PYTHON} -u ${SCRIPT} ${BASE_ARGS} \
  --dataset twitch \
  --backbone gat \
  --K 3 \
  --weight_decay 5e-5 \
  --tau 2 \
  --dropout 0 \
  --env_type graph \
  ${DBGMM_ARGS} \
  --result_name frontdoor_dbgmm_twitch_gat \
  --store

${PYTHON} -u ${SCRIPT} ${BASE_ARGS} \
  --dataset elliptic \
  --backbone gat \
  --K 5 \
  --weight_decay 0.0005 \
  --tau 2 \
  --dropout 0.1 \
  --env_type graph \
  ${DBGMM_ARGS} \
  --result_name frontdoor_dbgmm_elliptic_gat \
  --store

echo "All Dirichlet-Barycentric GMM Front-Door experiments finished."
