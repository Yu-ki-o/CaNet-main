# P007 - Discovering Dynamic Causal Space for DAG Structure Learning

## Basic Info

- Title: Discovering Dynamic Causal Space for DAG Structure Learning
- Authors: Fangfu Liu, Wenchang Ma, An Zhang, Xiang Wang, Yueqi Duan, Tat-Seng Chua
- Venue / Year: KDD 2023
- Source: `/public/wc/lunwen/Discovering Dynamic Causal Space for DAG Structure Learning.pdf`
- Tags: causal discovery, DAG structure learning, differentiable causal discovery, score function, DAG-ness-aware scoring, CASPER
- Extraction status: text extracted by `pdftotext`

## Core Problem

论文关注从纯 observational data 中学习 DAG causal structure。以 NOTEARS 为代表的 differentiable score-based DAG learners 把离散 DAG 约束变成连续 acyclicity constraint，从而能用梯度优化学习邻接矩阵。

论文指出现有方法的 score function 多数只衡量 data fitness，例如 least square loss、maximum likelihood、ELBO 等。这类 score function 通常与 DAG-ness 无关：训练过程中候选图还不是 DAG，但 score 仍然用静态拟合误差评价，容易导致 suboptimal DAG、local minimum 和噪声脆弱性。

## Main Idea

CASPER 提出 dynamic causal space，把 graph structure / DAG-ness 信息注入 score function。核心直觉是：候选图越不符合 DAG，评分空间就应该使用更复杂或更敏感的 measure；候选图逐渐接近 DAG 时，评分空间也动态调整，从而让 score 同时反映 data fitness 和 graph structure。

论文把这种结构感知的评分称为 causal structure distance。它通过 causal space mapper `T_phi` 将观测数据与重构数据映射到 causal space，再用受 `g(h(G))` 约束的 Lipschitz 函数族度量两者分布差异。其中 `h(G)` 是 DAG-ness function，`g(h(G))` 是 structure-aware descriptor。

## Method Components

- 输入：观测数据矩阵 `X`。
- DAG-fitting model `f_theta`：根据当前候选图 `G` 重构或生成 `X_hat`。
- Causal space mapper `T_phi`：把 `X` 和 `X_hat` 映射到 causal space。
- Structure-aware descriptor：用 `g(h(G))` 控制 `T_phi` 的 Lipschitz 范围，使 score 随 DAG-ness 动态变化。
- Score function：用 causal space 中的分布距离替代固定欧氏重构误差。
- DAG constraint：仍保留 NOTEARS 风格的 continuous acyclicity penalty。
- 优化：bilevel optimization。外层固定 `phi`，优化 `G` 和 `theta`；内层固定 `G/theta`，最大化 score 以学习 causal-space distance，并对 `phi` 做 clipping。

## Key Equations / Objectives

基础 SEM：

```text
X_j := f_j(X_pa(X_j)) + N_j
```

NOTEARS-style DAG-ness：

```text
h(G) = tr(exp(W o W)) - d = 0
```

CASPER 的 causal-space score：

```text
F_phi(X; G, theta)
  = E_{X ~ P_r}[T_phi(X)] - E_{X_hat ~ P_theta}[T_phi(X_hat)]
    + lambda * R_sparse(G)
```

整体优化：

```text
min_{G, theta} F_phi*(X; G, theta) + L_DAG(G)
s.t. phi* in argmax_phi F_phi(X; G, theta), phi in C(G)
```

其中 `C(G)` 约束 `T_phi` 的 Lipschitz norm 不超过 `g(h(G))`，实现 DAG-ness-aware scoring。

## Assumptions

- 结构学习目标可用 SEM 描述，且在附加假设下可识别。
- 观测数据中的 causal graph 可以通过 differentiable score-based framework 近似恢复。
- Score function 如果只度量 data fitness，会忽略训练中候选图非 DAG 的事实。
- DAG-ness-aware score 能更好地引导梯度优化，减少错误边和 local minimum。

## Strengths

- 对 differentiable DAG learning 的核心痛点有直接修正：score function 不再与 DAG-ness 脱钩。
- 可以作为 NOTEARS / DAG-GNN / GraN-DAG 等方法的增强框架，而不是完全替代。
- 在 synthetic linear/nonlinear 和 Sachs real heterogeneous dataset 上表现出更好的 SHD、SID、FDR、TPR。
- 对噪声和图密度变化更稳健，尤其适合候选图在训练早期高度 cyclic 的场景。

## Weaknesses / Risks

- 方法仍建立在 differentiable score-based causal discovery 框架上，不直接解决所有结构学习范式。
- Bilevel optimization 增加实现复杂度，需要额外的 inner loop、clipping 和稳定性控制。
- 原论文处理变量级 causal discovery，不是 graph OOD node classification；迁移到当前项目时只能借鉴 score/regularization 思想。
- 真实 graph OOD 中 latent feature DAG 的 ground truth 不可见，CASPER-style score 是否能稳定提升需要实验验证。

## Relation To Current CaNet Project

论文内容：

- P007 与 P005 都关注 DAG，但角度不同。P005 讨论如何把已知或构造的 DAG 注入 attention；P007 讨论如何让 DAG structure learning 的 score function 感知 DAG-ness。
- P007 对当前 `model_frontdoor_dag.py` 的 `A_feat` 很相关。当前模型已有 feature-only DAG、acyclicity penalty 和 DAG-derived mediator mask，但任务 loss / reconstruction-style score 仍可能主要由 label/data fitness 驱动。

项目推断：

- 当前 `GraphFrontDoorDAG` 可以加入轻量 CASPER-style regularizer：让 `h(A_feat)` 动态调节 DAG score、mediator score 或 independence penalty，而不是只把 DAG 当作额外 penalty。
- 如果未来引入 learnable DAG mask 或 feature DAG discovery，P007 提醒需要避免“只靠 classification loss + acyclicity”学习 DAG，否则可能得到能分类但结构不可信的图。
- P007 更适合增强 `model_frontdoor_dag.py`，不应优先改基础 `model_frontdoor.py`。

## Implementation Notes

- 关键文件：`model_frontdoor_dag.py`, `main_frontdoor_dag.py`。
- 可新增轻量模块：
  - `CausalSpaceScorer`：输入原始 hidden `z`、DAG-fitted reconstruction `z_hat` 和 `h(A_feat)`，输出 causal-space discrepancy。
  - `lambda_casper` / `casper_inner_steps` / `casper_clip_scale` 参数。
- 初版不要完整复现 bilevel CASPER，可先做：
  - 根据 `h(A_feat)` 动态缩放 DAG loss 或 mediator mask temperature。
  - 用一个小 MLP `T_phi` 估计 `z` 与 `z_hat` 的 causal-space distance。
  - 只在 stage2 小范围搜索 `lambda_casper`，确认不破坏 OOD accuracy。
- 如果做完整版本，需要额外 inner loop 更新 `T_phi`，训练复杂度会显著提高。

## Future Ideas

- CASPER-FeatureDAG: 用 DAG-ness-aware causal-space score 约束 `A_feat`，减少 feature DAG 的 cyclic/local-minimum 问题。
- DAG-ness-Aware Mediator Mask: 用 `h(A_feat)` 动态调节 mediator selection temperature，让训练早期更宽松、后期更结构化。
- CASPER-CIW: 把 P007 的 DAG-ness-aware score 与 P006 的 causal-strength masked independence 结合，避免 uniform independence 与结构学习互相冲突。
- CASPER-DAGMixer: 在 P005 latent mixer 前，先用 P007 score 学到更可信的 feature DAG 或 latent-token DAG mask。

## One-line Memory

P007 的关键启发是：学习 DAG 时不能只把 acyclicity 当惩罚项，score function 本身也应动态感知 DAG-ness；这可用于增强当前 feature DAG / mediator mask 的结构可信度。
