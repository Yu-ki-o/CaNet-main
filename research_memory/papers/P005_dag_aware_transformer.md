# P005 - DAG-aware Transformer for Causal Effect Estimation

## Basic Info

- Title: DAG-aware Transformer for Causal Effect Estimation
- Authors: Manqing Liu, David R. Bellamy, Andrew L. Beam
- Venue / Year: arXiv 2024
- Source: `/public/wc/lunwen/DAG-aware Transformer for Causal Effect Estimation.pdf`
- Tags: causal effect estimation, DAG-aware attention, ATE, CATE, IPTW, AIPW, proximal inference
- Extraction status: text extracted

## Core Problem

论文解决一般 causal effect estimation 中的模型错设和复杂因果结构问题。传统方法需要分别估计 propensity score、outcome regression 或 bridge function，且难以把已知 DAG 结构自然注入深度模型。

虽然它不是 graph OOD 论文，但对当前项目有两个价值：

- 如何把因果图/DAG 结构显式编码到 attention。
- 如何把 outcome model、propensity model、doubly robust estimation 等思想模块化。

## Main Idea

论文提出 DAG-aware Transformer：把 causal DAG 转成 attention mask，让 transformer 的注意力只能沿因果路径/允许的自注意力流动。

模型可以估计：

- propensity score `P(A|X)`
- outcome regression `P(Y|A,X)`
- bridge function `h(A,W,X)` for proximal inference

再接入 G-formula、IPTW、AIPW 或 proximal inference。

## Method Components

- 输入变量节点：treatment `A`、confounders `X`、outcome `Y`、可选 unmeasured confounding/proxy `U/W/Z`。
- DAG-aware attention mask：按因果图限制 attention。
- Transformer encoder：学习变量间结构化依赖。
- Raw input residual/concatenation：保留原始信息，避免 mask 后信息损失。
- Estimator heads：
  - outcome head
  - propensity head
  - bridge function head
- Training:
  - G-formula: outcome MSE
  - IPTW: treatment BCE
  - AIPW: outcome + propensity joint or separate training
  - proximal: kernel/NMMR objective

## Key Equations / Objectives

ATE:

```text
tau = E[Y^1 - Y^0]
```

CATE:

```text
tau(x) = E[Y^1 - Y^0 | X = x]
```

G-formula:

```text
ATE = E_X[mu(1, X) - mu(0, X)]
```

IPTW:

```text
pi(X) = P(A = 1 | X)
```

AIPW combines outcome model and propensity model, giving robustness when one nuisance model is correct.

## Assumptions

- 对标准 ATE/CATE：一致性、可交换性/无未测混杂、positivity。
- 对 proximal inference：需要合适 proxy 变量和 bridge function 假设。
- DAG 或至少变量间允许关系需要已知或可构造。

## Strengths

- 提供“把因果结构注入 attention”的具体机制。
- 可以启发当前 repo 的 `model_frontdoor_dag.py`：让 causal/spurious/mediator/context 的信息流更受控。
- AIPW 的思想可用于减少 mediator/context 估计错误带来的偏差。

## Weaknesses / Risks

- 当前 graph OOD 中真实 DAG 很难确定。
- 论文主要处理表格/因果变量，不直接处理图拓扑。
- 如果 DAG mask 错误，attention 会被错误约束。

## Relation To Current CaNet Project

论文内容：

- P005 不直接给 graph OOD 模型，但给了 DAG-aware attention 和 doubly robust estimator 的结构。

项目推断：

- 可以把当前 front-door DAG 模型里的 `causal -> mediator -> label`, `spurious/context -> label` 关系显式 mask 化。
- 可以让 `causal`, `spurious`, `env`, `mediator`, `logits` 这些 latent tokens 进入一个小型 DAG-aware transformer，而不是简单 concat/gate。

## Implementation Notes

- 可新增 `DAGAwareLatentMixer`：
  - tokens: causal z, spurious z, mediator/context z, env z, label query
  - mask: 只允许符合设定因果路径的信息流
- 优先小规模使用，不要直接对所有节点做 full transformer。
- 和 P002 的 global transformer 区分：
  - P002 处理图上节点的 global relation。
  - P005 处理 latent causal variables 的结构化 relation。

## Future Ideas

- FrontDoor-DAGMixer: 在 front-door logits 前加入 DAG-aware latent token mixer。
- DoublyRobust-FrontDoor: 同时学习 mediator outcome head 和 context propensity/gating head，借鉴 AIPW 降低偏差。
- Learnable DAG Mask: 初期用手工 DAG mask，后续用稀疏可学习 mask 并加 acyclicity 约束。

## One-line Memory

P005 的价值是方法工具箱：用 DAG-aware attention 控制因果变量间信息流，并借鉴 doubly robust causal estimation。
