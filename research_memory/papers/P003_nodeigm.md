# P003 - Direct Causal Inference for OOD Node Classification in IoT

## Basic Info

- Title: Direct Causal Inference for Out-of-Distribution Node Classification in Internet of Things
- Method name: NodeIGM
- Authors: Da Li, Liting Wang, Tao Liu, Zhiyun Lin
- Venue / Year: IEEE Internet of Things Journal 2026
- Source: `/public/wc/lunwen/Direct_Causal_Inference_for_Out-of-Distribution_Node_Classification_in_Internet_of_Things.pdf`
- Tags: graph OOD, node classification, direct causal subgraph inference, edge discriminator, environment mixup, deployment efficiency
- Extraction status: text extracted

## Core Problem

论文批评层级式环境推断方法会有两个问题：

- error accumulation：早期层环境推断错误会传到后续层。
- deployment overhead：推理时需要复杂环境/专家模块，部署成本高。

它提出直接学习 causal subgraph 和 environmental subgraph，避免逐层环境依赖。

## Main Idea

NodeIGM 用 learnable causal edge discriminator 直接给边打 causal importance score，将图划分为 causal subgraph 与 environmental subgraph。然后通过 global environment mixup 生成不同 spurious correlation 强度的环境，要求同一 causal pattern 在不同环境下预测一致。

推理时只保留 GNN encoder + classifier，不需要 edge discriminator 和 mixup loss，因此推理更轻。

## Method Components

- GNN encoder `f_GNN`：产生节点表示。
- Edge discriminator `f_edge`：为边 `(i,j)` 估计 causal importance score。
- Adaptive edge partitioning：按分数划分 causal/environmental edges。
- Hub preservation：保留高度数关键边，避免纯学习式筛边损失重要拓扑。
- Environment mixup：用混合比例 `alpha` 生成多个环境图。
- Consistency loss：不同环境下同一 causal pattern 的预测应一致。
- Dynamic loss weights：动态平衡 supervised loss 和 environment loss。
- 推理：仅使用 encoder + classifier。

## Key Equations / Objectives

总体目标由标准监督损失和环境 mixup 损失组成：

```text
L_total = w_std(t) * L_std + w_env(t) * L_env
```

其中 `L_env` 包含环境间 pairwise consistency / V-REx 风格的风险稳定项。

项目记忆：这和当前 front-door 的 `lambda_var` 有相似直觉，都是约束跨上下文/跨环境预测不要大幅波动。

## Assumptions

- 对节点标签真正稳定有用的是 causal subgraph。
- environmental subgraph 承载会随环境变化的 spurious correlation。
- 边的重要性可以通过节点表示学习出来。
- 高度数 hub 对图连通/语义有特殊价值，需要保留机制。

## Strengths

- 直接从边层面切 causal/environmental，比只从节点表示切 causal/spurious 更图原生。
- 推理阶段轻量，适合 Elliptic/IoT/大图部署。
- 环境 mixup 为“没有真实环境标签”提供了构造环境的路径。

## Weaknesses / Risks

- edge discriminator 学错时，可能直接删除关键因果边。
- hub preservation 的阈值需要调，且高度数不总是因果。
- 与 CaNet 的 pseudo environment/MoE 结合时，要避免两个环境机制互相打架。
- 对稠密图或超大边集，训练阶段边打分成本需要控制。

## Relation To Current CaNet Project

论文内容：

- P003 直接针对 node-level graph OOD，和当前项目任务一致。
- 它强调部署效率，这对当前 frontdoor/CaNet 增加模块后的复杂度有约束价值。

项目推断：

- 当前 `GraphFrontDoor` 里 causal/spurious 是 feature branch；可以加入 edge-level causal subgraph，让 mediator 更符合图因果结构。
- 可用 edge discriminator 生成两套 `edge_index`：
  - causal edge_index 给 causal/mediator branch。
  - environmental edge_index 给 spurious/context branch。

## Implementation Notes

- 新增模块：`CausalEdgeDiscriminator`。
- 修改位置：
  - `model_frontdoor.py` 的 forward 中，在 encoder 前或每层前生成 causal/environment edge sets。
  - `compute_frontdoor_loss` 中加入环境 mixup consistency。
- 新参数：
  - `--use_causal_edges`
  - `--edge_score_temp`
  - `--hub_degree_threshold`
  - `--lambda_env_mixup`
  - `--mixup_alpha`

## Future Ideas

- Edge-FrontDoor: 用 causal subgraph 表示作为 mediator，environmental subgraph 表示作为 front-door context。
- Edge-CaNet: MoE 专家只在 causal subgraph 上预测，spurious branch 在 environmental subgraph 上学习 uniform/环境信息。
- Inference-light FrontDoor: 训练时使用 edge discriminator，推理时缓存/丢弃辅助模块，只保留主分类器。

## One-line Memory

P003 提供了图原生的因果拆分方式：直接学习 causal/environmental edges，再用环境 mixup 约束稳定预测。
