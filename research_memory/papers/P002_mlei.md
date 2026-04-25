# P002 - Expand Horizon: Graph OOD Generalization via Multi-Level Environment Inference

## Basic Info

- Title: Expand Horizon: Graph Out-of-Distribution Generalization via Multi-Level Environment Inference
- Authors: Jiaqiang Zhang, Songcan Chen
- Venue / Year: AAAI 2025
- Source: `/public/wc/lunwen/33444-Article Text-37512-1-2-20250410.pdf`
- Tags: graph OOD, node classification, multi-level environment inference, global context, multi-hop ego-graph, transformer
- Extraction status: text extracted

## Core Problem

论文指出，CaNet/EERM/LiSA 等方法主要依赖低阶 ego-graph 推断环境，忽略了两个层面的信息：

- global information：节点所在的大范围图上下文。
- multi-granularity local information：不同 hop 的局部环境粒度。

直接堆深 GNN 捕获高阶信息会带来 over-smoothing，因此需要另一种方式扩大环境推断视野。

## Main Idea

MLEI 使用 multi-level environment inference：

- 用 linear graph transformer 建模 global relation，并推断 global environment。
- 用 multi-hop ego-graphs 做 layer-by-layer local environment inference。
- 将 global environment 作为概览信息辅助 local inference。
- 融合 global 和 local 两路预测，用因果分析导出的目标学习环境不敏感模式。

直觉：节点的环境不是单一低阶邻域，而是 macro/global + meso/micro local 的组合。

## Method Components

- Global branch：linear graph transformer，避免 vanilla transformer 的二次复杂度。
- Global environment inference：基于全局节点表示采样 pseudo environments。
- Local branch：多层/多 hop ego-graph 环境推断。
- Fusion：将 global environment 作为 general outline 注入 local view。
- Loss：`Loss = Loss_local + lambda * Loss_global`，两路都包含监督项和环境正则项。

## Key Equations / Objectives

论文沿用 CaNet 的干预/变分思想：

```text
E_q(E|G)[log p_theta(Y | G, E)] - KL(q_phi(E|G) || p0(E))
```

但 `q_phi(E|G)` 不再只来自低阶 ego-graph，而是拆成 global view 与 local multi-hop view。

最终目标：

```text
Loss = Loss_local + lambda * Loss_global
```

## Assumptions

- 环境是 graph OOD 的根因，但环境信息分布在多级结构里。
- 高阶环境不能简单靠堆深 GNN 获取，因为 over-smoothing 会损害节点表示。
- global environment 可以作为 local inference 的先验或概览。

## Strengths

- 非常适合补当前 CaNet/frontdoor 的短板：环境估计视野不够。
- 对 Arxiv、Elliptic 这种时间/快照/大图数据尤其有启发。
- 线性 transformer 提供了全局信息路径，计算上比普通 transformer 更可控。

## Weaknesses / Risks

- 模块复杂度明显高于原 CaNet。
- global transformer 对超大图的显存/时间仍需谨慎。
- global-local 融合如果设计不好，可能让环境信息反而泄漏到 causal branch。
- `lambda`, `K`, `tau`, global hidden size 等超参会增加搜索成本。

## Relation To Current CaNet Project

论文内容：

- P002 明确指出 CaNet 主要依赖 low-hop ego-graph，是短视环境建模。
- 它提出的 global + multi-hop local 可直接作为当前 `env_type=graph/node` 的升级方向。

项目推断：

- 当前 front-door 模型中的 mediator/spurious/context 可以从单一 encoder 表示升级为 multi-level 表示。
- 在 `model_frontdoor.py` 中，可以先做轻量版本：新增 global context encoder，仅输出环境 prototype/context，不替换主 GNN。

## Implementation Notes

- 可新增 `GlobalEnvironmentEncoder`：
  - 输入 `x, edge_index`
  - 输出 `global_env_logits` 或 `global_context`
  - 初版可用 SGC/APPNP/linear attention 替代完整 transformer。
- 在 `GraphFrontDoor.get_frontdoor_contexts()` 中融合 global context。
- 在 `grid_search_frontdoor_configs.py` 加参数：
  - `--use_global_env`
  - `--lambda_global_env`
  - `--global_context_dim`
  - `--multi_hop_env`

## Future Ideas

- FrontDoor-MLEI: front-door contexts 由 global environment prototypes 和 local spurious prototypes 共同构成。
- MLEI-DAG: global environment 只进入 spurious/context branch，不直接进入 causal logits，减少 shortcut。
- Arxiv/Elliptic 优先实验：这两个数据集更可能受益于全局/时间环境建模。

## One-line Memory

P002 是对 CaNet 环境推断的直接升级：用 global transformer + multi-hop local inference 扩大环境视野，避免只看低阶 ego-graph。
