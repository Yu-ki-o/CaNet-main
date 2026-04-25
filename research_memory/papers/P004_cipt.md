# P004 - Causal Interventional Prompt Tuning for Few-Shot OOD Generalization

## Basic Info

- Title: Causal Interventional Prompt Tuning for Few-Shot Out-of-Distribution Generalization
- Method name: CIPT
- Authors: Jie Wen, Yicheng Liu, Chao Huang, Chengliang Liu, Yong Xu, Xiaochun Cao
- Venue / Year: IEEE TPAMI 2026
- Source: `/public/wc/lunwen/Causal_Interventional_Prompt_Tuning_for_Few-Shot_Out-of-Distribution_Generalization.pdf`
- Tags: front-door adjustment, mediator, causal/non-causal decomposition, diversity augmentation, OOD
- Extraction status: text extracted

## Core Problem

论文研究的是 VLM few-shot OOD，不是图学习。但它和当前模型高度相关，因为它提供了一个可实现的 front-door adjustment 框架：

- 把输入表示拆成 causal representation 和 non-causal representation。
- 将 causal representation 作为 mediator。
- 用 diversity augmentation 丰富前门调整中需要平均的上下文。

## Main Idea

CIPT 构造 SCM：输入 `X` 到标签 `Y`，潜在混杂 `U` 同时影响 `X` 和 `Y`。方法引入 mediator `E` 表示纯 causal feature，通过 `X -> E -> Y` 估计 causal effect。

在 few-shot 中缺少多样数据，论文用 text-based diversity augmentation 模拟不同非因果上下文：固定 causal feature，改变上下文 prompt，让模型学会对非因果变化不敏感。

## Method Components

- Causal Decomposition:
  - 两个轻量 adapter 将图像特征分为 causal feature `e` 和 spurious/non-causal feature `s`。
  - causal feature 用于正确分类。
  - spurious feature 被约束不要承载标签因果信息。
- Front-door adjustment:
  - 将 `e` 作为 mediator。
  - 对多样上下文进行平均/融合，近似干预。
- Text-based Diversity Augmentation:
  - 用多种 prompt/template 构造非因果上下文。
  - 固定 causal feature，暴露给不同 context。
- Loss:
  - classification loss
  - decomposition loss
  - independence loss
  - augmented causal prediction loss

## Key Equations / Objectives

核心因果目标是从 `P(Y|X)` 转向 `P(Y|do(X))`，用 front-door adjustment 估计真实因果效应。

总损失形式：

```text
L = L_c + beta * L_de + gamma * L_ind
```

其中 `L_c` 包括 diversity augmented causal representation 的分类损失。

## Assumptions

- 存在可学习的 causal representation，可以作为 mediator。
- non-causal context 可以通过模板/文本语义近似生成。
- 固定 mediator、改变 context 的训练能提升 OOD 泛化。

## Strengths

- 给当前 `main_frontdoor.py` / `model_frontdoor.py` 的设计提供了直接蓝本。
- `causal/spurious adapter + diversity augmentation + front-door averaging` 结构清楚。
- 强调 mediator 和 spurious representation 的独立性，很适合加到图表示上。

## Weaknesses / Risks

- 原论文基于 CLIP/text prompt，图数据没有天然文本模板。
- 图上的 diversity augmentation 需要设计：环境 prototype、spurious prototype、edge perturbation、feature mask 都可能替代 text prompt。
- 如果 mediator 不是真因果表示，front-door 估计会变成普通数据增强。

## Relation To Current CaNet Project

论文内容：

- 当前 `GraphFrontDoor` 的 causal/spurious branch、context sampling 和 variance penalty 很像 CIPT 的图版本。

项目推断：

- 图模型中可以把 text templates 替换成：
  - pseudo environment prototypes from CaNet
  - global/local environment contexts from MLEI
  - environmental subgraph embeddings from NodeIGM
  - spurious feature prototypes across environments

## Implementation Notes

- 当前已有相关模块：
  - `lambda_ind`
  - `lambda_med`
  - `lambda_spu`
  - `lambda_fd`
  - `lambda_var`
  - `context_gate_temp`
- 可加强方向：
  - context bank 从简单原型升级为 multi-source context bank。
  - diversity augmentation 不只采样 context，还可对 environmental edges/features 做 mixup。
  - 加强 causal/spurious independence loss，结合 P006 CIW。

## Future Ideas

- Graph-CIPT: 用图环境原型替换 text prompt，做 graph front-door context augmentation。
- CIPT + MLEI: global/local environment contexts 作为多样上下文。
- CIPT + NodeIGM: environmental subgraph embeddings 作为非因果上下文。
- CIPT + CIW: 用 causal effect mask 指导 causal/spurious branch 的去相关。

## One-line Memory

P004 是当前 front-door 图模型的跨领域方法模板：mediator + causal/spurious decomposition + diversity augmentation。
