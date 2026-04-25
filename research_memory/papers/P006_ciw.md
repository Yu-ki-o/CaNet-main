# P006 - Causal-Guided Strength Differential Independence Sample Weighting

## Basic Info

- Title: Causal-Guided Strength Differential Independence Sample Weighting for Out-of-Distribution Generalization
- Method name: CIW
- Authors: Haoran Yu, Weifeng Liu, Yingjie Wang, Baodi Liu, Dapeng Tao, Honglong Chen
- Venue / Year: SSRN preprint, 2025
- Source: `/public/wc/lunwen/ssrn-5028078.pdf`
- Tags: OOD generalization, sample weighting, feature decorrelation, causal effect, cross-domain invariant DAG
- Extraction status: text extracted

## Core Problem

Independence sample weighting 通过学习样本权重消除特征间依赖，从而减少 spurious correlation。但已有方法往往粗暴地消除所有特征相关性，可能把 causal features 之间有用的稳定相关也破坏掉。

CIW 的问题意识是：不是所有相关性都该被同等消除。应根据特征对标签的 causal effect 强弱，差异化地做 decorrelation。

## Main Idea

CIW 分两阶段：

1. 学习 cross-domain invariant DAG，计算特征到标签的 causal effect。
2. 根据 causal effect 生成 strength guiding mask，指导 independence sample weighting：对非因果/弱因果特征之间更强去相关，对强因果特征之间保留相关。

直觉：消除伪相关，但保留因果特征之间的协同信息。

## Method Components

- Feature extractor：从输入得到特征表示。
- Cross-domain invariant DAG：
  - 把 features 和 label 视为 potential factors。
  - 通过 DAG reconstruction / score / contrastive prototype losses 学习跨域稳定因果图。
- Causal effect computation：
  - 计算 feature 到 label 的直接/间接/总 causal effect。
- Strength guiding mask：
  - causal effect 越强，decorrelation strength 越弱。
  - causal effect 越弱，decorrelation strength 越强。
- Independence sample weighting：
  - 学习样本权重，使加权后特征依赖按 mask 被差异化压制。
- Weighted classification：
  - 用 causal effects 和 sample weights 引导分类损失。

## Key Equations / Objectives

DAG 目标：

```text
L_DAG = L_G + L_CL-in + L_CL-cr
```

总目标：

```text
min L_cls + L_DAG
```

其中 `L_cls` 是结合 causal effects 和 sample weights 的加权分类损失。

## Assumptions

- 可以从多域数据中学习跨域稳定 DAG。
- feature-to-label causal effect 能指导哪些相关性该保留。
- 样本权重可以改变加权分布下的特征依赖结构。

## Strengths

- 对当前 causal/spurious branch 的 independence loss 有直接启发：不要一刀切去相关。
- 适合解决 front-door 中 `lambda_ind` 太强导致 causal 表示损失的问题。
- 可以作为辅助损失，不必替换主模型。

## Weaknesses / Risks

- 学 DAG 成本较高，论文也提到大规模数据上可能受限。
- 当前 graph node task 中“多域”可能来自 env split，但小图数据域数量有限。
- 如果 causal effect 估计不准，mask 会误导去相关。
- 原方法不是图专用，需要改造成 node representation / environment representation 版本。

## Relation To Current CaNet Project

论文内容：

- P006 提供 differential decorrelation 思路，核心是“基于因果强度决定去相关强度”。

项目推断：

- 当前 `GraphFrontDoor` 的 causal/spurious decorrelation 可以升级：
  - 原来：统一 `lambda_ind`。
  - 新版：按 feature/channel 的 causal score 调整 decorrelation mask。
- 对 `arxiv`/`elliptic` 这类大图，先做轻量版本：不学完整 DAG，只用梯度、attention、环境稳定性估计 causal score。

## Implementation Notes

- 可新增 `CausalStrengthMask`：
  - 输入 hidden representations 和 labels/envs。
  - 输出 channel-wise 或 feature-wise mask。
- 在 independence loss 中加入 mask：
  - 对低 causal score 的维度加强去相关。
  - 对高 causal score 的维度降低去相关。
- 初版避免完整 DAG，使用 moving-average feature-label association 或 gradient attribution。

## Future Ideas

- Masked-Independence FrontDoor: 用 causal strength mask 替代统一 decorrelation。
- Env-Stability CIW: feature 在不同环境中 label association 越稳定，causal score 越高。
- CIW for Spurious Branch: 只强力压制 spurious branch 与 causal branch 的弱因果通道相关性。

## One-line Memory

P006 的关键启发是：去相关不应一刀切，应按 feature-to-label causal strength 差异化消除 spurious dependence。
