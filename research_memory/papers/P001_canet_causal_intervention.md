# P001 - Graph Out-of-Distribution Generalization via Causal Intervention

## Basic Info

- Title: Graph Out-of-Distribution Generalization via Causal Intervention
- Authors: Qitian Wu, Fan Nie, Chenxiao Yang, Tianyi Bao, Junchi Yan
- Venue / Year: WWW 2024
- Source: `/public/wc/lunwen/Graph Out-of-Distribution Generalization via Causal Intervention1.pdf`
- Tags: graph OOD, node classification, causal intervention, latent environment, mixture-of-experts, CaNet
- Extraction status: text extracted

## Core Problem

论文关注节点级 graph OOD generalization。核心问题是：训练节点和测试节点处在不同环境分布下，普通 GNN 会利用环境敏感的伪相关，例如某种邻域属性只在训练环境中和标签相关，导致 OOD 节点预测退化。

论文把环境 `E` 视为未观测混杂变量。`E` 同时影响 ego-graph `G` 和标签 `Y`，普通最大似然训练会让预测依赖训练环境分布，从而捕获不稳定相关。

## Main Idea

CaNet 用因果干预视角近似学习 `P(Y | do(G))`。由于真实环境标签通常不可得，模型学习一个 environment estimator `q_phi(E|G)` 来推断 pseudo environments，再用 mixture-of-expert GNN predictor `p_theta(Y|G,E)` 按环境条件建模。

直觉：如果预测器能在多个伪环境条件下仍捕获稳定预测关系，就能减少训练环境混杂带来的偏差。

## Method Components

- 输入：节点特征、图结构、训练节点标签。
- 环境估计器：从节点 ego-graph 表示推断 pseudo environment 分布。
- 预测器：MoE GNN，每个环境对应一组传播/变换专家。
- GCN/GAT backbone：GCN 使用邻居聚合后与自身表示拼接；GAT 为每个环境维护注意力参数。
- 训练目标：监督分类项 + pseudo environment 正则项，目标来自 causal intervention 的变分下界。
- 推理：使用学到的环境估计和专家组合进行节点分类。

## Key Equations / Objectives

论文目标是近似 `log p_theta(Y_hat | do(G))`。由于不能直接枚举真实环境，用变分形式：

```text
E_{q_phi(E|G)}[log p_theta(Y_hat | G, E)] - KL(q_phi(E|G) || p0(E))
```

其中 `p0(E)` 是环境先验，通常鼓励 pseudo environments 不要完全贴合输入图而过拟合。

项目记忆：当前 repo 的 `model_canet.py` 与 `model_frontdoor.py` 都继承了这个思想：环境/上下文被作为对 GNN 表示或预测的调制变量。

## Assumptions

- 环境是节点 OOD shift 的根因之一。
- 环境不可观测，但可从 ego-graph 表示中推断出有用的 pseudo environments。
- 稳定预测关系在不同环境下更可泛化。
- 伪环境不必对应真实物理域，但需要在训练上有正则化价值。

## Strengths

- 非常贴合当前项目，是本 repo 的理论和代码根基。
- 不需要真实环境标签，适合 Cora/Citeseer/Pubmed/Arxiv/Twitch/Elliptic 这类环境标签不完整或含义不同的数据。
- MoE 结构和环境估计器给模型改造留下清晰接口。

## Weaknesses / Risks

- 环境估计多依赖低阶 ego-graph，可能“视野短”，这正是 P002 MLEI 批评和改进的点。
- `K` 过大可能引入冗余环境和过拟合。
- 环境估计和预测耦合，错误环境会影响专家学习。
- 对大图训练成本和环境采样稳定性敏感。

## Relation To Current CaNet Project

论文内容：

- 当前代码中的 `GraphOOD`, `CaNetConv`, `GraphFrontDoor` 都可以看成 CaNet 思路的延伸。
- `--K`, `--tau`, `--env_type`, `--variant`, `--backbone_type` 都与这篇论文的设计空间相关。

项目推断：

- 如果继续做 front-door 模型，P001 是主干；不要完全替换 CaNet，而应把 P002/P003/P004/P006 的模块作为对环境、因果子图、前门上下文和去相关损失的增强。
- `arxiv` 和 `elliptic` 上的 `--variant` 可以理解为论文实验配置中的图传播变体，属于 backbone 层面的局部选择。

## Implementation Notes

- 关键文件：`model_canet.py`, `model_frontdoor.py`, `model_frontdoor_dag.py`, `parse.py`, `main_frontdoor.py`。
- 可扩展点：
  - 将单层/低阶环境估计扩展为 P002 的 global + multi-hop local。
  - 将 pseudo environment 从节点表示推断改成 P003 的 causal/environmental subgraph partition。
  - 在 front-door branch 加入 P004 的 diversity augmentation 思路。
  - 加入 P006 的 causal-guided feature decorrelation 作为辅助正则。

## Future Ideas

- CaNet-MLEI: 在 CaNet 环境估计器前加入线性 graph transformer 的 global environment。
- CaNet-EdgeFD: 用 causal edge discriminator 生成 causal subgraph，把它作为 mediator。
- CaNet-CIW: 对隐藏表示做 causal-guided differential decorrelation，减少 spurious feature leakage。
- FrontDoor-CaNet: 把 CaNet pseudo environment prototypes 当作 front-door contexts。

## One-line Memory

P001 是当前项目的根论文：把 graph OOD 解释为 latent environment confounding，并用 pseudo environment + MoE GNN 近似 causal intervention。
