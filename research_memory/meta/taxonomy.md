# Taxonomy

This taxonomy is project-oriented, not a universal survey taxonomy.

## Task Axis

- Node-level OOD generalization: CaNet, MLEI, NodeIGM.
- General causal effect estimation: DAG-aware Transformer, useful as causal machinery rather than direct GNN OOD model.
- Non-graph OOD transfer: causal prompt tuning, useful for front-door mediator and diversity augmentation design.
- General OOD feature decorrelation: CIW, useful as an auxiliary regularizer or sample weighting module.

## Method Axis

- Environment inference: latent or pseudo environment estimation for invariant/expert learning.
- Mixture-of-experts: environment-conditioned predictors or adaptive expert ensembles.
- Causal intervention: front-door/back-door/do-calculus inspired objectives.
- Causal subgraph extraction: separate causal and environmental graph parts.
- Sample reweighting/decorrelation: rebalance samples/features to reduce spurious dependence.

## Current Project Focus

The current repo already contains CaNet-style environment inference, GCN/GAT backbones, and front-door variants. The highest-value memory threads are:

- Make environment inference richer without over-smoothing.
- Make front-door mediator/context design more graph-native and less prompt-specific.
- Reduce spurious feature/environment leakage by direct subgraph extraction or differential decorrelation.
- Keep inference-time cost acceptable for large node datasets such as Arxiv and Elliptic.
