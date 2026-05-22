# Taxonomy

This taxonomy is project-oriented, not a universal survey taxonomy.

## Task Axis

- Node-level OOD generalization: CaNet, MLEI, NodeIGM, CGRL.
- General causal effect estimation: DAG-aware Transformer, useful as causal machinery rather than direct GNN OOD model.
- General causal discovery / DAG structure learning: CASPER, useful for learning or regularizing latent feature DAGs rather than direct GNN OOD prediction.
- Topology-shift graph OOD: AdvDIFFormer, useful for reducing adjacency-propagation sensitivity under structural distribution shifts.
- Non-graph OOD transfer: causal prompt tuning, useful for front-door mediator and diversity augmentation design.
- General OOD feature decorrelation: CIW, useful as an auxiliary regularizer or sample weighting module.

## Method Axis

- Environment inference: latent or pseudo environment estimation for invariant/expert learning.
- Mixture-of-experts: environment-conditioned predictors or adaptive expert ensembles.
- Causal intervention: front-door/back-door/do-calculus inspired objectives.
- Causal representation stability: stabilize representation-label mutual information with graph-native regularization.
- DAG-ness-aware scoring: make differentiable structure learning scores aware of acyclicity violations.
- Advective diffusion / topology-reliance control: mix stable non-local interactions with weighted observed-topology propagation.
- Causal subgraph extraction: separate causal and environmental graph parts.
- Energy-based graph reconstruction: reconstruct graph posterior from causal representations to avoid label-spurious overfitting.
- Sample reweighting/decorrelation: rebalance samples/features to reduce spurious dependence.

## Current Project Focus

The current repo already contains CaNet-style environment inference, GCN/GAT backbones, and front-door variants. The highest-value memory threads are:

- Make environment inference richer without over-smoothing.
- Make front-door mediator/context design more graph-native and less prompt-specific.
- Make feature-DAG and latent-DAG learning structurally meaningful, not just task-fitting.
- Reduce sensitivity to topology shift in the base graph encoder before causal/front-door decomposition.
- Stabilize the causal mediator with EBR, class geometry, and MI diagnostics before adding more context mechanisms.
- Reduce spurious feature/environment leakage by direct subgraph extraction or differential decorrelation.
- Keep inference-time cost acceptable for large node datasets such as Arxiv and Elliptic.
