# Project Context

## Current Repo Understanding

The repo is based on CaNet-style graph OOD learning. It includes:

- CaNet-like environment estimator and mixture-of-experts GNN predictor.
- GCN/GAT backbones.
- Front-door variants in `main_frontdoor.py`, `model_frontdoor.py`, `main_frontdoor_dag.py`, and `model_frontdoor_dag.py`.
- Grid search scripts for stage1/stage2 front-door tuning.

## Current Model Theme

The active research direction appears to be:

```text
CaNet environment inference + front-door causal decomposition + graph OOD node classification
```

The strongest related papers among the current six are:

- Foundation: `P001`
- Environment upgrade: `P002`
- Graph-native causal split: `P003`
- Front-door design: `P004`
- DAG structure control: `P005`
- Better independence regularization: `P006`

## Current Implementation Bias

The current front-door model likely separates causal/spurious information mostly at representation level. This is useful, but may miss graph-native causal structure. `P003` suggests edge/subgraph-level separation as a next step.

The current environment/context source is likely local or prototype-based. `P002` suggests adding global and multi-hop local contexts.

The current independence penalty is likely uniform. `P006` suggests making it channel/feature-dependent based on causal strength.

## Dataset Notes

- Cora/Citeseer/Pubmed: small citation graphs; good for fast ablation and debugging.
- Arxiv: large temporal shift; likely benefits from global/multi-level environment inference.
- Twitch: geographic/domain shift; environment signals may be strong.
- Elliptic: temporal transaction snapshots; likely benefits from causal edge/subgraph and inference-light design.

## Near-Term Research Constraint

Add one mechanism at a time. The current model already has many losses and hyperparameters. For credible experiments:

1. Establish baseline front-door result.
2. Add one paper-inspired mechanism.
3. Run stage1 for core parameters if backbone/encoder changes.
4. Run stage2 for front-door objective weights.
5. Compare ID, OOD1/OOD2/OOD3, and training stability.
