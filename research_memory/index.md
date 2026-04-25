# Research Memory Index

This is the first project-level memory set for the current CaNet/front-door graph OOD work. It currently includes only the six user-selected papers most related to the current model.

## Paper Map

- `P001` CaNet: latent environment confounding + pseudo environment estimator + MoE GNN.
- `P002` MLEI: expands CaNet-like environment inference from low-hop ego-graphs to global + multi-hop local environments.
- `P003` NodeIGM: directly splits causal/environmental subgraphs with a causal edge discriminator and environment mixup.
- `P004` CIPT: front-door adjustment template with causal/spurious decomposition and diversity augmentation.
- `P005` DAG-aware Transformer: injects causal DAG structure into attention and connects to ATE/CATE/IPTW/AIPW.
- `P006` CIW: causal-effect-guided differential feature decorrelation and sample weighting.

## Current Knowledge Map

### Environment As Confounder

`P001` defines the central graph OOD story: environment `E` confounds ego-graph `G` and label `Y`. The current repo mostly follows this view.

`P002` updates the view: environment is not only low-hop local context; it has global and multi-hop granularity.

### Causal / Spurious Separation

`P004` separates causal and non-causal representations in feature space.

`P003` separates causal and environmental information in edge/subgraph space.

Project synthesis: a stronger graph front-door model should probably separate both features and edges.

### Front-Door Modeling

`P004` is the cleanest reference for front-door implementation logic: mediator, context augmentation, independence, and causal prediction averaging.

`P001` provides graph-specific pseudo environments that can become the context bank for `P004`-style front-door adjustment.

### Structured Causal Mixing

`P005` suggests replacing unconstrained concat/gating among latent variables with DAG-aware attention masks.

Project synthesis: current `model_frontdoor_dag.py` can be evolved into a small latent-token DAG mixer.

### Decorrelation

`P006` warns that uniform decorrelation can destroy useful causal feature correlations.

Project synthesis: the current `lambda_ind` loss should eventually become causal-strength-masked decorrelation.

## High-Priority Model Directions

1. Multi-Level FrontDoor: use `P002` global/local environment contexts inside the current front-door context bank.
2. Edge-Causal FrontDoor: use `P003` causal edge discriminator to produce mediator graph and environmental context graph.
3. DAG-Masked FrontDoor: use `P005` DAG-aware latent attention to control causal/spurious/context information flow.
4. CIW-Independence: replace uniform independence with `P006` causal-strength-guided decorrelation.

## How To Use This Memory

When starting model research:

1. Read `project_context.md`.
2. Read the paper card for the relevant mechanism.
3. Read matching method cards in `methods/`.
4. Check `ideas/model_variants.md` for implementable variants.
5. Use `experiments/stage1_stage2_protocol.md` when converting ideas into grid search.
