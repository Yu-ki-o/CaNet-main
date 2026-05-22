# P009 - CGRL: Causal-Guided Representation Learning for Graph Out-of-Distribution Generalization

## Basic Info

- Title: CGRL: Causal-Guided Representation Learning for Graph Out-of-Distribution Generalization
- Authors: Bowen Lu, Lianqiang Yang, Teng Li
- Venue / Year: arXiv 2026, preliminary work
- Source: `/public/wc/lunwen/CGRL Causal-Guided Representation Learning.pdf`
- arXiv: `2603.24304v1`
- Tags: graph OOD, node classification, causal representation learning, backdoor adjustment, energy-based reconstruction, mutual information stability
- Extraction status: text extracted

## Core Problem

CGRL studies why GNNs perform poorly under graph OOD shifts. The paper argues that common GNNs fit spurious correlations in ID data, so the mutual information between prediction representation `Hc` and label `Y` becomes unstable under shifted test data.

For this project, the useful point is not only accuracy improvement. CGRL provides a way to diagnose representation stability through `MI(Hc, Y)` and a graph reconstruction regularizer that can be attached to a causal mediator.

## Causal View

The paper builds a structural causal model with:

- environment noise `E`
- input graph `G`
- adjacency `A`
- node features/representations `X`
- prediction representation `Hc`
- intra-class representation distribution `Hc_intra`
- inter-class representation distribution `Hc_inter`
- label `Y`

Main non-causal paths:

- `Hc <- A <- G <- E -> Y`
- `Hc <- X <- G <- E -> Y`
- `Hc_inter <- Hc -> Hc_intra -> Y`

CGRL uses backdoor adjustment and derives:

```text
P_theta(Y | do(Hc)) = E_{P_theta(Gs)} P_theta(Y | Hc, Gs)
```

The paper then derives a lower bound that contains supervised prediction, graph posterior reconstruction, intra-class clustering, and inter-class separation terms.

## Method Components

### Re-Weight Representation Learning

Instead of directly using GNN output as `Hc`, CGRL learns node-wise reweighting. For GCN, it uses `K` branches over normalized neighbor aggregation. For GAT, it injects the reweight matrix into attention aggregation.

Project interpretation:

- The branch count `K` is a light multi-view causal representation mechanism.
- It can be treated as a representation-side alternative to our environment/context sampling.
- It may be useful when pseudo environments are noisy or unavailable.

### Energy-Based Reconstruction

CGRL reconstructs the graph posterior from `Hc` with an edge energy:

```text
E(u, v) = -Hc_v^T W_uv Hc_u
Q_phi(u, v) = exp(-E(u, v)) / sum_{u in N(v)} exp(-E(u, v))
```

The reconstruction loss is:

```text
L_rec = KL(Q_phi(Gs | Hc) || P_theta(Gs))
```

This regularizes the representation so it preserves a reasonable graph posterior instead of overfitting label-spurious noise.

### Loss Replacement

The original KL terms for `Hc_intra` and `Hc_inter` are intractable. CGRL proves that, under idealized convergence assumptions, replacement losses are asymptotically same-order:

- `L_intra`: cluster intra-class nodes.
- `L_inter`: separate inter-class nodes.
- `L_sup`: supervised classification.
- `L_rec`: energy-based graph reconstruction.

Overall:

```text
L = L_sup + L_rec + lambda_1 L_intra + lambda_2 L_inter
```

## Experimental Evidence

Datasets and shifts:

- Feature shifts: Cora, Citeseer, Pubmed.
- Temporal shift: Arxiv.
- Spatial shift: Twitch.
- GOOD benchmark covariate/concept shifts: GOODCora, GOODCBAS, GOODWebKB.

Baselines include ERM, IRM, DeepCoral, DANN, GroupDRO, Mixup, SRGNN, EERM, CaNet, and CIA-LRA.

Key reported patterns:

- CGRL improves over CaNet on feature, temporal, and spatial shifts in most reported GCN/GAT settings.
- On Pubmed feature shift, the reported gain over the second-best method is large.
- CGRL stabilizes `MI(Hc, Y)` over training, while vanilla GCN/GAT show stronger fluctuation under OOD shifts.
- Ablation shows all four losses matter; removing supervised loss hurts most, while removing `L_rec`, `L_intra`, or `L_inter` also reduces OOD performance.

## Strengths

- Environment labels are not required.
- Gives a direct causal story for unstable representation-label mutual information.
- EBM reconstruction is graph-native and attaches naturally to node classification.
- The loss design is modular: `L_rec`, `L_intra`, and `L_inter` can be tried separately.
- The paper reports results on the same datasets already used in this repo: Cora, Citeseer, Pubmed, Arxiv, Twitch, and GOOD.

## Weaknesses / Risks

- The paper is an arXiv preliminary work as of `2603.24304v1`.
- The theoretical same-order loss replacement depends on idealized assumptions such as convergence and ideal GNN behavior.
- Energy-based reconstruction can be expensive on large graphs unless edge sampling or sparse normalization is used.
- It does not implement front-door adjustment. It is closer to backdoor-guided deconfounding and representation regularization.
- It does not explicitly split mediator and spurious context; this repo still needs front-door-specific constraints such as HSIC or context averaging when using `P004` logic.

## Relation To Current CaNet Project

Paper content:

- `P009` says graph OOD failures can be viewed as unstable `MI(Hc, Y)` caused by spurious correlations and non-causal paths.
- It proposes EBM graph reconstruction from causal representation and class-structure losses as a practical representation regularizer.

Project connection:

- Current `model_gmm3_frontdoor_hsic_ebr.py` already contains a CGRL-style EBR term on the mediator `M`: `KL(Q_phi(G | M) || P_theta(G))`.
- The current model also uses HSIC to discourage dependence between mediator `M` and spurious variable `U`, which complements CGRL because CGRL itself does not explicitly implement a front-door mediator/context separation.
- CGRL suggests adding diagnostics for `MI(M, Y)` and `MI(U, Y)` or a proxy such as classifier confidence/entropy stability over epochs.
- CGRL's `L_intra` and `L_inter` provide a next low-risk addition to the current HSIC+EBR line: class-structure regularization on the mediator should be tested before adding more front-door context mechanisms.

## Implementation Notes

Minimal additions for this repo:

```text
M = causal mediator representation
U = spurious/context representation
L_ebr = KL(Q_phi(G | M) || P_theta(G))
L_intra = supervised-center or pairwise compactness loss on M by class
L_inter = class-center margin loss on M
loss += lambda_ebr * L_ebr + lambda_intra * L_intra + lambda_inter * L_inter
```

Practical defaults:

- Keep `lambda_ebr=0.05` as already used in the active HSIC+EBR script.
- Start `lambda_intra` and `lambda_inter` at small values such as `{0.01, 0.05, 0.1}` because the current model already has `loss_med`, `loss_fd`, and HSIC.
- Use edge sampling for Arxiv/Twitch; do not run dense edge-pair reconstruction.
- Log `loss_ebr_kl`, positive edge score, negative edge score when optional NCE/BCE is enabled, and mediator class compactness.

## Future Ideas

- CGRL-Mediator Compactness: add class-center compactness and margin separation to the causal mediator branch.
- MI Stability Diagnostics: track a kNN/soft-binned proxy of `MI(M, Y)` and `MI(U, Y)` across epochs for train/validation/OOD splits.
- EBR Prior Ablation: compare encoder prior vs uniform observed-neighbor prior in `compute_ebr_loss`.
- Branch Reweight Adapter: add a small `K`-branch reweighting module before mediator/spurious split, but only after the current HSIC+EBR baseline is stable.

## One-line Memory

P009 says graph OOD can be diagnosed as unstable representation-label mutual information from spurious correlations; for this repo, its most immediate use is CGRL-style EBR on the mediator plus class compactness/separation losses and MI-stability diagnostics.
