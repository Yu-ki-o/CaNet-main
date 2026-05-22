# Method Card - Causal-Guided Representation Learning

## Source Papers

- `P009` CGRL

## Core Idea

Learn node representations that keep stable label-relevant information under graph OOD shifts. `P009` frames the problem as unstable `MI(Hc, Y)` caused by spurious correlations and non-causal paths, then uses backdoor adjustment to motivate representation regularization.

## Main Ingredients

- Re-weight node representations instead of using raw GNN embeddings directly.
- Reconstruct graph posterior from the causal representation with energy-based edge scores.
- Cluster same-class nodes and separate different-class nodes in the causal representation space.
- Track mutual-information stability as a diagnostic, not only final OOD accuracy.

## Project Translation

For the current front-door models, map CGRL variables as:

```text
Hc -> M, the causal mediator branch
Gs -> observed graph or sampled ego-graph edges
P_theta(Gs) -> detached encoder/edge-score prior
Q_phi(Gs | Hc) -> EBM posterior reconstructed from M
```

The active `model_gmm3_frontdoor_hsic_ebr.py` already implements the most direct `P009` transfer:

```text
L_ebr = KL(Q_phi(G | M) || P_theta(G))
E(u, v) = -M_v^T W M_u
```

The next missing CGRL-style pieces are:

- `L_intra`: compact mediator representations within each class.
- `L_inter`: separate class centers or hard negative class pairs.
- MI-stability logging for `M` and `U`.

## Minimal Loss Sketch

```text
centers_y = class_mean(M_train, y_train)
L_intra = mean(||M_i - centers_{y_i}||^2)
L_inter = mean(max(0, margin - ||center_a - center_b||)^2)
loss += lambda_intra * L_intra + lambda_inter * L_inter
```

Use this as a lightweight proxy for `P009` before implementing the full same-order loss replacement derivation.

## Project Use

This method can improve the current HSIC+EBR front-door line:

- HSIC discourages mediator-spurious dependence.
- EBR keeps mediator structure graph-compatible.
- Intra/inter losses make the mediator label geometry more stable.
- MI diagnostics reveal whether OOD training instability is representation instability or context/front-door aggregation instability.

## Risks

- Strong class compactness can overfit training classes or collapse minority classes.
- `L_inter` may conflict with front-door context averaging if applied too heavily.
- EBR cost must stay sparse and sampled on large graphs.
- Mutual-information estimates can be noisy; use them as trend diagnostics rather than selection metrics at first.
