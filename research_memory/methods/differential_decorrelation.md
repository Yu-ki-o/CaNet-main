# Method Card - Causal-Guided Differential Decorrelation

## Source Papers

- `P006` CIW

## Core Idea

Do not decorrelate all features equally. Estimate feature-to-label causal strength and use it to decide how strongly to remove dependence between feature channels.

## Project Translation

Current independence losses can become:

```text
masked_independence_loss = sum_ij S_ij * dependence(z_i, z_j)
```

where `S_ij` is high for weak-causal/spurious channel pairs and low for strong-causal channel pairs.

## Lightweight Causal Strength Sources

Before implementing full cross-domain DAG:

- gradient attribution from label loss
- environment-stability of feature-label association
- mutual information with labels minus mutual information with environments
- moving-average classifier weights

## Project Use

This can improve `lambda_ind` in front-door models:

- uniform independence may suppress useful causal features.
- masked independence can preserve stable causal correlations.

## Risks

- Causal strength estimates can be noisy.
- Mask update schedule matters; freeze or EMA may be needed.
- Full DAG learning is likely too heavy for first graph implementation.
