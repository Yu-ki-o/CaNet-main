# Method Card - DAG-ness-Aware Structure Learning

## Source Papers

- `P007` CASPER
- Related to `P005` DAG-aware Transformer
- Related to `P006` CIW when causal strength depends on learned DAG quality

## Core Idea

When learning a DAG, do not let the score function measure only data/task fitness while acyclicity is handled as a detached penalty. The score itself should be aware of how DAG-like the current candidate graph is.

P007 implements this by defining a dynamic causal space. The metric used to compare observed and reconstructed distributions changes according to `h(G)`, the current DAG-ness violation. This gives the optimizer a structure-aware signal throughout training, not only at the final acyclicity constraint.

## Project Translation

Current `GraphFrontDoorDAG` already has a feature-only adjacency matrix:

```text
A_feat -> DAG-derived structural scores -> mediator mask
```

The risk is that `A_feat` may be learned mostly through classification/front-door losses plus an acyclicity penalty. P007 suggests adding a structure-aware score so that the learned feature DAG is useful as a causal structure, not merely a task-fitting regularizer.

## Lightweight Implementation Options

- Dynamic DAG loss scaling:

```text
lambda_dag_eff = lambda_dag * g(h(A_feat))
```

- DAG-ness-aware mediator temperature:

```text
mediator_temp_eff = schedule(mediator_temp, h(A_feat))
```

- Causal-space discrepancy over hidden representations:

```text
score = T_phi(z) - T_phi(z_hat)
clip(T_phi, -log(1 + h(A_feat)), log(1 + h(A_feat)))
```

where `z_hat` can be reconstructed by a small DAG-fitting module using `A_feat`.

## Project Use

Best target: `model_frontdoor_dag.py`.

Minimal first experiment:

- Keep the current feature DAG and front-door path.
- Add `lambda_casper=0` default.
- Add a cheap DAG-ness-aware auxiliary score on hidden features.
- Search only `lambda_casper` and perhaps one clipping/temperature parameter in stage2.

## Risks

- Full bilevel CASPER training may be too heavy for node classification experiments.
- A poorly designed `z_hat` reconstruction can optimize representation reconstruction instead of causal structure.
- If `h(A_feat)` is noisy early in training, dynamic scaling may destabilize the CIPT curriculum.

## Memory Link

Use this method when changing how `A_feat`, DAG regularization, mediator masks, or latent-token DAG masks are learned.
