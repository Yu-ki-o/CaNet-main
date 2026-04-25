# Method Card - DAG-Aware Causal Mixing

## Source Papers

- `P005` DAG-aware Transformer

## Core Idea

Represent causal variables as tokens and constrain attention with a DAG mask. This prevents arbitrary information flow between causal, spurious, environment, mediator, and label tokens.

## Project Translation

Possible latent tokens:

- `z_causal`
- `z_spurious`
- `z_env`
- `z_mediator`
- `z_context`
- `label_query`

Possible mask:

```text
z_causal -> z_mediator -> label_query
z_env -> z_spurious -> z_context
z_context -> label_query only through front-door augmentation path
z_spurious -/-> label_query direct path blocked or penalized
```

## Project Use

This belongs in `model_frontdoor_dag.py` more than base `model_frontdoor.py`.

## Minimal Version

Use a small multi-head attention block over 4-6 latent tokens per node, not full node-node transformer attention.

## Risks

- Hand-designed DAG can be wrong.
- A too-strict mask may reduce useful signal.
- Needs ablation against simple concat/gate.
