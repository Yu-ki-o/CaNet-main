# Method Card - Environment Inference

## Source Papers

- `P001` CaNet
- `P002` MLEI

## Core Idea

Environment is treated as a latent confounder behind graph OOD shifts. Since real environment labels are often unavailable, models infer pseudo environments from graph/node representations.

## P001 Version

CaNet infers pseudo environments from ego-graph representations and conditions a MoE GNN predictor on them.

Useful components:

- `q_phi(E|G)` environment estimator.
- `K` pseudo environment candidates.
- MoE graph propagation units.
- KL/regularization term against environment prior.

## P002 Update

MLEI argues low-hop ego-graph environment inference is short-sighted. It adds:

- global environment inference via linear graph transformer.
- multi-hop local environment inference.
- global-to-local fusion.

## Project Use

Current front-door contexts can be strengthened by replacing a single context bank with:

```text
context = fuse(local_spurious_prototype, global_environment_context, multi_hop_environment_context)
```

## Implementation Sketch

- Add optional `GlobalEnvironmentEncoder`.
- Add optional `multi_hop_env` context extraction.
- Keep the first version lightweight: global context should feed context/spurious branch, not directly dominate causal logits.

## Risks

- More environment machinery can overfit.
- Global context can leak label shortcuts.
- Needs careful ablation: local only, global only, local + global.
