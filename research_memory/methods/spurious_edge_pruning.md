# Method Card - Conservative Spurious-Edge Pruning

## Source Papers

- `P010` PrunE
- Related: `P003` NodeIGM

## Core Idea

Do not require the edge gate to perfectly identify causal edges. Learn a conservative scalar keep
mask and prune only low-confidence edges, while ERM preserves predictive structure.

## Project Translation

For the active layerwise node-enhancement and multi-ratio front-door model:

```text
scalar keep/prune gate -> layerwise node enhancement
informative environment gate -> multi-ratio front-door contexts
bottom-K discard gate -> ignored noise
```

This resolves the current over-strong assumption that every edge not routed to the useful branch is
a meaningful spurious environment edge.

## Node-Level Adaptation

PrunE was designed for graph classification. In a single full node-classification graph, use a
per-destination or degree-bucket budget:

```text
L_budget = mean_v (mean_{u -> v}(p_keep_uv) - eta_v)^2
```

Use the soft probabilities for bottom-K alignment and hard Gumbel masks only for forward routing.

## Risks

- A small keep budget can remove necessary local structure.
- Global bottom-K ranking can disproportionately prune low-degree or minority regions.
- Pruned edges are often uninformative; using them directly as front-door contexts can weaken
  environment diversity.
