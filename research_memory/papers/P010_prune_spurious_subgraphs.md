# P010 - Pruning Spurious Subgraphs for Graph OOD Generalization

## Basic Info

- Title: Pruning Spurious Subgraphs for Graph Out-of-Distribution Generalization
- Method: PrunE
- Venue / Year: NeurIPS 2025
- Source: `/public/wc/lunwen/NeurIPS-2025-pruning-spurious-subgraphs-for-graph-out-of-distribution-generalization-Paper-Conference.pdf`
- Code: `https://github.com/tianyao-aka/PrunE-GraphOOD`
- Tags: graph OOD, spurious-edge pruning, subgraph selector, Gumbel-Softmax, graph sparsity

## Core Problem

Most graph OOD methods directly identify invariant edges. PrunE argues that this is error-prone
when strongly label-correlated spurious edges resemble invariant edges. It instead conservatively
prunes edges that are easy to identify as uninformative/spurious, allowing ERM to preserve a more
complete predictive subgraph.

## Main Mechanism

PrunE learns a scalar keep probability for every edge and uses hard Gumbel-Softmax masks during
message passing. Two lightweight losses regularize the selector:

```text
L_edge_size = (mean(hard_keep_mask) - eta)^2
L_low_align = mean(|p_keep(bottom-K%) - epsilon|)
L = L_ERM + lambda_1 L_edge_size + lambda_2 L_low_align
```

The official code uses soft edge probabilities to select the bottom-K edges, a hard Gumbel mask
for propagation, and an ERM pretraining stage before enabling pruning.

## Assumptions And Limits

- Invariant edges are sufficiently predictive for ERM to preserve them under a suitable edge budget.
- Strongly correlated spurious edges may remain; the method prioritizes preserving invariant edges
  over perfectly separating causal and spurious structures.
- The paper and official implementation target graph classification. Full-graph node classification
  needs degree-aware or per-destination adaptations to avoid pruning minority/low-degree regions.
- The pruned bottom-K edges are mostly uninformative and should not automatically be treated as
  useful front-door environment contexts.

## Relation To Current Project

The active layerwise model currently treats its edge gate as a causal/useful discriminator and sends
the complement to the spurious context branch. P010 suggests a safer interpretation:

```text
scalar prune gate -> decide keep versus discard for layerwise enhancement
separate environment score -> decide which informative non-stable messages feed front-door contexts
```

For the current vector edge gate, a faithful transfer should add a scalar PrunE keep mask before the
optional channel-wise enhancement gate:

```text
final_message_gate_uvd = scalar_keep_uv * channel_gate_uvd
```

## Implementation Notes

- Preserve the soft keep probability before hard Gumbel sampling; PrunE losses must not operate
  only on the hard `0/1` gate.
- Use a conservative keep budget first, such as `eta in {0.75, 0.85, 0.95}` for node classification.
- Adapt the budget by destination node or degree bucket rather than relying only on one global edge
  ratio dominated by hubs.
- Reuse the existing decomposition warmup to pretrain node representations before enabling pruning.
- Do not directly use the lowest-probability pruned summary as `z_spurious`; it may mostly contain
  uninformative noise.

## Future Idea

PrunE-Ratio FrontDoor: generate multiple structural environments by varying the pruning budget,
then use the resulting removed/retained message differences as multi-ratio front-door contexts.

## One-line Memory

P010 replaces risky direct causal-edge identification with conservative spurious-edge pruning:
preserve predictive structure through ERM, prune only low-confidence edges, and keep pruning
separate from meaningful environment-context extraction.
