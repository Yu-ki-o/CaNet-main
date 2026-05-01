# Advective Diffusion For Topology Shift

Sources: `P008`

## Problem

Many graph OOD methods focus on feature, label, or environment confounding, but node representations can also fail because local adjacency propagation is highly sensitive to topology shift.

In the current repo, this matters because GCN/GAT-style backbones and front-door branches still depend on the observed graph. Even if the causal split is useful, topology-sensitive propagation can leak environment-specific structure into all downstream branches.

## Reusable Mechanism

Use two propagation channels:

- Stable non-local channel: global or approximate attention `C` over node embeddings.
- Observed topology channel: normalized adjacency or MPNN propagation `V`.

Combine them as:

```text
P = C + beta V
z = [z0, P z0, ..., P^K z0]
```

`beta` controls how much the model trusts observed graph topology.

## Design Pattern For CaNet / Front-Door

```text
raw features -> z0
z_stable = NonLocalDiffusion(z0)
z_topo = AdjPropagation(z0)
z = Mixer(z_stable, beta * z_topo)
causal/spurious/front-door modules consume z
```

Possible branch split:

- causal branch gets more stable non-local representation.
- mediator branch gets mixed representation.
- context/spurious branch may receive stronger topology signal because environments often manifest through topology.

## Implementation Cautions

- Dense attention can be too expensive for Arxiv. Prefer top-k attention, landmark attention, chunked attention, or low-rank approximations.
- `beta` should be searched; it is not universally best at `1`.
- Add this before large front-door/DAG changes, otherwise attribution will be unclear.
- Keep `K` small at first (`2` or `3`) to avoid over-smoothing and memory blowup.

## Diagnostics

Track:

- OOD accuracy under temporal/domain splits.
- validation-to-OOD gap as `beta` changes.
- sensitivity of logits or embeddings to edge dropout/rewiring.
- branch accuracy for causal/spurious heads if combined with front-door models.

## Project Hypothesis

For Arxiv and Twitch, a scalable advective encoder may improve OOD stability by reducing over-reliance on shifted observed topology while preserving useful local structure.
