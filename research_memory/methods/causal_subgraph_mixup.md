# Method Card - Causal Subgraph And Environment Mixup

## Source Papers

- `P003` NodeIGM

## Core Idea

Instead of only splitting hidden features into causal/spurious parts, split the graph structure into causal and environmental subgraphs. Then generate mixed environments and enforce consistent prediction across them.

## Components

- Learnable edge discriminator for causal importance.
- Causal subgraph and environmental subgraph partition.
- Hub preservation to avoid deleting important high-degree structure.
- Global environment mixup with different spurious strengths.
- Consistency / V-REx-style loss across generated environments.

## Project Use

For current front-door model:

```text
causal_edge_index -> mediator/causal branch
environment_edge_index -> spurious/context branch
mixup environments -> front-door diversity contexts
```

## Minimal Version

Do not implement full NodeIGM first. Start with:

- edge score MLP over `[h_i, h_j, |h_i-h_j|, h_i*h_j]`
- top-ratio causal edges
- remaining edges as environmental
- one consistency loss over two mixed edge sets

## Risks

- Edge discriminator can be unstable early in training.
- Hard edge sampling is non-smooth; use soft weights or straight-through only after baseline works.
- Large graphs need sparse implementation.
