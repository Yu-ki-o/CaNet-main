# Method Card - Multi-Level Graph Context

## Source Papers

- `P002` MLEI
- Related to `P001` CaNet

## Core Idea

Node environment is not a single neighborhood. It includes:

- micro: one-hop ego information.
- meso: multi-hop ego information.
- macro: global graph context.

## Project Use

Use multi-level context as a better source for front-door diversity:

```text
contexts = {
  local_spurious_context,
  multi_hop_context,
  global_environment_context
}
```

## Implementation Options

- SGC/APPNP features for multi-hop context.
- Linear attention/global pooling for macro context.
- Environment prototypes per level.
- Gated fusion with temperature.

## Risks

- Multi-level context can increase oversmoothing if implemented as deeper GNN.
- Global context should be regularized so it does not become a label shortcut.
