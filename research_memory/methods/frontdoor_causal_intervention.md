# Method Card - Front-Door Causal Intervention

## Source Papers

- `P004` CIPT
- `P001` CaNet as graph environment source

## Core Idea

Use a mediator representation to estimate a causal effect while reducing latent confounding. In graph OOD, mediator/context can be built from causal node representation plus environment/spurious context.

## P004 Template

- Decompose representation into causal `e` and spurious `s`.
- Treat causal `e` as mediator.
- Generate diverse non-causal contexts.
- Average or fuse predictions across contexts.
- Add independence/decomposition losses.

## Graph Translation

Replace text templates with graph contexts:

- CaNet pseudo environment prototypes (`P001`).
- MLEI global/local environment contexts (`P002`).
- NodeIGM environmental subgraph embeddings (`P003`).
- Spurious branch prototypes from current model.

## Project Use

Current `GraphFrontDoor` already follows this shape. The research opportunity is making the context bank more graph-native and better justified.

## Risks

- Front-door assumptions may not hold if mediator is not truly causal.
- Diversity augmentation can become arbitrary graph augmentation unless tied to environment/spurious mechanisms.
- Context averaging may smooth away useful minority-environment signals.
