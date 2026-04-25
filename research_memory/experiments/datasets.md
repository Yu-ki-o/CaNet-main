# Dataset Notes

## Cora / Citeseer / Pubmed

Use for quick debugging and small-scale ablations.

Potential issue: small graphs can make global context less meaningful and increase variance.

## Arxiv

Temporal OOD split. Strong candidate for:

- `P002` multi-level/global environment.
- `P004` front-door context diversity.
- `P006` masked independence if uniform independence hurts.

## Twitch

Likely geography/domain-driven shift. Environment signals may be strong and suitable for pseudo environment learning.

## Elliptic

Temporal transaction snapshots. Strong candidate for:

- `P003` causal edge/subgraph split.
- inference-light design.
- binary `pos_weight` tuning in stage2.

## Dataset-Specific Risk

Do not assume one mechanism helps every dataset. MLEI-like global context may help Arxiv/Elliptic more than Cora. Edge causal split may help Elliptic more than citation graphs if transaction structure matters.
