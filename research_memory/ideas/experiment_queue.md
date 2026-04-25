# Experiment Queue

## E001 - Baseline Audit

Goal: establish current front-door baseline for each dataset/backbone.

Run:

- existing `run_frontdoor.sh` commands.
- record best OOD metric and stability.

Why: needed before adding paper-inspired modules.

## E002 - Multi-Level Context Ablation

Sources: `P002`, `P004`

Compare:

- local context only
- global context only
- local + global context
- local + global + multi-hop context

Priority datasets: Arxiv, Elliptic.

## E003 - Edge-Causal Mini Prototype

Sources: `P003`

Implement soft edge scores and train with:

- original graph baseline
- causal-weighted graph
- environmental graph context

Start on Cora/Citeseer before Arxiv/Elliptic.

## E004 - Masked Independence Loss

Sources: `P006`

Compare:

- no independence loss
- uniform independence loss
- gradient/EMA causal-strength masked independence loss

Measure: OOD accuracy and whether causal branch accuracy drops.

## E005 - DAG Latent Mixer

Sources: `P005`

Compare:

- concat/gate baseline
- unrestricted latent attention
- DAG-masked latent attention

Run only after baseline front-door is stable.
