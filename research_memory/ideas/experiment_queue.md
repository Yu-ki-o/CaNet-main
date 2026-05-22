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

## E006 - DAG-ness-Aware Feature DAG

Sources: `P007`, `P005`

Compare:

- current `GraphFrontDoorDAG`
- dynamic DAG loss scaling only
- lightweight CASPER-style hidden causal-space score

Start with Cora/Citeseer/Pubmed and `lambda_casper=0` default. If stable, run stage2 search for `lambda_casper` and one temperature/clipping parameter.

Measure: OOD accuracy, validation stability, `h(A_feat)`, mediator mask entropy, and whether causal branch accuracy drops.

## E007 - Advective Encoder Topology-Reliance Sweep

Sources: `P008`

Compare:

- current GCN/GAT encoder
- non-local diffusion only (`beta=0`)
- advective diffusion with `beta in {0.2, 0.5, 0.8, 1.0}`
- current front-door DAG model with advective encoder plugged in

Start with Cora/Citeseer/Pubmed for debugging, then prioritize Arxiv and Twitch.

Measure: OOD accuracy, validation-to-OOD gap, memory/runtime, and embedding/logit sensitivity to edge dropout or edge rewiring.

## E008 - CGRL Mediator Stability

Sources: `P009`

Compare:

- current `model_gmm3_frontdoor_hsic_ebr.py` with `lambda_ebr=0`
- current default HSIC+EBR
- HSIC+EBR with mediator `L_intra`
- HSIC+EBR with mediator `L_intra + L_inter`

Start with Cora/Citeseer/Pubmed feature shifts, then run Arxiv/Twitch only if mediator compactness does not collapse minority classes.

Measure: OOD accuracy, validation-to-OOD gap, `loss_ebr_kl`, mediator class compactness, class-center margin, HSIC, and a lightweight MI or dependence proxy between mediator/spurious branches and labels.
