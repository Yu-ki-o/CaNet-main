# Update Log

## 2026-04-26

Added `P007` from `/public/wc/lunwen/Discovering Dynamic Causal Space for DAG Structure Learning.pdf`.

Changed files:

- Created `papers/P007_dynamic_causal_space.md`.
- Created `methods/dagness_aware_structure_learning.md`.
- Updated `meta/paper_registry.csv`.
- Updated `index.md`, `project_context.md`, `meta/taxonomy.md`.
- Updated `methods/dag_aware_causal_mixing.md`.
- Updated `ideas/model_variants.md` and `ideas/experiment_queue.md`.
- Updated `experiments/stage1_stage2_protocol.md`.

New insights:

- CASPER (`P007`) argues that differentiable DAG learners should not use a score function that measures only data/task fitness while acyclicity is treated as a detached penalty.
- For the current `GraphFrontDoorDAG`, this suggests a future `CASPER-FeatureDAG` variant: make `A_feat` learning DAG-ness-aware through dynamic scoring, causal-space discrepancy, or DAG-ness-dependent mediator/gate scheduling.

Unresolved uncertainties:

- Full bilevel CASPER may be too heavy for the current node OOD pipeline. A lightweight auxiliary score should be tried before implementing the full inner-loop scorer.

## 2026-04-25

Initialized project-level research memory from the 6 model-relevant PDFs selected by the user in `/public/wc/lunwen`.

Changed files:

- Created `README.md`, `index.md`, and `project_context.md`.
- Created 6 paper cards under `papers/`.
- Created method cards for CaNet-style environment inference, multi-level graph context, causal subgraph mixup, front-door causal intervention, DAG-aware causal effect estimation, and causal-guided differential decorrelation.
- Created `ideas/model_variants.md` and `ideas/experiment_queue.md`.
- Created `experiments/stage1_stage2_protocol.md` and `experiments/datasets.md`.
- Created `meta/paper_registry.csv`, `meta/taxonomy.md`, and `meta/self_prompt.md`.

Scope note:

- Other PDFs in `/public/wc/lunwen` were intentionally left out of this first memory pass. Add them later as `P008+` if they become relevant.
