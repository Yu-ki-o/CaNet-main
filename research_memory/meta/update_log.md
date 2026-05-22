# Update Log

## 2026-05-12

Added `P009` from `/public/wc/lunwen/CGRL Causal-Guided Representation Learning.pdf`.

Changed files:

- Created `papers/P009_cgrl.md`.
- Created `methods/causal_guided_representation_learning.md`.
- Updated `meta/paper_registry.csv`.
- Updated `index.md`, `project_context.md`, `meta/taxonomy.md`.
- Updated `ideas/model_variants.md` and `ideas/experiment_queue.md`.

New insights:

- CGRL (`P009`) treats graph OOD failure as unstable mutual information between prediction representation and labels caused by spurious correlations and non-causal paths.
- The current `model_gmm3_frontdoor_hsic_ebr.py` already implements the most direct CGRL transfer: EBM graph reconstruction on the causal mediator.
- The next low-risk extension is not another large context module, but mediator class compactness/separation plus MI-stability diagnostics.

Unresolved uncertainties:

- CGRL is preliminary arXiv work, and its same-order loss replacement assumptions are idealized.
- Strong mediator compactness may conflict with front-door context averaging or minority-class behavior, so it needs small weights and collapse diagnostics.

## 2026-05-01

Added `P008` from `/public/wc/lunwen/Supercharging_Graph_Trans.pdf`.

Changed files:

- Created `papers/P008_advdifformer.md`.
- Created `methods/advective_diffusion_topology_shift.md`.
- Updated `meta/paper_registry.csv`.
- Updated `index.md`, `project_context.md`, `meta/taxonomy.md`.
- Updated `ideas/model_variants.md` and `ideas/experiment_queue.md`.

New insights:

- AdvDIFFormer (`P008`) treats topology shift as environment-induced adjacency distribution change and warns that local adjacency diffusion can be highly sensitive under such shifts.
- For the current front-door/DAG pipeline, this suggests an `Advective-FrontDoor Encoder`: stable non-local attention plus weighted observed-topology propagation before causal/spurious decomposition.
- The topology reliance weight `beta` should be treated as an OOD hyperparameter, especially on Arxiv/Twitch/Elliptic.

Unresolved uncertainties:

- Dense all-pair attention may be too expensive for Arxiv; a top-k, low-rank, landmark, or chunked approximation is likely needed.
- It is unclear whether the stable non-local channel should feed all branches equally or be emphasized in the causal branch while topology-heavy representations feed context/spurious branches.

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

- Other PDFs in `/public/wc/lunwen` were intentionally left out of this first memory pass. Add them later as `P010+` if they become relevant.
