# Model Variants

Each idea cites source paper ids and is intended to be implementable in this repo.

## V001 - Multi-Level FrontDoor

Sources: `P001`, `P002`, `P004`

Add global/local environment contexts to `GraphFrontDoor`. Keep existing causal/spurious adapters, but replace the context bank with a fused context:

```text
context = gate(local_context, global_context, multi_hop_context)
```

Expected benefit: better Arxiv/Elliptic OOD performance where low-hop ego-graph is insufficient.

## V002 - Edge-Causal FrontDoor

Sources: `P003`, `P004`

Use a causal edge discriminator to create causal and environmental subgraphs:

- causal subgraph feeds mediator branch.
- environmental subgraph feeds spurious/context branch.
- environment mixup creates multiple front-door contexts.

Expected benefit: graph-native causal/spurious split.

## V003 - DAG-Masked Latent Mixer

Sources: `P005`, `P004`

Replace simple concat/gate among causal/spurious/context embeddings with a small DAG-aware transformer over latent tokens.

Expected benefit: prevent direct spurious-to-label leakage.

## V004 - Causal-Strength Independence

Sources: `P006`, `P004`

Replace uniform `lambda_ind` with masked decorrelation. Causal channel pairs are weakly penalized; spurious/weak-causal channel pairs are strongly penalized.

Expected benefit: less damage to useful causal feature correlations.

## V005 - Inference-Light Training

Sources: `P003`, `P001`

Use heavy causal/environment modules during training, but distill or cache into a simpler encoder/classifier for inference.

Expected benefit: better deployment cost on Elliptic and Arxiv.

## V006 - CASPER-FeatureDAG

Sources: `P007`, `P005`, `P006`

Add a DAG-ness-aware auxiliary score to `GraphFrontDoorDAG` so the feature DAG is not learned only from classification/front-door fitness plus an acyclicity penalty.

Minimal version:

```text
h_A = h(A_feat)
dag_score = causal_space_distance(T_phi(z), T_phi(z_hat), h_A)
loss += lambda_casper * dag_score
```

Expected benefit: more stable feature DAG and mediator mask learning, especially when `A_feat` otherwise falls into a task-fitting but structurally weak solution.

## Deprecated / Caution

- Do not combine V001 + V002 + V003 all at once. The loss surface and hyperparameter space will become hard to diagnose.
- Do not implement full cross-domain DAG from `P006` before a lightweight causal-strength mask baseline.
- Do not implement full bilevel `P007` CASPER before a lightweight DAG-ness-aware auxiliary score. The inner-loop scorer can destabilize current front-door training.
