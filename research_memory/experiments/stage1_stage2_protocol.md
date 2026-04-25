# Stage1 / Stage2 Protocol

## Current Design

Stage1 searches core optimization/encoder parameters:

- `lr`
- `weight_decay`
- `dropout`
- `hidden_channels`
- `K`

Stage2 fixes those core parameters through `CORE_DEFAULTS` and searches front-door objective parameters:

- `lambda_fd`
- `lambda_spu`
- `lambda_ind`
- `context_gate_temp`
- `fd_blend`
- `lambda_med` for non-binary datasets
- `pos_weight` for binary datasets

## How To Use With Research Variants

When adding a mechanism that changes encoder/context construction, rerun stage1 for affected dataset/backbone.

Examples:

- Adding MLEI-style global context changes encoder/context: rerun stage1.
- Only changing `lambda_ind` to masked independence: stage2 may be enough after a small sanity check.
- Adding edge discriminator changes graph encoder behavior: rerun stage1.
- Adding DAG latent mixer after representations: likely stage2 first, then stage1 if unstable.

## Updating Defaults

After stage1:

1. Open `search_logs_frontdoor/{dataset}/{backbone}/stage1/*top10.txt`.
2. Pick the best rank according to the chosen OOD metric.
3. Update `CORE_DEFAULTS[(dataset, backbone)]` in `grid_search_frontdoor_configs.py`.
4. Run stage2.

## Memory Link

Record any selected defaults or surprising results in this directory later, e.g. `experiments/frontdoor_results.md`.
