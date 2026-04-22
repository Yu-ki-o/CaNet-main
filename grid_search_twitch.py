from grid_search_common import run_grid_search


DATASET = "twitch"
SEARCH_STAGE = "stage1"  # choose from: stage1, stage2

BASE_CMD = [
    "python", "main.py",
    "--dataset", DATASET,
    "--backbone_type", "gcn",
    "--env_type", "graph",
    "--combine_result",
    "--store_result",
    "--epochs", "500",
    "--runs", "2",
    "--num_layers", "2",
    "--expert_agg", "mean",
    "--tau", "3",
    "--seed", "2024",
    "--device", "1",
    "--other_env_objective", "irm",
]

STAGE_GRIDS = {
    "stage1": {
        "lr": [0.01, 0.005, 0.001],
        "weight_decay": [5e-5],
        "dropout": [0.0, 0.2],
        "lamda": [0.1, 1.0, 2.0],
        "K": [4],
        "hidden_channels": [32, 64],
        "other_env_reduce": ["env"],
    },
    "stage2": {
        "lr": [0.0075, 0.005],
        "weight_decay": [1e-4, 5e-5],
        "dropout": [0.0, 0.05, 0.1],
        "lamda": [0.5, 1.0, 1.5],
        "K": [3, 4],
        "hidden_channels": [64],
        "other_env_reduce": ["sample", "env"],
        "use_causal_head": [True],
        "other_env_irm_lambda": [0.5, 1.0],
        "causal_align_weight": [0.05, 0.1],
    },
}


def main():
    run_grid_search(DATASET, SEARCH_STAGE, BASE_CMD, STAGE_GRIDS)


if __name__ == "__main__":
    main()
