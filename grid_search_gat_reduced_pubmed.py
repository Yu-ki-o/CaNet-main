from grid_search_gat_reduced_common import run_grid


DATASET = "pubmed"
BACKBONE = "gat"
SEARCH_STAGE = "stage1"

base_cmd = [
    "python", "main.py",
    "--dataset", DATASET,
    "--backbone_type", BACKBONE,
    "--env_type", "graph",
    "--combine_result",
    "--store_result",
    "--epochs", "500",
    "--runs", "2",
    "--num_layers", "2",
    "--expert_agg", "mean",
    "--tau", "1",
    "--device", "0",
    "--other_env_objective", "irm",
]

stage1_grid = {
    "lr": [0.01, 0.005],
    "weight_decay": [5e-5],
    "dropout": [0.1, 0.2],
    "lamda": [0.5, 1.0, 2.0],
    "K": [3],
    "hidden_channels": [64],
    "other_env_reduce": ["env"],
}

stage2_grid = {
    "lr": [0.0075, 0.005],
    "lr": [0.005,0.01],
    "weight_decay": [5e-5],
    "dropout": [0.2],
    "lamda": [1.0, 2.0,3.0],
    "K": [3],
    "hidden_channels": [64,32],
    "other_env_reduce": ["sample", "env"],
}


if __name__ == "__main__":
    run_grid(DATASET, SEARCH_STAGE, base_cmd, stage1_grid if SEARCH_STAGE == "stage1" else stage2_grid)
