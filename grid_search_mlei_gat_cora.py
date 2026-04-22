from grid_search_mlei_gat_common import run_grid_search


DATASET = "cora"
SEARCH_STAGE = "stage1"

base_cmd = [
    "python", "main_MLEI.py",
    "--dataset", DATASET,
    "--backbone_type", "gat",
    "--combine_result",
    "--store_result",
    "--epochs", "500",
    "--runs", "3",
    "--num_layers", "2",
    "--tau", "3",
    "--K", "3",
    "--hidden_channels", "64",
    "--device", "0",
]

stage1_grid = {
    "lr": [0.01, 0.005],
    "weight_decay": [0.0, 5e-5],
    "dropout": [0.1, 0.2],
    "lamda": [0.5, 1.0],
}

stage2_grid = {
    "lr": [0.0075, 0.005],
    "weight_decay": [0.0, 1e-5],
    "dropout": [0.15, 0.2],
    "lamda": [1.0, 1.5],
}

param_grid = stage1_grid if SEARCH_STAGE == "stage1" else stage2_grid

if __name__ == "__main__":
    run_grid_search(DATASET, base_cmd, param_grid, SEARCH_STAGE)

