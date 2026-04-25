SMALL_GRAPH_DATASETS = {"cora", "citeseer", "pubmed"}
BINARY_DATASETS = {"twitch", "elliptic"}
SUPPORTED_DATASETS = SMALL_GRAPH_DATASETS | {"arxiv", "twitch", "elliptic"}
SUPPORTED_BACKBONES = {"gcn", "gat"}


BASE_SETTINGS = {
    "cora": {
        "epochs": 500,
        "runs": 2,
        "combine_result": True,
        "seed": None,
        "num_layers": {"gcn": 2, "gat": 2},
        "variant": {"gcn": False, "gat": False},
    },
    "citeseer": {
        "epochs": 500,
        "runs": 2,
        "combine_result": True,
        "seed": None,
        "num_layers": {"gcn": 2, "gat": 2},
        "variant": {"gcn": False, "gat": False},
    },
    "pubmed": {
        "epochs": 500,
        "runs": 2,
        "combine_result": True,
        "seed": None,
        "num_layers": {"gcn": 2, "gat": 2},
        "variant": {"gcn": False, "gat": False},
    },
    "arxiv": {
        "epochs": 500,
        "runs": 2,
        "combine_result": False,
        "seed": None,
        "num_layers": {"gcn": 2, "gat": 2},
        "variant": {"gcn": True, "gat": False},
    },
    "twitch": {
        "epochs": 500,
        "runs": 2,
        "combine_result": False,
        "seed": 2024,
        "num_layers": {"gcn": 2, "gat": 2},
        "variant": {"gcn": False, "gat": False},
    },
    "elliptic": {
        "epochs": 500,
        "runs": 3,
        "combine_result": False,
        "seed": None,
        "num_layers": {"gcn": 3, "gat": 2},
        "variant": {"gcn": True, "gat": False},
    },
}


CORE_DEFAULTS = {
    ("cora", "gcn"): {
        "lr": 0.01,
        "weight_decay": 5e-5,
        "dropout": 0.2,
        "hidden_channels": 64,
        "K": 3,
    },
    ("cora", "gat"): {
        "lr": 0.01,
        "weight_decay": 0.0,
        "dropout": 0.2,
        "hidden_channels": 64,
        "K": 3,
    },
    ("citeseer", "gcn"): {
        "lr": 0.01,
        "weight_decay": 5e-5,
        "dropout": 0.1,
        "hidden_channels": 64,
        "K": 3,
    },
    ("citeseer", "gat"): {
        "lr": 0.01,
        "weight_decay": 0.0,
        "dropout": 0.2,
        "hidden_channels": 64,
        "K": 3,
    },
    ("pubmed", "gcn"): {
        "lr": 0.005,
        "weight_decay": 5e-5,
        "dropout": 0.2,
        "hidden_channels": 64,
        "K": 2,
    },
    ("pubmed", "gat"): {
        "lr": 0.01,
        "weight_decay": 5e-5,
        "dropout": 0.2,
        "hidden_channels": 64,
        "K": 3,
    },
    ("arxiv", "gcn"): {
        "lr": 0.01,
        "weight_decay": 5e-4,
        "dropout": 0.2,
        "hidden_channels": 64,
        "K": 3,
    },
    ("arxiv", "gat"): {
        "lr": 0.005,
        "weight_decay": 5e-5,
        "dropout": 0.2,
        "hidden_channels": 64,
        "K": 3,
    },
    ("twitch", "gcn"): {
        "lr": 0.01,
        "weight_decay": 5e-5,
        "dropout": 0.0,
        "hidden_channels": 64,
        "K": 3,
    },
    ("twitch", "gat"): {
        "lr": 0.01,
        "weight_decay": 5e-5,
        "dropout": 0.0,
        "hidden_channels": 64,
        "K": 3,
    },
    ("elliptic", "gcn"): {
        "lr": 0.01,
        "weight_decay": 1e-3,
        "dropout": 0.2,
        "hidden_channels": 32,
        "K": 3,
    },
    ("elliptic", "gat"): {
        "lr": 0.005,
        "weight_decay": 5e-4,
        "dropout": 0.1,
        "hidden_channels": 64,
        "K": 3,
    },
}


def _unique(values):
    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def _get_settings(dataset, backbone):
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return BASE_SETTINGS[dataset], CORE_DEFAULTS[(dataset, backbone)]


def build_frontdoor_search(dataset, backbone):
    settings, defaults = _get_settings(dataset, backbone)

    base_cmd = [
        "python", "main_frontdoor.py",
        "--dataset", dataset,
        "--backbone_type", backbone,
        "--store_result",
        "--epochs", str(settings["epochs"]),
        "--runs", str(settings["runs"]),
        "--num_layers", str(settings["num_layers"][backbone]),
        "--device", "1",
    ]

    if settings["combine_result"]:
        base_cmd.append("--combine_result")
    if settings["seed"] is not None:
        base_cmd.extend(["--seed", str(settings["seed"])])
    if settings["variant"][backbone]:
        base_cmd.append("--variant")

    stage_grids = {
        "stage1": build_stage1_grid(dataset, backbone, defaults),
        "stage2": build_stage2_grid(dataset, defaults),
    }
    return base_cmd, stage_grids


def build_stage1_grid(dataset, backbone, defaults):
    if dataset in SMALL_GRAPH_DATASETS:
        return {
            "lr": _unique([defaults["lr"], 0.005]),
            "weight_decay": _unique([defaults["weight_decay"], 5e-5]),
            "dropout": _unique([max(0.0, defaults["dropout"] - 0.1), defaults["dropout"]]),
            "hidden_channels": _unique([32, defaults["hidden_channels"]]),
            "K": _unique([2, defaults["K"]]),
        }

    if dataset == "arxiv":
        return {
            "lr": [0.01, 0.005, 0.001] if backbone == "gcn" else [0.005, 0.001],
            "weight_decay": _unique([defaults["weight_decay"], 1e-4]),
            "dropout": [0.1, 0.2, 0.3],
            "hidden_channels": [64, 128],
            "K": [2, 3],
        }

    if dataset == "twitch":
        return {
            "lr": [0.01, 0.005, 0.001],
            "weight_decay": _unique([defaults["weight_decay"], 1e-4]),
            "dropout": _unique([0.0, 0.1, 0.2]),
            "hidden_channels": [32, 64],
            "K": [2, 3],
        }

    if dataset == "elliptic":
        return {
            "lr": [0.01, 0.005] if backbone == "gcn" else [0.005, 0.001],
            "weight_decay": _unique([defaults["weight_decay"], 5e-4]),
            "dropout": [0.1, 0.2, 0.3] if backbone == "gcn" else [0.1, 0.2],
            "hidden_channels": [32, 64],
            "K": [3, 5],
        }

    raise ValueError(f"Unsupported dataset for stage1 grid: {dataset}")


def build_stage2_grid(dataset, defaults):
    # Stage 2 fixes the core optimization/encoder hyperparameters to the
    # current anchor values in CORE_DEFAULTS, then refines the front-door
    # objective weights. After stage1, you can update those anchors here.
    stage2_grid = {
        "lr": [defaults["lr"]],
        "weight_decay": [defaults["weight_decay"]],
        "dropout": [defaults["dropout"]],
        "hidden_channels": [defaults["hidden_channels"]],
        "K": [defaults["K"]],
        "lambda_fd": [0.5, 1.0],
        "lambda_spu": [0.05, 0.1],
        "lambda_ind": [0.05, 0.1],
        "context_gate_temp": [0.5, 1.0],
    }

    if dataset in BINARY_DATASETS:
        stage2_grid.update({
            "fd_blend": [0.3, 0.5],
            "pos_weight": [3.0, 5.0, 7.0],
        })
    else:
        stage2_grid.update({
            "fd_blend": [0.3, 0.5, 0.7],
            "lambda_med": [0.25, 0.5],
        })

    return stage2_grid
