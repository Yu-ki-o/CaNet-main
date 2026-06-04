import argparse
import importlib
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_undirected


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_model_specs(args):
    if not args.model_specs:
        return [(args.model_name, args.model_module, args.model_class)]

    specs = []
    for item in parse_csv(args.model_specs):
        if "=" in item:
            name, target = item.split("=", 1)
        else:
            target = item
            name = target.replace(":", ".")
        if ":" not in target:
            raise ValueError(
                f"Bad model spec {item!r}. Use name=module:Class or module:Class."
            )
        module_name, class_name = target.split(":", 1)
        specs.append((name, module_name, class_name))
    return specs


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def graph_dir_name(dataset_name):
    if dataset_name == "Weibo":
        return "Weibograph"
    return dataset_name + "graph"


def label_file_name(dataset_name):
    return os.path.join(dataset_name, dataset_name + "_label_All.txt")


def load_binary_labels(kpg_dir, dataset_name):
    label_path = os.path.join(kpg_dir, "data", label_file_name(dataset_name))
    labels = {}
    non_rumor = {"news", "non-rumor", "non-rumours", "true"}
    rumor = {"false", "unverified", "rumours"}

    with open(label_path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            raw_label = parts[0].lower()
            eid = parts[2]
            if raw_label in non_rumor:
                labels[eid] = 0
            elif raw_label in rumor:
                labels[eid] = 1
            else:
                raise ValueError(f"Unknown label {raw_label!r} in {label_path}")
    return labels


def sparse_triplet_to_dense(x_array, node_num, feat_dim):
    x = torch.zeros((node_num, feat_dim), dtype=torch.float32)
    if x_array.size == 0:
        return x
    rows = x_array[0].astype(np.int64)
    cols = x_array[1].astype(np.int64)
    vals = x_array[2].astype(np.float32)
    mask = (rows >= 0) & (rows < node_num) & (cols >= 0) & (cols < feat_dim)
    if mask.any():
        x[torch.from_numpy(rows[mask]), torch.from_numpy(cols[mask])] = torch.from_numpy(vals[mask])
    return x


def load_graph_npz(path, feat_dim):
    data = np.load(path, allow_pickle=True)
    edge_index = torch.as_tensor(data["edgeindex"], dtype=torch.long)
    rootindex = int(data["rootindex"])

    x_array = data["x"]
    if x_array.ndim == 2 and x_array.shape[0] == 3:
        max_from_x = int(x_array[0].max()) + 1 if x_array.shape[1] > 0 else 1
        max_from_e = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 1
        node_num = max(max_from_x, max_from_e, rootindex + 1)
        x = sparse_triplet_to_dense(x_array, node_num, feat_dim)
    else:
        node_num = x_array.shape[0]
        x = torch.as_tensor(x_array[:, :feat_dim], dtype=torch.float32)
        if x.size(1) < feat_dim:
            pad = torch.zeros((x.size(0), feat_dim - x.size(1)), dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

    return x, edge_index, rootindex


def stratified_split(indices, labels, train_ratio, valid_ratio, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for idx in indices:
        by_label[int(labels[idx])].append(idx)

    train_idx, valid_idx, test_idx = [], [], []
    for label_indices in by_label.values():
        rng.shuffle(label_indices)
        n = len(label_indices)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        if n_train == 0 and n > 0:
            n_train = 1
        train_idx.extend(label_indices[:n_train])
        valid_idx.extend(label_indices[n_train:n_train + n_valid])
        test_idx.extend(label_indices[n_train + n_valid:])

    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    rng.shuffle(test_idx)
    return train_idx, valid_idx, test_idx


def stratified_limit(indices, labels, max_count, seed):
    indices = [int(idx) for idx in indices]
    if max_count is None or int(max_count) <= 0 or len(indices) <= int(max_count):
        return indices

    max_count = int(max_count)
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for idx in indices:
        by_label[int(labels[idx])].append(idx)

    selected = []
    label_items = sorted(by_label.items(), key=lambda item: item[0])
    total = len(indices)
    remainders = []
    for label, label_indices in label_items:
        rng.shuffle(label_indices)
        exact = max_count * len(label_indices) / total
        take = int(exact)
        if take == 0 and len(label_indices) > 0 and max_count >= len(label_items):
            take = 1
        take = min(take, len(label_indices))
        selected.extend(label_indices[:take])
        remainders.append((exact - int(exact), label, label_indices[take:]))

    remaining = max_count - len(selected)
    for _, _, rest in sorted(remainders, reverse=True):
        while remaining > 0 and rest:
            selected.append(rest.pop(0))
            remaining -= 1

    if len(selected) > max_count:
        selected = selected[:max_count]
    rng.shuffle(selected)
    return selected


def attach_node_split_masks(data):
    num_graphs = int(data.y.size(0))
    train_graph = torch.zeros(num_graphs, dtype=torch.bool)
    train_graph[data.train_idx] = True
    valid_graph = torch.zeros(num_graphs, dtype=torch.bool)
    valid_graph[data.valid_idx] = True
    data.node_train_mask = train_graph[data.batch]
    data.node_valid_mask = valid_graph[data.batch]
    return data


def load_rumor_graph_dataset(args):
    datasets = parse_csv(args.rumor_datasets)
    train_datasets = parse_csv(args.train_datasets)
    test_datasets = parse_csv(args.test_datasets) if args.test_datasets else [
        name for name in datasets if name not in train_datasets
    ]

    xs, edges, batches, root_indices, ys, envs, graphs = [], [], [], [], [], [], []
    graph_indices_by_dataset = defaultdict(list)
    graph_labels = {}
    env_id = {name: i for i, name in enumerate(datasets)}
    node_offset = 0
    graph_id = 0

    for dataset_name in datasets:
        labels = load_binary_labels(args.kpg_dir, dataset_name)
        graph_dir = os.path.join(args.kpg_dir, "data", graph_dir_name(dataset_name))
        if not os.path.isdir(graph_dir):
            raise FileNotFoundError(f"Graph directory not found: {graph_dir}")

        for filename in sorted(os.listdir(graph_dir)):
            if not filename.endswith(".npz"):
                continue
            eid = filename[:-4]
            if eid not in labels:
                continue
            x_i, edge_i, root_i = load_graph_npz(os.path.join(graph_dir, filename), args.feat_dim)
            if args.undirected:
                edge_i = to_undirected(edge_i, num_nodes=x_i.size(0))

            label_i = labels[eid]
            env_i = env_id[dataset_name]
            graphs.append(
                Data(
                    x=x_i,
                    edge_index=edge_i,
                    y=torch.tensor([[label_i]], dtype=torch.long),
                    root_index=torch.tensor([root_i], dtype=torch.long),
                    graph_env=torch.tensor([env_i], dtype=torch.long),
                )
            )
            xs.append(x_i)
            if edge_i.numel() > 0:
                edges.append(edge_i + node_offset)
            batches.append(torch.full((x_i.size(0),), graph_id, dtype=torch.long))
            root_indices.append(node_offset + root_i)
            ys.append(label_i)
            envs.append(env_i)
            graph_indices_by_dataset[dataset_name].append(graph_id)
            graph_labels[graph_id] = label_i
            node_offset += x_i.size(0)
            graph_id += 1

    if not xs:
        raise RuntimeError("No rumor graphs were loaded.")

    if edges:
        edge_index = torch.cat(edges, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(
        x=torch.cat(xs, dim=0),
        edge_index=edge_index,
        y=torch.as_tensor(ys, dtype=torch.long).unsqueeze(1),
    )
    data.batch = torch.cat(batches, dim=0)
    data.root_index = torch.as_tensor(root_indices, dtype=torch.long)
    data.graph_env = torch.as_tensor(envs, dtype=torch.long)
    data.env = data.graph_env[data.batch]
    data.graph_count = len(ys)
    data.graphs = graphs
    data.env_num = len(datasets)
    data.train_env_num = len(train_datasets)
    data.env_names = datasets
    data.train_env_names = train_datasets
    data.test_env_names = test_datasets

    train_idx, valid_idx, test_in_idx = [], [], []
    for i, dataset_name in enumerate(train_datasets):
        split = stratified_split(
            graph_indices_by_dataset[dataset_name],
            graph_labels,
            args.train_ratio,
            args.valid_ratio,
            args.seed + i,
        )
        train_idx.extend(split[0])
        valid_idx.extend(split[1])
        test_in_idx.extend(split[2])

    train_idx = stratified_limit(train_idx, graph_labels, args.max_train_graphs, args.seed + 1001)
    valid_idx = stratified_limit(valid_idx, graph_labels, args.max_valid_graphs, args.seed + 1002)
    test_in_idx = stratified_limit(test_in_idx, graph_labels, args.max_test_in_graphs, args.seed + 1003)

    data.train_idx = torch.as_tensor(train_idx, dtype=torch.long)
    data.valid_idx = torch.as_tensor(valid_idx, dtype=torch.long)
    data.test_in_idx = torch.as_tensor(test_in_idx, dtype=torch.long)
    data.test_ood_idx = [
        torch.as_tensor(
            stratified_limit(
                graph_indices_by_dataset[name],
                graph_labels,
                args.max_ood_graphs,
                args.seed + 2000 + i,
            ),
            dtype=torch.long,
        )
        for i, name in enumerate(test_datasets)
    ]
    data.test_ood_full_idx = [
        torch.as_tensor(graph_indices_by_dataset[name], dtype=torch.long)
        for name in test_datasets
    ]

    return attach_node_split_masks(data)


def load_rumor_platform_dataset(args):
    return load_rumor_graph_dataset(args)


def pool_nodes(x, batch, num_graphs, readout, root_index=None):
    if readout == "root":
        if root_index is None:
            raise ValueError("graph_readout='root' requires data.root_index.")
        return x.index_select(0, root_index.to(device=x.device, dtype=torch.long))

    batch = batch.to(device=x.device, dtype=torch.long)
    out = []
    if readout in ("mean", "mean_max", "sum"):
        summed = x.new_zeros(num_graphs, x.size(-1))
        summed.index_add_(0, batch, x)
        if readout == "sum":
            out.append(summed)
        else:
            counts = torch.bincount(batch, minlength=num_graphs).to(device=x.device, dtype=x.dtype)
            out.append(summed / counts.clamp_min(1.0).unsqueeze(-1))

    if readout in ("max", "mean_max"):
        maxed = x.new_zeros(num_graphs, x.size(-1))
        for graph_id in range(num_graphs):
            graph_values = x[batch == graph_id]
            if graph_values.numel() > 0:
                maxed[graph_id] = graph_values.max(dim=0).values
        out.append(maxed)

    if not out:
        raise ValueError(f"Unsupported graph_readout: {readout}")
    return torch.cat(out, dim=-1) if len(out) > 1 else out[0]


class GraphReadoutClassifier(nn.Module):
    def __init__(self, base_cls, d, c, args, device):
        super().__init__()
        self.base = base_cls(d, c, args, device)
        self.c = c
        self.hidden_channels = int(args.hidden_channels)
        self.readout = args.graph_readout
        self.node_source = args.graph_node_source
        self.dropout = float(getattr(args, "readout_dropout", args.dropout))
        node_dim = c if self.node_source in ("logits", "mediator_logits") else self.hidden_channels
        pool_mult = 2 if self.readout == "mean_max" else 1
        self._head_in_dim = node_dim * pool_mult
        self.graph_head = nn.Linear(self._head_in_dim, c)

    def reset_parameters(self):
        if hasattr(self.base, "reset_parameters"):
            self.base.reset_parameters()
        if self.graph_head is not None and hasattr(self.graph_head, "reset_parameters"):
            self.graph_head.reset_parameters()

    def _ensure_head(self, in_dim, device):
        if in_dim != self._head_in_dim:
            raise ValueError(
                f"Graph readout produced dim={in_dim}, but graph head expects "
                f"{self._head_in_dim}. Set --graph_node_source to a hidden representation "
                "or adjust the wrapper configuration."
            )
        self.graph_head = self.graph_head.to(device)

    def _forward_base_logits(self, data, training=False):
        try:
            output = self.base(data.x, data.edge_index, training=training)
        except TypeError:
            output = self.base(data.x, data.edge_index)

        aux = data.x.new_zeros(())
        if isinstance(output, tuple):
            node_logits = output[0]
            if len(output) > 1 and torch.is_tensor(output[1]) and output[1].dim() == 0:
                aux = output[1]
        else:
            node_logits = output
        return node_logits, aux

    def _extract_mlei_features(self, data, training=False):
        details = self.base(
            data.x,
            data.edge_index,
            training=training,
            return_details=True,
        )
        source = self.node_source
        if source == "auto":
            source = "fused_repr"
        node_feat = details.get(source)
        if node_feat is None:
            raise ValueError(f"MLEI details do not contain graph_node_source={source!r}.")

        aux = data.x.new_zeros(())
        if training and hasattr(self.base, "_regularization_loss"):
            mask = getattr(data, "node_train_mask", None)
            if mask is not None:
                mask = mask.to(device=data.x.device)
                local_regs = [
                    self.base._regularization_loss(env[mask], logits[mask])
                    for env, logits in zip(details["local_envs"], details["local_env_logits"])
                    if mask.any()
                ]
                if local_regs:
                    aux = aux + torch.stack(local_regs).mean()
                if mask.any():
                    aux = aux + self.base._regularization_loss(
                        details["global_env"][mask],
                        details["global_env_logits"][mask],
                    )
        return node_feat, aux

    def _extract_encoded_features(self, data, training=False):
        encoded = self.base.encode_representation(data.x, data.edge_index, training=training)
        aux = data.x.new_zeros(())
        if not isinstance(encoded, tuple):
            return encoded, aux

        if len(encoded) == 2 and torch.is_tensor(encoded[1]) and encoded[1].dim() == 0:
            return encoded[0], encoded[1]

        source = self.node_source
        if source == "auto":
            source = "z_mediator" if hasattr(self.base, "frontdoor_logits_from_contexts") else "z"
        source_map = {
            "z": 0,
            "edge_summary": 1,
            "z_mediator": 3,
            "z_spurious": 4,
            "mediator_logits": 5,
        }
        if source not in source_map:
            raise ValueError(
                f"Unsupported graph_node_source={source!r} for encoded tuple. "
                f"Use one of {sorted(source_map)}."
            )
        return encoded[source_map[source]], aux

    def node_features(self, data, training=False):
        if self.node_source == "logits":
            return self._forward_base_logits(data, training=training)
        if hasattr(self.base, "encode_representation"):
            return self._extract_encoded_features(data, training=training)
        try:
            return self._extract_mlei_features(data, training=training)
        except TypeError:
            return self._forward_base_logits(data, training=training)

    def forward(self, data, training=False):
        node_feat, aux = self.node_features(data, training=training)
        num_graphs = int(data.y.size(0))
        graph_feat = pool_nodes(
            node_feat,
            data.batch,
            num_graphs,
            self.readout,
            root_index=getattr(data, "root_index", None),
        )
        graph_feat = F.dropout(graph_feat, p=self.dropout, training=training)
        self._ensure_head(graph_feat.size(-1), graph_feat.device)
        return self.graph_head(graph_feat), aux

    def loss_compute(self, data, criterion, args):
        if not hasattr(data, "node_train_mask"):
            data.node_train_mask = torch.ones(data.x.size(0), device=data.x.device, dtype=torch.bool)
        logits, aux = self.forward(data, training=True)
        if hasattr(data, "train_idx"):
            train_idx = data.train_idx.to(device=logits.device, dtype=torch.long)
        else:
            train_idx = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        target = data.y[train_idx].squeeze(1).long()
        loss = criterion(logits.index_select(0, train_idx), target)
        aux_weight = args.graph_aux_weight if args.graph_aux_weight >= 0.0 else args.lamda
        if torch.is_tensor(aux):
            loss = loss + aux_weight * aux
        return loss


def eval_acc(y_true, logits):
    pred = logits.argmax(dim=-1, keepdim=True)
    return (pred == y_true).float().mean().item()


def iter_chunks(indices, batch_size, shuffle=False, seed=0):
    values = [int(idx) for idx in indices]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(values)
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def make_graph_batch(graphs, graph_indices, device, training=False):
    batch = Batch.from_data_list([graphs[idx] for idx in graph_indices])
    batch = batch.to(device)
    batch.graph_count = int(batch.y.size(0))
    if training:
        batch.node_train_mask = torch.ones(batch.x.size(0), device=device, dtype=torch.bool)
    return batch


@torch.no_grad()
def evaluate_indices_minibatch(model, graphs, graph_indices, batch_size, device):
    if len(graph_indices) == 0:
        return 0.0
    model.eval()
    correct = 0
    total = 0
    for chunk in iter_chunks(graph_indices, batch_size, shuffle=False):
        batch = make_graph_batch(graphs, chunk, device, training=False)
        logits, _ = model(batch, training=False)
        pred = logits.argmax(dim=-1, keepdim=True)
        target = batch.y.to(device=logits.device)
        correct += int((pred == target).sum().item())
        total += int(target.numel())
    return correct / max(total, 1)


@torch.no_grad()
def evaluate(model, data, args=None, device=None):
    batch_size = int(getattr(args, "batch_size", 0)) if args is not None else 0
    if batch_size > 0:
        if device is None:
            device = next(model.parameters()).device
        result = [
            evaluate_indices_minibatch(model, data.graphs, data.train_idx.tolist(), batch_size, device),
            evaluate_indices_minibatch(model, data.graphs, data.valid_idx.tolist(), batch_size, device),
            evaluate_indices_minibatch(model, data.graphs, data.test_in_idx.tolist(), batch_size, device),
        ]
        for idx in data.test_ood_idx:
            result.append(evaluate_indices_minibatch(model, data.graphs, idx.tolist(), batch_size, device))
        return result

    model.eval()
    logits, _ = model(data, training=False)
    result = [
        eval_acc(data.y[data.train_idx], logits[data.train_idx]),
        eval_acc(data.y[data.valid_idx], logits[data.valid_idx]),
        eval_acc(data.y[data.test_in_idx], logits[data.test_in_idx]),
    ]
    for idx in data.test_ood_idx:
        result.append(eval_acc(data.y[idx], logits[idx]))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Run CaNet-style OOD models on KPG rumor datasets.")
    parser.add_argument("--canet_dir", type=str, default="/public/wc/CaNet-main")
    parser.add_argument("--kpg_dir", type=str, default="/public/wc/KPG-main")
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--model_module", type=str, default="model_canet")
    parser.add_argument("--model_class", type=str, default="CaNet")
    parser.add_argument(
        "--model_specs",
        type=str,
        default="",
        help="Comma-separated specs like canet=model_canet:CaNet,mlei=model_MLEI:MLEI,graph_cfam_nego=model_gmm3_reviewed1_graph_cfam_nego:GraphFrontDoorDAG.",
    )
    parser.add_argument("--rumor_datasets", type=str, default="Twitter15,Twitter16,Pheme")
    parser.add_argument("--train_datasets", type=str, default="Twitter15,Twitter16")
    parser.add_argument("--test_datasets", type=str, default="")
    parser.add_argument("--feat_dim", type=int, default=5000)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_graphs", type=int, default=0)
    parser.add_argument("--max_valid_graphs", type=int, default=0)
    parser.add_argument("--max_test_in_graphs", type=int, default=0)
    parser.add_argument("--max_ood_graphs", type=int, default=0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Graph mini-batch size. Use 1 for one graph at a time; 0 keeps the disconnected full-batch graph.",
    )
    parser.add_argument(
        "--graph_readout",
        type=str,
        default="mean_max",
        choices=["mean", "max", "sum", "mean_max", "root"],
    )
    parser.add_argument(
        "--graph_node_source",
        type=str,
        default="auto",
        help="Node representation to pool. auto uses fused_repr for MLEI and z_mediator for GraphFrontDoorDAG.",
    )
    parser.add_argument("--readout_dropout", type=float, default=0.0)
    parser.add_argument(
        "--graph_aux_weight",
        type=float,
        default=-1.0,
        help="Weight for model-specific node/environment auxiliary loss; negative reuses --lamda.",
    )
    parser.add_argument("--undirected", action="store_true", default=True)
    parser.add_argument("--directed", dest="undirected", action="store_false")
    parser.add_argument("--save_model_path", type=str, default="")

    sys.path.insert(0, "/public/wc/CaNet-main")
    from parse import parser_add_main_args

    parser_add_main_args(parser)
    parser.set_defaults(dataset="rumor")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.dataset = "rumor"
    fix_seed(args.seed)

    if args.canet_dir not in sys.path:
        sys.path.insert(0, args.canet_dir)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.device}")
    data = load_rumor_platform_dataset(args)
    c = int(data.y[data.y >= 0].max().item()) + 1
    d = data.x.size(1)
    args.env_num = data.env_num
    args.train_env_num = data.train_env_num

    print(
        f"rumor graph OOD data: graphs={data.graph_count} nodes={data.num_nodes} edges={data.edge_index.size(1)} "
        f"features={d} classes={c}"
    )
    print(
        f"train envs={data.train_env_names} test envs={data.test_env_names} "
        f"train={data.train_idx.numel()} valid={data.valid_idx.numel()} test_in={data.test_in_idx.numel()}"
    )
    for name, idx, full_idx in zip(data.test_env_names, data.test_ood_idx, data.test_ood_full_idx):
        suffix = "" if idx.numel() == full_idx.numel() else f" sampled_from={full_idx.numel()}"
        print(f"test_ood/{name}={idx.numel()}{suffix}")

    model_specs = parse_model_specs(args)
    criterion = nn.CrossEntropyLoss()

    full_batch = int(args.batch_size) <= 0
    if full_batch:
        data = data.to(device)

    for model_name, module_name, class_name in model_specs:
        print(f"\n=== Model: {model_name} ({module_name}:{class_name}) ===")
        module = importlib.import_module(module_name)
        base_cls = getattr(module, class_name)
        best_valid = -1.0
        best_result = None
        all_results = []

        for run in range(args.runs):
            fix_seed(args.seed + run)
            model = GraphReadoutClassifier(base_cls, d, c, args, device).to(device)
            if hasattr(model, "reset_parameters"):
                model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            for epoch in range(args.epochs):
                model.train()
                if full_batch:
                    optimizer.zero_grad()
                    loss_out = model.loss_compute(data, criterion, args)
                    loss = loss_out[0] if isinstance(loss_out, (tuple, list)) else loss_out
                    loss.backward()
                    optimizer.step()
                else:
                    total_loss = 0.0
                    total_graphs = 0
                    for chunk in iter_chunks(
                        data.train_idx.tolist(),
                        int(args.batch_size),
                        shuffle=True,
                        seed=args.seed + run * 100000 + epoch,
                    ):
                        batch = make_graph_batch(data.graphs, chunk, device, training=True)
                        optimizer.zero_grad()
                        loss_out = model.loss_compute(batch, criterion, args)
                        loss = loss_out[0] if isinstance(loss_out, (tuple, list)) else loss_out
                        loss.backward()
                        optimizer.step()
                        total_loss += float(loss.detach()) * len(chunk)
                        total_graphs += len(chunk)
                    loss = torch.tensor(total_loss / max(total_graphs, 1), device=device)

                result = evaluate(model, data, args=args, device=device)
                if result[1] > best_valid:
                    best_valid = result[1]
                    best_result = result
                    if args.save_model_path:
                        root, ext = os.path.splitext(args.save_model_path)
                        save_path = args.save_model_path
                        if len(model_specs) > 1:
                            save_path = f"{root}_{model_name}{ext or '.pt'}"
                        save_dir = os.path.dirname(save_path)
                        if save_dir:
                            os.makedirs(save_dir, exist_ok=True)
                        torch.save(model.state_dict(), save_path)

                if epoch % args.display_step == 0:
                    msg = (
                        f"{model_name} Run {run:02d} Epoch {epoch:03d} Loss {float(loss):.4f} "
                        f"Train {result[0] * 100:.2f}% Valid {result[1] * 100:.2f}% "
                        f"TestIn {result[2] * 100:.2f}%"
                    )
                    for name, value in zip(data.test_env_names, result[3:]):
                        msg += f" OOD-{name} {value * 100:.2f}%"
                    print(msg)

            all_results.append(best_result)

        mean_result = np.mean(np.asarray(all_results), axis=0)
        msg = (
            f"{model_name} Mean best-valid result: Train {mean_result[0] * 100:.2f}% "
            f"Valid {mean_result[1] * 100:.2f}% TestIn {mean_result[2] * 100:.2f}%"
        )
        for name, value in zip(data.test_env_names, mean_result[3:]):
            msg += f" OOD-{name} {value * 100:.2f}%"
        print(msg)


if __name__ == "__main__":
    main()
