import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
from torch_geometric.data import ShaDowKHopSampler

from logger import Logger
from dataset import *
from data_utils import normalize, gen_normalized_adjs, to_sparse_tensor, \
    load_fixed_splits, rand_splits, get_gpu_memory_map, count_parameters, reindex_env
from eval import evaluate_full, eval_acc, eval_rocauc, eval_f1
from parse import parser_add_main_args
from model_canet_layerwise_node_enhance import *
import time


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
parser.add_argument('--edge_score_temp', type=float, default=2.0)
parser.add_argument('--edge_blend', type=float, default=0.2)
parser.add_argument(
    '--edge_feat_mode',
    type=str,
    default='mul',
    choices=[
        'mul',
        'diff',
        'signed_diff',
        'degree',
        'mul_diff',
        'mul_signed_diff',
        'concat',
        'concat_diff',
        'mul_degree',
        'diff_degree',
        'mul_diff_degree',
        'mul_signed_diff_degree',
    ],
)
parser.add_argument('--edge_gate_mode', type=str, default='vector', choices=['scalar', 'vector'])
parser.add_argument('--use_layerwise_local_igm', action='store_true')
parser.add_argument(
    '--layerwise_local_igm_include_last',
    action='store_false',
    dest='layerwise_local_igm_skip_last',
)
parser.set_defaults(layerwise_local_igm_skip_last=True)
parser.add_argument(
    '--disable_layerwise_final_edge_fuse',
    action='store_false',
    dest='layerwise_final_edge_fuse',
)
parser.set_defaults(layerwise_final_edge_fuse=True)
parser.add_argument('--layerwise_gate_target', type=float, default=0.5)
parser.add_argument('--lambda_layerwise_gate', type=float, default=0.0)
parser.add_argument('--lambda_enhance_sem', type=float, default=0.0)
parser.add_argument('--enhance_sem_mode', type=str, default='cosine', choices=['cosine', 'mse'])
parser.add_argument('--disable_node_edge_norm', action='store_true')
parser.add_argument('--use_graph_cfam', action='store_true')
parser.add_argument(
    '--disable_final_graph_cfam',
    action='store_false',
    dest='use_final_graph_cfam',
)
parser.set_defaults(use_final_graph_cfam=True)
parser.add_argument('--graph_cfam_residual_blend', type=float, default=0.1)
parser.add_argument('--use_pre_gnn_graph_cfam', action='store_true')
parser.add_argument('--pre_graph_cfam_blend', type=float, default=0.1)
parser.add_argument('--pre_graph_cfam_residual_blend', type=float, default=0.0)
parser.add_argument('--graph_cfam_gate_temp', type=float, default=1.0)
parser.add_argument('--graph_cfam_gate_target', type=float, default=0.5)
parser.add_argument('--lambda_graph_cfam_gate', type=float, default=0.0)
parser.add_argument('--lambda_graph_delf', type=float, default=0.0)
parser.add_argument('--graph_delf_top_frac', type=float, default=0.2)
parser.add_argument('--graph_delf_margin', type=float, default=0.2)
parser.add_argument('--graph_delf_shortcut_weight', type=float, default=0.5)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
# multi-graph datasets, divide graphs into train/valid/test
if args.dataset == 'twitch':
    dataset = load_twitch_dataset(args.data_dir, train_num=3)
elif args.dataset == 'elliptic':
    dataset = load_elliptic_dataset(args.data_dir, train_num=5)
# single-graph datasets, divide nodes into train/valid/test
elif args.dataset == 'arxiv':
    dataset = load_arxiv_dataset(args.data_dir, train_num=3)
# synthetic datasets, add spurious node features
elif args.dataset in ('cora', 'citeseer', 'pubmed'):
    dataset = load_synthetic_dataset(args.data_dir, args.dataset, train_num=3, combine=args.combine_result)
else:
    raise ValueError('Invalid dataname')

if len(dataset.y.shape) == 1:
    dataset.y = dataset.y.unsqueeze(1)

c = max(dataset.y.max().item() + 1, dataset.y.shape[1])
d = dataset.x.shape[1]
n = dataset.num_nodes

print(f"dataset {args.dataset}: all nodes {dataset.num_nodes} | edges {dataset.edge_index.size(1)} | "
      + f"classes {c} | feats {d}")
print(f"train nodes {dataset.train_idx.shape[0]} | valid nodes {dataset.valid_idx.shape[0]} | "
      f"test in nodes {dataset.test_in_idx.shape[0]}")
m = ""
for i in range(len(dataset.test_ood_idx)):
    m += f"test ood{i+1} nodes {dataset.test_ood_idx[i].shape[0]} "
print(m)
print(f'[INFO] env numbers: {dataset.env_num} train env numbers: {dataset.train_env_num}')

### Load method ###
is_multilabel = args.dataset in ('proteins', 'ppi')

model = CaNet(d, c, args, device).to(device)

if args.dataset in ('elliptic', 'twitch'):
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
else:
    criterion = nn.CrossEntropyLoss(reduction='mean')

if args.dataset in ('twitch'):
    eval_func = eval_rocauc
elif args.dataset in ('elliptic'):
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

tr_acc, val_acc = [], []

### Training loop ###
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float('-inf')

    dataset.x, dataset.y, dataset.edge_index, dataset.env = \
        dataset.x.to(device), dataset.y.to(device), dataset.edge_index.to(device), dataset.env.to(device)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss_compute(dataset, criterion, args)
        loss.backward()
        optimizer.step()
        result = evaluate_full(model, dataset, eval_func)
        logger.add_result(run, result)

        tr_acc.append(result[0])
        val_acc.append(result[2])

        if epoch % args.display_step == 0:
            m = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test In: {100 * result[2]:.2f}% '
            for i in range(len(result)-3):
                m += f'Test OOD{i+1}: {100 * result[i+3]:.2f}% '
            print(m)
    logger.print_statistics(run)


logger.print_statistics()
if args.store_result:
    logger.output(args)
