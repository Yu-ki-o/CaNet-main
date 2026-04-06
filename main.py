#针对于纯ciw的实现

import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from logger import Logger
from dataset import *
from data_utils import normalize, gen_normalized_adjs, to_sparse_tensor, \
    load_fixed_splits, rand_splits, get_gpu_memory_map, count_parameters, reindex_env
from eval import evaluate_full, eval_acc, eval_rocauc, eval_f1
from parse import parser_add_main_args
from model import GraphCIW # 仅导入重构后的 GraphCIW
import time
import torch.optim as optim
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

# 将数据集中的训练环境数传递给 args，供 GraphCIW 初始化原型（Prototypes）矩阵使用
args.train_env_num = dataset.train_env_num 

model = GraphCIW(d, c, args, device).to(device)

# 【关键修正】: 将 reduction 改为 'none'。
# 因为 GraphCIW 需要逐节点进行独立性加权（Sample Weighting），必须保留 loss 的 Instance 维度
if args.dataset in ('elliptic', 'twitch'):
    # 【终极补丁：类别不平衡对抗机制】
        # 方案一：通过 argparse 手动控制（推荐，方便你后续写脚本跑 Grid Search）
        # 如果 args 里没有传 pos_weight，默认给个 5.0 (意味着把正样本的 Loss 放大 5 倍)
    weight_val = getattr(args, 'pos_weight', 5.0)   
    pos_weight_tensor = torch.tensor([weight_val]).to(device)
        
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_tensor)
    # criterion = nn.BCEWithLogitsLoss(reduction='none')
else:
    criterion = nn.CrossEntropyLoss(reduction='none')

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
    
    # 隔离 DAG 参数，避免 L2 weight_decay 破坏其稀疏性
    dag_param_names = ['A', 'M_dag.weight']
    base_params = [p for n, p in model.named_parameters() if n not in dag_param_names]
    dag_params = [p for n, p in model.named_parameters() if n in dag_param_names]

    optimizer = torch.optim.Adam([
        {'params': base_params, 'weight_decay': args.weight_decay},
        {'params': dag_params, 'weight_decay': 0.0}
    ], lr=args.lr)
    
    best_val = float('-inf')

    dataset.x, dataset.y, dataset.edge_index, dataset.env = \
        dataset.x.to(device), dataset.y.to(device), dataset.edge_index.to(device), dataset.env.to(device)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        # loss = model.loss_compute(dataset, criterion, args)
        # loss.backward()
        total_loss, l_cls, l_ind, l_dag, l_cl = model.loss_compute(dataset, criterion, args)
        total_loss.backward()
        optimizer.step()

        result = evaluate_full(model, dataset, eval_func)
        logger.add_result(run, result)

        tr_acc.append(result[0])
        val_acc.append(result[2])

        if epoch % args.display_step == 0:
            #新增代码3：在打印信息中加入当前的真实 LR
            # current_lr = optimizer.param_groups[0]['lr']

            # 【核心修改】：在控制台极其直观地打印出 4 个 Loss 的实时博弈情况！
            loss_str = f"Loss [Total: {total_loss.item():.4f} | Cls: {l_cls:.4f} | Ind: {l_ind:.6f} | DAG: {l_dag:.4f} | CL: {l_cl:.4f}]"

            m = f'Epoch: {epoch:02d}, Loss: {loss_str}\n\tTrain: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test In: {100 * result[2]:.2f}% '
            for i in range(len(result)-3):
                m += f'Test OOD{i+1}: {100 * result[i+3]:.2f}% '
            print(m)
    logger.print_statistics(run)

logger.print_statistics()
if args.store_result:
    logger.output(args)