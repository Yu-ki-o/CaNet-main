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
from model_test import *
import time
import os
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter # 新增 TensorBoard

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


# ==========================================
# [新增] TensorBoard 目录配置与初始化
# ==========================================
# 自动创建形如: ./runs/cora/May13_14-30-00_lamda_1.0/ 的目录
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
# 这里我帮你把 lamda 的值也放进了文件夹名，方便你在面板里区分消融实验！
run_name = f"{current_time}_lamda_{args.lamda}" 
log_dir = os.path.join('.', 'runs', args.dataset, run_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logging activated. Logs will be saved to: {log_dir}")
# ==========================================


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


        # ==========================================
        # [新增] 向 TensorBoard 写入当前 Epoch 的数据
        # 考虑到 args.runs 可能会跑多次实验，我们用 global_step = run * args.epochs + epoch 来连续记录
        # ==========================================
        global_step = run * args.epochs + epoch
        # 记录训练 Loss
        writer.add_scalar('Loss/Train_Loss', loss.item(), global_step)
        # 记录各项评估指标 (根据你原本 result 返回的顺序)
        # result[0]: Train, result[1]: Valid, result[2]: Test_In
        writer.add_scalar('Metrics/1_Train', result[0] * 100, global_step)
        writer.add_scalar('Metrics/2_Valid', result[1] * 100, global_step)
        writer.add_scalar('Metrics/3_Test_In', result[2] * 100, global_step)
        # 动态记录多个 OOD 测试集的表现 (result[3:])
        for i in range(len(result) - 3):
            writer.add_scalar(f'Metrics/4_Test_OOD_{i+1}', result[i+3] * 100, global_step)
        # ==========================================



        if epoch % args.display_step == 0:
            m = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test In: {100 * result[2]:.2f}% '
            for i in range(len(result)-3):
                m += f'Test OOD{i+1}: {100 * result[i+3]:.2f}% '
            print(m)
    logger.print_statistics(run)


logger.print_statistics()
if args.store_result:
    logger.output(args)



# [新增] 关闭 TensorBoard 记录器
writer.close()
print("[INFO] TensorBoard writer closed.")