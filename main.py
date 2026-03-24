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
from model import *
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

# ==================== 核心创新点：ICA 因果特征解耦 ====================
from sklearn.decomposition import FastICA
import warnings
import time

print(f"\n[ICA前] 原始节点特征维度: {dataset.x.shape}")
start_time = time.time()

# 1. 提取全局特征，并强制转换为 float64 防止溢出
original_x = dataset.x.cpu().numpy().astype(np.float64)

# 【关键】：提取仅属于训练集 (In-Distribution) 的特征，ICA 只能在这个集合上学习！
train_idx = dataset.train_idx.cpu().numpy()
train_x = original_x[train_idx]

# 2. 清洗可能存在的 NaN 和 Inf
original_x = np.nan_to_num(original_x, nan=0.0, posinf=0.0, neginf=0.0)
train_x = np.nan_to_num(train_x, nan=0.0, posinf=0.0, neginf=0.0)

# 3. 靶向修复极低方差特征（严格基于 train_x 的统计量来判断）
stds = np.std(train_x, axis=0)
bad_cols = stds < 1e-5
if bad_cols.any():
    print(f"[警告] 训练集中检测到 {bad_cols.sum()} 个极低方差特征，施加靶向高斯扰动...")
    # 对训练集和全局分别施加扰动，防止除零
    train_x[:, bad_cols] += np.random.normal(0, 1e-2, size=(train_x.shape[0], bad_cols.sum()))
    original_x[:, bad_cols] += np.random.normal(0, 1e-2, size=(original_x.shape[0], bad_cols.sum()))

# 4. 手动 Z-score 标准化：【必须严格使用 train_x 的均值和方差】
train_mean = train_x.mean(axis=0)
train_std = train_x.std(axis=0) + 1e-5

train_x = (train_x - train_mean) / train_std
original_x = (original_x - train_mean) / train_std # 测试集必须复用训练集的 mean 和 std

n_components = min(train_x.shape[1], 512) 

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ica = FastICA(n_components=n_components, 
                  random_state=42, 
                  max_iter=2000, 
                  tol=1e-3, 
                  whiten='unit-variance')
    
    # 5. 【学术严谨核心】：ICA 只能在训练集上 fit
    print("[INFO] 正在 In-Distribution (训练集) 特征上拟合 ICA 规则...")
    ica.fit(train_x)
    
    # 6. 利用学到的解耦矩阵，去 transform 包含测试集的整张图
    print("[INFO] 正在将 ICA 解耦规则泛化(transform)到全图...")
    disentangled_x = ica.transform(original_x)

# 换回 float32，无缝衔接 PyTorch
dataset.x = torch.tensor(disentangled_x, dtype=torch.float32)
print(f"[ICA后] 解耦特征维度: {dataset.x.shape}, 耗时: {time.time() - start_time:.2f} 秒\n")
# ======================================================================

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
# 监控自适应融合模块学到的最终权重比例
if args.env_type in ['local_global', 'combined_vn']:
    local_weight, global_weight = model.env_enc[0].get_fusion_weights()
    module_name = "Transformer" if args.env_type == 'local_global' else "Virtual Node"
    
    print(f"\n[{args.dataset} 数据集] Run {run} 最终信息分配倾向 ({args.env_type})：")
    print(f"  ➜ 局部结构 (GCN) 权重: {local_weight*100:.2f}%")
    print(f"  ➜ 全局宏观 ({module_name}) 权重: {global_weight*100:.2f}%\n")
if args.store_result:
    logger.output(args)