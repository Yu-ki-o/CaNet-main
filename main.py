#针对于纯ciw的实现

import argparse
import datetime
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
from model import GraphCIW 
import time
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

if args.dataset == 'twitch':
    dataset = load_twitch_dataset(args.data_dir, train_num=3)
elif args.dataset == 'elliptic':
    dataset = load_elliptic_dataset(args.data_dir, train_num=5)
elif args.dataset == 'arxiv':
    dataset = load_arxiv_dataset(args.data_dir, train_num=3)
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

is_multilabel = args.dataset in ('proteins', 'ppi')
args.train_env_num = dataset.train_env_num 

model = GraphCIW(d, c, args, device).to(device)

if args.dataset in ('elliptic', 'twitch'):
    weight_val = getattr(args, 'pos_weight', 5.0)   
    pos_weight_tensor = torch.tensor([weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_tensor)
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

for run in range(args.runs):
    model.reset_parameters()
    # ---------------------------------------------------------
    # 初始化 TensorBoard Writer (按数据集、时间戳和 run 序号归类)
    # ---------------------------------------------------------
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/{args.dataset}/{current_time}_run{run}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] TensorBoard log directory: {log_dir}")
    
    # 【整改点 1】：必须把 dag_gate 纳入因果图的专属优化器
    dag_param_names = ['A', 'M_dag.weight', 'dag_gate']
    base_params = [p for n, p in model.named_parameters() if n not in dag_param_names]
    dag_params = [p for n, p in model.named_parameters() if n in dag_param_names]

    optimizer_model = torch.optim.Adam(base_params, lr=args.lr, weight_decay=args.weight_decay)
    
    lr_dag = getattr(args, 'lr_dag', args.lr)
    optimizer_dag = torch.optim.Adam(dag_params, lr=lr_dag, weight_decay=0.0)
    
    best_val = float('-inf')

    dataset.x, dataset.y, dataset.edge_index, dataset.env = \
        dataset.x.to(device), dataset.y.to(device), dataset.edge_index.to(device), dataset.env.to(device)

    for epoch in range(args.epochs):
        model.train()
        
        # ==========================================
        # 阶段一：冻结主干网络，专属训练因果拓扑图
        # ==========================================
        # 第 1 次前向传播：使用当前的主干网络提取特征
        _, loss_dag_opt, _, _, _, _ = model.loss_compute(dataset, criterion, args)
        
        optimizer_dag.zero_grad()
        # 此时只需单独 backward，不需要 retain_graph 了，因为计算图马上要重构
        loss_dag_opt.backward() 
        torch.nn.utils.clip_grad_norm_(dag_params, max_norm=2.0)
        optimizer_dag.step()
        
        # ==========================================
        # 阶段二：冻结因果图，训练主干网络与分类器
        # ==========================================
        # 第 2 次前向传播：这是论文的核心！
        # 此时模型内部使用的是刚刚 step() 更新过、具备更强跨域不变提取能力的 A 矩阵！
        loss_model_opt, _, l_cls, l_ind, l_dag, l_cl = model.loss_compute(dataset, criterion, args)
        
        optimizer_model.zero_grad()
        loss_model_opt.backward()
        torch.nn.utils.clip_grad_norm_(base_params, max_norm=2.0)
        optimizer_model.step()

        result = evaluate_full(model, dataset, eval_func)
        logger.add_result(run, result)

        tr_acc.append(result[0])
        val_acc.append(result[2])
        
        # 1. 记录各种 Loss
        writer.add_scalar('Loss_Optimizer/Model_Total', loss_model_opt.item(), epoch)
        writer.add_scalar('Loss_Optimizer/DAG_Total', loss_dag_opt.item(), epoch)
        writer.add_scalar('Loss_Detail/Classification', l_cls, epoch)
        writer.add_scalar('Loss_Detail/Independence', l_ind, epoch)
        writer.add_scalar('Loss_Detail/DAG_Reconstruct', l_dag, epoch)
        writer.add_scalar('Loss_Detail/Contrastive', l_cl, epoch)
        
        # 2. 记录各项评测指标
        writer.add_scalar('Performance/Train', result[0], epoch)
        writer.add_scalar('Performance/Valid', result[1], epoch)
        writer.add_scalar('Performance/Test_In', result[2], epoch)
        
        for i in range(len(result)-3):
            writer.add_scalar(f'Performance/Test_OOD{i+1}', result[i+3], epoch)

        if epoch % args.display_step == 0:
            total_loss_val = loss_model_opt.item() + loss_dag_opt.item()
            loss_str = f"Loss [Total: {total_loss_val:.4f} | Cls: {l_cls:.4f} | Ind: {l_ind:.6f} | DAG: {l_dag:.4f} | CL: {l_cl:.4f}]"

            m = f'Epoch: {epoch:02d}, {loss_str}\n\tTrain: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test In: {100 * result[2]:.2f}% '
            for i in range(len(result)-3):
                m += f'Test OOD{i+1}: {100 * result[i+3]:.2f}% '
            print(m)
            
    logger.print_statistics(run)

logger.print_statistics()
if args.store_result:
    logger.output(args)