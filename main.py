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
from torch.utils.tensorboard import SummaryWriter  

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
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/{args.dataset}/{current_time}_run{run}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] TensorBoard log directory: {log_dir}")
    
    dag_param_names = ['A', 'M_dag.weight', 'dag_gate','log_vars']
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
        
        warmup_epochs = 100.0
        anneal_rate = min(1.0, epoch / warmup_epochs)
        model.lambda_l1 = getattr(args, 'lambda_l1', 1e-5) * (0.01 + 0.99 * anneal_rate)
        model.lambda_dag = getattr(args, 'lambda_dag', 0.1) * (0.01 + 0.99 * anneal_rate)
        
        optimizer_dag.zero_grad()
        optimizer_model.zero_grad()

        # ==========================================================
        # 🌟 修复：单次前向传播！保证所有梯度都基于同一个子采样 Batch！
        # ==========================================================
        loss_model_opt, loss_dag_opt, l_cls, l_dag, l_cl = model.loss_compute(dataset, criterion, args)
        
        # 阶段一：仅优化全局 DAG 物理结构，保留计算图供模型使用
        loss_dag_opt.backward(retain_graph=True) 
        torch.nn.utils.clip_grad_norm_(dag_params, max_norm=2.0)
        optimizer_dag.step()
        
        # 阶段二：联合优化 Transformer 表征重组与分类器
        loss_model_opt.backward()
        torch.nn.utils.clip_grad_norm_(base_params, max_norm=2.0)
        optimizer_model.step()

        # 模型评估
        result = evaluate_full(model, dataset, eval_func)
        logger.add_result(run, result)

        tr_acc.append(result[0])
        val_acc.append(result[2])
        
        # ==========================================================
        # 🌟 修复：增广拉格朗日外部更新策略
        # ==========================================================
        h_A_val = model.current_h_A.item()
        
        # 频率控制：每 50 个 Epoch 进行一次评估 (防止 rho 瞬间爆炸)
        if epoch > 0 and epoch % 50 == 0:
            # 宽容度控制：只要环状结构没有明显下降 (下降不足 10%)，就温和惩罚 (x2)
            if h_A_val > 0.90 * model.h_A_last.item():
                model.rho *= 2.0
            
            # 更新拉格朗日乘子
            model.lagrange_multiplier += model.rho * h_A_val
            model.h_A_last.fill_(h_A_val)
            
            # 上限控制：防止 rho 过大引发计算图 NaN 崩溃
            model.rho.clamp_(max=1e6)


        # TensorBoard 监控
        writer.add_scalar('Loss_Optimizer/Model_Total', loss_model_opt.item(), epoch)
        writer.add_scalar('Loss_Optimizer/DAG_Total', loss_dag_opt.item(), epoch)
        writer.add_scalar('Loss_Detail/Classification', l_cls, epoch)
        writer.add_scalar('Loss_Detail/DAG_Reconstruct', l_dag, epoch)
        writer.add_scalar('Loss_Detail/Contrastive', l_cl, epoch)
        
        writer.add_scalar('Performance/Train', result[0], epoch)
        writer.add_scalar('Performance/Valid', result[1], epoch)
        writer.add_scalar('Performance/Test_In', result[2], epoch)
        
        for i in range(len(result)-3):
            writer.add_scalar(f'Performance/Test_OOD{i+1}', result[i+3], epoch)

        # 终端控制台打印
        if epoch % args.display_step == 0:
            total_loss_val = loss_model_opt.item() + loss_dag_opt.item()
            loss_str = f"Loss [Total: {total_loss_val:.4f} | Cls: {l_cls:.4f} | DAG: {l_dag:.4f} | CL: {l_cl:.4f}]"

            m = f'Epoch: {epoch:02d}, {loss_str}\n\tTrain: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test In: {100 * result[2]:.2f}% '
            for i in range(len(result)-3):
                m += f'Test OOD{i+1}: {100 * result[i+3]:.2f}% '
            print(m)
            
    logger.print_statistics(run)

logger.print_statistics()
if args.store_result:
    logger.output(args)