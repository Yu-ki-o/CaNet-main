import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from eval import eval_acc, eval_f1, eval_rocauc, evaluate_full
from ica_utils import infer_pseudo_envs_with_ica
from logger import Logger
from model_ori_ciw import GraphCIW
from parse import parser_add_main_args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def add_ciw_args(parser):
    parser.add_argument('--rff_dim', type=int, default=128,
                        help='random Fourier feature dimension used in CIW independence estimation')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='EMA coefficient for CIW global memory')
    parser.add_argument('--global_size', type=int, default=2048,
                        help='size of the global memory used by CIW')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='EMA momentum for CIW prototypes')
    parser.add_argument('--lambda_l1', type=float, default=1e-5,
                        help='L1 sparsity weight inside the DAG regularizer')
    parser.add_argument('--lambda_ind', type=float, default=0.1,
                        help='weight of the CIW independence loss')
    parser.add_argument('--lambda_cl', type=float, default=0.1,
                        help='weight of the CIW contrastive loss')
    parser.add_argument('--lambda_dag', type=float, default=0.1,
                        help='weight of the CIW DAG reconstruction loss')
    parser.add_argument('--lambda_med', type=float, default=0.5,
                        help='weight of the mediator-only supervision loss')
    parser.add_argument('--lambda_spu', type=float, default=0.1,
                        help='weight of the spurious-branch uniformity loss')
    parser.add_argument('--lambda_fd', type=float, default=0.5,
                        help='weight of the front-door aggregation loss')
    parser.add_argument('--lambda_var', type=float, default=0.05,
                        help='weight of the cross-environment front-door variance penalty')
    parser.add_argument('--mediator_temp', type=float, default=8.0,
                        help='temperature of the DAG-based soft mediator selector')
    parser.add_argument('--low_temp', type=float, default=8.0,
                        help='temperature used to detect low-label-effect features')
    parser.add_argument('--low_threshold', type=float, default=0.35,
                        help='threshold for identifying low-label-effect features')
    parser.add_argument('--mediator_threshold', type=float, default=0.5,
                        help='threshold for activating mediator dimensions from the DAG effect scores')
    parser.add_argument('--pollution_coeff', type=float, default=1.0,
                        help='penalty coefficient for feature pollution from low-effect nodes')
    parser.add_argument('--fd_blend', type=float, default=0.5,
                        help='blend ratio between mediator logits and front-door aggregated logits')


def sanitize_name(name):
    safe_name = "".join(
        ch if ch.isalnum() or ch in ('-', '_', '.') else '_'
        for ch in str(name).strip()
    ).strip('._')
    return safe_name


def set_requires_grad(parameters, flag):
    for parameter in parameters:
        parameter.requires_grad_(flag)


parser = argparse.ArgumentParser(description='GraphCIW Training Pipeline')
parser_add_main_args(parser)
add_ciw_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

if args.dataset == 'twitch':
    dataset = load_twitch_dataset(args.data_dir, train_num=3)
elif args.dataset == 'elliptic':
    dataset = load_elliptic_dataset(args.data_dir, train_num=5)
elif args.dataset == 'arxiv':
    dataset = load_arxiv_dataset(args.data_dir, train_num=3)
elif args.dataset in ('cora', 'citeseer', 'pubmed'):
    dataset = load_synthetic_dataset(
        args.data_dir,
        args.dataset,
        train_num=3,
        combine=args.combine_result,
    )
else:
    raise ValueError('Invalid dataname')

if args.infer_env:
    dataset = infer_pseudo_envs_with_ica(
        dataset,
        env_num=args.infer_env_num,
        n_components=args.infer_env_components,
        num_iters=args.infer_env_iters,
        seed=args.seed,
    )

if len(dataset.y.shape) == 1:
    dataset.y = dataset.y.unsqueeze(1)

args.train_env_num = int(dataset.train_env_num)

c = max(dataset.y.max().item() + 1, dataset.y.shape[1])
d = dataset.x.shape[1]

print(
    f"dataset {args.dataset}: all nodes {dataset.num_nodes} | edges {dataset.edge_index.size(1)} | "
    f"classes {c} | feats {d}"
)
print(
    f"train nodes {dataset.train_idx.shape[0]} | valid nodes {dataset.valid_idx.shape[0]} | "
    f"test in nodes {dataset.test_in_idx.shape[0]}"
)
m = ""
for i in range(len(dataset.test_ood_idx)):
    m += f"test ood{i + 1} nodes {dataset.test_ood_idx[i].shape[0]} "
print(m)
print(f'[INFO] env numbers: {dataset.env_num} train env numbers: {dataset.train_env_num}')

model = GraphCIW(d, c, args, device).to(device)

if args.dataset in ('elliptic', 'twitch'):
    pos_weight = torch.full((c,), float(args.pos_weight), device=device)
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
else:
    criterion = nn.CrossEntropyLoss(reduction='none')

if args.dataset == 'twitch':
    eval_func = eval_rocauc
elif args.dataset == 'elliptic':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)
print('MODEL:', model)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
run_name = sanitize_name(args.result_name)
if not run_name:
    run_name = (
        f"{current_time}_ciw_ind_{args.lambda_ind}_dag_{args.lambda_dag}"
        f"_cl_{args.lambda_cl}_fd_{args.lambda_fd}"
    )
log_dir = os.path.join('.', 'runs', args.dataset, 'ciw', run_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logging activated. Logs will be saved to: {log_dir}")

dataset.x = dataset.x.to(device)
dataset.y = dataset.y.to(device)
dataset.edge_index = dataset.edge_index.to(device)
dataset.env = dataset.env.to(device)

for run in range(args.runs):
    model.reset_parameters()
    main_params = list(model.main_parameters())
    dag_params = list(model.dag_parameters())
    optimizer_main = torch.optim.Adam(main_params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_dag = torch.optim.Adam(dag_params, lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()

        set_requires_grad(main_params, True)
        set_requires_grad(dag_params, False)
        optimizer_main.zero_grad()
        main_losses = model.compute_losses(
            dataset,
            criterion,
            args,
            detach_causal_graph=True,
            update_state=True,
        )
        main_objective = (
            main_losses['loss_cls']
            + model.lambda_ind * main_losses['loss_ind']
            + model.lambda_cl * main_losses['loss_cl']
            + model.lambda_med * main_losses['loss_med']
            + model.lambda_spu * main_losses['loss_spu']
            + model.lambda_fd * main_losses['loss_fd']
            + model.lambda_var * main_losses['loss_var']
        )
        main_objective.backward()
        optimizer_main.step()

        set_requires_grad(main_params, False)
        set_requires_grad(dag_params, True)
        optimizer_dag.zero_grad()
        dag_losses = model.compute_losses(
            dataset,
            criterion,
            args,
            detach_causal_graph=False,
            update_state=False,
        )
        dag_objective = (
            dag_losses['loss_cls']
            + model.lambda_ind * dag_losses['loss_ind']
            + model.lambda_dag * dag_losses['loss_dag']
            + model.lambda_cl * dag_losses['loss_cl']
            + model.lambda_med * dag_losses['loss_med']
            + model.lambda_spu * dag_losses['loss_spu']
            + model.lambda_fd * dag_losses['loss_fd']
            + model.lambda_var * dag_losses['loss_var']
        )
        dag_objective.backward()
        optimizer_dag.step()

        set_requires_grad(main_params, True)
        set_requires_grad(dag_params, True)

        result = evaluate_full(model, dataset, eval_func)
        logger.add_result(run, result)

        global_step = run * args.epochs + epoch
        writer.add_scalar('Loss/Total', dag_losses['total_loss'].item(), global_step)
        writer.add_scalar('Loss/Cls', dag_losses['loss_cls'].item(), global_step)
        writer.add_scalar('Loss/Ind', (model.lambda_ind * dag_losses['loss_ind']).item(), global_step)
        writer.add_scalar('Loss/DAG', (model.lambda_dag * dag_losses['loss_dag']).item(), global_step)
        writer.add_scalar('Loss/CL', (model.lambda_cl * dag_losses['loss_cl']).item(), global_step)
        writer.add_scalar('Loss/Med', (model.lambda_med * dag_losses['loss_med']).item(), global_step)
        writer.add_scalar('Loss/Spu', (model.lambda_spu * dag_losses['loss_spu']).item(), global_step)
        writer.add_scalar('Loss/FD', (model.lambda_fd * dag_losses['loss_fd']).item(), global_step)
        writer.add_scalar('Loss/Var', (model.lambda_var * dag_losses['loss_var']).item(), global_step)
        writer.add_scalar('Graph/MediatorGate', dag_losses['mediator_gate_mean'].item(), global_step)
        writer.add_scalar('Graph/CausalScore', dag_losses['causal_score_mean'].item(), global_step)
        writer.add_scalar('Graph/PollutionScore', dag_losses['pollution_score_mean'].item(), global_step)
        writer.add_scalar('Metrics/1_Train', result[0] * 100, global_step)
        writer.add_scalar('Metrics/2_Valid', result[1] * 100, global_step)
        writer.add_scalar('Metrics/3_Test_In', result[2] * 100, global_step)
        for i in range(len(result) - 3):
            writer.add_scalar(f'Metrics/4_Test_OOD_{i + 1}', result[i + 3] * 100, global_step)

        if epoch % args.display_step == 0:
            msg = (
                f"Epoch: {epoch:02d}, Loss: {dag_losses['total_loss'].item():.4f}, "
                f"Cls: {dag_losses['loss_cls'].item():.4f}, "
                f"Ind: {(model.lambda_ind * dag_losses['loss_ind']).item():.4f}, "
                f"DAG: {(model.lambda_dag * dag_losses['loss_dag']).item():.4f}, "
                f"CL: {(model.lambda_cl * dag_losses['loss_cl']).item():.4f}, "
                f"Med: {(model.lambda_med * dag_losses['loss_med']).item():.4f}, "
                f"Spu: {(model.lambda_spu * dag_losses['loss_spu']).item():.4f}, "
                f"FD: {(model.lambda_fd * dag_losses['loss_fd']).item():.4f}, "
                f"Var: {(model.lambda_var * dag_losses['loss_var']).item():.4f}, "
                f"Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, "
                f"Test In: {100 * result[2]:.2f}% "
            )
            for i in range(len(result) - 3):
                msg += f"Test OOD{i + 1}: {100 * result[i + 3]:.2f}% "
            print(msg)

    logger.print_statistics(run)

logger.print_statistics()
if args.store_result:
    logger.output(args)

writer.close()
print("[INFO] TensorBoard writer closed.")
