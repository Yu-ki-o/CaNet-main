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
from model_frontdoor import GraphFrontDoor
from parse import parser_add_main_args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def add_frontdoor_args(parser):
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='EMA momentum for environment-specific spurious prototypes')
    parser.add_argument('--lambda_ind', type=float, default=0.1,
                        help='weight of mediator-spurious decorrelation')
    parser.add_argument('--lambda_med', type=float, default=0.5,
                        help='weight of the causal-branch supervision loss')
    parser.add_argument('--lambda_spu', type=float, default=0.1,
                        help='weight of the spurious-branch uniformity loss')
    parser.add_argument('--lambda_fd', type=float, default=0.5,
                        help='weight of the intervention-branch supervision loss')
    parser.add_argument('--lambda_var', type=float, default=0.05,
                        help='weight of the cross-context front-door variance penalty')
    parser.add_argument('--fd_blend', type=float, default=0.5,
                        help='blend ratio between causal logits and intervention logits')
    parser.add_argument('--context_gate_temp', type=float, default=1.0,
                        help='temperature of the adaptive context gating inside diversity augmentation')


def sanitize_name(name):
    safe_name = "".join(
        ch if ch.isalnum() or ch in ('-', '_', '.') else '_'
        for ch in str(name).strip()
    ).strip('._')
    return safe_name


parser = argparse.ArgumentParser(description='Graph Front-Door Training Pipeline')
parser_add_main_args(parser)
add_frontdoor_args(parser)
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

model = GraphFrontDoor(d, c, args, device).to(device)

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
    run_name = f"{current_time}_frontdoor_cipt_fd_{args.lambda_fd}_ind_{args.lambda_ind}"
log_dir = os.path.join('.', 'runs', args.dataset, 'frontdoor', run_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logging activated. Logs will be saved to: {log_dir}")

dataset.x = dataset.x.to(device)
dataset.y = dataset.y.to(device)
dataset.edge_index = dataset.edge_index.to(device)
dataset.env = dataset.env.to(device)

for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        losses = model.compute_losses(dataset, criterion, args, update_state=True)
        losses['total_loss'].backward()
        optimizer.step()
        model.apply_state_update(losses.get('state_payload'))
        result = evaluate_full(model, dataset, eval_func)
        logger.add_result(run, result)

        global_step = run * args.epochs + epoch
        writer.add_scalar('Loss/Total', losses['total_loss'].item(), global_step)
        writer.add_scalar('Loss/Cls', losses['loss_cls'].item(), global_step)
        writer.add_scalar('Loss/Ind', (model.lambda_ind * losses['loss_ind']).item(), global_step)
        writer.add_scalar('Loss/Med', (model.lambda_med * losses['loss_med']).item(), global_step)
        writer.add_scalar('Loss/Spu', (model.lambda_spu * losses['loss_spu']).item(), global_step)
        writer.add_scalar('Loss/FD', (model.lambda_fd * losses['loss_fd']).item(), global_step)
        writer.add_scalar('Loss/Var', (model.lambda_var * losses['loss_var']).item(), global_step)
        writer.add_scalar('Graph/CausalNorm', losses['causal_norm_mean'].item(), global_step)
        writer.add_scalar('Graph/SpuriousNorm', losses['spurious_norm_mean'].item(), global_step)
        writer.add_scalar('Graph/NumContexts', losses['num_contexts'].item(), global_step)
        writer.add_scalar('Metrics/1_Train', result[0] * 100, global_step)
        writer.add_scalar('Metrics/2_Valid', result[1] * 100, global_step)
        writer.add_scalar('Metrics/3_Test_In', result[2] * 100, global_step)
        for i in range(len(result) - 3):
            writer.add_scalar(f'Metrics/4_Test_OOD_{i + 1}', result[i + 3] * 100, global_step)

        if epoch % args.display_step == 0:
            msg = (
                f"Epoch: {epoch:02d}, Loss: {losses['total_loss'].item():.4f}, "
                f"Cls: {losses['loss_cls'].item():.4f}, "
                f"Ind: {(model.lambda_ind * losses['loss_ind']).item():.4f}, "
                f"Med: {(model.lambda_med * losses['loss_med']).item():.4f}, "
                f"Spu: {(model.lambda_spu * losses['loss_spu']).item():.4f}, "
                f"FD: {(model.lambda_fd * losses['loss_fd']).item():.4f}, "
                f"Var: {(model.lambda_var * losses['loss_var']).item():.4f}, "
                f"Ctx: {int(losses['num_contexts'].item())}, "
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
print('[INFO] TensorBoard writer closed.')
