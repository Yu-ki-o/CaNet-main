import argparse
import math
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
from model_frontdoor_dag import GraphFrontDoorDAG
from parse import parser_add_main_args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_frontdoor_dag_args(parser):
    parser.set_defaults(use_cipt_schedule=True, use_cosine_lr=True)
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='EMA momentum for spurious context prototypes')
    parser.add_argument('--lambda_l1', type=float, default=1e-5,
                        help='L1 sparsity weight inside the DAG regularizer')
    parser.add_argument('--lambda_ind', type=float, default=0.1,
                        help='weight of mediator-spurious decorrelation')
    parser.add_argument('--lambda_dag', type=float, default=0.1,
                        help='weight of DAG regularization')
    parser.add_argument('--lambda_med', type=float, default=0.5,
                        help='weight of mediator-only supervision loss')
    parser.add_argument('--lambda_spu', type=float, default=0.1,
                        help='weight of spurious branch environment loss')
    parser.add_argument('--lambda_fd', type=float, default=0.5,
                        help='weight of front-door aggregation loss')
    parser.add_argument('--lambda_var', type=float, default=0.05,
                        help='weight of cross-context front-door variance penalty')
    parser.add_argument('--lambda_env', type=float, default=0.1,
                        help='weight of environment-uniform loss on mediator branch')
    parser.add_argument('--lambda_inv', type=float, default=0.1,
                        help='weight of cross-environment prediction invariance loss')
    parser.add_argument('--lambda_gate', type=float, default=0.0,
                        help='optional mean-gate sparsity regularizer inside the DAG loss')
    parser.add_argument('--mediator_temp', type=float, default=8.0,
                        help='temperature of the DAG-based soft mediator selector')
    parser.add_argument('--low_temp', type=float, default=8.0,
                        help='temperature used to detect low-score features')
    parser.add_argument('--low_threshold', type=float, default=0.35,
                        help='threshold for identifying low-score features')
    parser.add_argument('--mediator_threshold', type=float, default=0.5,
                        help='threshold for activating mediator dimensions')
    parser.add_argument('--pollution_coeff', type=float, default=1.0,
                        help='penalty coefficient for feature pollution from low-score nodes')
    parser.add_argument('--fd_blend', type=float, default=0.5,
                        help='blend ratio between mediator logits and front-door aggregated logits')
    parser.add_argument('--disable_cipt_schedule', action='store_false', dest='use_cipt_schedule',
                        help='disable the CIPT-style curriculum that warms up decomposition before full intervention')
    parser.add_argument('--decomp_warmup_epochs', type=int, default=50,
                        help='warmup epochs that emphasize causal decomposition before front-door intervention')
    parser.add_argument('--intervention_ramp_epochs', type=int, default=100,
                        help='epochs used to smoothly ramp front-door and invariance losses after warmup')
    parser.add_argument('--min_intervention_scale', type=float, default=0.0,
                        help='minimum scale applied to intervention-related losses during the warmup stage')
    parser.add_argument('--disable_cosine_lr', action='store_false', dest='use_cosine_lr',
                        help='disable cosine annealing and keep a constant learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='minimum learning rate for cosine annealing')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='gradient clipping norm for stable few-shot front-door training; <= 0 disables it')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='stop a run if validation does not improve for this many epochs; default 0 disables')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='minimum validation improvement required to reset early stopping')


def sanitize_name(name):
    safe_name = "".join(
        ch if ch.isalnum() or ch in ('-', '_', '.') else '_'
        for ch in str(name).strip()
    ).strip('._')
    return safe_name


def capture_lambda_state(model):
    lambda_names = (
        'lambda_dag',
        'lambda_med',
        'lambda_spu',
        'lambda_fd',
        'lambda_var',
        'lambda_ind',
        'lambda_env',
        'lambda_inv',
    )
    return {name: float(getattr(model, name)) for name in lambda_names}


def restore_lambda_state(model, lambda_state):
    for name, value in lambda_state.items():
        setattr(model, name, value)


def cosine_rampup(progress):
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * progress)


def apply_cipt_schedule(model, base_lambdas, epoch, args):
    """
    CIPT-inspired curriculum:
    1) stabilize mediator / spurious decomposition first,
    2) then gradually activate intervention-related objectives.
    """
    restore_lambda_state(model, base_lambdas)

    if not args.use_cipt_schedule:
        return {'intervention_scale': 1.0, 'dag_scale': 1.0}

    warmup_epochs = max(0, int(args.decomp_warmup_epochs))
    ramp_epochs = max(0, int(args.intervention_ramp_epochs))
    min_scale = float(args.min_intervention_scale)

    if epoch < warmup_epochs:
        intervention_scale = min_scale
    elif ramp_epochs == 0:
        intervention_scale = 1.0
    else:
        progress = (epoch - warmup_epochs + 1) / float(ramp_epochs)
        intervention_scale = min_scale + (1.0 - min_scale) * cosine_rampup(progress)

    intervention_scale = min(max(intervention_scale, 0.0), 1.0)
    dag_scale = 0.5 + 0.5 * intervention_scale

    for name in ('lambda_fd', 'lambda_var', 'lambda_env', 'lambda_inv'):
        setattr(model, name, base_lambdas[name] * intervention_scale)
    setattr(model, 'lambda_dag', base_lambdas['lambda_dag'] * dag_scale)

    return {
        'intervention_scale': intervention_scale,
        'dag_scale': dag_scale,
    }


parser = argparse.ArgumentParser(description='Graph Front-Door DAG Training Pipeline')
parser_add_main_args(parser)
add_frontdoor_dag_args(parser)
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

model = GraphFrontDoorDAG(d, c, args, device).to(device)

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
        f"{current_time}_fd_dag_d{args.lambda_dag}_fd{args.lambda_fd}"
        f"_warm{args.decomp_warmup_epochs}_ramp{args.intervention_ramp_epochs}"
    )
log_dir = os.path.join('.', 'runs', args.dataset, 'frontdoor_dag', run_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logging activated. Logs will be saved to: {log_dir}")
print(
    f"[INFO] Training recipe | CIPT schedule: {args.use_cipt_schedule} | "
    f"cosine lr: {args.use_cosine_lr} | warmup: {args.decomp_warmup_epochs} | "
    f"ramp: {args.intervention_ramp_epochs} | grad clip: {args.grad_clip}"
)

dataset.x = dataset.x.to(device)
dataset.y = dataset.y.to(device)
dataset.edge_index = dataset.edge_index.to(device)
dataset.env = dataset.env.to(device)

base_lambdas = capture_lambda_state(model)

for run in range(args.runs):
    model.reset_parameters()
    restore_lambda_state(model, base_lambdas)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.use_cosine_lr and args.epochs > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=min(args.min_lr, args.lr),
        )
    best_valid = float('-inf')
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(args.epochs):
        schedule_state = apply_cipt_schedule(model, base_lambdas, epoch, args)
        model.train()
        optimizer.zero_grad()
        losses = model.compute_losses(dataset, criterion, args, update_state=True)
        losses['total_loss'].backward()

        grad_norm = None
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        model.apply_state_update(losses.get('state_payload'))

        result = evaluate_full(model, dataset, eval_func)
        logger.add_result(run, result)
        current_lr = optimizer.param_groups[0]['lr']

        global_step = run * args.epochs + epoch
        writer.add_scalar('Loss/Total', losses['total_loss'].item(), global_step)
        writer.add_scalar('Loss/Cls', losses['loss_cls'].item(), global_step)
        writer.add_scalar('Loss/Med', losses['loss_med'].item(), global_step)
        writer.add_scalar('Loss/FD', losses['loss_fd'].item(), global_step)
        writer.add_scalar('Loss/Ind', (model.lambda_ind * losses['loss_ind']).item(), global_step)
        writer.add_scalar('Loss/DAG', (model.lambda_dag * losses['loss_dag']).item(), global_step)
        writer.add_scalar('Loss/Spu', (model.lambda_spu * losses['loss_spu']).item(), global_step)
        writer.add_scalar('Loss/EnvMed', (model.lambda_env * losses['loss_env_med']).item(), global_step)
        writer.add_scalar('Loss/Inv', (model.lambda_inv * losses['loss_inv']).item(), global_step)
        writer.add_scalar('Loss/Var', (model.lambda_var * losses['loss_var']).item(), global_step)
        writer.add_scalar('Graph/MediatorGate', losses['mediator_gate_mean'].item(), global_step)
        writer.add_scalar('Graph/CausalScore', losses['causal_score_mean'].item(), global_step)
        writer.add_scalar('Graph/PollutionScore', losses['pollution_score_mean'].item(), global_step)
        writer.add_scalar('Graph/NumContexts', losses['num_contexts'].item(), global_step)
        writer.add_scalar('Schedule/LR', current_lr, global_step)
        writer.add_scalar('Schedule/InterventionScale', schedule_state['intervention_scale'], global_step)
        writer.add_scalar('Schedule/DAGScale', schedule_state['dag_scale'], global_step)
        if grad_norm is not None:
            writer.add_scalar('Grad/Norm', grad_norm.item(), global_step)
        writer.add_scalar('Metrics/1_Train', result[0] * 100, global_step)
        writer.add_scalar('Metrics/2_Valid', result[1] * 100, global_step)
        writer.add_scalar('Metrics/3_Test_In', result[2] * 100, global_step)
        for i in range(len(result) - 3):
            writer.add_scalar(f'Metrics/4_Test_OOD_{i + 1}', result[i + 3] * 100, global_step)

        valid_score = result[1]
        if valid_score > best_valid + args.early_stop_min_delta:
            best_valid = valid_score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch % args.display_step == 0:
            msg = (
                f"Epoch: {epoch:03d}, Loss: {losses['total_loss'].item():.4f}, "
                f"Cls: {losses['loss_cls'].item():.4f}, "
                f"Med: {(model.lambda_med * losses['loss_med']).item():.4f}, "
                f"FD: {(model.lambda_fd * losses['loss_fd']).item():.4f}, "
                f"Ind: {(model.lambda_ind * losses['loss_ind']).item():.4f}, "
                f"DAG: {(model.lambda_dag * losses['loss_dag']).item():.4f}, "
                f"Spu: {(model.lambda_spu * losses['loss_spu']).item():.4f}, "
                f"EnvMed: {(model.lambda_env * losses['loss_env_med']).item():.4f}, "
                f"Inv: {(model.lambda_inv * losses['loss_inv']).item():.4f}, "
                f"Var: {(model.lambda_var * losses['loss_var']).item():.4f}, "
                f"LR: {current_lr:.6f}, "
                f"IntScale: {schedule_state['intervention_scale']:.3f}, "
                f"Ctx: {int(losses['num_contexts'].item())}, "
                f"Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, "
                f"Test In: {100 * result[2]:.2f}% "
            )
            for i in range(len(result) - 3):
                msg += f"Test OOD{i + 1}: {100 * result[i + 3]:.2f}% "
            print(msg)

        if scheduler is not None:
            scheduler.step()

        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            print(
                f"[INFO] Early stopping run {run + 1:02d} at epoch {epoch + 1}. "
                f"Best valid {100 * best_valid:.2f}% was reached at epoch {best_epoch + 1}."
            )
            break

    logger.print_statistics(run)

logger.print_statistics()
if args.store_result:
    logger.output(args)

writer.close()
print('[INFO] TensorBoard writer closed.')
