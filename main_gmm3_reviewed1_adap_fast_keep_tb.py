import argparse
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            print('[WARN] tensorboard is not installed; scalar logging is disabled.')

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from dataset import *
from eval import eval_acc, eval_f1, eval_rocauc, evaluate_full
from ica_utils import infer_pseudo_envs_with_ica
from logger import Logger
from model_gmm3_reviewed1_adap_fast_keep_tb import GraphFrontDoorAdapter
from parse import parser_add_main_args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_name(name):
    safe_name = ''.join(
        ch if ch.isalnum() or ch in ('-', '_', '.') else '_'
        for ch in str(name).strip()
    ).strip('._')
    return safe_name


def cosine_rampup(progress):
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * progress)


def add_adapter_frontdoor_args(parser):
    parser.set_defaults(use_cipt_schedule=True, use_cosine_lr=True)
    parser.add_argument('--lambda_med', type=float, default=0.25,
                        help='weight of mediator-only supervised loss')
    parser.add_argument('--lambda_fd', type=float, default=0.5,
                        help='weight of front-door supervised loss')
    parser.add_argument('--lambda_spu', type=float, default=0.05,
                        help='weight of spurious-branch label-uniform loss')
    parser.add_argument('--lambda_ind', type=float, default=0.05,
                        help='weight of mediator/spurious independence loss')
    parser.add_argument('--fd_blend', type=float, default=0.5,
                        help='blend ratio between mediator logits and front-door logits')
    parser.add_argument('--eval_pred_mode', type=str, default='blend',
                        choices=['blend', 'mediator', 'frontdoor'])

    parser.add_argument('--edge_score_temp', type=float, default=2.0,
                        help='temperature for local edge gate logits')
    parser.add_argument('--edge_blend', type=float, default=0.2,
                        help='strength of local edge-aware enhancement')
    parser.add_argument('--edge_feat_mode', type=str, default='mul',
                        choices=['mul', 'diff', 'signed_diff', 'degree',
                                 'mul_diff', 'mul_signed_diff',
                                 'concat', 'concat_diff',
                                 'mul_degree', 'diff_degree',
                                 'mul_diff_degree', 'mul_signed_diff_degree'])

    parser.add_argument('--disable_virtual_node_enhance', action='store_false',
                        dest='use_virtual_node_enhance',
                        help='disable same-class ego-out virtual-edge enhancement')
    parser.set_defaults(use_virtual_node_enhance=True)
    parser.add_argument('--virtual_k', type=int, default=3,
                        help='random same-class ego-out nodes aggregated for each train node per GNN layer')
    parser.add_argument('--virtual_sample_pool', type=int, default=12,
                        help='deprecated: virtual edges now sample K nodes directly from a cached ego-out pool')
    parser.add_argument('--virtual_blend', type=float, default=0.2,
                        help='strength of ego-out virtual-edge enhancement')
    parser.add_argument('--virtual_score_temp', type=float, default=1.0,
                        help='temperature for virtual-edge attention')
    parser.add_argument('--virtual_diff_bias', type=float, default=1.0,
                        help='positive bias for dissimilar same-class virtual edges')
    parser.add_argument('--virtual_exclude_hops', type=int, default=-1,
                        help='deprecated: virtual edges always exclude num_layers-hop ego nodes')

    parser.add_argument('--disable_cipt_schedule', action='store_false',
                        dest='use_cipt_schedule',
                        help='disable warmup/ramp schedule for front-door losses')
    parser.add_argument('--decomp_warmup_epochs', type=int, default=50)
    parser.add_argument('--intervention_ramp_epochs', type=int, default=100)
    parser.add_argument('--min_intervention_scale', type=float, default=0.0)
    parser.add_argument('--disable_cosine_lr', action='store_false', dest='use_cosine_lr')
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--early_stop_patience', type=int, default=0)
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0)

    # Deprecated compatibility switches accepted so old reviewed1 commands do
    # not fail, but they are intentionally unused in this clean adapter model.
    parser.add_argument('--lambda_dag', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--lambda_dag_label', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--lambda_env', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--dag_latent_dim', type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument('--gmm_sample_k', type=int, default=0, help=argparse.SUPPRESS)


def capture_lambda_state(model):
    names = ('lambda_med', 'lambda_fd', 'lambda_spu', 'lambda_ind')
    return {name: float(getattr(model, name)) for name in names}


def restore_lambda_state(model, lambda_state):
    for name, value in lambda_state.items():
        setattr(model, name, value)


def apply_cipt_schedule(model, base_lambdas, epoch, args):
    restore_lambda_state(model, base_lambdas)
    if not args.use_cipt_schedule:
        return {'intervention_scale': 1.0}

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

    model.lambda_fd = base_lambdas['lambda_fd'] * intervention_scale
    return {'intervention_scale': intervention_scale}


parser = argparse.ArgumentParser(description='Clean Adapter Front-Door Training Pipeline')
parser_add_main_args(parser)
add_adapter_frontdoor_args(parser)
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
m = ''
for i in range(len(dataset.test_ood_idx)):
    m += f"test ood{i + 1} nodes {dataset.test_ood_idx[i].shape[0]} "
print(m)
print(f'[INFO] env numbers: {dataset.env_num} train env numbers: {dataset.train_env_num}')

model = GraphFrontDoorAdapter(d, c, args, device).to(device)

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
        f"{current_time}_fd_adap_virtual{args.virtual_k}_{args.edge_feat_mode}"
        f"_fd{args.lambda_fd}_spu{args.lambda_spu}_ind{args.lambda_ind}"
    )
log_dir = os.path.join('.', 'runs', args.dataset, 'frontdoor_adap', run_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logging activated. Logs will be saved to: {log_dir}")
print(
    f"[INFO] Training recipe: clean adapter front-door | "
    f"virtual ego-out: {args.use_virtual_node_enhance} "
    f"(per-layer random K={args.virtual_k}, blend={args.virtual_blend}, "
    f"exclude_hops=num_layers({args.num_layers}), "
    f"diff_bias={args.virtual_diff_bias}) | "
    f"edge feat: {args.edge_feat_mode} | local blend: {args.edge_blend} | "
    f"lambda med/fd/spu/ind: {args.lambda_med}/{args.lambda_fd}/{args.lambda_spu}/{args.lambda_ind}"
)

dataset.x = dataset.x.to(device)
dataset.y = dataset.y.to(device)
dataset.edge_index = dataset.edge_index.to(device)
dataset.env = dataset.env.to(device)
model.prepare_graph_cache(dataset.edge_index, dataset.x.size(0))

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
        model.prepare_virtual_edges(
            dataset.edge_index,
            dataset.x.size(0),
            labels=dataset.y,
            train_idx=dataset.train_idx,
        )
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
        writer.add_scalar('Loss/Med', (model.lambda_med * losses['loss_med']).item(), global_step)
        writer.add_scalar('Loss/FD', (model.lambda_fd * losses['loss_fd']).item(), global_step)
        writer.add_scalar('Loss/Spu', (model.lambda_spu * losses['loss_spu']).item(), global_step)
        writer.add_scalar('Loss/Ind', (model.lambda_ind * losses['loss_ind']).item(), global_step)
        writer.add_scalar('Graph/NumVirtualEdges', losses['num_virtual_edges'].item(), global_step)
        writer.add_scalar('Graph/VirtualAlphaMean', losses['virtual_alpha_mean'].item(), global_step)
        writer.add_scalar('Graph/NumContexts', losses['num_contexts'].item(), global_step)
        writer.add_scalar('Graph/CausalEnergy', losses['causal_score_mean'].item(), global_step)
        writer.add_scalar('Graph/SpuriousEnergy', losses['pollution_score_mean'].item(), global_step)
        writer.add_scalar('Schedule/LR', current_lr, global_step)
        writer.add_scalar('Schedule/InterventionScale', schedule_state['intervention_scale'], global_step)
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
                f"Spu: {(model.lambda_spu * losses['loss_spu']).item():.4f}, "
                f"Ind: {(model.lambda_ind * losses['loss_ind']).item():.4f}, "
                f"VirtualEdges: {int(losses['num_virtual_edges'].item())}, "
                f"Ctx: {int(losses['num_contexts'].item())}, "
                f"LR: {current_lr:.6f}, IntScale: {schedule_state['intervention_scale']:.3f}, "
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
