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
from model_frontdoor_trisplit import GraphFrontDoorTriSplit
from parse import parser_add_main_args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_frontdoor_trisplit_args(parser):
    parser.set_defaults(use_cipt_schedule=True, use_cosine_lr=True)

    # Context / front-door state.
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='EMA momentum for environment context statistics')
    parser.add_argument('--fd_blend', type=float, default=0.5,
                        help='base blend ratio between causal logits and front-door logits')
    parser.add_argument('--context_mode', type=str, default='both', choices=['gmm', 'proto', 'both', 'none'],
                        help='front-door context source: GMM samples, deterministic prototypes, both, or none')
    parser.add_argument('--disable_spu_gmm', action='store_false', dest='use_spu_gmm',
                        help='disable GMM sampling for environment contexts')
    parser.set_defaults(use_spu_gmm=True)
    parser.add_argument('--gmm_sample_k', type=int, default=0,
                        help='number of GMM-sampled environment contexts; <=0 uses K')
    parser.add_argument('--gmm_min_var', type=float, default=1e-4,
                        help='minimum diagonal variance used by environment-context GMM')
    parser.add_argument('--gmm_max_std', type=float, default=0.2,
                        help='maximum std for GMM context sampling; <=0 disables clipping')
    parser.add_argument('--eval_gmm_noise', action='store_true',
                        help='add deterministic GMM noise at evaluation; default uses environment means only')

    # TriSplit gate.
    parser.add_argument('--split_temp', type=float, default=1.0,
                        help='temperature for channel-wise causal/env/residual softmax gates')
    parser.add_argument('--split_prior_causal', type=float, default=0.5,
                        help='target average mass for causal channels')
    parser.add_argument('--split_prior_env', type=float, default=0.3,
                        help='target average mass for environment channels')
    parser.add_argument('--split_prior_residual', type=float, default=0.2,
                        help='target average mass for residual channels')

    # Loss weights.
    parser.add_argument('--lambda_causal', type=float, default=1.0,
                        help='weight of causal-only supervised loss')
    parser.add_argument('--lambda_spu', type=float, default=0.05,
                        help='weight of pseudo-environment discovery on env branch')
    parser.add_argument('--lambda_fd', type=float, default=0.5,
                        help='weight of front-door aggregation loss')
    parser.add_argument('--lambda_env', type=float, default=0.05,
                        help='weight of environment-uniform loss on causal branch')
    parser.add_argument('--lambda_residual', type=float, default=0.01,
                        help='weight of residual null loss')
    parser.add_argument('--lambda_orth', type=float, default=0.01,
                        help='weight of causal/env/residual decorrelation loss')
    parser.add_argument('--lambda_gate_balance', type=float, default=0.01,
                        help='weight of average gate-prior matching loss')
    parser.add_argument('--lambda_gate_entropy', type=float, default=0.001,
                        help='weight of gate entropy minimization loss')
    parser.add_argument('--lambda_var', type=float, default=0.0,
                        help='optional cross-context front-door prediction variance penalty')
    parser.add_argument('--lambda_inv', type=float, default=0.0,
                        help='optional pseudo-env invariance penalty on causal predictions')
    parser.add_argument('--pseudo_env_balance', type=float, default=1.0,
                        help='balance weight inside label-free pseudo-environment discovery')

    # Backward-compatible no-op DAG flags, so old launch scripts do not crash.
    parser.add_argument('--lambda_l1', type=float, default=0.0,
                        help='deprecated in TriSplit')
    parser.add_argument('--lambda_ind', type=float, default=0.0,
                        help='deprecated alias; use lambda_orth')
    parser.add_argument('--lambda_dag', type=float, default=0.0,
                        help='deprecated in TriSplit')
    parser.add_argument('--lambda_med', type=float, default=0.0,
                        help='deprecated alias; use lambda_causal')
    parser.add_argument('--lambda_fd_aug', type=float, default=0.0,
                        help='deprecated in TriSplit')
    parser.add_argument('--lambda_gate', type=float, default=0.0,
                        help='deprecated in TriSplit')
    parser.add_argument('--lambda_dag_label', type=float, default=0.0,
                        help='deprecated in TriSplit')
    parser.add_argument('--lambda_sem', type=float, default=0.0,
                        help='deprecated in TriSplit')
    parser.add_argument('--lambda_spu_y', type=float, default=0.0,
                        help='deprecated in TriSplit')
    parser.add_argument('--dag_latent_dim', type=int, default=16,
                        help='deprecated in TriSplit')
    parser.add_argument('--mediator_temp', type=float, default=8.0,
                        help='deprecated in TriSplit')
    parser.add_argument('--low_temp', type=float, default=8.0,
                        help='deprecated in TriSplit')
    parser.add_argument('--low_threshold', type=float, default=0.35,
                        help='deprecated in TriSplit')
    parser.add_argument('--mediator_threshold', type=float, default=0.5,
                        help='deprecated in TriSplit')
    parser.add_argument('--pollution_coeff', type=float, default=1.0,
                        help='deprecated in TriSplit')
    parser.add_argument('--edge_pollution_coeff', type=float, default=0.5,
                        help='deprecated in TriSplit')
    parser.add_argument('--causal_support_coeff', type=float, default=0.5,
                        help='deprecated in TriSplit')
    parser.add_argument('--disable_dag_mixer', action='store_false', dest='use_dag_mixer',
                        help='deprecated in TriSplit; DAG mixer is always disabled')
    parser.set_defaults(use_dag_mixer=False)
    parser.add_argument('--dag_mixer_heads', type=int, default=1,
                        help='deprecated in TriSplit')
    parser.add_argument('--dag_mixer_layers', type=int, default=1,
                        help='deprecated in TriSplit')

    # Encoder extras retained from the DAG-Core version.
    parser.add_argument('--edge_score_temp', type=float, default=2.0,
                        help='temperature for edge semantic gate logits; larger values produce smoother gates')
    parser.add_argument('--edge_blend', type=float, default=0.2,
                        help='residual strength of edge-aware neighbor aggregation')
    parser.add_argument('--use_global_info', action='store_true',
                        help='inject global information before TriSplit')
    parser.add_argument('--global_info_mode', type=str, default='advective', choices=['linear', 'advective'],
                        help="'linear' uses global linear attention; 'advective' mixes it with local topology")
    parser.add_argument('--global_alpha', type=float, default=0.2,
                        help='residual strength of the global information channel')
    parser.add_argument('--global_beta', type=float, default=0.5,
                        help='topology reliance weight beta for advective global-local mixing')
    parser.add_argument('--global_steps', type=int, default=1,
                        help='number of lightweight advective propagation steps')
    parser.add_argument('--global_local_source', type=str, default='gcn', choices=['edge', 'gcn'],
                        help="'edge' uses learned edge-gated aggregation; 'gcn' uses raw GCN propagation")

    # Training schedule / optimization.
    parser.add_argument('--disable_cipt_schedule', action='store_false', dest='use_cipt_schedule',
                        help='disable CIPT-style curriculum')
    parser.add_argument('--decomp_warmup_epochs', type=int, default=50,
                        help='warmup epochs that emphasize representation splitting before intervention')
    parser.add_argument('--intervention_ramp_epochs', type=int, default=100,
                        help='epochs used to smoothly ramp front-door and invariance losses after warmup')
    parser.add_argument('--min_intervention_scale', type=float, default=0.0,
                        help='minimum scale applied to intervention-related losses during warmup')
    parser.add_argument('--disable_cosine_lr', action='store_false', dest='use_cosine_lr',
                        help='disable cosine annealing and keep a constant learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='minimum learning rate for cosine annealing')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='gradient clipping norm; <=0 disables it')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='stop a run if validation does not improve for this many epochs; 0 disables')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='minimum validation improvement required to reset early stopping')
    parser.add_argument('--binary_loss_mode', type=str, default='original_bce', choices=['original_bce', 'ce'],
                        help="for twitch/elliptic: keep original multi-output BCE or use CE when labels are single-column")

    # Deprecated prototype mixup flags kept for launch compatibility.
    parser.add_argument('--proto_aug_k', type=int, default=1,
                        help='deprecated in TriSplit')
    parser.add_argument('--proto_mix_alpha', type=float, default=1.0,
                        help='deprecated in TriSplit')


def sanitize_name(name):
    safe_name = "".join(
        ch if ch.isalnum() or ch in ('-', '_', '.') else '_'
        for ch in str(name).strip()
    ).strip('._')
    return safe_name


def capture_lambda_state(model):
    lambda_names = (
        'lambda_causal',
        'lambda_spu',
        'lambda_fd',
        'lambda_env',
        'lambda_residual',
        'lambda_orth',
        'lambda_gate_balance',
        'lambda_gate_entropy',
        'lambda_var',
        'lambda_inv',
        'lambda_fd_aug',
        'lambda_dag',
        'lambda_dag_label',
        'lambda_spu_y',
    )
    return {name: float(getattr(model, name, 0.0)) for name in lambda_names}


def restore_lambda_state(model, lambda_state):
    for name, value in lambda_state.items():
        if hasattr(model, name):
            setattr(model, name, value)
    # Keep deprecated aliases in sync for any old helper that reads them.
    if hasattr(model, 'lambda_causal'):
        model.lambda_med = model.lambda_causal
    if hasattr(model, 'lambda_orth'):
        model.lambda_ind = model.lambda_orth


def cosine_rampup(progress):
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * progress)


def apply_cipt_schedule(model, base_lambdas, epoch, args):
    """
    CIPT-inspired curriculum:
    1) learn causal/env/residual split first;
    2) gradually activate front-door/invariance objectives;
    3) also ramp fd_blend, so warmup classification is purely causal.
    """
    restore_lambda_state(model, base_lambdas)

    if not args.use_cipt_schedule:
        model.current_fd_blend = model.fd_blend
        model.current_intervention_scale = 1.0
        return {'intervention_scale': 1.0, 'decomp_scale': 1.0}

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

    for name in ('lambda_fd', 'lambda_env', 'lambda_var', 'lambda_inv'):
        if name in base_lambdas and hasattr(model, name):
            setattr(model, name, base_lambdas[name] * intervention_scale)

    model.current_fd_blend = model.fd_blend * intervention_scale
    model.current_intervention_scale = intervention_scale

    return {
        'intervention_scale': intervention_scale,
        'decomp_scale': 1.0,
    }


def load_selected_dataset(args):
    if args.dataset == 'twitch':
        return load_twitch_dataset(args.data_dir, train_num=3)
    if args.dataset == 'elliptic':
        return load_elliptic_dataset(args.data_dir, train_num=5)
    if args.dataset == 'arxiv':
        return load_arxiv_dataset(args.data_dir, train_num=3)
    if args.dataset in ('cora', 'citeseer', 'pubmed'):
        return load_synthetic_dataset(
            args.data_dir,
            args.dataset,
            train_num=3,
            combine=args.combine_result,
        )
    raise ValueError('Invalid dataname')


parser = argparse.ArgumentParser(description='Graph Front-Door TriSplit Training Pipeline')
parser_add_main_args(parser)
add_frontdoor_trisplit_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

dataset = load_selected_dataset(args)

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

model = GraphFrontDoorTriSplit(d, c, args, device).to(device)

if args.dataset in ('elliptic', 'twitch'):
    if args.binary_loss_mode == 'ce' and c > 1 and dataset.y.shape[1] == 1:
        criterion = nn.CrossEntropyLoss(reduction='none')
        print('[INFO] Using CrossEntropyLoss for binary/multiclass single-column labels.')
    else:
        pos_weight = torch.full((c,), float(args.pos_weight), device=device)
        criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        print('[INFO] Using original multi-output BCEWithLogitsLoss for twitch/elliptic.')
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
        f"{current_time}_fd_trisplit_c{args.lambda_causal}_fd{args.lambda_fd}"
        f"_spu{args.lambda_spu}_env{args.lambda_env}_ctx{args.context_mode}"
        f"_warm{args.decomp_warmup_epochs}_ramp{args.intervention_ramp_epochs}"
    )
log_dir = os.path.join('.', 'runs', args.dataset, 'frontdoor_trisplit', run_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logging activated. Logs will be saved to: {log_dir}")
print(
    f"[INFO] Training recipe: TriSplit | CIPT schedule: {args.use_cipt_schedule} | "
    f"cosine lr: {args.use_cosine_lr} | warmup: {args.decomp_warmup_epochs} | "
    f"ramp: {args.intervention_ramp_epochs} | grad clip: {args.grad_clip} | "
    f"context mode: {args.context_mode} | GMM contexts: {args.use_spu_gmm} | "
    f"GMM sample k: {args.gmm_sample_k if args.gmm_sample_k > 0 else args.K} | "
    f"eval GMM noise: {args.eval_gmm_noise} | fd blend: {args.fd_blend} | "
    f"split prior: ({args.split_prior_causal}, {args.split_prior_env}, {args.split_prior_residual}) | "
    f"global info: {args.use_global_info} ({args.global_info_mode}, "
    f"alpha={args.global_alpha}, beta={args.global_beta}, steps={args.global_steps}, "
    f"local={args.global_local_source})"
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
        writer.add_scalar('Loss/Causal', (model.lambda_causal * losses['loss_causal']).item(), global_step)
        writer.add_scalar('Loss/FD', (model.lambda_fd * losses['loss_fd']).item(), global_step)
        writer.add_scalar('Loss/Spu', (model.lambda_spu * losses['loss_spu']).item(), global_step)
        writer.add_scalar('Loss/EnvCausal', (model.lambda_env * losses['loss_env_causal']).item(), global_step)
        writer.add_scalar('Loss/Residual', (model.lambda_residual * losses['loss_residual']).item(), global_step)
        writer.add_scalar('Loss/Orth', (model.lambda_orth * losses['loss_orth']).item(), global_step)
        writer.add_scalar('Loss/GateBalance', (model.lambda_gate_balance * losses['loss_gate_balance']).item(), global_step)
        writer.add_scalar('Loss/GateEntropy', (model.lambda_gate_entropy * losses['loss_gate_entropy']).item(), global_step)
        writer.add_scalar('Loss/Var', (model.lambda_var * losses['loss_var']).item(), global_step)
        writer.add_scalar('Loss/Inv', (model.lambda_inv * losses['loss_inv']).item(), global_step)
        writer.add_scalar('Gate/CausalMean', losses['gate_causal_mean'].item(), global_step)
        writer.add_scalar('Gate/EnvMean', losses['gate_env_mean'].item(), global_step)
        writer.add_scalar('Gate/ResidualMean', losses['gate_residual_mean'].item(), global_step)
        writer.add_scalar('Gate/EntropyValue', losses['gate_entropy_value'].item(), global_step)
        writer.add_scalar('Context/NumContexts', losses['num_contexts'].item(), global_step)
        writer.add_scalar('Context/NumProtoContexts', losses['num_proto_contexts'].item(), global_step)
        writer.add_scalar('Context/NumGMMContexts', losses['num_gmm_contexts'].item(), global_step)
        writer.add_scalar('Schedule/LR', current_lr, global_step)
        writer.add_scalar('Schedule/InterventionScale', schedule_state['intervention_scale'], global_step)
        writer.add_scalar('Schedule/FDBlend', model.current_fd_blend, global_step)
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
                f"Causal: {(model.lambda_causal * losses['loss_causal']).item():.4f}, "
                f"FD: {(model.lambda_fd * losses['loss_fd']).item():.4f}, "
                f"Spu: {(model.lambda_spu * losses['loss_spu']).item():.4f}, "
                f"Env: {(model.lambda_env * losses['loss_env_causal']).item():.4f}, "
                f"Res: {(model.lambda_residual * losses['loss_residual']).item():.4f}, "
                f"Orth: {(model.lambda_orth * losses['loss_orth']).item():.4f}, "
                f"GateBal: {(model.lambda_gate_balance * losses['loss_gate_balance']).item():.4f}, "
                f"GateEnt: {(model.lambda_gate_entropy * losses['loss_gate_entropy']).item():.4f}, "
                f"LR: {current_lr:.6f}, "
                f"IntScale: {schedule_state['intervention_scale']:.3f}, "
                f"FDBlend: {model.current_fd_blend:.3f}, "
                f"Gate: C/E/R={losses['gate_causal_mean'].item():.2f}/"
                f"{losses['gate_env_mean'].item():.2f}/"
                f"{losses['gate_residual_mean'].item():.2f}, "
                f"Ctx: {int(losses['num_contexts'].item())}, "
                f"ProtoCtx: {int(losses['num_proto_contexts'].item())}, "
                f"GMMCtx: {int(losses['num_gmm_contexts'].item())}, "
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
