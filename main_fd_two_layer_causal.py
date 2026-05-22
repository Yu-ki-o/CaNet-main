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
from model_fd_two_layer_adap import GraphSoftCutCIPT
from parse import parser_add_main_args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_signed_cipt_args(parser):
    """DAG-free signed-operator CIPT arguments."""
    parser.set_defaults(
        use_cipt_schedule=True,
        use_cosine_lr=True,
        relation_mode='signed_hadamard',
        edge_feat_mode='signed_hadamard',
        use_node_enhance=True,
        use_neighbor_denoise=False,
        use_dag_module=False,
        use_dag_mixer=True,
        use_node_conditioned_context=True,
    )
    parser.add_argument('--lambda_med', type=float, default=1.0,
                        help='CIPT Lc weight: mediator-only supervised loss')
    parser.add_argument('--lambda_spu', type=float, default=2.0,
                        help='CIPT Lde weight: push spurious label prediction to uniform')
    parser.add_argument('--lambda_ind', type=float, default=5.0,
                        help='CIPT Lind weight: mediator-spurious independence')
    parser.add_argument('--lambda_fd', type=float, default=0.5,
                        help='weight of front-door/context intervention loss')
    parser.add_argument('--lambda_env', type=float, default=0.05,
                        help='make mediator branch pseudo-environment-uniform')
    parser.add_argument('--lambda_spu_env', type=float, default=0.05,
                        help='encourage spurious branch to form balanced pseudo environments')
    parser.add_argument('--lambda_var', type=float, default=0.0,
                        help='optional variance penalty across selected contexts')
    parser.add_argument('--lambda_operator', type=float, default=0.1,
                        help='weight of signed f1/f2 two-hop operator consistency')
    parser.add_argument('--lambda_context', type=float, default=0.0,
                        help='optional loss weight for solving f1(x_origin, context) ~= a_i')
    parser.add_argument('--lambda_dag', type=float, default=0.0,
                        help='DAG acyclicity/sparsity loss weight; active only with --use_dag_module')
    parser.add_argument('--lambda_dag_label', type=float, default=0.0,
                        help='DAG-to-label supervision weight; active only with --use_dag_module')
    parser.add_argument('--lambda_l1', type=float, default=1e-5,
                        help='L1 sparsity coefficient inside DAG regularization')
    parser.add_argument('--lambda_gate', type=float, default=0.0,
                        help='optional mediator gate compactness regularizer inside DAG loss')
    parser.add_argument('--lambda_bismooth', type=float, default=0.0,
                        help='compatibility placeholder')
    parser.add_argument('--lambda_bismooth_cls', type=float, default=0.0,
                        help='compatibility placeholder')
    parser.add_argument('--lambda_layerwise_gate', type=float, default=0.0,
                        help='compatibility placeholder')

    relation_choices = [
        'signed_hadamard',
        'signed_qk',
        'signed_concat_diff',
        'signed_concat_diff_degree',
        'signed_cosine_hadamard',
        'signed_diff',
        # backward-compatible aliases
        'hadamard', 'mul', 'stable_qk', 'concat_diff', 'diff', 'degree',
    ]
    parser.add_argument('--relation_mode', type=str, default='signed_hadamard', choices=relation_choices,
                        help='operator -- form. signed_hadamard uses projected per-dim product')
    parser.add_argument('--edge_feat_mode', type=str, default='signed_hadamard', choices=relation_choices,
                        help='backward-compatible alias of relation_mode')
    parser.add_argument('--score_scale', type=float, default=1.0,
                        help='signed interval is [-score_scale, score_scale]')
    parser.add_argument('--edge_score_temp', type=float, default=1.0,
                        help='temperature for f1/f2/f3 signed mappings')
    parser.add_argument('--edge_blend', type=float, default=0.2,
                        help='residual strength of edge-aware enhancement')
    parser.add_argument('--softcut_lambda', type=float, default=0.1,
                        help='dynamic correction ratio for f2 after anchoring to f1')
    parser.add_argument('--softcut_epsilon', type=float, default=0.05,
                        help='minimum absolute f1 strength before f2 dynamic correction is trusted')
    parser.add_argument('--softcut_sign_temp', type=float, default=5.0,
                        help='temperature for f1/f2 same-sign trust gate')
    parser.add_argument('--softcut_strength_temp', type=float, default=5.0,
                        help='temperature for f1 strength trust gate')
    parser.add_argument('--softcut_margin', type=float, default=0.0,
                        help='margin before f2-f1 discrepancy enters spurious branch')
    parser.add_argument('--softcut_env_scale', type=float, default=0.5,
                        help='scale of f2-f1 discrepancy routed to spurious context')
    parser.add_argument('--operator_margin', type=float, default=0.1,
                        help='hinge margin in two-hop operator loss')
    parser.add_argument('--twohop_sample_size', type=int, default=512,
                        help='number of sampled u->v->t triplets per step for operator loss')

    parser.add_argument('--use_node_enhance', action='store_true', dest='use_node_enhance',
                        help='enable useful signed neighbor enhancement')
    parser.add_argument('--disable_node_enhance', action='store_false', dest='use_node_enhance',
                        help='disable useful signed neighbor enhancement')
    parser.add_argument('--use_neighbor_denoise', action='store_true', dest='use_neighbor_denoise',
                        help='enable soft subtraction of spurious neighbor summary')
    parser.add_argument('--disable_neighbor_denoise', action='store_false', dest='use_neighbor_denoise',
                        help='disable soft subtraction of spurious neighbor summary')
    parser.add_argument('--noise_subtract_alpha', type=float, default=0.05,
                        help='strength for subtracting the spurious neighbor component')
    parser.add_argument('--noise_gate_temp', type=float, default=1.0,
                        help='temperature for the node-level spurious subtraction gate')

    parser.add_argument('--use_dag_module', action='store_true', dest='use_dag_module',
                        help='enable DAG gate and move environment context extraction into DAG branch')
    parser.add_argument('--disable_dag_module', action='store_false', dest='use_dag_module',
                        help='disable DAG/context branch and keep node-enhancement-only classification')
    parser.add_argument('--dag_latent_dim', type=int, default=16,
                        help='compact node/edge latent dimension used by the optional DAG')
    parser.add_argument('--disable_dag_mixer', action='store_false', dest='use_dag_mixer',
                        help='use the simpler context mixer instead of DAG-aware latent mixer')
    parser.add_argument('--dag_mixer_heads', type=int, default=1,
                        help='number of attention heads in the optional DAG-aware context mixer')
    parser.add_argument('--dag_mixer_layers', type=int, default=2,
                        help='number of layers in the optional DAG-aware context mixer')
    parser.add_argument('--mediator_temp', type=float, default=8.0,
                        help='temperature for DAG mediator gate')
    parser.add_argument('--mediator_threshold', type=float, default=0.5,
                        help='threshold applied to DAG causal-pollution score')
    parser.add_argument('--low_temp', type=float, default=8.0,
                        help='temperature for separating low/high DAG label-effect dimensions')
    parser.add_argument('--low_threshold', type=float, default=0.35,
                        help='threshold for low/high DAG label-effect dimensions')
    parser.add_argument('--pollution_coeff', type=float, default=1.0,
                        help='DAG pollution penalty coefficient')
    parser.add_argument('--edge_pollution_coeff', type=float, default=0.5,
                        help='DAG edge-to-node pollution coefficient')
    parser.add_argument('--causal_support_coeff', type=float, default=0.5,
                        help='DAG causal support coefficient')

    parser.add_argument('--fd_blend', type=float, default=0.5,
                        help='blend ratio between mediator logits and front-door logits')
    parser.add_argument('--use_node_conditioned_context', action='store_true', dest='use_node_conditioned_context',
                        help='enable per-node solving f1(x_origin, context) ~= a_i')
    parser.add_argument('--disable_node_conditioned_context', action='store_false', dest='use_node_conditioned_context',
                        help='disable per-node context solving and average over context bank')
    parser.add_argument('--node_context_topk', type=int, default=1,
                        help='number of contexts selected from the bank per node; <=0 uses K or 1')
    parser.add_argument('--context_bank_type', type=str, default='prototype',
                        choices=['prototype', 'memory', 'prototype_memory'],
                        help='candidate bank for solving f1(x_origin, context) ~= a_i')
    parser.add_argument('--context_memory_size', type=int, default=512,
                        help='max memory-bank contexts if context_bank_type uses memory; 0 means all')
    parser.add_argument('--disable_context_detach', action='store_false', dest='context_detach',
                        help='allow gradients through context bank values')
    parser.set_defaults(context_detach=True)
    parser.add_argument('--pseudo_env_balance', type=float, default=1.0,
                        help='balance term inside pseudo-environment discovery')

    # Deprecated compatibility switches. They do not enable GMM in this version.
    parser.add_argument('--disable_spu_gmm', action='store_true', dest='deprecated_disable_spu_gmm',
                        help='deprecated: GMM is removed in the bank-based version')
    parser.add_argument('--gmm_sample_k', type=int, default=0,
                        help='deprecated: ignored; use context_bank_type/context_memory_size instead')
    parser.add_argument('--gmm_min_var', type=float, default=1e-4,
                        help='deprecated: ignored')
    parser.add_argument('--gmm_max_std', type=float, default=1.0,
                        help='deprecated: ignored')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='deprecated: ignored because GMM/EMA context sampling is removed')

    parser.add_argument('--disable_cipt_schedule', action='store_false', dest='use_cipt_schedule',
                        help='disable CIPT warmup/ramp schedule')
    parser.add_argument('--decomp_warmup_epochs', type=int, default=50,
                        help='warmup epochs emphasizing mediator/spurious decomposition')
    parser.add_argument('--intervention_ramp_epochs', type=int, default=100,
                        help='epochs used to ramp front-door and operator losses')
    parser.add_argument('--min_intervention_scale', type=float, default=0.0,
                        help='minimum scale for intervention/operator/context losses during warmup')
    parser.add_argument('--disable_cosine_lr', action='store_false', dest='use_cosine_lr',
                        help='disable cosine annealing learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='minimum learning rate for cosine annealing')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='gradient clipping norm; <=0 disables')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='stop if validation does not improve for this many epochs; 0 disables')
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
        'lambda_med', 'lambda_spu', 'lambda_ind', 'lambda_fd',
        'lambda_env', 'lambda_spu_env', 'lambda_var', 'lambda_operator',
        'lambda_context', 'lambda_dag', 'lambda_dag_label', 'lambda_bismooth',
        'lambda_bismooth_cls', 'lambda_layerwise_gate',
    )
    return {name: float(getattr(model, name)) for name in lambda_names if hasattr(model, name)}


def restore_lambda_state(model, lambda_state):
    for name, value in lambda_state.items():
        setattr(model, name, value)


def cosine_rampup(progress):
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * progress)


def apply_cipt_schedule(model, base_lambdas, epoch, args):
    restore_lambda_state(model, base_lambdas)
    if not args.use_cipt_schedule:
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

    for name in (
        'lambda_fd', 'lambda_env', 'lambda_spu_env', 'lambda_var',
        'lambda_operator', 'lambda_context', 'lambda_dag', 'lambda_dag_label',
    ):
        if name in base_lambdas:
            setattr(model, name, base_lambdas[name] * intervention_scale)
    return {'intervention_scale': intervention_scale, 'decomp_scale': 1.0}


parser = argparse.ArgumentParser(description='DAG-free Signed-Operator CIPT Two-Layer GNN Training Pipeline')
parser_add_main_args(parser)
add_signed_cipt_args(parser)
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

model = GraphSoftCutCIPT(d, c, args, device).to(device)

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
        f"{current_time}_signed_cipt_{args.relation_mode}_med{args.lambda_med}_spu{args.lambda_spu}"
        f"_ind{args.lambda_ind}_op{args.lambda_operator}_fd{args.lambda_fd}"
    )
log_dir = os.path.join('.', 'runs', args.dataset, 'signed_operator_cipt', run_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logging activated. Logs will be saved to: {log_dir}")
print(
    f"[INFO] Training recipe: DAG-free Signed-Operator CIPT | schedule: {args.use_cipt_schedule} | "
    f"cosine lr: {args.use_cosine_lr} | warmup: {args.decomp_warmup_epochs} | "
    f"ramp: {args.intervention_ramp_epochs} | grad clip: {args.grad_clip} | "
    f"relation: {args.relation_mode} | score_scale: {args.score_scale} | "
    f"softcut_lambda: {args.softcut_lambda} | epsilon: {args.softcut_epsilon} | "
    f"env_scale: {args.softcut_env_scale} | twohop: {args.twohop_sample_size} | "
    f"context bank: {args.context_bank_type} (memory_size={args.context_memory_size}, detach={args.context_detach}) | "
    f"node-conditioned solve: {args.use_node_conditioned_context} (topk={args.node_context_topk}) | "
    f"node enhance: {args.use_node_enhance} | "
    f"neighbor denoise: {args.use_neighbor_denoise} (alpha={args.noise_subtract_alpha}, temp={args.noise_gate_temp}) | "
    f"DAG/context: {args.use_dag_module} (mixer={args.use_dag_mixer}, dim={args.dag_latent_dim})"
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
        writer.add_scalar('Loss/ClsFinal', losses['loss_cls'].item(), global_step)
        writer.add_scalar('Loss/MediatorLc', (model.lambda_med * losses['loss_med']).item(), global_step)
        writer.add_scalar('Loss/SpuriousUniformLde', (model.lambda_spu * losses['loss_spu']).item(), global_step)
        writer.add_scalar('Loss/IndependenceLind', (model.lambda_ind * losses['loss_ind']).item(), global_step)
        writer.add_scalar('Loss/FrontDoor', (model.lambda_fd * losses['loss_fd']).item(), global_step)
        writer.add_scalar('Loss/EnvMed', (model.lambda_env * losses['loss_env_med']).item(), global_step)
        writer.add_scalar('Loss/SpuEnv', (model.lambda_spu_env * losses['loss_spu_env']).item(), global_step)
        writer.add_scalar('Loss/ContextVariance', (model.lambda_var * losses['loss_var']).item(), global_step)
        writer.add_scalar('Loss/SignedOperator', (model.lambda_operator * losses['loss_operator']).item(), global_step)
        writer.add_scalar('Loss/ContextSolve', (model.lambda_context * losses['loss_context']).item(), global_step)
        writer.add_scalar('Graph/SignedF1Mean', losses['signed_f1_mean'].item(), global_step)
        writer.add_scalar('Graph/SignedF2Mean', losses['signed_f2_mean'].item(), global_step)
        writer.add_scalar('Graph/SoftCutSignedMean', losses['softcut_inv_gate_mean'].item(), global_step)
        writer.add_scalar('Graph/EnvRouteMean', losses['softcut_env_gate_mean'].item(), global_step)
        writer.add_scalar('Graph/ContextTargetMean', losses['context_target_mean'].item(), global_step)
        writer.add_scalar('Graph/ContextMatchMean', losses['context_match_mean'].item(), global_step)
        writer.add_scalar('Graph/NumContextsPerNode', losses['num_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumBankContexts', losses['num_bank_contexts'].item(), global_step)
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
                f"MedLc: {(model.lambda_med * losses['loss_med']).item():.4f}, "
                f"SpuLde: {(model.lambda_spu * losses['loss_spu']).item():.4f}, "
                f"Ind: {(model.lambda_ind * losses['loss_ind']).item():.4f}, "
                f"FD: {(model.lambda_fd * losses['loss_fd']).item():.4f}, "
                f"EnvMed: {(model.lambda_env * losses['loss_env_med']).item():.4f}, "
                f"SpuEnv: {(model.lambda_spu_env * losses['loss_spu_env']).item():.4f}, "
                f"Var: {(model.lambda_var * losses['loss_var']).item():.4f}, "
                f"SignedOp: {(model.lambda_operator * losses['loss_operator']).item():.4f}, "
                f"CtxSolve: {(model.lambda_context * losses['loss_context']).item():.4f}, "
                f"F1: {losses['signed_f1_mean'].item():.3f}, "
                f"F2: {losses['signed_f2_mean'].item():.3f}, "
                f"Soft: {losses['softcut_inv_gate_mean'].item():.3f}, "
                f"EnvRoute: {losses['softcut_env_gate_mean'].item():.3f}, "
                f"Target: {losses['context_target_mean'].item():.3f}, "
                f"Match: {losses['context_match_mean'].item():.3f}, "
                f"Ctx: {int(losses['num_contexts'].item())}, "
                f"Bank: {int(losses['num_bank_contexts'].item())}, "
                f"LR: {current_lr:.6f}, "
                f"IntScale: {schedule_state['intervention_scale']:.3f}, "
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
