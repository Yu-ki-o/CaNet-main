#当前模型默认开始edge_blend 为0.2，默认未开启边去噪（neighbor denoise），默认edge_feat_mode为'mul'，默认gate_mode为'residual'，默认前门混合器（front-door mixer）开启，默认使用CIPT课程安排，默认使用余弦学习率调度。其他参数可通过命令行调整。


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
from model_frontdoor_gate import GraphFrontDoorDAG
from parse import parser_add_main_args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_frontdoor_gate_args(parser):
    parser.set_defaults(use_cipt_schedule=True, use_cosine_lr=True)
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='EMA momentum for spurious context statistics')

    # Lightweight front-door gate losses. Old DAG argument names are kept where
    # possible so existing launch scripts do not break, but DAG losses default to 0.
    parser.add_argument('--lambda_med', type=float, default=0.2,
                        help='weight of mediator supervised loss CE(classifier(z_m), y)')
    parser.add_argument('--lambda_fd', type=float, default=0.05,
                        help='weight of front-door intervention loss')
    parser.add_argument('--lambda_spu', type=float, default=0.01,
                        help='weight of pseudo-environment clustering loss on z_spurious')
    parser.add_argument('--lambda_env', type=float, default=0.0,
                        help='weight of environment-uniform loss on z_mediator')
    parser.add_argument('--lambda_spu_y', type=float, default=0.0,
                        help='optional tiny label-uniform loss on spurious branch; keep very small')
    parser.add_argument('--lambda_gate', type=float, default=0.0,
                        help='optional gate confidence regularizer')
    parser.add_argument('--lambda_var', type=float, default=0.0,
                        help='optional cross-context prediction variance regularizer')
    parser.add_argument('--lambda_ind', type=float, default=0.0,
                        help='optional mediator/spurious decorrelation loss')

    # Deprecated DAG weights kept for logging and old commands.
    parser.add_argument('--lambda_l1', type=float, default=0.0)
    parser.add_argument('--lambda_dag', type=float, default=0.0)
    parser.add_argument('--lambda_dag_label', type=float, default=0.0)
    parser.add_argument('--lambda_fd_aug', type=float, default=0.0)
    parser.add_argument('--lambda_inv', type=float, default=0.0)
    parser.add_argument('--lambda_global_env', type=float, default=0.0)
    parser.add_argument('--lambda_sem', type=float, default=0.0)
    parser.add_argument('--dag_latent_dim', type=int, default=16,
                        help='deprecated; kept for old scripts')

    parser.add_argument('--pseudo_env_balance', type=float, default=1.0,
                        help='balance weight inside pseudo-environment discovery')

    # Edge-aware representation enhancement.
    parser.add_argument('--edge_score_temp', type=float, default=2.0,
                        help='temperature for edge semantic gate logits; larger values produce smoother gates')
    parser.add_argument('--edge_blend', type=float, default=0.2,
                        help='residual strength of edge-aware neighbor aggregation')
    parser.add_argument('--use_neighbor_denoise', action='store_true',
                        help='subtract a gated low-relevance neighbor residual summary from node representation')
    parser.add_argument('--noise_subtract_alpha', type=float, default=0.1,
                        help='strength for subtracting the low-relevance neighbor component')
    parser.add_argument('--noise_gate_temp', type=float, default=1.0,
                        help='temperature for the node-level low-relevance subtraction gate')
    parser.add_argument('--edge_feat_mode', type=str, default='mul',
                        choices=['mul', 'diff', 'degree', 'mul_diff', 'mul_degree', 'diff_degree', 'mul_diff_degree'],
                        help='edge feature used by the relation MLP')

    # Front-door mediator-context gate.
    parser.add_argument('--frontdoor_gate_mode', type=str, default='residual', choices=['residual', 'mask'],
                        help="'residual': z_m=LN(h+h*g), z_s=LN(h+h*(1-g)); 'mask': z_m=LN(h*g), z_s=LN(h*(1-g))")
    parser.add_argument('--frontdoor_gate_temp', type=float, default=1.0,
                        help='temperature for the mediator-context gate')
    parser.add_argument('--disable_frontdoor_mixer', action='store_false', dest='use_frontdoor_mixer',
                        help='disable masked front-door mixer and use concat fuser')
    parser.set_defaults(use_frontdoor_mixer=True)
    parser.add_argument('--frontdoor_mixer_heads', type=int, default=1,
                        help='attention heads in the lightweight front-door mixer')
    parser.add_argument('--frontdoor_mixer_layers', type=int, default=1,
                        help='number of masked latent attention layers')

    # Global information/context channel.
    parser.add_argument('--use_global_info', action='store_true',
                        help='inject MLEI/AdvDIFFormer-style global information before front-door splitting')
    parser.add_argument('--disable_global_contexts', action='store_false', dest='use_global_contexts',
                        help='disable global-attention context prototypes in the front-door context bank')
    parser.set_defaults(use_global_contexts=None)
    parser.add_argument('--global_info_mode', type=str, default='advective', choices=['linear', 'advective'])
    parser.add_argument('--global_alpha', type=float, default=0.2)
    parser.add_argument('--global_beta', type=float, default=0.5)
    parser.add_argument('--global_steps', type=int, default=1)
    parser.add_argument('--global_local_source', type=str, default='gcn', choices=['edge', 'gcn'])
    parser.add_argument('--global_context_weight', type=float, default=1.0)

    # Context construction / sampling. Defaults are conservative to reduce OOD2/3 fluctuation.
    parser.add_argument('--fd_blend', type=float, default=0.1,
                        help='blend ratio between mediator logits and front-door aggregated logits')
    parser.add_argument('--disable_proto_context', action='store_false', dest='use_proto_context',
                        help='disable deterministic pseudo-env prototype contexts')
    parser.set_defaults(use_proto_context=True)
    parser.add_argument('--disable_spu_gmm', action='store_false', dest='use_spu_gmm',
                        help='disable GMM sampling for spurious environment contexts')
    parser.set_defaults(use_spu_gmm=True)
    parser.add_argument('--gmm_sample_k', type=int, default=1,
                        help='number of GMM-sampled spurious contexts; <=0 uses K')
    parser.add_argument('--gmm_min_var', type=float, default=1e-4)
    parser.add_argument('--gmm_max_std', type=float, default=0.5,
                        help='maximum std for GMM context sampling; <=0 disables clipping')
    parser.add_argument('--no_context_detach', action='store_false', dest='context_detach',
                        help='allow context prototypes to backpropagate; default detaches for stability')
    parser.set_defaults(context_detach=True)

    # Training schedule / stability.
    parser.add_argument('--disable_cipt_schedule', action='store_false', dest='use_cipt_schedule',
                        help='disable curriculum that warms up decomposition before full intervention')
    parser.add_argument('--decomp_warmup_epochs', type=int, default=30)
    parser.add_argument('--intervention_ramp_epochs', type=int, default=80)
    parser.add_argument('--min_intervention_scale', type=float, default=0.0)
    parser.add_argument('--disable_cosine_lr', action='store_false', dest='use_cosine_lr')
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--early_stop_patience', type=int, default=0)
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0)

def sanitize_name(name):
    safe_name = "".join(
        ch if ch.isalnum() or ch in ('-', '_', '.') else '_'
        for ch in str(name).strip()
    ).strip('._')
    return safe_name


def capture_lambda_state(model):
    lambda_names = (
        'lambda_dag',
        'lambda_dag_label',
        'lambda_med',
        'lambda_spu',
        'lambda_fd',
        'lambda_fd_aug',
        'lambda_var',
        'lambda_ind',
        'lambda_env',
        'lambda_inv',
        'lambda_global_env',
        'lambda_spu_y',
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

    for name in ('lambda_fd', 'lambda_fd_aug', 'lambda_var', 'lambda_env', 'lambda_inv'):
        setattr(model, name, base_lambdas[name] * intervention_scale)
    setattr(model, 'lambda_dag', base_lambdas['lambda_dag'] * dag_scale)

    return {
        'intervention_scale': intervention_scale,
        'dag_scale': dag_scale,
    }


parser = argparse.ArgumentParser(description='Lightweight Front-Door Gate Training Pipeline')
parser_add_main_args(parser)
add_frontdoor_gate_args(parser)
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
        f"{current_time}_fd_gate_d{args.lambda_dag}_dl{args.lambda_dag_label}_fd{args.lambda_fd}"
        f"_warm{args.decomp_warmup_epochs}_ramp{args.intervention_ramp_epochs}"
    )
log_dir = os.path.join('.', 'runs', args.dataset, 'frontdoor_gate', run_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logging activated. Logs will be saved to: {log_dir}")
print(
    f"[INFO] Training recipe: FrontDoor-Gate | CIPT schedule: {args.use_cipt_schedule} | "
    f"cosine lr: {args.use_cosine_lr} | warmup: {args.decomp_warmup_epochs} | "
    f"ramp: {args.intervention_ramp_epochs} | grad clip: {args.grad_clip} | "
    f"FD mixer: {args.use_frontdoor_mixer} | gate mode: {args.frontdoor_gate_mode} | edge feat: {args.edge_feat_mode} | "
    f"neighbor denoise: {args.use_neighbor_denoise} (alpha={args.noise_subtract_alpha}, "
    f"temp={args.noise_gate_temp}) | "
    f"GMM contexts: {args.use_spu_gmm} | "
    f"GMM sample k: {args.gmm_sample_k if args.gmm_sample_k > 0 else args.K} | "
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
        writer.add_scalar('Loss/Med', losses['loss_med'].item(), global_step)
        writer.add_scalar('Loss/FD', losses['loss_fd'].item(), global_step)
        writer.add_scalar('Loss/FDAug', (model.lambda_fd_aug * losses['loss_fd_aug']).item(), global_step)
        writer.add_scalar('Loss/Ind', (model.lambda_ind * losses['loss_ind']).item(), global_step)
        writer.add_scalar('Loss/DAG', (model.lambda_dag * losses['loss_dag']).item(), global_step)
        writer.add_scalar('Loss/DAGLabel', (model.lambda_dag_label * losses['loss_dag_label']).item(), global_step)
        writer.add_scalar('Loss/Spu', (model.lambda_spu * losses['loss_spu']).item(), global_step)
        writer.add_scalar('Loss/SpuY', (model.lambda_spu_y * losses['loss_spu_y']).item(), global_step)
        writer.add_scalar('Loss/EnvMed', (model.lambda_env * losses['loss_env_med']).item(), global_step)
        writer.add_scalar('Loss/Inv', (model.lambda_inv * losses['loss_inv']).item(), global_step)
        writer.add_scalar('Loss/Var', (model.lambda_var * losses['loss_var']).item(), global_step)
        writer.add_scalar('Loss/GlobalEnv', (model.lambda_global_env * losses['loss_global_env']).item(), global_step)
        writer.add_scalar('Graph/MediatorGate', losses['mediator_gate_mean'].item(), global_step)
        writer.add_scalar('Graph/CausalScore', losses['causal_score_mean'].item(), global_step)
        writer.add_scalar('Graph/PollutionScore', losses['pollution_score_mean'].item(), global_step)
        writer.add_scalar('Graph/CounterexamplePenalty', losses['counterexample_penalty_mean'].item(), global_step)
        writer.add_scalar('Graph/CounterexamplePenaltyBatch', losses['counterexample_penalty_batch_mean'].item(), global_step)
        writer.add_scalar('Graph/NumContexts', losses['num_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumMixedContexts', losses['num_mixed_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumGMMContexts', losses['num_gmm_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumGlobalContexts', losses['num_global_contexts'].item(), global_step)
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
                f"FDAug: {(model.lambda_fd_aug * losses['loss_fd_aug']).item():.4f}, "
                f"Ind: {(model.lambda_ind * losses['loss_ind']).item():.4f}, "
                f"DAG: {(model.lambda_dag * losses['loss_dag']).item():.4f}, "
                f"DAGLabel: {(model.lambda_dag_label * losses['loss_dag_label']).item():.4f}, "
                f"Spu: {(model.lambda_spu * losses['loss_spu']).item():.4f}, "
                f"SpuY: {(model.lambda_spu_y * losses['loss_spu_y']).item():.4f}, "
                f"EnvMed: {(model.lambda_env * losses['loss_env_med']).item():.4f}, "
                f"Inv: {(model.lambda_inv * losses['loss_inv']).item():.4f}, "
                f"Var: {(model.lambda_var * losses['loss_var']).item():.4f}, "
                f"GlobalEnv: {(model.lambda_global_env * losses['loss_global_env']).item():.4f}, "
                f"LR: {current_lr:.6f}, "
                f"IntScale: {schedule_state['intervention_scale']:.3f}, "
                f"CEPen: {losses['counterexample_penalty_mean'].item():.3f}, "
                f"Ctx: {int(losses['num_contexts'].item())}, "
                f"GMMCtx: {int(losses['num_gmm_contexts'].item())}, "
                f"GlobalCtx: {int(losses['num_global_contexts'].item())}, "
                f"MixCtx: {int(losses['num_mixed_contexts'].item())}, "
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
