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
from model_cfam_cipt_nego import GraphCFAMCIPTNeGo
from parse import parser_add_main_args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_cfam_cipt_nego_args(parser):
    parser.set_defaults(use_cipt_schedule=True, use_cosine_lr=True, use_dag_mixer=True)
    parser.add_argument('--lambda_fd', type=float, default=0.5,
                        help='weight of the front-door aggregation loss')
    parser.add_argument('--fd_blend', type=float, default=0.5,
                        help='blend ratio between mediator and front-door logits')
    parser.add_argument('--eval_pred_mode', type=str, default='blend',
                        choices=['blend', 'mediator', 'frontdoor'],
                        help='prediction branch used during evaluation')
    parser.add_argument('--edge_feat_mode', type=str, default='mul',
                        choices=['mul', 'diff', 'signed_diff', 'degree',
                                 'mul_diff', 'mul_signed_diff',
                                 'concat', 'concat_diff',
                                 'mul_degree', 'diff_degree',
                                 'mul_diff_degree', 'mul_signed_diff_degree'],
                        help='edge feature used by the Graph-CFAM local gate')
    parser.add_argument('--edge_gate_mode', type=str, default='vector',
                        choices=['scalar', 'vector'],
                        help='scalar or dimension-wise Graph-CFAM edge gate')
    parser.add_argument('--edge_score_temp', type=float, default=2.0,
                        help='temperature for edge gate logits')
    parser.add_argument('--edge_blend', type=float, default=0.2,
                        help='strength of causal local Graph-CFAM enhancement')
    parser.add_argument('--graph_cfam_residual_blend', type=float, default=0.1,
                        help='strength of graph high-pass residual in Graph-CFAM')
    parser.add_argument('--graph_cfam_gate_temp', type=float, default=1.0,
                        help='temperature for dimension-wise Graph-CFAM causal-local gate')
    parser.add_argument('--graph_cfam_gate_target', type=float, default=0.5,
                        help='target mean Graph-CFAM gate')
    parser.add_argument('--lambda_graph_cfam_gate', type=float, default=0.0,
                        help='weight of optional Graph-CFAM gate balance loss')
    parser.add_argument('--lambda_graph_delf', type=float, default=0.0,
                        help='weight of Graph-DELF hard shortcut decoupling loss')
    parser.add_argument('--graph_delf_top_frac', type=float, default=0.2,
                        help='fraction of hard/shortcut-heavy train nodes used by Graph-DELF')
    parser.add_argument('--graph_delf_margin', type=float, default=0.2,
                        help='cosine margin for pushing mediator away from shortcut prototypes')
    parser.add_argument('--graph_delf_shortcut_weight', type=float, default=0.5,
                        help='weight of shortcut-prototype push term inside Graph-DELF')
    parser.add_argument('--disable_env_contexts', action='store_false',
                        dest='use_env_contexts',
                        help='disable pseudo-environment spurious contexts')
    parser.set_defaults(use_env_contexts=True)
    parser.add_argument('--env_context_weight', type=float, default=1.0,
                        help='scale of pseudo-environment front-door contexts')
    parser.add_argument('--env_context_momentum', type=float, default=0.9,
                        help='EMA momentum for inference-time environment context bank')
    parser.add_argument('--detach_env_context', action='store_true',
                        dest='env_context_detach',
                        help='stop front-door gradients through context construction')
    parser.set_defaults(env_context_detach=False)
    parser.add_argument('--use_nego_prompt', action='store_true',
                        help='enable NeGo-lite negative prompt loss')
    parser.add_argument('--use_nego_context', action='store_true',
                        help='add NeGo-lite contexts to the front-door context bank')
    parser.add_argument('--lambda_nego', type=float, default=0.0,
                        help='weight of NeGo-lite prompt loss')
    parser.add_argument('--nego_temp', type=float, default=0.2,
                        help='temperature for NeGo prompt/prototype matching')
    parser.add_argument('--nego_context_weight', type=float, default=1.0,
                        help='scale of NeGo front-door contexts')
    parser.add_argument('--nego_momentum', type=float, default=0.9,
                        help='EMA momentum for inference-time NeGo contexts')
    parser.add_argument('--disable_nego_detach_source', action='store_false',
                        dest='nego_detach_source',
                        help='allow gradients from NeGo prompt loss into its source branch')
    parser.set_defaults(nego_detach_source=True)
    parser.add_argument('--nego_source', type=str, default='spurious',
                        choices=['spurious', 'mediator', 'z'],
                        help='representation source for NeGo negative prompts')
    parser.add_argument('--fd_context_source', type=str, default='mixed',
                        choices=['mixed', 'env_only', 'nego_only'],
                        help='front-door context source')
    parser.add_argument('--disable_dag_mixer', action='store_false',
                        dest='use_dag_mixer',
                        help='disable masked latent mixer and use simple M+C fuser')
    parser.add_argument('--dag_mixer_heads', type=int, default=1,
                        help='attention heads in the masked front-door mixer')
    parser.add_argument('--dag_mixer_layers', type=int, default=2,
                        help='number of masked front-door mixer layers')
    parser.add_argument('--disable_cipt_schedule', action='store_false',
                        dest='use_cipt_schedule',
                        help='disable warmup/ramp schedule for front-door and NeGo losses')
    parser.add_argument('--decomp_warmup_epochs', type=int, default=50,
                        help='epochs before intervention losses ramp up')
    parser.add_argument('--intervention_ramp_epochs', type=int, default=100,
                        help='epochs used to ramp intervention losses')
    parser.add_argument('--min_intervention_scale', type=float, default=0.0,
                        help='minimum scale for intervention losses during warmup')
    parser.add_argument('--disable_cosine_lr', action='store_false',
                        dest='use_cosine_lr',
                        help='disable cosine learning-rate schedule')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='minimum learning rate for cosine annealing')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='gradient clipping norm; <=0 disables clipping')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='stop if validation does not improve for this many epochs')
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
        'lambda_fd',
        'lambda_graph_cfam_gate',
        'lambda_graph_delf',
        'lambda_nego',
    )
    return {name: float(getattr(model, name)) for name in lambda_names}


def restore_lambda_state(model, lambda_state):
    for name, value in lambda_state.items():
        setattr(model, name, value)


def cosine_rampup(progress):
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * progress)


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

    for name in ('lambda_fd', 'lambda_graph_delf', 'lambda_nego'):
        setattr(model, name, base_lambdas[name] * intervention_scale)
    return {'intervention_scale': intervention_scale}


parser = argparse.ArgumentParser(description='CFAM-CIPT-NeGo Front-Door Training Pipeline')
parser_add_main_args(parser)
add_cfam_cipt_nego_args(parser)
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

model = GraphCFAMCIPTNeGo(d, c, args, device).to(device)

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
        f"{current_time}_cfam_cipt_nego_fd{args.lambda_fd}"
        f"_nego{args.lambda_nego}_gcfam{args.lambda_graph_cfam_gate}"
        f"_delf{args.lambda_graph_delf}"
    )
log_dir = os.path.join('.', 'runs', args.dataset, 'cfam_cipt_nego', run_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logging activated. Logs will be saved to: {log_dir}")
print(
    f"[INFO] Training recipe: CFAM -> CIPT dual adapters -> NeGo -> front-door | "
    f"CIPT schedule: {args.use_cipt_schedule} | cosine lr: {args.use_cosine_lr} | "
    f"FD lambda: {args.lambda_fd} | NeGo lambda: {args.lambda_nego} | "
    f"Graph-CFAM gate lambda: {args.lambda_graph_cfam_gate} | "
    f"Graph-DELF lambda: {args.lambda_graph_delf} | "
    f"context source: {args.fd_context_source} | eval pred: {args.eval_pred_mode}"
)

dataset.x = dataset.x.to(device)
dataset.y = dataset.y.to(device)
dataset.edge_index = dataset.edge_index.to(device)
if hasattr(dataset, 'env'):
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

        result = evaluate_full(model, dataset, eval_func, args=args)
        logger.add_result(run, result)
        current_lr = optimizer.param_groups[0]['lr']
        global_step = run * args.epochs + epoch

        writer.add_scalar('Loss/Total', losses['total_loss'].item(), global_step)
        writer.add_scalar('Loss/Cls', losses['loss_cls'].item(), global_step)
        writer.add_scalar('Loss/FD', (model.lambda_fd * losses['loss_fd']).item(), global_step)
        writer.add_scalar('Loss/GraphCFAMGate', (model.lambda_graph_cfam_gate * losses['loss_graph_cfam_gate']).item(), global_step)
        writer.add_scalar('Loss/GraphDELF', (model.lambda_graph_delf * losses['loss_graph_delf']).item(), global_step)
        writer.add_scalar('Loss/NeGo', (model.lambda_nego * losses['loss_nego']).item(), global_step)
        writer.add_scalar('Diag/GraphCFAMGateMean', losses['graph_cfam_gate_mean'].item(), global_step)
        writer.add_scalar('Diag/EdgeGateMean', losses['edge_gate_mean'].item(), global_step)
        writer.add_scalar('Diag/NeGoExtraScore', losses['nego_extra_score'].item(), global_step)
        writer.add_scalar('Diag/NeGoSelfScore', losses['nego_self_score'].item(), global_step)
        writer.add_scalar('Graph/NumContexts', losses['num_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumEnvContexts', losses['num_env_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumNeGoContexts', losses['num_nego_contexts'].item(), global_step)
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
                f"FD: {(model.lambda_fd * losses['loss_fd']).item():.4f}, "
                f"GCFAMGate: {(model.lambda_graph_cfam_gate * losses['loss_graph_cfam_gate']).item():.4f}, "
                f"GDELF: {(model.lambda_graph_delf * losses['loss_graph_delf']).item():.4f}, "
                f"NeGo: {(model.lambda_nego * losses['loss_nego']).item():.4f}, "
                f"GateMean: {losses['graph_cfam_gate_mean'].item():.3f}, "
                f"Ctx: {int(losses['num_contexts'].item())}, "
                f"EnvCtx: {int(losses['num_env_contexts'].item())}, "
                f"NeGoCtx: {int(losses['num_nego_contexts'].item())}, "
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
