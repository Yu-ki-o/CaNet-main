import argparse
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
from model_gmm3_reviewed1_graph_cfam_nego_rscgate import GraphFrontDoorDAG
from parse import parser_add_main_args


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_frontdoor_args(parser):
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='reserved EMA momentum for compatible experiment names')
    parser.add_argument('--lambda_fd', type=float, default=0.5,
                        help='weight of front-door aggregated prediction loss')
    parser.add_argument('--fd_blend', type=float, default=0.5,
                        help='blend ratio between mediator logits and front-door logits')
    parser.add_argument('--eval_pred_mode', type=str, default='mediator',
                        choices=['blend', 'mediator', 'frontdoor'],
                        help='prediction path used during evaluation')

    parser.add_argument('--use_graph_cfam', action='store_true', default=True,
                        help='enable Graph-CFAM node enhancement')
    parser.add_argument('--disable_graph_cfam', action='store_false', dest='use_graph_cfam',
                        help='disable Graph-CFAM for a plain-backbone ablation')
    parser.add_argument('--use_pre_gnn_graph_cfam', action='store_true',
                        help='apply Graph-CFAM before the first GNN layer')
    parser.add_argument('--disable_final_graph_cfam', action='store_false',
                        dest='use_final_graph_cfam',
                        help='skip the final post-GNN Graph-CFAM pass')
    parser.set_defaults(use_final_graph_cfam=True)
    parser.add_argument('--layerwise_graph_cfam_include_last', action='store_false',
                        dest='layerwise_local_igm_skip_last',
                        help='also apply layer-wise Graph-CFAM after the final GNN layer before final CFAM')
    parser.set_defaults(layerwise_local_igm_skip_last=True)
    parser.add_argument('--pre_graph_cfam_blend', type=float, default=0.1,
                        help='causal-local blend for pre-GNN Graph-CFAM')
    parser.add_argument('--pre_graph_cfam_residual_blend', type=float, default=0.0,
                        help='high-pass residual blend for pre-GNN Graph-CFAM')
    parser.add_argument('--graph_cfam_residual_blend', type=float, default=0.1,
                        help='strength of graph high-pass residual in Graph-CFAM')
    parser.add_argument('--graph_cfam_gate_temp', type=float, default=1.0,
                        help='temperature for Graph-CFAM dimension gate')
    parser.add_argument('--graph_cfam_gate_target', type=float, default=0.5,
                        help='target mean Graph-CFAM gate')
    parser.add_argument('--lambda_graph_cfam_gate', type=float, default=0.0,
                        help='weight of Graph-CFAM gate balance regularization')

    parser.add_argument('--use_energy_rsc_gate', action='store_true', default=True,
                        help='enable energy-guided reliability-aware RSC gate')
    parser.add_argument('--disable_energy_rsc_gate', action='store_false',
                        dest='use_energy_rsc_gate',
                        help='disable Energy-guided RSC gate')
    parser.add_argument('--lambda_energy_gate_rec', type=float, default=0.0,
                        help='weight of edge reconstruction loss for Energy RSC reliability')
    parser.add_argument('--energy_rsc_top_frac', type=float, default=0.2,
                        help='fraction of high-gate low-reliability dimensions challenged by RSC')
    parser.add_argument('--energy_rsc_second_weight', type=float, default=0.5,
                        help='strength of the second-stage complementary RSC gate')
    parser.add_argument('--energy_rsc_reliability_temp', type=float, default=1.0,
                        help='temperature for converting edge reconstruction margin to reliability')
    parser.add_argument('--energy_rsc_reliability_floor', type=float, default=0.05,
                        help='minimum reliability multiplier')
    parser.add_argument('--energy_rsc_edge_sample', type=int, default=4096,
                        help='maximum positive edges sampled for Energy RSC')
    parser.add_argument('--disable_energy_rsc_detach_reliability', action='store_false',
                        dest='energy_rsc_detach_reliability',
                        help='allow gradients through the Energy RSC reliability score')
    parser.set_defaults(energy_rsc_detach_reliability=True)

    parser.add_argument('--edge_feat_mode', type=str, default='mul',
                        choices=['mul', 'diff', 'signed_diff', 'degree',
                                 'mul_diff', 'mul_signed_diff',
                                 'concat', 'concat_diff',
                                 'mul_degree', 'diff_degree',
                                 'mul_diff_degree', 'mul_signed_diff_degree'],
                        help='edge feature used by the relation gate')
    parser.add_argument('--edge_gate_mode', type=str, default='vector',
                        choices=['scalar', 'vector'],
                        help='scalar uses one edge gate; vector uses one gate per hidden dimension')
    parser.add_argument('--edge_score_temp', type=float, default=2.0,
                        help='temperature for edge gate logits')
    parser.add_argument('--edge_blend', type=float, default=0.2,
                        help='causal local blend inside Graph-CFAM')
    parser.add_argument('--disable_node_edge_norm', action='store_true',
                        help='skip LayerNorm after plain node-edge fusion when Graph-CFAM is disabled')
    parser.add_argument('--direct_z_spurious_mode', type=str, default='shortcut',
                        choices=['shortcut', 'zero', 'z_adapter'],
                        help='spurious branch used for multi-ratio front-door contexts')

    parser.add_argument('--multi_ratio_spurious_source', type=str, default='self',
                        choices=['self', 'shuffle'],
                        help='source of node-level spurious contexts')
    parser.add_argument('--multi_ratio_spurious_ratios', type=str, default='0,0.33,0.67,1.0',
                        help='comma-separated spurious retention ratios')
    parser.add_argument('--lambda_multi_ratio_fd', type=float, default=0.5,
                        help='weight of mean supervised loss across multi-ratio contexts')
    parser.add_argument('--lambda_multi_ratio_fd_worst', type=float, default=0.2,
                        help='weight of worst-ratio supervised loss')
    parser.add_argument('--lambda_multi_ratio_fd_cons', type=float, default=0.1,
                        help='weight of prediction consistency among ratios')

    parser.add_argument('--use_cosine_lr', action='store_true', default=True,
                        help='enable cosine learning-rate decay')
    parser.add_argument('--disable_cosine_lr', action='store_false', dest='use_cosine_lr',
                        help='use a constant learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='minimum learning rate for cosine decay')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='gradient clipping norm; <= 0 disables')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='stop if validation does not improve for this many epochs')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='minimum validation improvement for early stopping')


def sanitize_name(name):
    safe_name = "".join(
        ch if ch.isalnum() or ch in ('-', '_', '.') else '_'
        for ch in str(name).strip()
    ).strip('._')
    return safe_name


parser = argparse.ArgumentParser(description='Graph-CFAM Energy-RSC Front-Door Training')
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
ood_msg = ""
for i in range(len(dataset.test_ood_idx)):
    ood_msg += f"test ood{i + 1} nodes {dataset.test_ood_idx[i].shape[0]} "
print(ood_msg)
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
    run_name = f"{current_time}_cfam_rsc_fd_lfd{args.lambda_fd}_mr{args.lambda_multi_ratio_fd}"
log_dir = os.path.join('.', 'runs', args.dataset, 'cfam_rsc_frontdoor', run_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logging activated. Logs will be saved to: {log_dir}")
print(
    f"[INFO] Training recipe: Graph-CFAM={args.use_graph_cfam} "
    f"(pre={args.use_pre_gnn_graph_cfam}, final={args.use_final_graph_cfam}, "
    f"gate_lambda={args.lambda_graph_cfam_gate}) | "
    f"Energy-RSC={args.use_energy_rsc_gate} "
    f"(rec_lambda={args.lambda_energy_gate_rec}, top_frac={args.energy_rsc_top_frac}, "
    f"second={args.energy_rsc_second_weight}) | "
    f"multi-ratio contexts={args.multi_ratio_spurious_ratios} "
    f"(source={args.multi_ratio_spurious_source}, mean={args.lambda_multi_ratio_fd}, "
    f"worst={args.lambda_multi_ratio_fd_worst}, cons={args.lambda_multi_ratio_fd_cons}) | "
    f"front-door lambda={args.lambda_fd}, fd_blend={args.fd_blend}, eval={args.eval_pred_mode} | "
    f"cosine lr={args.use_cosine_lr}, grad_clip={args.grad_clip}"
)

dataset.x = dataset.x.to(device)
dataset.y = dataset.y.to(device)
dataset.edge_index = dataset.edge_index.to(device)
dataset.env = dataset.env.to(device)

for run in range(args.runs):
    model.reset_parameters()
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
        writer.add_scalar('Loss/FD', (model.lambda_fd * losses['loss_fd']).item(), global_step)
        writer.add_scalar('Loss/GraphCFAMGate', (model.lambda_graph_cfam_gate * losses['loss_graph_cfam_gate']).item(), global_step)
        writer.add_scalar('Loss/EnergyGateRec', (model.lambda_energy_gate_rec * losses['loss_energy_gate_rec']).item(), global_step)
        writer.add_scalar('Loss/MultiRatioFD', (model.lambda_multi_ratio_fd * losses['loss_multi_ratio_fd']).item(), global_step)
        writer.add_scalar('Loss/MultiRatioFDWorst', (model.lambda_multi_ratio_fd_worst * losses['loss_multi_ratio_fd_worst']).item(), global_step)
        writer.add_scalar('Loss/MultiRatioFDCons', (model.lambda_multi_ratio_fd_cons * losses['loss_multi_ratio_fd_cons']).item(), global_step)
        writer.add_scalar('Gate/GraphCFAMMean', losses['graph_cfam_gate_mean'].item(), global_step)
        writer.add_scalar('Gate/GraphCFAMLayers', losses['graph_cfam_layers'].item(), global_step)
        writer.add_scalar('Gate/EnergyReliability', losses['energy_gate_reliability_mean'].item(), global_step)
        writer.add_scalar('Gate/EnergyRSCMask', losses['energy_gate_rsc_mask_mean'].item(), global_step)
        writer.add_scalar('Gate/EnergySecondGate', losses['energy_gate_second_mean'].item(), global_step)
        writer.add_scalar('Graph/NumContexts', losses['num_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumMultiRatioContexts', losses['num_multi_ratio_contexts'].item(), global_step)
        writer.add_scalar('Graph/MediatorNorm', losses['mediator_norm'].item(), global_step)
        writer.add_scalar('Graph/SpuriousNorm', losses['spurious_norm'].item(), global_step)
        writer.add_scalar('Schedule/LR', current_lr, global_step)
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
                f"Med: {losses['loss_med'].item():.4f}, "
                f"FD: {(model.lambda_fd * losses['loss_fd']).item():.4f}, "
                f"GCFAMGate: {(model.lambda_graph_cfam_gate * losses['loss_graph_cfam_gate']).item():.4f}, "
                f"EGateRec: {(model.lambda_energy_gate_rec * losses['loss_energy_gate_rec']).item():.4f}, "
                f"MRFD: {((model.lambda_multi_ratio_fd * losses['loss_multi_ratio_fd']) + (model.lambda_multi_ratio_fd_worst * losses['loss_multi_ratio_fd_worst']) + (model.lambda_multi_ratio_fd_cons * losses['loss_multi_ratio_fd_cons'])).item():.4f}, "
                f"GCFAMMean: {losses['graph_cfam_gate_mean'].item():.3f}, "
                f"ERel: {losses['energy_gate_reliability_mean'].item():.3f}, "
                f"ERSC: {losses['energy_gate_rsc_mask_mean'].item():.3f}, "
                f"E2Gate: {losses['energy_gate_second_mean'].item():.3f}, "
                f"Ctx: {int(losses['num_contexts'].item())}, "
                f"LR: {current_lr:.6f}, "
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
