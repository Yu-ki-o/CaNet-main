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
from model_gmm3_reviewed1_multibranch import GraphFrontDoorDAG
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
                        help='EMA momentum for spurious context statistics')
    parser.add_argument('--lambda_l1', type=float, default=1e-5,
                        help='L1 sparsity weight inside the DAG regularizer')
    parser.add_argument('--lambda_ind', type=float, default=0.0,
                        help='deprecated in DAG-Core: mediator-spurious decorrelation is logged as zero')
    parser.add_argument('--lambda_dag', type=float, default=0.05,
                        help='weight of DAG acyclicity/sparsity regularization')
    parser.add_argument('--lambda_med', type=float, default=0.0,
                        help='deprecated in DAG-Core: mediator-only supervision is disabled')
    parser.add_argument('--lambda_spu', type=float, default=0.05,
                        help='weight of spurious branch pseudo-environment loss')
    parser.add_argument('--lambda_fd', type=float, default=0.5,
                        help='weight of front-door aggregation loss')
    parser.add_argument('--lambda_fd_aug', type=float, default=0.0,
                        help='deprecated in DAG-Core: mixed-context extra supervision is disabled')
    parser.add_argument('--lambda_var', type=float, default=0.0,
                        help='deprecated in DAG-Core: cross-context variance penalty is disabled')
    parser.add_argument('--lambda_env', type=float, default=0.05,
                        help='weight of environment-uniform loss on mediator branch')
    parser.add_argument('--lambda_inv', type=float, default=0.0,
                        help='deprecated in DAG-Core: cross-environment prediction invariance is disabled')
    parser.add_argument('--lambda_gate', type=float, default=0.0,
                        help='optional mean-gate sparsity regularizer inside the DAG loss')
    parser.add_argument('--lambda_dag_label', type=float, default=0.05,
                        help='weight of direct DAG node/edge latent -> label supervision')
    parser.add_argument('--lambda_sem', type=float, default=0.0,
                        help='deprecated in DAG-Core: semantic reconstruction is removed')
    parser.add_argument('--lambda_spu_y', type=float, default=0.0,
                        help='deprecated in DAG-Core: environment-conditioned spurious label supervision is disabled')
    parser.add_argument('--dag_latent_dim', type=int, default=16,
                        help='bottleneck dimension used for node/edge variables in the learned DAG')
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
    parser.add_argument('--edge_pollution_coeff', type=float, default=0.5,
                        help='penalty coefficient for pseudo-env-sensitive edge influence')
    parser.add_argument('--causal_support_coeff', type=float, default=0.5,
                        help='bonus for support from high-label-effect bottleneck features')
    parser.add_argument('--counterexample_coeff', type=float, default=0.0,
                        help='penalty strength for DAG dimensions that fail on hard counterexample samples')
    parser.add_argument('--counterexample_top_frac', type=float, default=0.2,
                        help='fraction of hardest train samples used to estimate per-dimension counterexample penalty')
    parser.add_argument('--counterexample_momentum', type=float, default=0.9,
                        help='EMA momentum for the per-dimension counterexample penalty')
    parser.add_argument('--pseudo_env_balance', type=float, default=1.0,
                        help='balance weight inside label-free pseudo-environment discovery')
    parser.add_argument('--edge_env_momentum', type=float, default=0.9,
                        help='EMA momentum for edge-dimension pseudo-env sensitivity')
    parser.add_argument('--edge_score_temp', type=float, default=2.0,
                        help='temperature for edge semantic gate logits; larger values produce smoother gates')
    parser.add_argument('--edge_blend', type=float, default=0.2,
                        help='residual strength of edge-aware neighbor aggregation')
    parser.add_argument('--node_enhance_branches', type=int, default=3,
                        help='number of CGRL-style branches in the edge-aware node enhancement block')
    parser.add_argument('--node_enhance_combine', type=str, default='mean',
                        choices=['mean', 'sum', 'learned'],
                        help='how to combine multi-branch node enhancement deltas')
    parser.add_argument('--use_neighbor_denoise', action='store_true',
                        help='subtract a gated low-relevance neighbor summary from the node representation')
    parser.add_argument('--noise_subtract_alpha', type=float, default=0.1,
                        help='strength for subtracting the low-relevance neighbor component')
    parser.add_argument('--noise_gate_temp', type=float, default=1.0,
                        help='temperature for the node-level low-relevance subtraction gate')
    parser.add_argument('--edge_feat_mode', type=str, default='mul',
                        choices=['mul', 'diff', 'degree', 'mul_diff', 'concat', 'concat_diff',
                                 'mul_degree', 'diff_degree', 'mul_diff_degree'],
                        help=(
                            'edge feature used by the edge reliability MLP: '
                            'mul=h_src*h_dst, diff=|h_src-h_dst|, concat=[h_src,h_dst], '
                            'degree=max normalized log degree, or combinations such as '
                            'mul_diff, concat_diff, diff_degree and mul_diff_degree'
                        ))
    parser.add_argument('--use_layerwise_local_igm', action='store_true',
                        help='apply Local-IGM edge routing after each GNN layer before the next layer')
    parser.add_argument('--layerwise_local_igm_include_last', action='store_false',
                        dest='layerwise_local_igm_skip_last',
                        help='also apply layer-wise Local-IGM after the final GNN layer')
    parser.set_defaults(layerwise_local_igm_skip_last=True)
    parser.add_argument('--disable_layerwise_final_edge_fuse', action='store_false',
                        dest='layerwise_final_edge_fuse',
                        help='keep final edge summary for DAG variables but do not fuse it into z after layer-wise routing')
    parser.set_defaults(layerwise_final_edge_fuse=True)
    parser.add_argument('--layerwise_gate_target', type=float, default=0.5,
                        help='target mean edge gate for optional layer-wise gate budget regularization')
    parser.add_argument('--lambda_layerwise_gate', type=float, default=0.0,
                        help='weight for the optional layer-wise gate budget loss')
    parser.add_argument('--use_layerwise_spurious_contexts', action='store_true',
                        help='add each GNN layer spurious branch to the front-door environment context bank')
    parser.add_argument('--layerwise_spurious_context_weight', type=float, default=1.0,
                        help='scale of layer-wise spurious environment contexts')
    parser.add_argument('--disable_layerwise_spurious_context_detach', action='store_false',
                        dest='layerwise_spurious_context_detach',
                        help='allow gradients to flow through layer-wise spurious contexts')
    parser.set_defaults(layerwise_spurious_context_detach=True)
    parser.add_argument('--use_global_info', action='store_true',
                        help='inject MLEI/AdvDIFFormer-style global information before DAG/front-door splitting')
    parser.add_argument('--disable_global_contexts', action='store_false', dest='use_global_contexts',
                        help='disable MLEI global-attention context prototypes in the front-door context bank')
    parser.set_defaults(use_global_contexts=None)
    parser.add_argument('--global_info_mode', type=str, default='advective', choices=['linear', 'advective'],
                        help="'linear' uses MLEI-style global linear attention; 'advective' mixes it with local topology")
    parser.add_argument('--global_alpha', type=float, default=0.2,
                        help='residual strength of the global information channel')
    parser.add_argument('--global_beta', type=float, default=0.5,
                        help='topology reliance weight beta for advective global-local mixing')
    parser.add_argument('--global_steps', type=int, default=1,
                        help='number of lightweight advective propagation steps')
    parser.add_argument('--global_local_source', type=str, default='gcn', choices=['edge', 'gcn'],
                        help="'edge' uses learned edge-gated aggregation as AdvDIFFormer V; 'gcn' uses raw GCN propagation")
    parser.add_argument('--global_context_weight', type=float, default=1.0,
                        help='scale of global-attention environment prototypes used as front-door contexts')
    parser.add_argument('--lambda_global_env', type=float, default=0.0,
                        help='weight of local/global pseudo-environment consistency')
    parser.add_argument('--use_local_bismooth', action='store_true',
                        help='enable node-aware bi-smoothed local mediator consistency')
    parser.add_argument('--lambda_bismooth', type=float, default=0.0,
                        help='weight of local bi-smoothed mediator consistency loss')
    parser.add_argument('--lambda_bismooth_cls', type=float, default=0.0,
                        help='optional supervised loss on the mediator from bi-smoothed graphs')
    parser.add_argument('--bismooth_edge_drop', type=float, default=0.1,
                        help='edge deletion probability for local node-aware bi-smoothing')
    parser.add_argument('--bismooth_node_drop', type=float, default=0.05,
                        help='node deletion probability for local node-aware bi-smoothing; incident edges are removed')
    parser.add_argument('--bismooth_samples', type=int, default=1,
                        help='number of randomized bi-smoothed graphs per training step')
    parser.add_argument('--bismooth_consistency', type=str, default='cosine', choices=['cosine', 'mse'],
                        help='mediator consistency metric between clean and bi-smoothed graphs')
    parser.add_argument('--bismooth_drop_train_nodes', action='store_false', dest='bismooth_keep_train_nodes',
                        help='allow training/query nodes themselves to be node-dropped in bi-smoothing')
    parser.set_defaults(bismooth_keep_train_nodes=True)
    parser.add_argument('--bismooth_singleton', type=str, default='exclude', choices=['include', 'exclude'],
                        help='exclude isolated nodes from the local bi-smoothing loss, following node-aware-exclude')
    parser.add_argument('--fd_blend', type=float, default=0.5,
                        help='blend ratio between mediator logits and front-door aggregated logits')
    parser.add_argument('--proto_aug_k', type=int, default=1,
                        help='deprecated: prototype mixup is replaced by GMM context sampling in the DAG model')
    parser.add_argument('--proto_mix_alpha', type=float, default=1.0,
                        help='deprecated: prototype mixup is replaced by GMM context sampling in the DAG model')
    parser.add_argument('--disable_spu_gmm', action='store_false', dest='use_spu_gmm',
                        help='disable GMM sampling for spurious environment contexts')
    parser.set_defaults(use_spu_gmm=True)
    parser.add_argument('--gmm_sample_k', type=int, default=3,
                        help='number of GMM-sampled spurious contexts; <=0 uses K')
    parser.add_argument('--gmm_min_var', type=float, default=1e-4,
                        help='minimum diagonal variance used by spurious-context GMM')
    parser.add_argument('--gmm_max_std', type=float, default=1.0,
                        help='maximum std for GMM context sampling; <=0 disables clipping')
    parser.add_argument('--disable_dag_mixer', action='store_false', dest='use_dag_mixer',
                        help='disable DAG-masked latent mixing and use the old concat fuser')
    parser.set_defaults(use_dag_mixer=True)
    parser.add_argument('--dag_mixer_heads', type=int, default=1,
                        help='attention heads in the DAG-masked latent mixer')
    parser.add_argument('--dag_mixer_layers', type=int, default=2,
                        help='number of masked latent attention layers')
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
        'lambda_bismooth',
        'lambda_bismooth_cls',
        'lambda_layerwise_gate',
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

    for name in ('lambda_fd', 'lambda_fd_aug', 'lambda_var', 'lambda_env', 'lambda_inv', 'lambda_bismooth', 'lambda_bismooth_cls', 'lambda_layerwise_gate'):
        setattr(model, name, base_lambdas[name] * intervention_scale)
    setattr(model, 'lambda_dag', base_lambdas['lambda_dag'] * dag_scale)

    return {
        'intervention_scale': intervention_scale,
        'dag_scale': dag_scale,
    }


parser = argparse.ArgumentParser(description='Graph Front-Door DAG-Core Training Pipeline')
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
        f"{current_time}_fd_dag_core_d{args.lambda_dag}_dl{args.lambda_dag_label}_fd{args.lambda_fd}"
        f"_warm{args.decomp_warmup_epochs}_ramp{args.intervention_ramp_epochs}"
    )
log_dir = os.path.join('.', 'runs', args.dataset, 'frontdoor_dag_core', run_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logging activated. Logs will be saved to: {log_dir}")
print(
    f"[INFO] Training recipe: DAG-Core | CIPT schedule: {args.use_cipt_schedule} | "
    f"cosine lr: {args.use_cosine_lr} | warmup: {args.decomp_warmup_epochs} | "
    f"ramp: {args.intervention_ramp_epochs} | grad clip: {args.grad_clip} | "
    f"DAG mixer: {args.use_dag_mixer} | edge feat: {args.edge_feat_mode} | "
    f"node enhance branches: {args.node_enhance_branches} "
    f"(combine={args.node_enhance_combine}) | "
    f"neighbor denoise: {args.use_neighbor_denoise} (alpha={args.noise_subtract_alpha}, "
    f"temp={args.noise_gate_temp}) | "
    f"GMM contexts: {args.use_spu_gmm} | "
    f"GMM sample k: {args.gmm_sample_k if args.gmm_sample_k > 0 else args.K} | "
    f"layerwise local IGM: {args.use_layerwise_local_igm} "
    f"(skip_last={args.layerwise_local_igm_skip_last}, final_fuse={args.layerwise_final_edge_fuse}, "
    f"target={args.layerwise_gate_target}, lambda={args.lambda_layerwise_gate}) | "
    f"layerwise spurious ctx: {args.use_layerwise_spurious_contexts} "
    f"(weight={args.layerwise_spurious_context_weight}, "
    f"detach={args.layerwise_spurious_context_detach}) | "
    f"local bi-smooth: {args.use_local_bismooth} "
    f"(lambda={args.lambda_bismooth}, cls={args.lambda_bismooth_cls}, "
    f"pe={args.bismooth_edge_drop}, pn={args.bismooth_node_drop}, "
    f"samples={args.bismooth_samples}, keep_train={args.bismooth_keep_train_nodes}, "
    f"singleton={args.bismooth_singleton}) | "
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
        writer.add_scalar('Loss/BiSmooth', (model.lambda_bismooth * losses['loss_bismooth']).item(), global_step)
        writer.add_scalar('Loss/BiSmoothCls', (model.lambda_bismooth_cls * losses['loss_bismooth_cls']).item(), global_step)
        writer.add_scalar('Loss/LayerwiseGate', (model.lambda_layerwise_gate * losses['loss_layerwise_gate']).item(), global_step)
        writer.add_scalar('Graph/BiSmoothValidRatio', losses['bismooth_valid_ratio'].item(), global_step)
        writer.add_scalar('Graph/LayerwiseGateMean', losses['layerwise_gate_mean'].item(), global_step)
        writer.add_scalar('Graph/LayerwiseGateLayers', losses['layerwise_gate_layers'].item(), global_step)
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
                f"BiSmooth: {(model.lambda_bismooth * losses['loss_bismooth']).item():.4f}, "
                f"BiSmoothCls: {(model.lambda_bismooth_cls * losses['loss_bismooth_cls']).item():.4f}, "
                f"LayerGate: {(model.lambda_layerwise_gate * losses['loss_layerwise_gate']).item():.4f}, "
                f"LayerGateMean: {losses['layerwise_gate_mean'].item():.3f}, "
                f"LayerGateLayers: {int(losses['layerwise_gate_layers'].item())}, "
                f"BiValid: {losses['bismooth_valid_ratio'].item():.3f}, "
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
