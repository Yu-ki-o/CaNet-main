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
from model_frontdoor2_ica_hardsplit_dbgmm import GraphFrontDoor
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
    parser.add_argument('--ind_loss_type', type=str, default='mi', choices=['cosine', 'corr', 'hsic', 'mi'],
                        help='independence loss between causal and spurious representations')
    parser.add_argument('--hsic_sigma', type=float, default=0.0,
                        help='RBF kernel sigma for HSIC; <= 0 uses a median-distance heuristic')
    parser.add_argument('--hsic_max_samples', type=int, default=256,
                        help='maximum nodes used by the O(n^2) HSIC loss')
    parser.add_argument('--lambda_env_causal', type=float, default=0.0,
                        help='weight of environment-uniform loss on the causal branch')
    parser.add_argument('--lambda_spu_env', type=float, default=0.05,
                        help='weight of label-free pseudo-environment discovery on the spurious branch')
    parser.add_argument('--lambda_split_gate', type=float, default=0.01,
                        help='weight of complementary causal/spurious split-gate regularization')
    parser.add_argument('--gate_binary_weight', type=float, default=0.1,
                        help='binary pressure inside the split-gate regularizer')
    parser.add_argument('--lambda_context_recon', type=float, default=0.1,
                        help='weight of observed-pair causal/context composer reconstruction')
    parser.add_argument('--lambda_ica_decor', type=float, default=0.01,
                        help='weight of off-diagonal covariance penalty on the ICA-projected hidden representation')
    parser.add_argument('--lambda_ica_orth', type=float, default=0.01,
                        help='weight of orthogonality penalty on the learnable ICA projection matrix')
    parser.add_argument('--lambda_med', type=float, default=0.5,
                        help='weight of the causal-branch supervision loss')
    parser.add_argument('--lambda_spu', type=float, default=0.1,
                        help='backward-compatible alias; lambda_spu_y is used by the new environment-conditioned spurious label head')
    parser.add_argument('--lambda_spu_y', type=float, default=0.05,
                        help='weight of the environment-conditioned spurious label prediction loss')
    parser.add_argument('--lambda_fd', type=float, default=0.5,
                        help='weight of the intervention-branch supervision loss')
    parser.add_argument('--lambda_fd_aug', type=float, default=0.5,
                        help='weight of per-context label-preserving supervision for prototype augmentation')
    parser.add_argument('--lambda_var', type=float, default=0.05,
                        help='weight of the cross-context front-door variance penalty')
    parser.add_argument('--fd_blend', type=float, default=0.5,
                        help='blend ratio between causal logits and intervention logits')
    parser.add_argument('--context_gate_temp', type=float, default=1.0,
                        help='temperature of the adaptive context gating inside diversity augmentation')
    parser.add_argument('--causal_ratio', type=float, default=0.5,
                        help='ratio of hidden dimensions assigned to the hard causal split when causal_dim <= 0')
    parser.add_argument('--causal_dim', type=int, default=0,
                        help='explicit number of hidden dimensions assigned to causal features; <=0 uses causal_ratio')
    parser.add_argument('--disable_ica', action='store_true',
                        help='disable the learnable ICA/whitening projection before the hard split')
    parser.add_argument('--env_emb_dim', type=int, default=16,
                        help='environment embedding dimension for the spurious conditional label head')
    parser.add_argument('--proto_aug_k', type=int, default=3,
                        help='number of mixed environment prototypes added during training')
    parser.add_argument('--proto_mix_alpha', type=float, default=1.0,
                        help='Beta distribution alpha for environment prototype mixup')
    parser.add_argument('--use_true_env', action='store_true',
                        help='use dataset.env for GMM, prototypes, spurious environment CE, and spurious conditional label head when available')
    parser.add_argument('--use_true_env_for_gmm', action='store_true',
                        help='use dataset.env rather than pseudo environments to update Dirichlet-Barycentric GMM statistics')
    parser.add_argument('--use_true_env_for_prototypes', action='store_true',
                        help='use dataset.env rather than pseudo environments to update front-door spurious prototypes')
    parser.add_argument('--use_true_env_for_spu_env', action='store_true',
                        help='train the spurious environment classifier with CE on dataset.env when available')
    parser.add_argument('--use_true_env_for_spu_y', action='store_true',
                        help='condition the spurious label head on dataset.env when available')
    parser.add_argument('--use_spu_gmm', action='store_true',
                        help='use Dirichlet-Barycentric GMM virtual spurious contexts from EMA environment Gaussians')
    parser.add_argument('--gmm_alpha', type=float, default=0.0,
                        help='small blend weight for logits computed from GMM-sampled spurious contexts')
    parser.add_argument('--gmm_sample_k', type=int, default=0,
                        help='number of Dirichlet-Barycentric GMM virtual spurious contexts')
    parser.add_argument('--gmm_min_var', type=float, default=1e-4,
                        help='minimum diagonal variance for each spurious-environment Gaussian')
    parser.add_argument('--gmm_max_std', type=float, default=1.0,
                        help='maximum diagonal std for GMM context sampling; <=0 disables clipping')
    parser.add_argument('--gmm_cap_by_fd_k', action='store_true',
                        help='cap Dirichlet-Barycentric GMM contexts by K/fd_sample_k to reproduce the old budget')
    parser.add_argument('--virtual_dir_alpha', type=float, default=0.5,
                        help='Dirichlet concentration for virtual environment weights; <1 creates sparse realistic mixtures')
    parser.add_argument('--virtual_between_scale', type=float, default=0.15,
                        help='scale of between-environment variance added to each barycentric Gaussian')
    parser.add_argument('--virtual_sample_temp', type=float, default=0.35,
                        help='noise temperature for sampling around the barycentric virtual mean')
    parser.add_argument('--virtual_maha_max', type=float, default=4.0,
                        help='maximum averaged Mahalanobis distance to observed environment Gaussians; <=0 disables realism gate')
    parser.add_argument('--virtual_eval_noise', action='store_true',
                        help='add stochastic noise to Dirichlet-Barycentric GMM contexts during evaluation')
    parser.add_argument('--lambda_bootstrap', type=float, default=0.0,
                        help='weight of FLOOD-style bootstrap self-supervision during training')
    parser.add_argument('--use_tta_rl', action='store_true',
                        help='adapt the front-door context policy on each test split with an unlabeled bandit objective')
    parser.add_argument('--tta_rl_steps', type=int, default=1,
                        help='number of test-time RL adaptation steps per evaluation split')
    parser.add_argument('--tta_rl_lr', type=float, default=1e-3,
                        help='learning rate for the test-time context policy')
    parser.add_argument('--ttt_feat_drop', type=float, default=0.1,
                        help='feature dropout used to build test-time bootstrap views')
    parser.add_argument('--ttt_edge_drop', type=float, default=0.1,
                        help='edge dropout used to build test-time bootstrap views')
    parser.add_argument('--ttt_ema', type=float, default=0.99,
                        help='EMA momentum for the bootstrap target encoder')
    parser.add_argument('--ttt_reward_conf', type=float, default=1.0,
                        help='weight of prediction-confidence reward in test-time RL')
    parser.add_argument('--ttt_reward_consistency', type=float, default=0.5,
                        help='weight of augmentation-consistency reward in test-time RL')
    parser.add_argument('--ttt_policy_entropy', type=float, default=0.01,
                        help='entropy bonus for the test-time context policy')


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

args.train_env_num = int(getattr(dataset, 'train_env_num', args.K))

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
if hasattr(dataset, 'env_num') and hasattr(dataset, 'train_env_num'):
    print(f'[INFO] env numbers: {dataset.env_num} train env numbers: {dataset.train_env_num}')
else:
    print(f'[INFO] no environment labels found; using {args.K} model-inferred pseudo environments')

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
if hasattr(dataset, 'env'):
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
        model.update_bootstrap_target()
        model.apply_state_update(losses.get('state_payload'))
        result = evaluate_full(model, dataset, eval_func, args if args.use_tta_rl else None)
        logger.add_result(run, result)

        global_step = run * args.epochs + epoch
        writer.add_scalar('Loss/Total', losses['total_loss'].item(), global_step)
        writer.add_scalar('Loss/Cls', losses['loss_cls'].item(), global_step)
        writer.add_scalar('Loss/Ind', (model.lambda_ind * losses['loss_ind']).item(), global_step)
        writer.add_scalar('Loss/Med', (model.lambda_med * losses['loss_med']).item(), global_step)
        writer.add_scalar('Loss/SpuY', (model.lambda_spu_y * losses['loss_spu']).item(), global_step)
        writer.add_scalar('Loss/FD', (model.lambda_fd * losses['loss_fd']).item(), global_step)
        writer.add_scalar('Loss/FDAug', (model.lambda_fd_aug * losses['loss_fd_aug']).item(), global_step)
        writer.add_scalar('Loss/Var', (model.lambda_var * losses['loss_var']).item(), global_step)
        writer.add_scalar('Loss/EnvCausal', (model.lambda_env_causal * losses['loss_env_causal']).item(), global_step)
        writer.add_scalar('Loss/SpuEnv', (model.lambda_spu_env * losses['loss_spu_env']).item(), global_step)
        writer.add_scalar('Loss/SplitGate', (model.lambda_split_gate * losses['loss_split_gate']).item(), global_step)
        writer.add_scalar('Loss/ContextRecon', (model.lambda_context_recon * losses['loss_context_recon']).item(), global_step)
        writer.add_scalar('Loss/ICA_Decor', (model.lambda_ica_decor * losses['loss_ica_decor']).item(), global_step)
        writer.add_scalar('Loss/ICA_Orth', (model.lambda_ica_orth * losses['loss_ica_orth']).item(), global_step)
        writer.add_scalar('Loss/Bootstrap', (model.lambda_bootstrap * losses['loss_bootstrap']).item(), global_step)
        writer.add_scalar('Graph/CausalNorm', losses['causal_norm_mean'].item(), global_step)
        writer.add_scalar('Graph/SpuriousNorm', losses['spurious_norm_mean'].item(), global_step)
        writer.add_scalar('Graph/SplitGateMean', losses['split_gate_mean'].item(), global_step)
        writer.add_scalar('Graph/SplitGateStd', losses['split_gate_std'].item(), global_step)
        writer.add_scalar('Graph/NumContexts', losses['num_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumMixedContexts', losses['num_mixed_contexts'].item(), global_step)
        writer.add_scalar('Graph/NumGMMContexts', losses['num_gmm_contexts'].item(), global_step)
        writer.add_scalar('Graph/UsedTrueEnv', losses['used_true_env'].item(), global_step)
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
                f"SpuY: {(model.lambda_spu_y * losses['loss_spu']).item():.4f}, "
                f"FD: {(model.lambda_fd * losses['loss_fd']).item():.4f}, "
                f"FDAug: {(model.lambda_fd_aug * losses['loss_fd_aug']).item():.4f}, "
                f"Var: {(model.lambda_var * losses['loss_var']).item():.4f}, "
                f"EnvC: {(model.lambda_env_causal * losses['loss_env_causal']).item():.4f}, "
                f"SpuE: {(model.lambda_spu_env * losses['loss_spu_env']).item():.4f}, "
                f"Gate: {(model.lambda_split_gate * losses['loss_split_gate']).item():.4f}, "
                f"Recon: {(model.lambda_context_recon * losses['loss_context_recon']).item():.4f}, "
                f"ICAdec: {(model.lambda_ica_decor * losses['loss_ica_decor']).item():.4f}, "
                f"ICAorth: {(model.lambda_ica_orth * losses['loss_ica_orth']).item():.4f}, "
                f"Boot: {(model.lambda_bootstrap * losses['loss_bootstrap']).item():.4f}, "
                f"GateMean: {losses['split_gate_mean'].item():.3f}, "
                f"GateStd: {losses['split_gate_std'].item():.3f}, "
                f"Ctx: {int(losses['num_contexts'].item())}, "
                f"MixCtx: {int(losses['num_mixed_contexts'].item())}, "
                f"GMMCtx: {int(losses['num_gmm_contexts'].item())}, "
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
