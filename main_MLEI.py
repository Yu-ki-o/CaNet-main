import argparse
import random
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from logger import Logger
from dataset import *
from eval import evaluate_full, eval_acc, eval_rocauc, eval_f1
from parse import parser_add_main_args
from model_MLEI import *
from ica_utils import infer_pseudo_envs_with_ica
from torch.utils.tensorboard import SummaryWriter


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _evaluate_split(y, out, indices, eval_func):
    return eval_func(y[indices], out[indices])


@torch.no_grad()
def evaluate_mlei_diagnostics(model, dataset, eval_func):
    model.eval()

    y_cpu = dataset.y.cpu()
    details = model(dataset.x, dataset.edge_index, return_details=True)

    def collect_metrics(logits):
        out_cpu = logits.cpu()
        result = [
            _evaluate_split(y_cpu, out_cpu, dataset.train_idx, eval_func),
            _evaluate_split(y_cpu, out_cpu, dataset.valid_idx, eval_func),
            _evaluate_split(y_cpu, out_cpu, dataset.test_in_idx, eval_func),
        ]
        for split_idx in dataset.test_ood_idx:
            result.append(_evaluate_split(y_cpu, out_cpu, split_idx, eval_func))
        return result

    branch_results = {
        'fused': collect_metrics(details["fused_logits"]),
        'local': collect_metrics(details["local_logits"]),
        'global': collect_metrics(details["global_logits"]),
    }

    train_idx = dataset.train_idx

    def entropy(probs):
        probs = probs.clamp_min(1e-12)
        return float((-(probs * probs.log()).sum(dim=-1).mean()).item())

    env_stats = {}
    global_env = details["global_env"]
    env_stats["Env/Global_Entropy_Train"] = entropy(global_env[train_idx])
    env_stats["Env/Global_MaxProb_Train"] = float(global_env[train_idx].max(dim=-1).values.mean().item())
    for env_id in range(global_env.size(1)):
        env_stats[f"Env/Global_Usage_Train/env{env_id}"] = float(global_env[train_idx, env_id].mean().item())

    for layer_id, local_env in enumerate(details["local_envs"], start=1):
        env_stats[f"Env/Layer_{layer_id}_Entropy_Train"] = entropy(local_env[train_idx])
        env_stats[f"Env/Layer_{layer_id}_MaxProb_Train"] = float(local_env[train_idx].max(dim=-1).values.mean().item())
        for env_id in range(local_env.size(1)):
            env_stats[f"Env/Layer_{layer_id}_Usage_Train/env{env_id}"] = float(local_env[train_idx, env_id].mean().item())

    return branch_results, env_stats


parser = argparse.ArgumentParser(description='MLEI Training Pipeline')
parser_add_main_args(parser)
parser.add_argument(
    '--mlei_pred_mode',
    type=str,
    default='fused',
    choices=['local', 'global', 'fused'],
    help='prediction branch used during evaluation/inference for MLEI',
)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

if args.dataset == 'twitch':
    dataset = load_twitch_dataset(args.data_dir, train_num=3)
elif args.dataset == 'elliptic':
    dataset = load_elliptic_dataset(args.data_dir, train_num=5)
elif args.dataset == 'arxiv':
    dataset = load_arxiv_dataset(args.data_dir, train_num=3)
elif args.dataset in ('cora', 'citeseer', 'pubmed'):
    dataset = load_synthetic_dataset(args.data_dir, args.dataset, train_num=3, combine=args.combine_result)
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

c = max(dataset.y.max().item() + 1, dataset.y.shape[1])
d = dataset.x.shape[1]

print(f"dataset {args.dataset}: all nodes {dataset.num_nodes} | edges {dataset.edge_index.size(1)} | "
      + f"classes {c} | feats {d}")
print(f"train nodes {dataset.train_idx.shape[0]} | valid nodes {dataset.valid_idx.shape[0]} | "
      f"test in nodes {dataset.test_in_idx.shape[0]}")
m = ""
for i in range(len(dataset.test_ood_idx)):
    m += f"test ood{i+1} nodes {dataset.test_ood_idx[i].shape[0]} "
print(m)
print(f'[INFO] env numbers: {dataset.env_num} train env numbers: {dataset.train_env_num}')

model = CaNet(d, c, args, device).to(device)

if args.dataset in ('elliptic', 'twitch'):
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
else:
    criterion = nn.CrossEntropyLoss(reduction='mean')

if args.dataset in ('twitch'):
    eval_func = eval_rocauc
elif args.dataset in ('elliptic'):
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
run_name = f"{current_time}_MLEI_lamda_{args.lamda}"
log_dir = os.path.join('.', 'runs', 'MLEI', args.dataset, run_name)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"[INFO] TensorBoard logging activated. Logs will be saved to: {log_dir}")

for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset.x = dataset.x.to(device)
    dataset.y = dataset.y.to(device)
    dataset.edge_index = dataset.edge_index.to(device)
    dataset.env = dataset.env.to(device)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss_compute(dataset, criterion, args)
        loss.backward()
        optimizer.step()

        branch_results, env_stats = evaluate_mlei_diagnostics(model, dataset, eval_func)
        result = branch_results['fused']
        logger.add_result(run, result)

        global_step = run * args.epochs + epoch
        writer.add_scalar('Loss/Train_Loss', loss.item(), global_step)
        for name, value in getattr(model, "_last_loss_breakdown", {}).items():
            writer.add_scalar(f'LossBreakdown/{name}', float(value.item()), global_step)
        writer.add_scalar('Metrics/1_Train', result[0] * 100, global_step)
        writer.add_scalar('Metrics/2_Valid', result[1] * 100, global_step)
        writer.add_scalar('Metrics/3_Test_In', result[2] * 100, global_step)
        for i in range(len(result) - 3):
            writer.add_scalar(f'Metrics/4_Test_OOD_{i+1}', result[i + 3] * 100, global_step)

        for branch_name in ('local', 'global', 'fused'):
            branch_result = branch_results[branch_name]
            prefix = f'Branches/{branch_name}'
            writer.add_scalar(f'{prefix}/Train', branch_result[0] * 100, global_step)
            writer.add_scalar(f'{prefix}/Valid', branch_result[1] * 100, global_step)
            writer.add_scalar(f'{prefix}/Test_In', branch_result[2] * 100, global_step)
            for i in range(len(branch_result) - 3):
                writer.add_scalar(f'{prefix}/Test_OOD_{i+1}', branch_result[i + 3] * 100, global_step)

        for name, value in env_stats.items():
            writer.add_scalar(name, value, global_step)

        if epoch % args.display_step == 0:
            msg = (
                f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                f'Train: {100 * result[0]:.2f}%, '
                f'Valid: {100 * result[1]:.2f}%, '
                f'Test In: {100 * result[2]:.2f}% '
            )
            for i in range(len(result) - 3):
                msg += f'Test OOD{i+1}: {100 * result[i + 3]:.2f}% '
            print(msg)
    logger.print_statistics(run)

logger.print_statistics()
if args.store_result:
    logger.output(args)

writer.close()
print("[INFO] TensorBoard writer closed.")
