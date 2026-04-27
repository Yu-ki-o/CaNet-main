import copy

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def eval_f1(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)


def _predict_with_optional_tta(model, dataset, idx, args):
    use_tta = bool(getattr(args, 'use_tta_rl', False)) if args is not None else False
    if not use_tta:
        with torch.no_grad():
            return model(dataset.x, dataset.edge_index)

    state = copy.deepcopy(model.state_dict())
    model.adapt_test_time_policy(
        dataset.x,
        dataset.edge_index,
        idx,
        steps=int(getattr(args, 'tta_rl_steps', 1)),
        lr=float(getattr(args, 'tta_rl_lr', 1e-3)),
    )
    with torch.no_grad():
        out = model(dataset.x, dataset.edge_index, use_context_policy=True)
    model.load_state_dict(state)
    model.eval()
    return out


def evaluate_full(model, dataset, eval_func, args=None):
    model.eval()

    train_idx, valid_idx, test_in_idx, test_ood_idx = dataset.train_idx, dataset.valid_idx, dataset.test_in_idx, dataset.test_ood_idx
    y = dataset.y.cpu()
    with torch.no_grad():
        base_out = model(dataset.x, dataset.edge_index).cpu()

    train_acc = eval_func(y[train_idx], base_out[train_idx])
    valid_acc = eval_func(y[valid_idx], base_out[valid_idx])

    test_in_out = _predict_with_optional_tta(model, dataset, test_in_idx, args).cpu()
    test_in_acc = eval_func(y[test_in_idx], test_in_out[test_in_idx])
    test_ood_accs = []
    for t in test_ood_idx:
        test_out = _predict_with_optional_tta(model, dataset, t, args).cpu()
        test_ood_accs.append(eval_func(y[t], test_out[t]))
    result = [train_acc, valid_acc, test_in_acc] + test_ood_accs

    return result
