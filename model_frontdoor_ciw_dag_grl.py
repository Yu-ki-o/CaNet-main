"""
Graph Front-Door CIW-DAG + GRL adversarial disentanglement.

This is a drop-in variant of GraphFrontDoorCIWDAG. It keeps the original
DAG-CIW/front-door path, and adds EGOG-style gradient reversal losses:
  1) causal/mediator representation should not predict environment;
  2) spurious representation should not predict task label.

Use it to test whether GRL is a better disentanglement constraint than the
original uniform losses on PubMed/Twitch and other OOD node datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from model_frontdoor_ciw_dag import GraphFrontDoorCIWDAG


class _GradientReverseFn(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return _GradientReverseFn.apply(x, lambd)


class GraphFrontDoorCIWDAGGRL(GraphFrontDoorCIWDAG):
    """CIW-DAG model with optional EGOG-style GRL disentanglement."""

    def __init__(self, d_in, c, args, device):
        super().__init__(d_in, c, args, device)

        self.use_grl = bool(getattr(args, 'use_grl', False))
        self.lambda_grl_env = float(getattr(args, 'lambda_grl_env', 0.0))
        self.lambda_grl_y = float(getattr(args, 'lambda_grl_y', 0.0))
        self.grl_env_coeff = float(getattr(args, 'grl_env_coeff', 1.0))
        self.grl_y_coeff = float(getattr(args, 'grl_y_coeff', 1.0))
        self.grl_replace_uniform = bool(getattr(args, 'grl_replace_uniform', False))
        self.grl_use_pseudo_env = bool(getattr(args, 'grl_use_pseudo_env', True))

        rep_dim = self._infer_rep_dim(args)
        env_out_dim = max(1, int(getattr(self, 'num_envs', getattr(args, 'train_env_num', 1))))

        self.grl_env_head = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            nn.ReLU(),
            nn.Linear(rep_dim, env_out_dim),
        )
        self.grl_y_head = nn.Sequential(
            nn.Linear(rep_dim, rep_dim),
            nn.ReLU(),
            nn.Linear(rep_dim, self.c),
        )
        self._reset_grl_modules()

    def _infer_rep_dim(self, args):
        # The hidden representation dimension should match classifier/env_classifier input.
        for module_name in ('classifier', 'env_classifier'):
            module = getattr(self, module_name, None)
            if isinstance(module, nn.Linear):
                return int(module.in_features)
            if hasattr(module, 'modules'):
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        return int(m.in_features)
        for name in ('hidden_channels', 'hidden_dim', 'channels', 'num_hidden'):
            if hasattr(args, name):
                return int(getattr(args, name))
        raise ValueError('Cannot infer representation dimension for GRL heads.')

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'grl_env_head'):
            self._reset_grl_modules()

    def _reset_grl_modules(self):
        for module in (self.grl_env_head, self.grl_y_head):
            for m in module.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()

    def _env_targets(self, data, train_idx, env_logits_spu):
        env = getattr(data, 'env', None)
        if env is not None and self.num_envs > 1:
            return env[train_idx].view(-1).long()
        if self.grl_use_pseudo_env and env_logits_spu is not None and env_logits_spu.size(-1) > 1:
            return env_logits_spu.detach().argmax(dim=-1).view(-1).long()
        return None

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx

        # Be robust to small differences between GraphFrontDoorDAG versions:
        # some versions append extra diagnostics to forward(), so positional
        # unpacking with a fixed length can fail.  The tensors used below keep
        # the same positions as the original CIW-DAG implementation.
        forward_out = self.forward(x, edge_index, training=True)
        z_all = forward_out[1]
        dag_vars_all = forward_out[3]
        z_mediator_all = forward_out[4]
        z_spurious_all = forward_out[5]
        mediator_gate = forward_out[6]
        causal_score = forward_out[7]
        pollution_score = forward_out[8]
        dag_total = forward_out[9]
        mediator_logits_all = forward_out[10]

        y_tr = y[train_idx]
        med_tr = z_mediator_all[train_idx]
        spu_tr = z_spurious_all[train_idx]
        dag_vars_tr = dag_vars_all[train_idx]
        edge_latent_tr = dag_vars_tr[:, self.edge_var_slice]
        mediator_logits_tr = mediator_logits_all[train_idx]
        env_logits_spu = self.env_classifier(spu_tr)
        env_probs_spu = F.softmax(env_logits_spu, dim=-1) if self.num_envs > 1 else None

        contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(spu_tr, env_probs_spu, training=True),
            training=True,
        )
        num_gmm_contexts = 0 if contexts is None else int(contexts.size(0))
        fd_logits_tr, fd_stack_tr = self.frontdoor_logits_from_contexts(med_tr, spu_tr, contexts)
        final_logits_tr = self.blend_logits(mediator_logits_tr, fd_logits_tr)

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        loss_dag = self.dag_regularization_loss(mediator_gate, dag_total)
        loss_dag_rec, loss_dag_feat, loss_dag_label_rec = self.dag_reconstruction_loss(
            dag_vars_all,
            y,
            train_idx,
            criterion,
            args,
            env=getattr(data, 'env', None),
        )
        loss_dag_label = loss_dag_rec

        if self.num_envs > 1:
            env_logits_med = self.env_classifier(med_tr)
            loss_env_med = self.compute_env_uniform_loss(env_logits_med)
            loss_spu = self.compute_pseudo_env_loss(env_logits_spu)
        else:
            loss_env_med = self.A_feat.new_zeros(())
            loss_spu = self.compute_uniform_loss(self.classifier(spu_tr))

        zero = self.A_feat.new_zeros(())
        loss_grl_env = zero
        loss_grl_y = zero
        if self.use_grl:
            env_target = self._env_targets(data, train_idx, env_logits_spu)
            if env_target is not None and self.grl_env_head[-1].out_features > 1:
                env_logits_from_med = self.grl_env_head(grad_reverse(med_tr, self.grl_env_coeff))
                loss_grl_env = F.cross_entropy(env_logits_from_med, env_target)

            y_logits_from_spu = self.grl_y_head(grad_reverse(spu_tr, self.grl_y_coeff))
            loss_grl_y = self.compute_supervised_loss(y_logits_from_spu, y_tr, criterion, args).mean()

            if self.grl_replace_uniform:
                loss_env_med = zero
                loss_spu = zero

        loss_med = zero
        loss_fd_aug = zero
        loss_var = zero
        loss_ind = zero
        loss_sem = zero
        loss_degree = zero
        loss_spu_y = zero
        loss_inv = zero
        loss_causal_mask_cls = zero
        if self.lambda_causal_mask_cls > 0.0:
            causal_expand = self.dag_gate_expander(causal_score.unsqueeze(0)).squeeze(0)
            causal_mask = torch.sigmoid(self.mediator_temp * (causal_expand - self.mediator_threshold))
            z_causal_mask = z_all * causal_mask.unsqueeze(0)
            loss_causal_mask_cls = self.compute_supervised_loss(
                self.classifier(z_causal_mask[train_idx]), y_tr, criterion, args
            ).mean()

        total_loss = (
            loss_cls
            + self.lambda_fd * loss_fd
            + self.lambda_dag * loss_dag
            + self.lambda_dag_rec * loss_dag_rec
            + self.lambda_spu * loss_spu
            + self.lambda_env * loss_env_med
            + self.lambda_causal_mask_cls * loss_causal_mask_cls
            + self.lambda_grl_env * loss_grl_env
            + self.lambda_grl_y * loss_grl_y
        )

        state_payload = None
        if update_state:
            state_payload = {
                'spu_tr': spu_tr.detach(),
                'env_probs_tr': env_probs_spu.detach() if env_probs_spu is not None else None,
                'edge_latent_tr': edge_latent_tr.detach(),
            }

        num_contexts = 0 if contexts is None else int(contexts.size(0))
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_med': loss_med,
            'loss_fd': loss_fd,
            'loss_fd_aug': loss_fd_aug,
            'loss_var': loss_var,
            'loss_ind': loss_ind,
            'loss_dag': loss_dag,
            'loss_dag_label': loss_dag_label,
            'loss_dag_rec': loss_dag_rec,
            'loss_dag_feat': loss_dag_feat,
            'loss_dag_label_rec': loss_dag_label_rec,
            'loss_sem': loss_sem,
            'loss_degree': loss_degree,
            'loss_spu_y': loss_spu_y,
            'loss_spu': loss_spu,
            'loss_env_med': loss_env_med,
            'loss_inv': loss_inv,
            'loss_causal_mask_cls': loss_causal_mask_cls,
            'loss_grl_env': loss_grl_env,
            'loss_grl_y': loss_grl_y,
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'causal_score_mean': causal_score.mean().detach(),
            'pollution_score_mean': pollution_score.mean().detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_mixed_contexts': torch.tensor(0.0, device=x.device),
            'num_gmm_contexts': torch.tensor(float(num_gmm_contexts), device=x.device),
            'state_payload': state_payload,
        }
