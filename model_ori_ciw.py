import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import SAGEConv


class NodeWeightGenerator(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(NodeWeightGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )

    def reset_parameters(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        nn.init.constant_(self.net[2].bias, 1.0)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class GraphCIW(nn.Module):
    def __init__(self, d_in, c, args, device):
        super(GraphCIW, self).__init__()
        self.device = device
        self.d = args.hidden_channels
        self.c = c
        self.num_envs = args.train_env_num

        self.conv1 = SAGEConv(d_in, self.d)
        self.conv2 = SAGEConv(self.d, self.d)
        self.classifier = nn.Linear(self.d, c)
        self.spurious_classifier = nn.Linear(self.d, c)
        self.weight_net = NodeWeightGenerator(self.d, max(1, self.d // 2))

        self.fd_fuser = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.fd_norm = nn.LayerNorm(self.d)

        self.A = Parameter(torch.zeros(self.d + self.c, self.d + self.c))
        self.label_reconstructor = nn.Linear(self.d + self.c, self.c, bias=False)

        self.rff_dim = getattr(args, 'rff_dim', 128)
        self.register_buffer('omega', torch.randn(1, self.d, self.rff_dim))
        self.register_buffer('phi', torch.rand(1, self.d, self.rff_dim) * 2 * math.pi)

        self.alpha = getattr(args, 'alpha', 0.9)
        self.global_size = getattr(args, 'global_size', 2048)
        self.register_buffer('z_global', torch.zeros(self.global_size, self.d))
        self.register_buffer('w_global', torch.zeros(self.global_size, 1))
        self.register_buffer('global_valid_mask', torch.zeros(self.global_size, dtype=torch.bool))

        self.gamma = getattr(args, 'gamma', 0.99)
        self.tau = getattr(args, 'tau', 1.0)
        self.register_buffer('proto_in', torch.zeros(self.num_envs, c, self.d))
        self.register_buffer('proto_cr', torch.zeros(c, self.d))
        self.register_buffer('proto_in_valid', torch.zeros(self.num_envs, c, dtype=torch.bool))
        self.register_buffer('proto_cr_valid', torch.zeros(c, dtype=torch.bool))
        self.register_buffer('proto_spu_env', torch.zeros(self.num_envs, self.d))
        self.register_buffer('proto_spu_env_valid', torch.zeros(self.num_envs, dtype=torch.bool))

        self.dropout = getattr(args, 'dropout', 0.0)
        self.lambda_l1 = getattr(args, 'lambda_l1', 1e-5)
        self.lambda_ind = getattr(args, 'lambda_ind', 0.1)
        self.lambda_cl = getattr(args, 'lambda_cl', 0.1)
        self.lambda_dag = getattr(args, 'lambda_dag', 0.1)
        self.lambda_med = getattr(args, 'lambda_med', 0.5)
        self.lambda_spu = getattr(args, 'lambda_spu', 0.1)
        self.lambda_fd = getattr(args, 'lambda_fd', 0.5)
        self.lambda_var = getattr(args, 'lambda_var', 0.05)

        self.mediator_temp = getattr(args, 'mediator_temp', 8.0)
        self.low_temp = getattr(args, 'low_temp', 8.0)
        self.low_threshold = getattr(args, 'low_threshold', 0.35)
        self.mediator_threshold = getattr(args, 'mediator_threshold', 0.5)
        self.pollution_coeff = getattr(args, 'pollution_coeff', 1.0)
        self.fd_blend = getattr(args, 'fd_blend', 0.5)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.classifier.reset_parameters()
        self.spurious_classifier.reset_parameters()
        self.weight_net.reset_parameters()
        self.label_reconstructor.reset_parameters()
        for module in self.fd_fuser:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.fd_norm.reset_parameters()
        nn.init.uniform_(self.A, -0.01, 0.01)

        self.z_global.zero_()
        self.w_global.zero_()
        self.global_valid_mask.zero_()
        self.proto_in.zero_()
        self.proto_cr.zero_()
        self.proto_in_valid.zero_()
        self.proto_cr_valid.zero_()
        self.proto_spu_env.zero_()
        self.proto_spu_env_valid.zero_()

    def dag_parameters(self):
        yield self.A
        for parameter in self.label_reconstructor.parameters():
            yield parameter

    def main_parameters(self):
        dag_param_ids = {id(parameter) for parameter in self.dag_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in dag_param_ids:
                yield parameter

    def get_masked_A(self):
        A_masked = self.A - torch.diag(torch.diag(self.A))
        A_masked = A_masked.clone()
        A_masked[self.d:, :] = 0.0
        return A_masked

    def _sample_global_bank(self, z_local, w_local):
        if z_local.size(0) == 0:
            return z_local, w_local

        sample_size = min(self.global_size, z_local.size(0))
        if sample_size == z_local.size(0):
            return z_local.detach(), w_local.detach()

        indices = torch.linspace(
            0,
            z_local.size(0) - 1,
            steps=sample_size,
            device=z_local.device,
        ).round().long()
        return z_local[indices].detach(), w_local[indices].detach()

    def load_global_info(self, z_local, w_local):
        if self.global_valid_mask.any():
            valid_size = int(self.global_valid_mask.sum().item())
            z_concat = torch.cat([z_local, self.z_global[:valid_size]], dim=0)
            w_concat = torch.cat([w_local, self.w_global[:valid_size]], dim=0)
            return z_concat, w_concat
        return z_local, w_local

    def update_global_memory(self, z_local, w_local):
        z_bank, w_bank = self._sample_global_bank(z_local, w_local)
        bank_size = z_bank.size(0)
        if bank_size == 0:
            return

        if not self.global_valid_mask.any():
            self.z_global[:bank_size].copy_(z_bank)
            self.w_global[:bank_size].copy_(w_bank)
            self.global_valid_mask[:bank_size] = True
            return

        old_size = int(self.global_valid_mask.sum().item())
        blend_size = min(old_size, bank_size)
        if blend_size > 0:
            self.z_global[:blend_size].mul_(self.alpha).add_(z_bank[:blend_size], alpha=1.0 - self.alpha)
            self.w_global[:blend_size].mul_(self.alpha).add_(w_bank[:blend_size], alpha=1.0 - self.alpha)

        if bank_size > old_size:
            self.z_global[old_size:bank_size].copy_(z_bank[old_size:bank_size])
            self.w_global[old_size:bank_size].copy_(w_bank[old_size:bank_size])
            self.global_valid_mask[old_size:bank_size] = True

    def get_valid_in_domain_prototypes(self):
        valid_pairs = self.proto_in_valid.nonzero(as_tuple=False)
        if valid_pairs.numel() == 0:
            return None, None

        env_idx = valid_pairs[:, 0]
        class_idx = valid_pairs[:, 1]
        proto_features = self.proto_in[env_idx, class_idx]
        return proto_features, class_idx

    def _normalize_score(self, values, default_value):
        if values.numel() == 0:
            return values

        v_min = values.min()
        v_max = values.max()
        if (v_max - v_min).abs() < 1e-4:
            return (values - values.detach()) + default_value
        return (values - v_min) / (v_max - v_min + 1e-8)

    def get_causal_effect_and_mask(self):
        A_no_diag = self.get_masked_A()
        A_sq = A_no_diag * A_no_diag
        C_tot = torch.matrix_exp(A_sq)

        label_slice = C_tot[:self.d, self.d:]
        if self.proto_cr_valid.any():
            label_slice = label_slice[:, self.proto_cr_valid]

        if label_slice.numel() == 0:
            causal_score = torch.full((self.d,), 0.5, device=A_no_diag.device)
        else:
            causal_score = self._normalize_score(label_slice.mean(dim=1), default_value=0.5)

        feature_slice = C_tot[:self.d, :self.d]
        feature_slice = feature_slice - torch.diag(torch.diag(feature_slice))
        symmetric_flow = 0.5 * (feature_slice + feature_slice.t())

        low_weight = torch.sigmoid(self.low_temp * (self.low_threshold - causal_score))
        low_weight = low_weight / low_weight.sum().clamp_min(1e-8)
        pollution_score = torch.matmul(symmetric_flow, low_weight)
        pollution_score = self._normalize_score(pollution_score, default_value=0.0)

        mediator_logit = causal_score - self.pollution_coeff * pollution_score - self.mediator_threshold
        mediator_gate = torch.sigmoid(self.mediator_temp * mediator_logit)

        mediator_expand = mediator_gate.unsqueeze(1)
        spurious_expand = (1.0 - mediator_gate).unsqueeze(1)
        S = mediator_expand @ spurious_expand.t()
        S = S + S.t()
        return causal_score, pollution_score, mediator_gate, S

    def update_prototypes(self, z_local, z_mediator, y, envs):
        if y.numel() == 0:
            return

        unique_classes = torch.unique(y)
        for class_value in unique_classes:
            class_idx = int(class_value.item())
            mask_c = (y.squeeze() == class_value)
            if not mask_c.any():
                continue

            z_med_c_mean = z_mediator[mask_c].mean(dim=0)
            if self.proto_cr_valid[class_idx]:
                updated = self.gamma * self.proto_cr[class_idx] + (1 - self.gamma) * z_med_c_mean.detach()
            else:
                updated = z_med_c_mean.detach()
            self.proto_cr[class_idx] = F.normalize(updated, dim=0)
            self.proto_cr_valid[class_idx] = True

            if envs is None:
                continue

            for env_idx in range(self.num_envs):
                mask_ce = mask_c & (envs.squeeze() == env_idx)
                if not mask_ce.any():
                    continue

                z_loc_ce_mean = z_local[mask_ce].mean(dim=0)
                if self.proto_in_valid[env_idx, class_idx]:
                    updated = self.gamma * self.proto_in[env_idx, class_idx] + (1 - self.gamma) * z_loc_ce_mean.detach()
                else:
                    updated = z_loc_ce_mean.detach()
                self.proto_in[env_idx, class_idx] = F.normalize(updated, dim=0)
                self.proto_in_valid[env_idx, class_idx] = True

    def update_spurious_env_prototypes(self, z_spurious, envs):
        if envs is None or envs.numel() == 0:
            return

        env_values = envs.squeeze().long()
        for env_idx in range(self.num_envs):
            mask_e = env_values == env_idx
            if not mask_e.any():
                continue

            z_spu_mean = z_spurious[mask_e].mean(dim=0)
            if self.proto_spu_env_valid[env_idx]:
                updated = self.gamma * self.proto_spu_env[env_idx] + (1 - self.gamma) * z_spu_mean.detach()
            else:
                updated = z_spu_mean.detach()
            self.proto_spu_env[env_idx] = F.normalize(updated, dim=0)
            self.proto_spu_env_valid[env_idx] = True

    def compute_contrastive_loss(self, z_local, z_mediator, y, envs):
        zero = z_local.new_zeros(())
        if y.numel() == 0:
            return zero

        y_idx = y.squeeze().long()
        z_med_norm = F.normalize(z_mediator, dim=1)
        z_loc_norm = F.normalize(z_local, dim=1)

        loss_cl_cr = zero
        valid_classes = self.proto_cr_valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_classes.numel() > 0:
            logits_cr = torch.matmul(z_med_norm, self.proto_cr[valid_classes].t()) / self.tau
            class_map = torch.full((self.c,), -1, dtype=torch.long, device=y.device)
            class_map[valid_classes] = torch.arange(valid_classes.numel(), device=y.device)
            targets_cr = class_map[y_idx]
            valid_samples = targets_cr >= 0
            if valid_samples.any():
                loss_cl_cr = F.cross_entropy(logits_cr[valid_samples], targets_cr[valid_samples])

        loss_cl_in = zero
        if envs is not None and envs.numel() > 0:
            e_idx = envs.squeeze().long()
            protos_e = self.proto_in[e_idx]
            valid_mask = self.proto_in_valid[e_idx]
            logits_in = torch.bmm(protos_e, z_loc_norm.unsqueeze(2)).squeeze(2) / self.tau
            logits_in = logits_in.masked_fill(~valid_mask, -1e9)
            positive_valid = valid_mask.gather(1, y_idx.unsqueeze(1)).squeeze(1)
            if positive_valid.any():
                loss_cl_in = F.cross_entropy(logits_in[positive_valid], y_idx[positive_valid])

        return loss_cl_in + loss_cl_cr

    def dag_reconstruction_loss_on_prototypes(self):
        zero = self.A.new_zeros(())
        proto_features, labels = self.get_valid_in_domain_prototypes()
        if proto_features is None or labels.numel() == 0:
            return zero

        proto_features = F.layer_norm(proto_features, (self.d,))
        label_factor = F.one_hot(labels, num_classes=self.c).float()
        factors = torch.cat([proto_features, label_factor], dim=1)

        A_no_diag = self.get_masked_A()
        parent_messages = torch.matmul(factors, A_no_diag)
        feature_recon = parent_messages[:, :self.d]
        label_logits = self.label_reconstructor(parent_messages)

        loss_rec_x = F.mse_loss(feature_recon, factors[:, :self.d])
        loss_rec_y = F.cross_entropy(label_logits, labels)

        A_sq = A_no_diag * A_no_diag
        h_A = torch.trace(torch.matrix_exp(A_sq)) - (self.d + self.c)
        h_A_clipped = torch.clamp(h_A, min=-10.0, max=10.0)
        loss_dag_reg = 0.5 * (h_A_clipped ** 2) + self.lambda_l1 * torch.norm(A_no_diag, p=1)
        return loss_rec_x + loss_rec_y + loss_dag_reg

    def compute_weighted_independence_loss(self, z_local, w_local, S):
        zero = z_local.new_zeros(())
        if z_local.size(0) <= 1:
            return zero

        z_concat, w_concat = self.load_global_info(z_local, w_local)
        w_concat = w_concat.clamp_min(1e-4)
        w_concat = w_concat / w_concat.sum().clamp_min(1e-8)

        z_mean = torch.mean(z_concat, dim=0, keepdim=True)
        z_std = torch.std(z_concat, dim=0, keepdim=True) + 1e-5
        z_concat_norm = (z_concat - z_mean) / z_std
        z_expand = z_concat_norm.unsqueeze(-1)
        z_rff = math.sqrt(2.0) * torch.cos(z_expand * self.omega + self.phi)

        N, D, R = z_rff.shape
        z_rff_flat = z_rff.view(N, D * R)

        weighted_mean = torch.sum(w_concat * z_rff_flat, dim=0, keepdim=True)
        z_centered = z_rff_flat - weighted_mean
        cov = torch.matmul((w_concat * z_centered).t(), z_centered)
        cov_sq = cov ** 2
        cov_blocks_sq = cov_sq.view(D, R, D, R).sum(dim=(1, 3)) / (R * R)

        ind_loss = torch.sum(torch.triu(S * cov_blocks_sq, diagonal=1))
        num_pairs = max((self.d * (self.d - 1)) / 2.0, 1.0)
        return ind_loss / num_pairs

    def encode_representation(self, x, edge_index, training=False, detach_causal_graph=True):
        x = F.dropout(x, self.dropout, training=training)
        z = F.relu(self.conv1(x, edge_index))
        z = F.dropout(z, self.dropout, training=training)
        z = self.conv2(z, edge_index)

        w = self.weight_net(z.detach())
        causal_score, pollution_score, mediator_gate, S = self.get_causal_effect_and_mask()
        if detach_causal_graph:
            causal_score = causal_score.detach()
            pollution_score = pollution_score.detach()
            mediator_gate = mediator_gate.detach()
            S = S.detach()

        z_mediator = z * mediator_gate.unsqueeze(0)
        z_spurious = z * (1.0 - mediator_gate).unsqueeze(0)
        z_mediator = F.dropout(z_mediator, self.dropout, training=training)
        z_spurious = F.dropout(z_spurious, self.dropout, training=training)
        mediator_logits = self.classifier(z_mediator)
        return z, w, S, z_mediator, z_spurious, mediator_logits, mediator_gate, causal_score, pollution_score

    def get_frontdoor_contexts(self, z_spurious=None, envs=None):
        context_map = {}

        if self.proto_spu_env_valid.any():
            valid_envs = self.proto_spu_env_valid.nonzero(as_tuple=False).squeeze(-1).tolist()
            for env_idx in valid_envs:
                context_map[int(env_idx)] = self.proto_spu_env[env_idx].detach()

        if z_spurious is not None and envs is not None and envs.numel() > 0:
            env_values = envs.squeeze().long()
            for env_idx in range(self.num_envs):
                mask_e = env_values == env_idx
                if mask_e.any():
                    context_map[env_idx] = z_spurious[mask_e].mean(dim=0).detach()

        if not context_map:
            return None

        ordered = [context_map[idx] for idx in sorted(context_map.keys())]
        return torch.stack(ordered, dim=0)

    def frontdoor_logits_from_contexts(self, z_mediator, contexts):
        base_logits = self.classifier(z_mediator)
        if contexts is None or contexts.size(0) == 0:
            return base_logits, None

        num_contexts = contexts.size(0)
        mediator_expand = z_mediator.unsqueeze(1).expand(-1, num_contexts, -1)
        context_expand = contexts.unsqueeze(0).expand(z_mediator.size(0), -1, -1)
        fused_input = torch.cat([mediator_expand, context_expand], dim=-1)
        fused = self.fd_fuser(fused_input.reshape(-1, self.d * 2)).view(z_mediator.size(0), num_contexts, self.d)
        fused = self.fd_norm(fused + mediator_expand)
        logits_stack = self.classifier(fused.reshape(-1, self.d)).view(z_mediator.size(0), num_contexts, self.c)
        fd_logits = logits_stack.mean(dim=1)
        return fd_logits, logits_stack

    def blend_logits(self, base_logits, fd_logits):
        if fd_logits is None:
            return base_logits
        return (1.0 - self.fd_blend) * base_logits + self.fd_blend * fd_logits

    def forward(self, x, edge_index, training=False, detach_causal_graph=True):
        (
            z,
            w,
            S,
            z_mediator,
            z_spurious,
            mediator_logits,
            mediator_gate,
            causal_score,
            pollution_score,
        ) = self.encode_representation(
            x,
            edge_index,
            training=training,
            detach_causal_graph=detach_causal_graph,
        )

        fd_logits, _ = self.frontdoor_logits_from_contexts(z_mediator, self.get_frontdoor_contexts())
        logits = self.blend_logits(mediator_logits, fd_logits)

        if training:
            return (
                logits,
                z,
                w,
                S,
                z_mediator,
                z_spurious,
                mediator_gate,
                causal_score,
                pollution_score,
                mediator_logits,
                fd_logits,
            )
        return logits

    def compute_supervised_loss(self, logits, y, criterion, args):
        if args.dataset in ('twitch', 'elliptic'):
            if y.shape[1] == 1 and logits.shape[1] > 1:
                true_label = F.one_hot(y.squeeze().long(), logits.shape[1]).float()
            else:
                true_label = y.float()
            sup_loss = criterion(logits, true_label)
            if sup_loss.dim() > 1:
                sup_loss = sup_loss.mean(dim=1)
            return sup_loss
        return criterion(logits, y.squeeze().long())

    def compute_uniform_spurious_loss(self, logits):
        if logits.size(-1) <= 1:
            return logits.new_zeros(())
        log_probs = F.log_softmax(logits, dim=-1)
        uniform = torch.full_like(log_probs, 1.0 / logits.size(-1))
        return F.kl_div(log_probs, uniform, reduction='batchmean')

    def compute_frontdoor_variance_loss(self, logits_stack):
        if logits_stack is None or logits_stack.size(1) <= 1:
            return self.A.new_zeros(())
        probs = torch.softmax(logits_stack, dim=-1)
        return probs.var(dim=1, unbiased=False).mean()

    def compute_losses(self, data, criterion, args, detach_causal_graph=True, update_state=True):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx
        envs = data.env if hasattr(data, 'env') else None

        (
            _,
            z_local,
            w_local,
            S,
            z_mediator,
            z_spurious,
            mediator_gate,
            causal_score,
            pollution_score,
            mediator_logits_all,
            _,
        ) = self.forward(
            x,
            edge_index,
            training=True,
            detach_causal_graph=detach_causal_graph,
        )

        z_tr = z_local[train_idx]
        w_tr = w_local[train_idx]
        y_tr = y[train_idx]
        med_tr = z_mediator[train_idx]
        spu_tr = z_spurious[train_idx]
        env_tr = envs[train_idx] if envs is not None else None
        mediator_logits_tr = mediator_logits_all[train_idx]

        contexts = self.get_frontdoor_contexts(spu_tr, env_tr)
        fd_logits_tr, fd_stack_tr = self.frontdoor_logits_from_contexts(med_tr, contexts)
        final_logits_tr = self.blend_logits(mediator_logits_tr, fd_logits_tr)

        S_for_ind = S.detach() if detach_causal_graph else S
        loss_ind = self.compute_weighted_independence_loss(z_tr.detach(), w_tr, S_for_ind)

        if update_state:
            self.update_prototypes(z_tr, med_tr, y_tr, env_tr)
            self.update_spurious_env_prototypes(spu_tr, env_tr)

        loss_cl = self.compute_contrastive_loss(z_tr, med_tr, y_tr, env_tr)
        loss_dag = self.dag_reconstruction_loss_on_prototypes()

        sup_loss_raw = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args)
        w_tr_norm = w_tr.squeeze(-1) / torch.clamp(w_tr.squeeze(-1).mean(), min=1e-3)
        loss_cls = torch.mean(w_tr_norm.detach() * sup_loss_raw)

        loss_med = self.compute_supervised_loss(mediator_logits_tr, y_tr, criterion, args).mean()
        loss_spu = self.compute_uniform_spurious_loss(self.spurious_classifier(spu_tr))
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        loss_var = self.compute_frontdoor_variance_loss(fd_stack_tr)

        if update_state:
            self.update_global_memory(z_tr.detach(), w_tr.detach())

        total_loss = (
            loss_cls
            + self.lambda_ind * loss_ind
            + self.lambda_dag * loss_dag
            + self.lambda_cl * loss_cl
            + self.lambda_med * loss_med
            + self.lambda_spu * loss_spu
            + self.lambda_fd * loss_fd
            + self.lambda_var * loss_var
        )
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_ind': loss_ind,
            'loss_dag': loss_dag,
            'loss_cl': loss_cl,
            'loss_med': loss_med,
            'loss_spu': loss_spu,
            'loss_fd': loss_fd,
            'loss_var': loss_var,
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'causal_score_mean': causal_score.mean().detach(),
            'pollution_score_mean': pollution_score.mean().detach(),
        }

    def loss_compute(self, data, criterion, args):
        losses = self.compute_losses(data, criterion, args, detach_causal_graph=True, update_state=True)
        return (
            losses['total_loss'],
            losses['loss_cls'].item(),
            (self.lambda_ind * losses['loss_ind']).item(),
            (self.lambda_dag * losses['loss_dag']).item(),
            (self.lambda_cl * losses['loss_cl']).item(),
        )
