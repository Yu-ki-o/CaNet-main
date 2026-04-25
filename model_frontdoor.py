import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_sparse import SparseTensor, matmul


def gcn_backbone_conv(x, edge_index):
    """
    CaNet-style GCN propagation used as the shared graph encoder backbone.
    """
    num_nodes = x.size(0)
    row, col = edge_index
    deg = degree(col, num_nodes).float()
    deg_in = (1.0 / deg[col]).sqrt()
    deg_out = (1.0 / deg[row]).sqrt()
    value = torch.ones_like(row, dtype=x.dtype) * deg_in * deg_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
    return matmul(adj, x)


class FrontDoorBackboneLayer(nn.Module):
    """
    Single shared graph encoder layer.

    This mirrors the CaNet backbone choices so the paper-style front-door
    model can reuse the same graph encoder family under OOD evaluation.
    """

    def __init__(self, in_features, out_features, backbone_type='gcn', residual=True, variant=False):
        super().__init__()
        self.backbone_type = backbone_type
        self.out_features = out_features
        self.residual = residual
        self.variant = variant

        if backbone_type == 'gcn':
            self.weight = nn.Parameter(torch.FloatTensor(in_features * 2, out_features))
        elif backbone_type == 'gat':
            self.leakyrelu = nn.LeakyReLU()
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            self.att = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        else:
            raise NotImplementedError(
                f"Front-door backbone_type='{backbone_type}' is not implemented. "
                "Use 'gcn' or 'gat' to match the CaNet-style backbone."
            )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.backbone_type == 'gat':
            nn.init.xavier_uniform_(self.att.data, gain=1.414)

    def specialspmm(self, edge_index, values, size, h):
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=values, sparse_sizes=size)
        return matmul(adj, h)

    def forward(self, x, edge_index):
        if self.backbone_type == 'gcn':
            if self.variant: #不做标准的gcn归一化，目的是让度数较大的节点的影响力更强
                adj = torch.sparse_coo_tensor(
                    edge_index,
                    torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype),
                    size=(x.size(0), x.size(0)),
                )
                h_neigh = torch.sparse.mm(adj, x)
            else:
                h_neigh = gcn_backbone_conv(x, edge_index) #标准的GCN归一化
            h = torch.cat([h_neigh, x], dim=1)
            out = torch.matmul(h, self.weight)
        else:
            h = torch.matmul(x, self.weight)
            num_nodes = x.size(0)
            att_edge_index, _ = remove_self_loops(edge_index)
            att_edge_index, _ = add_self_loops(att_edge_index, num_nodes=num_nodes)
            edge_h = torch.cat([h[att_edge_index[0]], h[att_edge_index[1]]], dim=1)
            logits = self.leakyrelu(torch.matmul(edge_h, self.att)).squeeze(1)
            logits = logits - logits.max()
            edge_e = torch.exp(logits)

            eps = 1e-8
            denom = self.specialspmm(
                att_edge_index,
                edge_e,
                torch.Size([num_nodes, num_nodes]),
                torch.ones(num_nodes, 1, device=x.device, dtype=x.dtype),
            ) + eps
            out = self.specialspmm(att_edge_index, edge_e, torch.Size([num_nodes, num_nodes]), h)
            out = out / denom

        if self.residual:
            out = out + x
        return out


class GraphFrontDoor(nn.Module):
    """
    Paper-style non-DAG front-door model for graph OOD.

    Mapping the CIPT framework to graphs:
    1) a shared graph encoder extracts node representations;
    2) two lightweight adapters decouple them into causal and spurious views;
    3) the causal branch predicts labels while the spurious branch is pushed
       toward a uniform label distribution;
    4) diverse environment-specific spurious contexts are sampled and combined
       with the causal feature through an adaptive augmentation module to
       approximate the front-door intervention.
    """

    def __init__(self, d_in, c, args, device):
        super().__init__()
        self.device = device
        self.d = args.hidden_channels
        self.c = c
        self.num_envs = max(1, int(args.train_env_num))
        self.num_layers = max(1, int(getattr(args, 'num_layers', 2)))
        self.backbone_type = getattr(args, 'backbone_type', 'gcn')
        self.variant = getattr(args, 'variant', False)
        self.dropout = getattr(args, 'dropout', 0.0)
        self.gamma = getattr(args, 'gamma', 0.99)
        self.fd_blend = getattr(args, 'fd_blend', 0.5)
        self.fd_sample_k = max(0, int(getattr(args, 'K', 0)))
        self.context_sample_seed = int(getattr(args, 'seed', 0))
        self.context_gate_temp = float(getattr(args, 'context_gate_temp', 1.0))

        self.lambda_med = getattr(args, 'lambda_med', 0.5)
        self.lambda_spu = getattr(args, 'lambda_spu', 0.1)
        self.lambda_fd = getattr(args, 'lambda_fd', 0.5)
        self.lambda_var = getattr(args, 'lambda_var', 0.05)
        self.lambda_ind = getattr(args, 'lambda_ind', 0.1)

        self.act_fn = nn.ReLU()
        self.input_proj = nn.Linear(d_in, self.d)
        self.backbone_layers = nn.ModuleList([
            FrontDoorBackboneLayer(
                self.d,
                self.d,
                backbone_type=self.backbone_type,
                residual=True,
                variant=self.variant,
            )
            for _ in range(self.num_layers)
        ])

        # Paper-style causal decomposition: two lightweight adapters.
        self.causal_adapter = nn.Linear(self.d, self.d)
        self.spurious_adapter = nn.Linear(self.d, self.d)
        self.causal_norm = nn.LayerNorm(self.d)
        self.spurious_norm = nn.LayerNorm(self.d)

        self.classifier = nn.Linear(self.d, c)
        self.spurious_classifier = nn.Linear(self.d, c)
        self.fd_classifier = nn.Linear(self.d, c)

        # Adaptive diversity augmentation (graph analogue of text-based DA).
        self.query_proj = nn.Linear(self.d, self.d)
        self.key_proj = nn.Linear(self.d, self.d)
        self.value_proj = nn.Linear(self.d, self.d)
        self.aug_norm = nn.LayerNorm(self.d)

        self.register_buffer('proto_spu_env', torch.zeros(self.num_envs, self.d))
        self.register_buffer('proto_spu_env_valid', torch.zeros(self.num_envs, dtype=torch.bool))

        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for layer in self.backbone_layers:
            layer.reset_parameters()
        self.causal_adapter.reset_parameters()
        self.spurious_adapter.reset_parameters()
        self.causal_norm.reset_parameters()
        self.spurious_norm.reset_parameters()
        self.classifier.reset_parameters()
        self.spurious_classifier.reset_parameters()
        self.fd_classifier.reset_parameters()
        self.query_proj.reset_parameters()
        self.key_proj.reset_parameters()
        self.value_proj.reset_parameters()
        self.aug_norm.reset_parameters()
        self.proto_spu_env.zero_()
        self.proto_spu_env_valid.zero_()

    def encode_backbone(self, x, edge_index, training=False):
        x = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(self.input_proj(x))
        for layer in self.backbone_layers:
            h = F.dropout(h, self.dropout, training=training)
            h = self.act_fn(layer(h, edge_index))
        return h

    def decompose_representation(self, h, training=False):
        z_causal = self.causal_norm(h + self.causal_adapter(h))
        z_spurious = self.spurious_norm(h + self.spurious_adapter(h))
        z_causal = F.dropout(z_causal, self.dropout, training=training)
        z_spurious = F.dropout(z_spurious, self.dropout, training=training)
        causal_logits = self.classifier(z_causal)
        spurious_logits = self.spurious_classifier(z_spurious)
        return z_causal, z_spurious, causal_logits, spurious_logits

    def encode_representation(self, x, edge_index, training=False):
        h = self.encode_backbone(x, edge_index, training=training)
        z_causal, z_spurious, causal_logits, spurious_logits = self.decompose_representation(
            h,
            training=training,
        )
        return h, z_causal, z_spurious, causal_logits, spurious_logits

    def update_spurious_env_prototypes(self, z_spurious, envs):
        if envs is None or envs.numel() == 0:
            return
        env_values = envs.squeeze().long()
        for env_idx in range(self.num_envs):
            mask_e = env_values == env_idx
            if not mask_e.any():
                continue
            vec = z_spurious[mask_e].mean(dim=0).detach()
            if self.proto_spu_env_valid[env_idx]:
                vec = self.gamma * self.proto_spu_env[env_idx] + (1.0 - self.gamma) * vec
            self.proto_spu_env[env_idx] = F.normalize(vec, dim=0)
            self.proto_spu_env_valid[env_idx] = True

    @torch.no_grad()
    def apply_state_update(self, state_payload):
        if state_payload is None:
            return
        self.update_spurious_env_prototypes(
            state_payload['spu_tr'],
            state_payload['env_tr'],
        )

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
                    context_map[int(env_idx)] = z_spurious[mask_e].mean(dim=0).detach()

        if not context_map:
            return None
        ordered = [context_map[idx] for idx in sorted(context_map.keys())]
        return torch.stack(ordered, dim=0)

    def sample_frontdoor_contexts(self, contexts, training=False):
        if contexts is None or contexts.size(0) == 0:
            return contexts

        if self.fd_sample_k <= 0 or contexts.size(0) <= self.fd_sample_k:
            return contexts

        num_contexts = contexts.size(0)
        if training:
            indices = torch.randperm(num_contexts, device=contexts.device)[:self.fd_sample_k]
        else:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(self.context_sample_seed + num_contexts)
            indices = torch.randperm(num_contexts, generator=generator)[:self.fd_sample_k]
            indices = indices.to(contexts.device)
        return contexts.index_select(0, indices)

    def interventional_logits_from_contexts(self, z_causal, contexts, training=False):
        base_logits = self.fd_classifier(z_causal)
        if contexts is None or contexts.size(0) == 0:
            return base_logits, None

        num_contexts = contexts.size(0)
        causal_expand = z_causal.unsqueeze(1).expand(-1, num_contexts, -1)
        context_expand = contexts.unsqueeze(0).expand(z_causal.size(0), -1, -1)

        query = self.query_proj(causal_expand)
        key = self.key_proj(context_expand)
        value = self.value_proj(context_expand)

        gate_score = (query * key).sum(dim=-1, keepdim=True) / math.sqrt(self.d)
        gate = torch.sigmoid(self.context_gate_temp * gate_score)
        aug = self.aug_norm(causal_expand + gate * value)
        aug = F.dropout(aug, self.dropout, training=training)

        logits_stack = self.fd_classifier(aug.reshape(-1, self.d)).view(z_causal.size(0), num_contexts, self.c)
        fd_logits = logits_stack.mean(dim=1)
        return fd_logits, logits_stack

    def blend_logits(self, causal_logits, fd_logits):
        if fd_logits is None:
            return causal_logits
        return (1.0 - self.fd_blend) * causal_logits + self.fd_blend * fd_logits

    def forward(self, x, edge_index, training=False):
        h, z_causal, z_spurious, causal_logits, spurious_logits = self.encode_representation(
            x,
            edge_index,
            training=training,
        )

        contexts = self.sample_frontdoor_contexts(
            self.get_frontdoor_contexts(),
            training=training,
        )
        fd_logits, fd_stack = self.interventional_logits_from_contexts(
            z_causal,
            contexts,
            training=training,
        )
        logits = self.blend_logits(causal_logits, fd_logits)

        if training:
            return (
                logits,
                h,
                z_causal,
                z_spurious,
                causal_logits,
                spurious_logits,
                fd_logits,
                fd_stack,
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

    def compute_independence_loss(self, z_causal, z_spurious):
        if z_causal.numel() == 0:
            return z_causal.new_zeros(())
        z_causal = F.normalize(z_causal, dim=1)
        z_spurious = F.normalize(z_spurious, dim=1)
        corr = (z_causal * z_spurious).sum(dim=1)
        return 0.5 * (corr ** 2).mean()

    def compute_frontdoor_variance_loss(self, logits_stack):
        if logits_stack is None or logits_stack.size(1) <= 1:
            return logits_stack.new_zeros(()) if logits_stack is not None else self.classifier.weight.new_zeros(())
        probs = torch.softmax(logits_stack, dim=-1)
        return probs.var(dim=1, unbiased=False).mean()

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx
        envs = data.env if hasattr(data, 'env') else None

        (
            _,
            _,
            z_causal_all,
            z_spurious_all,
            causal_logits_all,
            spurious_logits_all,
            _,
            _,
        ) = self.forward(x, edge_index, training=True)

        y_tr = y[train_idx]
        causal_tr = z_causal_all[train_idx]
        spurious_tr = z_spurious_all[train_idx]
        env_tr = envs[train_idx] if envs is not None else None
        causal_logits_tr = causal_logits_all[train_idx]
        spurious_logits_tr = spurious_logits_all[train_idx]

        contexts = self.sample_frontdoor_contexts(
            self.get_frontdoor_contexts(spurious_tr, env_tr),
            training=True,
        )
        fd_logits_tr, fd_stack_tr = self.interventional_logits_from_contexts(
            causal_tr,
            contexts,
            training=True,
        )
        final_logits_tr = self.blend_logits(causal_logits_tr, fd_logits_tr)

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_med = self.compute_supervised_loss(causal_logits_tr, y_tr, criterion, args).mean()
        loss_spu = self.compute_uniform_spurious_loss(spurious_logits_tr)
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        loss_var = self.compute_frontdoor_variance_loss(fd_stack_tr)
        loss_ind = self.compute_independence_loss(causal_tr, spurious_tr)

        total_loss = (
            loss_cls
            + self.lambda_med * loss_med
            + self.lambda_spu * loss_spu
            + self.lambda_fd * loss_fd
            + self.lambda_var * loss_var
            + self.lambda_ind * loss_ind
        )

        state_payload = None
        if update_state:
            state_payload = {
                'spu_tr': spurious_tr.detach(),
                'env_tr': env_tr.detach() if env_tr is not None else None,
            }

        num_contexts = 0 if contexts is None else int(contexts.size(0))
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_med': loss_med,
            'loss_spu': loss_spu,
            'loss_fd': loss_fd,
            'loss_var': loss_var,
            'loss_ind': loss_ind,
            'causal_norm_mean': causal_tr.norm(dim=1).mean().detach(),
            'spurious_norm_mean': spurious_tr.norm(dim=1).mean().detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'state_payload': state_payload,
        }

    def loss_compute(self, data, criterion, args):
        losses = self.compute_losses(data, criterion, args, update_state=True)
        return (
            losses['total_loss'],
            losses['loss_cls'].item(),
            (self.lambda_ind * losses['loss_ind']).item(),
            (self.lambda_fd * losses['loss_fd']).item(),
        )
