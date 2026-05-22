import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, softmax
from torch_sparse import SparseTensor, matmul


def gcn_backbone_conv(x, edge_index):
    num_nodes = x.size(0)
    row, col = edge_index
    deg = degree(col, num_nodes).float().clamp_min(1.0)
    deg_in = (1.0 / deg[col]).sqrt()
    deg_out = (1.0 / deg[row]).sqrt()
    value = torch.ones_like(row, dtype=x.dtype) * deg_in * deg_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
    return matmul(adj, x)


class FrontDoorBackboneLayer(nn.Module):
    """CaNet-style backbone layer with stable GCN/GAT choices."""

    def __init__(self, in_features, out_features, backbone_type='gcn', residual=True, variant=False):
        super().__init__()
        self.backbone_type = backbone_type
        self.out_features = out_features
        self.residual = residual
        self.variant = variant

        if backbone_type == 'gcn':
            self.weight = nn.Parameter(torch.FloatTensor(in_features * 2, out_features))
        elif backbone_type == 'gat':
            self.leakyrelu = nn.LeakyReLU(0.2)
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            self.att = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        else:
            raise NotImplementedError("Use backbone_type='gcn' or 'gat'.")
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
            if self.variant:
                adj = torch.sparse_coo_tensor(
                    edge_index,
                    torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype),
                    size=(x.size(0), x.size(0)),
                )
                h_neigh = torch.sparse.mm(adj, x)
            else:
                h_neigh = gcn_backbone_conv(x, edge_index)
            h = torch.cat([h_neigh, x], dim=1)
            out = torch.matmul(h, self.weight)
        else:
            h = torch.matmul(x, self.weight)
            num_nodes = x.size(0)
            att_edge_index, _ = remove_self_loops(edge_index)
            att_edge_index, _ = add_self_loops(att_edge_index, num_nodes=num_nodes)
            src, dst = att_edge_index
            edge_h = torch.cat([h[src], h[dst]], dim=1)
            logits = self.leakyrelu(torch.matmul(edge_h, self.att)).squeeze(1)
            # Per-destination softmax. This is more stable than subtracting the
            # global maximum over all graph edges.
            alpha = softmax(logits, dst, num_nodes=num_nodes)
            out = self.specialspmm(att_edge_index, alpha, torch.Size([num_nodes, num_nodes]), h)

        if self.residual:
            out = out + x
        return out


class GlobalLinearAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = float(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for module in (self.query_proj, self.key_proj, self.value_proj, self.out_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        self.norm.reset_parameters()

    def forward(self, x, training=False):
        num_nodes = max(1, x.size(0))
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        q = q / q.norm(p='fro').clamp_min(1e-12)
        k = k / k.norm(p='fro').clamp_min(1e-12)
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        kv = torch.matmul(k.transpose(0, 1), v)
        k_sum = torch.matmul(k.transpose(0, 1), ones)
        attn_norm = 1.0 + torch.matmul(q, k_sum) / num_nodes
        global_repr = v + torch.matmul(q, kv) / num_nodes
        global_repr = global_repr / attn_norm.clamp_min(1e-12)
        global_repr = self.out_proj(global_repr)
        global_repr = F.dropout(global_repr, self.dropout, training=training)
        return self.norm(global_repr)


class AdvectiveGlobalMixer(nn.Module):
    def __init__(self, hidden_dim, beta=0.5, steps=1, dropout=0.0):
        super().__init__()
        self.beta = float(beta)
        self.steps = max(1, int(steps))
        self.global_attn = GlobalLinearAttention(hidden_dim, dropout=dropout)
        self.proj = nn.Linear(hidden_dim * (self.steps + 1), hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = float(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.global_attn.reset_parameters()
        self.proj.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x, edge_index, training=False, local_fn=None):
        states = [x]
        h = x
        for _ in range(self.steps):
            h_global = self.global_attn(h, training=training)
            h_local = gcn_backbone_conv(h, edge_index) if local_fn is None else local_fn(h, edge_index, training=training)
            h = h_global + self.beta * h_local
            h = F.dropout(h, self.dropout, training=training)
            states.append(h)
        return self.norm(self.proj(torch.cat(states, dim=-1)))


class FrontDoorLatentMixer(nn.Module):
    """Small masked mixer for front-door intervention without a learned DAG."""

    def __init__(self, hidden_dim, num_heads=1, num_layers=1, dropout=0.0):
        super().__init__()
        if hidden_dim % max(1, int(num_heads)) != 0:
            num_heads = 1
        self.label_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, max(1, int(num_heads)), dropout=dropout, batch_first=True)
            for _ in range(max(1, int(num_layers)))
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.attn_layers])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
            for _ in self.attn_layers
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.attn_layers])
        allowed = torch.tensor([
            [1, 0, 0, 0],  # mediator keeps itself
            [0, 1, 0, 0],  # spurious keeps itself
            [0, 1, 1, 0],  # context can absorb spurious
            [1, 0, 1, 1],  # label sees mediator and context, not direct spurious
        ], dtype=torch.bool)
        self.register_buffer('blocked_attn_mask', ~allowed)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.label_query, std=0.02)
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, nn.MultiheadAttention):
                module._reset_parameters()
            elif hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, z_mediator, z_spurious, context):
        if z_spurious is None:
            z_spurious = torch.zeros_like(z_mediator)
        tokens = torch.stack([z_mediator, z_spurious, context], dim=1)
        label_query = self.label_query.expand(z_mediator.size(0), -1, -1)
        tokens = torch.cat([tokens, label_query], dim=1)
        blocked = self.blocked_attn_mask.to(tokens.device)
        attn_mask = tokens.new_zeros(blocked.shape).masked_fill(blocked, -1e9)
        for attn, norm, ffn, ffn_norm in zip(self.attn_layers, self.norms, self.ffns, self.ffn_norms):
            out, _ = attn(tokens, tokens, tokens, attn_mask=attn_mask, need_weights=False)
            tokens = norm(tokens + out)
            tokens = ffn_norm(tokens + ffn(tokens))
        return tokens[:, 3, :]


class GraphFrontDoorDAG(nn.Module):
    """
    Lightweight Front-door Mediator-Context Gate model.

    This class keeps the old name for training-script compatibility, but it no
    longer learns a DAG. It replaces the two free CIPT adapters / DAG mediator
    selector with one shared gate:
        z_mediator = Gate(h)
        z_spurious = Complement(Gate(h))
    and uses front-door context sampling from z_spurious.
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
        self.dropout = float(getattr(args, 'dropout', 0.0))
        self.gamma = float(getattr(args, 'gamma', 0.99))
        self.fd_blend = float(getattr(args, 'fd_blend', 0.1))
        self.fd_sample_k = max(0, int(getattr(args, 'K', 0)))
        self.context_sample_seed = int(getattr(args, 'seed', 0))
        self.act_fn = nn.ReLU()

        self.input_proj = nn.Linear(d_in, self.d)
        self.backbone_layers = nn.ModuleList([
            FrontDoorBackboneLayer(self.d, self.d, backbone_type=self.backbone_type, residual=True, variant=self.variant)
            for _ in range(self.num_layers)
        ])

        self.use_global_info = bool(getattr(args, 'use_global_info', False))
        requested_global_contexts = getattr(args, 'use_global_contexts', None)
        if requested_global_contexts is None:
            requested_global_contexts = self.use_global_info
        self.use_global_contexts = bool(requested_global_contexts)
        self.global_info_mode = getattr(args, 'global_info_mode', 'advective')
        self.global_alpha = float(getattr(args, 'global_alpha', 0.2))
        self.global_beta = float(getattr(args, 'global_beta', 0.5))
        self.global_steps = max(1, int(getattr(args, 'global_steps', 1)))
        self.global_local_source = getattr(args, 'global_local_source', 'gcn')
        if self.use_global_info or self.use_global_contexts:
            if self.global_info_mode == 'linear':
                self.global_encoder = GlobalLinearAttention(self.d, dropout=self.dropout)
            elif self.global_info_mode == 'advective':
                self.global_encoder = AdvectiveGlobalMixer(self.d, beta=self.global_beta, steps=self.global_steps, dropout=self.dropout)
            else:
                raise ValueError("global_info_mode must be 'linear' or 'advective'.")
            self.global_fuse_norm = nn.LayerNorm(self.d)
            self.global_context_proj = nn.Sequential(nn.Linear(self.d, self.d), nn.ReLU(), nn.Linear(self.d, self.d))
            self.global_context_norm = nn.LayerNorm(self.d)
        else:
            self.global_encoder = None
            self.global_fuse_norm = None
            self.global_context_proj = None
            self.global_context_norm = None
        self.global_context_weight = max(0.0, float(getattr(args, 'global_context_weight', 1.0)))
        self.global_context_detach = bool(getattr(args, 'global_context_detach', True))

        # Relation-aware node representation enhancement.
        self.edge_score_temp = max(1e-3, float(getattr(args, 'edge_score_temp', 2.0)))
        self.edge_blend = max(0.0, float(getattr(args, 'edge_blend', 0.2)))
        self.use_neighbor_denoise = bool(getattr(args, 'use_neighbor_denoise', False))
        self.noise_subtract_alpha = max(0.0, float(getattr(args, 'noise_subtract_alpha', 0.1)))
        self.noise_gate_temp = max(1e-3, float(getattr(args, 'noise_gate_temp', 1.0)))
        self.edge_feat_mode = getattr(args, 'edge_feat_mode', 'mul')
        edge_feat_dim = self._get_edge_feat_dim(self.edge_feat_mode)
        self.edge_pair_encoder = nn.Sequential(nn.Linear(edge_feat_dim, self.d), nn.ReLU(), nn.Linear(self.d, self.d))
        self.edge_score_head = nn.Linear(self.d, 1)
        self.edge_summary_norm = nn.LayerNorm(self.d)
        self.node_edge_fuser = nn.Sequential(nn.Linear(self.d * 3, self.d), nn.ReLU(), nn.Linear(self.d, self.d))
        self.node_edge_norm = nn.LayerNorm(self.d)
        self.noise_summary_norm = nn.LayerNorm(self.d)
        self.node_noise_fuser = nn.Sequential(nn.Linear(self.d * 3, self.d), nn.ReLU(), nn.Linear(self.d, self.d))
        self.node_noise_gate = nn.Linear(self.d * 3, 1)

        # Front-door mediator-context gate.
        self.gate_mode = getattr(args, 'frontdoor_gate_mode', 'residual')
        self.gate_temp = max(1e-3, float(getattr(args, 'frontdoor_gate_temp', 1.0)))
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d, self.d),
        )
        self.mediator_norm = nn.LayerNorm(self.d)
        self.spurious_norm = nn.LayerNorm(self.d)

        self.classifier = nn.Linear(self.d, c)
        self.fd_classifier = nn.Linear(self.d, c)
        self.spurious_label_head = nn.Linear(self.d, c)
        self.env_classifier = nn.Linear(self.d, self.num_envs)

        self.use_frontdoor_mixer = bool(getattr(args, 'use_frontdoor_mixer', True))
        self.fd_mixer = FrontDoorLatentMixer(
            self.d,
            num_heads=getattr(args, 'frontdoor_mixer_heads', 1),
            num_layers=getattr(args, 'frontdoor_mixer_layers', 1),
            dropout=self.dropout,
        )
        self.fd_fuser = nn.Sequential(nn.Linear(self.d * 2, self.d), nn.ReLU(), nn.Linear(self.d, self.d))
        self.fd_norm = nn.LayerNorm(self.d)

        # Context sampling.
        self.use_spu_gmm = bool(getattr(args, 'use_spu_gmm', True))
        requested_gmm_sample_k = int(getattr(args, 'gmm_sample_k', 1))
        if requested_gmm_sample_k <= 0:
            requested_gmm_sample_k = self.fd_sample_k
        self.gmm_sample_k = max(0, requested_gmm_sample_k)
        self.gmm_min_var = max(1e-6, float(getattr(args, 'gmm_min_var', 1e-4)))
        self.gmm_max_std = max(0.0, float(getattr(args, 'gmm_max_std', 0.5)))
        self.use_proto_context = bool(getattr(args, 'use_proto_context', True))
        self.context_detach = bool(getattr(args, 'context_detach', True))

        self.register_buffer('proto_spu_env', torch.zeros(self.num_envs, self.d))
        self.register_buffer('proto_spu_env_valid', torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer('gmm_spu_mean', torch.zeros(self.num_envs, self.d))
        self.register_buffer('gmm_spu_var', torch.ones(self.num_envs, self.d))
        self.register_buffer('gmm_spu_valid', torch.zeros(self.num_envs, dtype=torch.bool))

        # Loss weights. Keep old names for main-script compatibility.
        self.lambda_med = float(getattr(args, 'lambda_med', 0.2))
        self.lambda_fd = float(getattr(args, 'lambda_fd', 0.05))
        self.lambda_spu = float(getattr(args, 'lambda_spu', 0.01))
        self.lambda_env = float(getattr(args, 'lambda_env', 0.0))
        self.lambda_spu_y = float(getattr(args, 'lambda_spu_y', 0.0))
        self.lambda_gate = float(getattr(args, 'lambda_gate', 0.0))
        self.lambda_var = float(getattr(args, 'lambda_var', 0.0))
        self.lambda_ind = float(getattr(args, 'lambda_ind', 0.0))
        self.lambda_dag = float(getattr(args, 'lambda_dag', 0.0))
        self.lambda_dag_label = float(getattr(args, 'lambda_dag_label', 0.0))
        self.lambda_fd_aug = float(getattr(args, 'lambda_fd_aug', 0.0))
        self.lambda_inv = float(getattr(args, 'lambda_inv', 0.0))
        self.lambda_global_env = float(getattr(args, 'lambda_global_env', 0.0))
        self.lambda_l1 = 0.0
        self.pseudo_env_balance = float(getattr(args, 'pseudo_env_balance', 1.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for layer in self.backbone_layers:
            layer.reset_parameters()
        if self.global_encoder is not None:
            self.global_encoder.reset_parameters()
            self.global_fuse_norm.reset_parameters()
            self._reset_module_parameters(self.global_context_proj)
            self.global_context_norm.reset_parameters()
        self._reset_module_parameters(self.edge_pair_encoder)
        self.edge_score_head.reset_parameters()
        self.edge_summary_norm.reset_parameters()
        self._reset_module_parameters(self.node_edge_fuser)
        nn.init.zeros_(self.node_edge_fuser[-1].weight)
        nn.init.zeros_(self.node_edge_fuser[-1].bias)
        self.node_edge_norm.reset_parameters()
        self.noise_summary_norm.reset_parameters()
        self._reset_module_parameters(self.node_noise_fuser)
        nn.init.zeros_(self.node_noise_fuser[-1].weight)
        nn.init.zeros_(self.node_noise_fuser[-1].bias)
        self.node_noise_gate.reset_parameters()
        nn.init.zeros_(self.node_noise_gate.weight)
        nn.init.zeros_(self.node_noise_gate.bias)
        self._reset_module_parameters(self.gate_mlp)
        gate_init_bias = float(getattr(self, 'gate_init_bias', 2.0))
        nn.init.constant_(self.gate_mlp[-1].bias, gate_init_bias)
        self.mediator_norm.reset_parameters()
        self.spurious_norm.reset_parameters()
        self.classifier.reset_parameters()
        self.fd_classifier.reset_parameters()
        self.spurious_label_head.reset_parameters()
        self.env_classifier.reset_parameters()
        self.fd_mixer.reset_parameters()
        self._reset_module_parameters(self.fd_fuser)
        self.fd_norm.reset_parameters()
        self.proto_spu_env.zero_()
        self.proto_spu_env_valid.zero_()
        self.gmm_spu_mean.zero_()
        self.gmm_spu_var.fill_(1.0)
        self.gmm_spu_valid.zero_()

    def _reset_module_parameters(self, module):
        for sub_module in module.modules():
            if sub_module is module:
                continue
            if hasattr(sub_module, 'reset_parameters'):
                sub_module.reset_parameters()

    def _get_edge_feat_dim(self, mode):
        if mode in ('mul', 'diff'):
            return self.d
        if mode == 'degree':
            return 1
        if mode == 'mul_diff':
            return 2 * self.d
        if mode in ('mul_degree', 'diff_degree'):
            return self.d + 1
        if mode == 'mul_diff_degree':
            return 2 * self.d + 1
        raise ValueError("edge_feat_mode must be one of mul, diff, degree, mul_diff, mul_degree, diff_degree, mul_diff_degree.")

    def build_edge_feat(self, h_src, h_dst, deg_src, deg_dst, deg_max):
        mul_feat = h_src * h_dst
        diff_feat = torch.abs(h_src - h_dst)
        deg_pair = torch.maximum(torch.log1p(deg_src), torch.log1p(deg_dst)) / deg_max.clamp_min(1.0)
        deg_pair = deg_pair.unsqueeze(-1)
        mode = self.edge_feat_mode
        if mode == 'mul':
            return mul_feat
        if mode == 'diff':
            return diff_feat
        if mode == 'degree':
            return deg_pair
        if mode == 'mul_diff':
            return torch.cat([mul_feat, diff_feat], dim=-1)
        if mode == 'mul_degree':
            return torch.cat([mul_feat, deg_pair], dim=-1)
        if mode == 'diff_degree':
            return torch.cat([diff_feat, deg_pair], dim=-1)
        if mode == 'mul_diff_degree':
            return torch.cat([mul_feat, diff_feat, deg_pair], dim=-1)
        raise ValueError(f'Unknown edge_feat_mode={mode}')

    def compute_edge_summaries(self, h, edge_index, training=False):
        if edge_index.numel() == 0:
            zero = h.new_zeros(h.size())
            return zero, zero, None
        src, dst = edge_index
        deg = degree(dst, h.size(0)).to(device=h.device, dtype=h.dtype).clamp_min(1.0)
        deg_max = torch.log1p(deg).max().clamp_min(1.0)
        h_src, h_dst = h[src], h[dst]
        edge_feat = self.build_edge_feat(h_src, h_dst, deg[src], deg[dst], deg_max)
        edge_hidden = self.edge_pair_encoder(edge_feat)
        edge_hidden = F.dropout(edge_hidden, self.dropout, training=training)
        edge_gate = torch.sigmoid(self.edge_score_head(edge_hidden).squeeze(-1) / self.edge_score_temp)
        norm = deg[src].pow(-0.5) * deg[dst].pow(-0.5)
        useful_weight = torch.nan_to_num(norm * edge_gate, nan=0.0, posinf=0.0, neginf=0.0)
        noise_weight = torch.nan_to_num(norm * (1.0 - edge_gate), nan=0.0, posinf=0.0, neginf=0.0)
        useful_summary = h.new_zeros(h.size())
        useful_summary.index_add_(0, dst, useful_weight.unsqueeze(-1) * h_src)
        useful_summary = self.edge_summary_norm(useful_summary)
        noise_summary = h.new_zeros(h.size())
        # Use residual shift as the low-relevance component. This is safer than
        # treating the whole neighbor representation as noise.
        noise_summary.index_add_(0, dst, noise_weight.unsqueeze(-1) * (h_src - h_dst))
        noise_summary = self.noise_summary_norm(noise_summary)
        return useful_summary, noise_summary, edge_gate

    def compute_edge_semantic_summary(self, h, edge_index, training=False):
        useful_summary, _, _ = self.compute_edge_summaries(h, edge_index, training=training)
        return useful_summary

    def fuse_node_edge_representation(self, h, edge_summary, noise_summary=None, training=False):
        useful_delta = self.node_edge_fuser(torch.cat([h, edge_summary, h * edge_summary], dim=-1))
        useful_delta = F.dropout(useful_delta, self.dropout, training=training)
        fused = h + self.edge_blend * useful_delta
        if self.use_neighbor_denoise and self.noise_subtract_alpha > 0.0 and noise_summary is not None:
            noise_delta = self.node_noise_fuser(torch.cat([h, noise_summary, h * noise_summary], dim=-1))
            noise_delta = F.dropout(noise_delta, self.dropout, training=training)
            noise_gate = torch.sigmoid(self.node_noise_gate(torch.cat([h, noise_summary, torch.abs(h - noise_summary)], dim=-1)) / self.noise_gate_temp)
            fused = fused - self.noise_subtract_alpha * noise_gate * noise_delta
        return self.node_edge_norm(fused)

    def encode_backbone(self, x, edge_index, training=False):
        x = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(self.input_proj(x))
        for layer in self.backbone_layers:
            h = F.dropout(h, self.dropout, training=training)
            h = self.act_fn(layer(h, edge_index))
        h_global_context = None
        if self.global_encoder is not None:
            if self.global_info_mode == 'linear':
                h_global = self.global_encoder(h, training=training)
            else:
                local_fn = self.compute_edge_semantic_summary if self.global_local_source == 'edge' else None
                h_global = self.global_encoder(h, edge_index, training=training, local_fn=local_fn)
            h_global_context = self.global_context_norm(self.global_context_proj(h_global))
            if self.use_global_info:
                h = h + self.global_alpha * self.global_fuse_norm(h_global)
        edge_summary, noise_summary, edge_gate = self.compute_edge_summaries(h, edge_index, training=training)
        z = self.fuse_node_edge_representation(h, edge_summary, noise_summary=noise_summary, training=training)
        return z, edge_summary, h_global_context, edge_gate

    def decompose_representation(self, z, training=False):
        gate = torch.sigmoid(self.gate_mlp(z) / self.gate_temp)
        if self.gate_mode == 'mask':
            z_mediator = self.mediator_norm(z * gate)
            z_spurious = self.spurious_norm(z * (1.0 - gate))
        elif self.gate_mode == 'residual':
            z_mediator = self.mediator_norm(z + z * gate)
            z_spurious = self.spurious_norm(z + z * (1.0 - gate))
        else:
            raise ValueError("frontdoor_gate_mode must be 'residual' or 'mask'.")
        z_mediator = F.dropout(z_mediator, self.dropout, training=training)
        z_spurious = F.dropout(z_spurious, self.dropout, training=training)
        med_logits = self.classifier(z_mediator)
        return z_mediator, z_spurious, med_logits, gate

    def encode_representation(self, x, edge_index, training=False):
        z, edge_summary, h_global_context, _ = self.encode_backbone(x, edge_index, training=training)
        z_mediator, z_spurious, med_logits, mediator_gate = self.decompose_representation(z, training=training)
        return z, edge_summary, z_mediator, z_spurious, med_logits, mediator_gate, h_global_context

    def compute_pseudo_env_probs(self, z_spurious):
        if self.num_envs <= 1 or z_spurious is None or z_spurious.numel() == 0:
            return z_spurious.new_ones(z_spurious.size(0), 1)
        return F.softmax(self.env_classifier(z_spurious), dim=-1)

    def get_frontdoor_contexts(self, z_spurious=None, env_probs=None):
        if not self.use_proto_context:
            return None
        context_map = {}
        if self.proto_spu_env_valid.any():
            for env_idx in self.proto_spu_env_valid.nonzero(as_tuple=False).squeeze(-1).tolist():
                context_map[int(env_idx)] = self.proto_spu_env[env_idx].detach()
        if z_spurious is not None and z_spurious.numel() > 0:
            if env_probs is None:
                env_probs = self.compute_pseudo_env_probs(z_spurious).detach()
            env_probs = env_probs.detach().clamp_min(0.0)
            values = z_spurious.detach() if self.context_detach else z_spurious
            for env_idx in range(env_probs.size(1)):
                weight = env_probs[:, env_idx]
                mass = weight.sum()
                if mass > 1e-6:
                    vec = (weight.unsqueeze(-1) * values).sum(dim=0) / mass.clamp_min(1e-6)
                    context_map[int(env_idx)] = F.normalize(vec.detach(), dim=0)
        if not context_map:
            return None
        return torch.stack([context_map[idx] for idx in sorted(context_map.keys())], dim=0)

    def get_global_contexts(self, h_global=None, env_probs=None):
        if not self.use_global_contexts or h_global is None or h_global.numel() == 0 or self.global_context_weight <= 0.0:
            return None
        global_values = h_global.detach() if self.global_context_detach else h_global
        if env_probs is None:
            env_probs = self.compute_pseudo_env_probs(global_values)
        env_probs = env_probs.detach().clamp_min(0.0)
        contexts = []
        for env_idx in range(env_probs.size(1)):
            weight = env_probs[:, env_idx]
            mass = weight.sum()
            if mass > 1e-6:
                context_vec = (weight.unsqueeze(-1) * global_values).sum(dim=0) / mass.clamp_min(1e-6)
                contexts.append(F.normalize(context_vec, dim=0))
        if not contexts:
            return None
        return self.global_context_weight * torch.stack(contexts, dim=0)

    def merge_frontdoor_contexts(self, *context_sets):
        contexts = [ctx for ctx in context_sets if ctx is not None and ctx.numel() > 0]
        if not contexts:
            return None
        return torch.cat(contexts, dim=0)

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

    def _fit_spurious_gmm_stats(self, z_spurious, env_probs):
        means = z_spurious.new_zeros(self.num_envs, self.d)
        vars_ = z_spurious.new_ones(self.num_envs, self.d)
        valid = torch.zeros(self.num_envs, device=z_spurious.device, dtype=torch.bool)
        if z_spurious is None or z_spurious.numel() == 0:
            return means, vars_, valid
        if env_probs is None or env_probs.numel() == 0 or env_probs.size(-1) != self.num_envs:
            env_probs = self.compute_pseudo_env_probs(z_spurious)
        env_probs = env_probs.detach().clamp_min(0.0)
        z_detached = z_spurious.detach()
        for env_idx in range(self.num_envs):
            weights = env_probs[:, env_idx]
            mass = weights.sum()
            if mass <= 1e-8:
                continue
            mean = (z_detached * weights.unsqueeze(-1)).sum(dim=0) / mass.clamp_min(1e-8)
            centered = z_detached - mean
            var = (centered.pow(2) * weights.unsqueeze(-1)).sum(dim=0) / mass.clamp_min(1e-8)
            means[env_idx] = mean
            vars_[env_idx] = var.clamp_min(self.gmm_min_var)
            valid[env_idx] = True
        return means, vars_, valid

    def sample_gmm_contexts(self, z_spurious=None, env_probs=None, training=False):
        if not self.use_spu_gmm or self.gmm_sample_k <= 0:
            return None
        if training and z_spurious is not None:
            means, vars_, valid = self._fit_spurious_gmm_stats(z_spurious, env_probs)
        else:
            means, vars_, valid = self.gmm_spu_mean, self.gmm_spu_var, self.gmm_spu_valid
        valid_envs = valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_envs.numel() == 0:
            return None
        sample_k = min(self.gmm_sample_k, self.fd_sample_k) if self.fd_sample_k > 0 else self.gmm_sample_k
        if sample_k <= 0:
            return None
        if training:
            env_indices = valid_envs[torch.randint(valid_envs.numel(), (sample_k,), device=valid_envs.device)]
        else:
            repeat = (sample_k + valid_envs.numel() - 1) // valid_envs.numel()
            env_indices = valid_envs.repeat(repeat)[:sample_k]
        mean = means.index_select(0, env_indices)
        std = vars_.index_select(0, env_indices).clamp_min(self.gmm_min_var).sqrt()
        if self.gmm_max_std > 0.0:
            std = std.clamp_max(self.gmm_max_std)
        if training:
            noise = torch.randn_like(mean)
        else:
            generator = torch.Generator(device=mean.device)
            generator.manual_seed(self.context_sample_seed + sample_k + int(valid_envs.numel()))
            noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        return F.normalize(mean + noise * std, dim=1)

    def frontdoor_logits_from_contexts(self, z_mediator, z_spurious, contexts):
        base_logits = self.fd_classifier(z_mediator)
        if contexts is None or contexts.size(0) == 0:
            return base_logits, None
        num_contexts = contexts.size(0)
        mediator_expand = z_mediator.unsqueeze(1).expand(-1, num_contexts, -1)
        spurious_expand = z_spurious.unsqueeze(1).expand(-1, num_contexts, -1)
        context_expand = contexts.unsqueeze(0).expand(z_mediator.size(0), -1, -1)
        if self.use_frontdoor_mixer:
            fused = self.fd_mixer(
                mediator_expand.reshape(-1, self.d),
                spurious_expand.reshape(-1, self.d),
                context_expand.reshape(-1, self.d),
            ).view(z_mediator.size(0), num_contexts, self.d)
            fused = self.fd_norm(fused + mediator_expand)
        else:
            fused = self.fd_fuser(torch.cat([mediator_expand, context_expand], dim=-1).reshape(-1, self.d * 2))
            fused = fused.view(z_mediator.size(0), num_contexts, self.d)
            fused = self.fd_norm(fused + mediator_expand)
        logits_stack = self.fd_classifier(fused.reshape(-1, self.d)).view(z_mediator.size(0), num_contexts, self.c)
        return logits_stack.mean(dim=1), logits_stack

    def blend_logits(self, med_logits, fd_logits):
        if fd_logits is None:
            return med_logits
        return (1.0 - self.fd_blend) * med_logits + self.fd_blend * fd_logits

    def forward(self, x, edge_index, training=False):
        z, edge_summary, z_mediator, z_spurious, med_logits, mediator_gate, h_global_context = self.encode_representation(
            x, edge_index, training=training
        )
        contexts = self.merge_frontdoor_contexts(
            self.sample_frontdoor_contexts(self.sample_gmm_contexts(training=training), training=training),
            self.get_global_contexts(h_global_context),
            self.get_frontdoor_contexts(),
        )
        fd_logits, fd_stack = self.frontdoor_logits_from_contexts(z_mediator, z_spurious, contexts)
        logits = self.blend_logits(med_logits, fd_logits)
        if training:
            return logits, z, edge_summary, z_mediator, z_spurious, mediator_gate, med_logits, fd_logits, fd_stack, h_global_context
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

    def compute_env_uniform_loss(self, logits):
        if logits is None or logits.size(-1) <= 1:
            return self.classifier.weight.new_zeros(())
        log_probs = F.log_softmax(logits, dim=-1)
        uniform = torch.full_like(log_probs, 1.0 / logits.size(-1))
        return F.kl_div(log_probs, uniform, reduction='batchmean')

    def compute_pseudo_env_loss(self, env_logits):
        if env_logits is None or env_logits.size(-1) <= 1:
            return self.classifier.weight.new_zeros(())
        probs = F.softmax(env_logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        mean_probs = probs.mean(dim=0)
        uniform = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
        balance = F.kl_div(mean_probs.clamp_min(1e-8).log(), uniform, reduction='sum')
        return entropy + self.pseudo_env_balance * balance

    def compute_spurious_label_uniform_loss(self, z_spurious):
        if self.lambda_spu_y <= 0.0:
            return self.classifier.weight.new_zeros(())
        return self.compute_env_uniform_loss(self.spurious_label_head(z_spurious))

    def compute_frontdoor_variance_loss(self, logits_stack):
        if logits_stack is None or logits_stack.size(1) <= 1:
            return self.classifier.weight.new_zeros(())
        probs = torch.softmax(logits_stack, dim=-1)
        return probs.var(dim=1, unbiased=False).mean()

    def compute_independence_loss(self, z_mediator, z_spurious):
        if self.lambda_ind <= 0.0 or z_mediator.numel() == 0:
            return self.classifier.weight.new_zeros(())
        z_med = F.normalize(z_mediator, dim=1)
        z_spu = F.normalize(z_spurious, dim=1)
        return 0.5 * ((z_med * z_spu).sum(dim=1) ** 2).mean()

    def compute_gate_loss(self, gate):
        if self.lambda_gate <= 0.0 or gate is None:
            return self.classifier.weight.new_zeros(())
        # Encourage confident but not collapsed partitions. Very small weight only.
        return (gate * (1.0 - gate)).mean()

    @torch.no_grad()
    def update_spurious_env_prototypes(self, z_spurious, env_probs=None):
        if not self.use_proto_context or z_spurious is None or z_spurious.numel() == 0:
            return
        if env_probs is None or env_probs.numel() == 0 or env_probs.size(-1) != self.num_envs:
            env_probs = self.compute_pseudo_env_probs(z_spurious)
        env_probs = env_probs.detach().clamp_min(0.0)
        z_detached = z_spurious.detach()
        for env_idx in range(env_probs.size(1)):
            weights = env_probs[:, env_idx]
            mass = weights.sum()
            if mass <= 1e-8:
                continue
            vec = (z_detached * weights.unsqueeze(-1)).sum(dim=0) / mass.clamp_min(1e-8)
            if self.proto_spu_env_valid[env_idx]:
                vec = self.gamma * self.proto_spu_env[env_idx] + (1.0 - self.gamma) * vec
            self.proto_spu_env[env_idx] = F.normalize(vec, dim=0)
            self.proto_spu_env_valid[env_idx] = True

    @torch.no_grad()
    def update_spurious_gmm(self, z_spurious, env_probs=None):
        if not self.use_spu_gmm or z_spurious is None or z_spurious.numel() == 0:
            return
        means, vars_, valid = self._fit_spurious_gmm_stats(z_spurious, env_probs)
        for env_idx in valid.nonzero(as_tuple=False).squeeze(-1).tolist():
            mean = means[env_idx]
            var = vars_[env_idx]
            if self.gmm_spu_valid[env_idx]:
                mean = self.gamma * self.gmm_spu_mean[env_idx] + (1.0 - self.gamma) * mean
                var = self.gamma * self.gmm_spu_var[env_idx] + (1.0 - self.gamma) * var
            self.gmm_spu_mean[env_idx] = mean
            self.gmm_spu_var[env_idx] = var.clamp_min(self.gmm_min_var)
            self.gmm_spu_valid[env_idx] = True

    @torch.no_grad()
    def apply_state_update(self, state_payload):
        if state_payload is None:
            return
        self.update_spurious_env_prototypes(state_payload.get('spu_tr'), state_payload.get('env_probs_tr'))
        self.update_spurious_gmm(state_payload.get('spu_tr'), state_payload.get('env_probs_tr'))

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx
        logits_all, _, _, z_mediator_all, z_spurious_all, mediator_gate, med_logits_all, _, _, h_global_all = self.forward(
            x, edge_index, training=True
        )
        y_tr = y[train_idx]
        med_tr = z_mediator_all[train_idx]
        spu_tr = z_spurious_all[train_idx]
        gate_tr = mediator_gate[train_idx]
        med_logits_tr = med_logits_all[train_idx]
        env_logits_spu = self.env_classifier(spu_tr) if self.num_envs > 1 else None
        env_probs_spu = F.softmax(env_logits_spu, dim=-1) if env_logits_spu is not None else None

        proto_contexts = self.get_frontdoor_contexts(spu_tr, env_probs_spu)
        gmm_contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(spu_tr, env_probs_spu, training=True), training=True
        )
        global_contexts = self.get_global_contexts(
            h_global_all,
            self.compute_pseudo_env_probs(h_global_all) if h_global_all is not None else None,
        )
        contexts = self.merge_frontdoor_contexts(proto_contexts, gmm_contexts, global_contexts)
        fd_logits_tr, fd_stack_tr = self.frontdoor_logits_from_contexts(med_tr, spu_tr, contexts)
        final_logits_tr = self.blend_logits(med_logits_tr, fd_logits_tr)

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_med = self.compute_supervised_loss(med_logits_tr, y_tr, criterion, args).mean()
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        loss_spu = self.compute_pseudo_env_loss(env_logits_spu)
        loss_env_med = self.compute_env_uniform_loss(self.env_classifier(med_tr)) if self.num_envs > 1 else self.classifier.weight.new_zeros(())
        loss_spu_y = self.compute_spurious_label_uniform_loss(spu_tr)
        loss_var = self.compute_frontdoor_variance_loss(fd_stack_tr)
        loss_ind = self.compute_independence_loss(med_tr, spu_tr)
        loss_gate = self.compute_gate_loss(gate_tr)

        zero = self.classifier.weight.new_zeros(())
        loss_dag = zero
        loss_dag_label = zero
        loss_fd_aug = zero
        loss_sem = zero
        loss_degree = zero
        loss_inv = zero
        loss_global_env = zero

        total_loss = (
            loss_cls
            + self.lambda_med * loss_med
            + self.lambda_fd * loss_fd
            + self.lambda_spu * loss_spu
            + self.lambda_env * loss_env_med
            + self.lambda_spu_y * loss_spu_y
            + self.lambda_var * loss_var
            + self.lambda_ind * loss_ind
            + self.lambda_gate * loss_gate
        )

        state_payload = None
        if update_state:
            state_payload = {
                'spu_tr': spu_tr.detach(),
                'env_probs_tr': env_probs_spu.detach() if env_probs_spu is not None else None,
            }

        num_contexts = 0 if contexts is None else int(contexts.size(0))
        num_gmm_contexts = 0 if gmm_contexts is None else int(gmm_contexts.size(0))
        num_global_contexts = 0 if global_contexts is None else int(global_contexts.size(0))
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
            'loss_sem': loss_sem,
            'loss_degree': loss_degree,
            'loss_spu_y': loss_spu_y,
            'loss_spu': loss_spu,
            'loss_env_med': loss_env_med,
            'loss_inv': loss_inv,
            'loss_global_env': loss_global_env,
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'causal_score_mean': gate_tr.mean().detach(),
            'pollution_score_mean': (1.0 - gate_tr).mean().detach(),
            'counterexample_penalty_mean': zero.detach(),
            'counterexample_penalty_batch_mean': zero.detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_mixed_contexts': torch.tensor(0.0, device=x.device),
            'num_gmm_contexts': torch.tensor(float(num_gmm_contexts), device=x.device),
            'num_global_contexts': torch.tensor(float(num_global_contexts), device=x.device),
            'state_payload': state_payload,
        }

    def loss_compute(self, data, criterion, args):
        losses = self.compute_losses(data, criterion, args, update_state=True)
        return (
            losses['total_loss'],
            losses['loss_cls'].item(),
            (self.lambda_ind * losses['loss_ind']).item(),
            (self.lambda_dag * losses['loss_dag']).item(),
            (self.lambda_fd * losses['loss_fd']).item(),
        )
