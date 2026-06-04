import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, softmax
from torch_sparse import SparseTensor, matmul


def gcn_backbone_conv(x, edge_index):
    num_nodes = x.size(0)
    row, col = edge_index
    deg = degree(col, num_nodes).float().clamp_min(1.0)
    deg_in = deg[col].pow(-0.5)
    deg_out = deg[row].pow(-0.5)
    value = torch.ones_like(row, dtype=x.dtype) * deg_in * deg_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
    return matmul(adj, x)


class FrontDoorBackboneLayer(nn.Module):
    """CaNet-style graph encoder layer used before CFAM node enhancement."""

    def __init__(self, in_features, out_features, backbone_type='gcn', residual=True, variant=False):
        super().__init__()
        self.backbone_type = backbone_type
        self.out_features = out_features
        self.residual = residual
        self.variant = variant

        if backbone_type == 'gcn':
            self.weight = Parameter(torch.FloatTensor(in_features * 2, out_features))
        elif backbone_type == 'gat':
            self.leakyrelu = nn.LeakyReLU()
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
            self.att = Parameter(torch.FloatTensor(2 * out_features, 1))
        else:
            raise NotImplementedError("Use backbone_type='gcn' or 'gat'.")
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.out_features ** 0.5)
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
            out = torch.matmul(torch.cat([h_neigh, x], dim=1), self.weight)
        else:
            h = torch.matmul(x, self.weight)
            num_nodes = x.size(0)
            att_edge_index, _ = remove_self_loops(edge_index)
            att_edge_index, _ = add_self_loops(att_edge_index, num_nodes=num_nodes)
            edge_h = torch.cat([h[att_edge_index[0]], h[att_edge_index[1]]], dim=1)
            logits = self.leakyrelu(torch.matmul(edge_h, self.att)).squeeze(1)
            alpha = softmax(logits, att_edge_index[1], num_nodes=num_nodes)
            out = self.specialspmm(att_edge_index, alpha, torch.Size([num_nodes, num_nodes]), h)
        return out + x if self.residual else out


class GraphFrontDoorDAG(nn.Module):
    """
    Compact Graph-CFAM + Energy-RSC + multi-ratio front-door model.

    Kept path:
    1) GNN backbone encodes node states.
    2) Graph-CFAM builds smooth/residual local signals and enhances causal nodes.
    3) Energy-guided RSC protects reliable dimensions and challenges shortcuts.
    4) Multi-ratio spurious contexts provide the environment mixture.
    5) Front-door aggregation predicts through mediator + mixed environment contexts.
    """

    def __init__(self, d_in, c, args, device):
        super().__init__()
        self.device = device
        self.d = int(args.hidden_channels)
        self.c = int(c)
        self.num_envs = max(1, int(getattr(args, 'train_env_num', 1)))
        self.num_layers = max(1, int(getattr(args, 'num_layers', 2)))
        self.backbone_type = getattr(args, 'backbone_type', 'gcn')
        self.variant = bool(getattr(args, 'variant', False))
        self.dropout = float(getattr(args, 'dropout', 0.0))
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

        self.classifier = nn.Linear(self.d, self.c)
        self.fd_classifier = nn.Linear(self.d, self.c)
        self.env_classifier = nn.Linear(self.d, self.num_envs)

        self.edge_feat_mode = getattr(args, 'edge_feat_mode', 'mul')
        self.edge_gate_mode = getattr(args, 'edge_gate_mode', 'vector')
        if self.edge_gate_mode not in ('scalar', 'vector'):
            self.edge_gate_mode = 'vector'
        self.edge_score_temp = max(1e-3, float(getattr(args, 'edge_score_temp', 2.0)))
        self.edge_blend = max(0.0, float(getattr(args, 'edge_blend', 0.2)))
        edge_gate_out_dim = 1 if self.edge_gate_mode == 'scalar' else self.d
        self.edge_pair_encoder = nn.Sequential(
            nn.Linear(self._get_edge_feat_dim(self.edge_feat_mode), self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.edge_score_head = nn.Linear(self.d, edge_gate_out_dim)
        self.edge_summary_norm = nn.LayerNorm(self.d)
        self.noise_summary_norm = nn.LayerNorm(self.d)
        self.node_edge_fuser = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.use_node_edge_norm = not bool(getattr(args, 'disable_node_edge_norm', False))
        self.node_edge_norm = nn.LayerNorm(self.d)

        self.use_graph_cfam = bool(getattr(args, 'use_graph_cfam', True))
        self.use_pre_gnn_graph_cfam = bool(getattr(args, 'use_pre_gnn_graph_cfam', False))
        self.use_final_graph_cfam = bool(getattr(args, 'use_final_graph_cfam', True))
        self.layerwise_local_igm_skip_last = bool(getattr(args, 'layerwise_local_igm_skip_last', True))
        self.pre_graph_cfam_blend = max(0.0, float(getattr(args, 'pre_graph_cfam_blend', 0.1)))
        self.pre_graph_cfam_residual_blend = max(0.0, float(getattr(args, 'pre_graph_cfam_residual_blend', 0.0)))
        self.graph_cfam_residual_blend = max(0.0, float(getattr(args, 'graph_cfam_residual_blend', 0.1)))
        self.graph_cfam_gate_temp = max(1e-3, float(getattr(args, 'graph_cfam_gate_temp', 1.0)))
        self.graph_cfam_gate_target = min(max(float(getattr(args, 'graph_cfam_gate_target', 0.5)), 0.0), 1.0)
        self.lambda_graph_cfam_gate = max(0.0, float(getattr(args, 'lambda_graph_cfam_gate', 0.0)))
        self.graph_cfam_gate = nn.Sequential(
            nn.Linear(self.d * 5, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.graph_cfam_norm = nn.LayerNorm(self.d)

        self.use_energy_rsc_gate = bool(getattr(args, 'use_energy_rsc_gate', True))
        self.lambda_energy_gate_rec = max(0.0, float(getattr(args, 'lambda_energy_gate_rec', 0.0)))
        self.energy_rsc_top_frac = min(max(float(getattr(args, 'energy_rsc_top_frac', 0.2)), 0.0), 1.0)
        self.energy_rsc_second_weight = max(0.0, float(getattr(args, 'energy_rsc_second_weight', 0.5)))
        self.energy_rsc_reliability_temp = max(1e-3, float(getattr(args, 'energy_rsc_reliability_temp', 1.0)))
        self.energy_rsc_reliability_floor = min(max(float(getattr(args, 'energy_rsc_reliability_floor', 0.05)), 0.0), 1.0)
        self.energy_rsc_edge_sample = max(1, int(getattr(args, 'energy_rsc_edge_sample', 4096)))
        self.energy_rsc_detach_reliability = bool(getattr(args, 'energy_rsc_detach_reliability', True))
        self.energy_rsc_second_gate = nn.Sequential(
            nn.Linear(self.d * 5, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.edge_energy_diag = Parameter(torch.ones(self.d))

        bottleneck_dim = max(16, self.d // 2)
        self.spurious_adapter = nn.Sequential(
            nn.Linear(self.d, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(bottleneck_dim, self.d),
        )
        self.spurious_norm = nn.LayerNorm(self.d)
        self.direct_z_spurious_mode = getattr(args, 'direct_z_spurious_mode', 'shortcut')
        if self.direct_z_spurious_mode not in ('shortcut', 'zero', 'z_adapter'):
            self.direct_z_spurious_mode = 'shortcut'

        self.fd_fuser = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.fd_norm = nn.LayerNorm(self.d)
        self.fd_blend = float(getattr(args, 'fd_blend', 0.5))
        self.eval_pred_mode = getattr(args, 'eval_pred_mode', 'mediator')
        if self.eval_pred_mode not in ('blend', 'mediator', 'frontdoor'):
            self.eval_pred_mode = 'mediator'

        self.multi_ratio_spurious_source = getattr(args, 'multi_ratio_spurious_source', 'self')
        if self.multi_ratio_spurious_source not in ('self', 'shuffle'):
            self.multi_ratio_spurious_source = 'self'
        self.multi_ratio_spurious_ratios = self._parse_ratios(
            getattr(args, 'multi_ratio_spurious_ratios', '0,0.33,0.67,1.0')
        )
        self.lambda_fd = max(0.0, float(getattr(args, 'lambda_fd', 0.5)))
        self.lambda_multi_ratio_fd = max(0.0, float(getattr(args, 'lambda_multi_ratio_fd', 0.5)))
        self.lambda_multi_ratio_fd_worst = max(0.0, float(getattr(args, 'lambda_multi_ratio_fd_worst', 0.2)))
        self.lambda_multi_ratio_fd_cons = max(0.0, float(getattr(args, 'lambda_multi_ratio_fd_cons', 0.1)))
        self.context_sample_seed = int(getattr(args, 'seed', 0))

        self._last_graph_cfam_gate_loss = None
        self._last_graph_cfam_gate_mean = None
        self._last_graph_cfam_layers = 0
        self._last_energy_gate_rec_loss = None
        self._last_energy_gate_reliability_mean = None
        self._last_energy_gate_rsc_mask_mean = None
        self._last_energy_gate_second_mean = None
        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for layer in self.backbone_layers:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        self.fd_classifier.reset_parameters()
        self.env_classifier.reset_parameters()
        self._reset_module_parameters(self.edge_pair_encoder)
        self.edge_score_head.reset_parameters()
        self.edge_summary_norm.reset_parameters()
        self.noise_summary_norm.reset_parameters()
        self._reset_module_parameters(self.node_edge_fuser)
        nn.init.zeros_(self.node_edge_fuser[-1].weight)
        nn.init.zeros_(self.node_edge_fuser[-1].bias)
        self.node_edge_norm.reset_parameters()
        self._reset_module_parameters(self.graph_cfam_gate)
        nn.init.zeros_(self.graph_cfam_gate[-1].weight)
        nn.init.zeros_(self.graph_cfam_gate[-1].bias)
        self.graph_cfam_norm.reset_parameters()
        self._reset_module_parameters(self.energy_rsc_second_gate)
        nn.init.zeros_(self.energy_rsc_second_gate[-1].weight)
        nn.init.zeros_(self.energy_rsc_second_gate[-1].bias)
        nn.init.ones_(self.edge_energy_diag)
        self._reset_module_parameters(self.spurious_adapter)
        self.spurious_norm.reset_parameters()
        self._reset_module_parameters(self.fd_fuser)
        self.fd_norm.reset_parameters()
        self._clear_diagnostics()

    def _reset_module_parameters(self, module):
        for sub_module in module.modules():
            if sub_module is module:
                continue
            if hasattr(sub_module, 'reset_parameters'):
                sub_module.reset_parameters()

    def _clear_diagnostics(self):
        self._last_graph_cfam_gate_loss = None
        self._last_graph_cfam_gate_mean = None
        self._last_graph_cfam_layers = 0
        self._last_energy_gate_rec_loss = None
        self._last_energy_gate_reliability_mean = None
        self._last_energy_gate_rsc_mask_mean = None
        self._last_energy_gate_second_mean = None

    def _parse_ratios(self, ratio_text):
        ratios = []
        for item in str(ratio_text).split(','):
            item = item.strip()
            if not item:
                continue
            try:
                ratios.append(min(max(float(item), 0.0), 1.0))
            except ValueError:
                continue
        return tuple(ratios) if ratios else (0.0, 0.33, 0.67, 1.0)

    def _get_edge_feat_dim(self, mode):
        if mode in ('mul', 'diff', 'signed_diff'):
            return self.d
        if mode == 'degree':
            return 1
        if mode in ('mul_diff', 'mul_signed_diff', 'concat'):
            return 2 * self.d
        if mode == 'concat_diff':
            return 3 * self.d
        if mode in ('mul_degree', 'diff_degree'):
            return self.d + 1
        if mode in ('mul_diff_degree', 'mul_signed_diff_degree'):
            return 2 * self.d + 1
        raise ValueError(f"Unknown edge_feat_mode='{mode}'.")

    def build_edge_feat(self, h_src, h_dst, deg_src, deg_dst, deg_max):
        mul_feat = h_src * h_dst
        signed_diff_feat = h_src - h_dst
        diff_feat = torch.abs(signed_diff_feat)
        concat_feat = torch.cat([h_src, h_dst], dim=-1)
        deg_pair = torch.maximum(torch.log1p(deg_src), torch.log1p(deg_dst))
        deg_pair = (deg_pair / deg_max.clamp_min(1.0)).unsqueeze(-1)

        if self.edge_feat_mode == 'mul':
            return mul_feat
        if self.edge_feat_mode == 'diff':
            return diff_feat
        if self.edge_feat_mode == 'signed_diff':
            return signed_diff_feat
        if self.edge_feat_mode == 'degree':
            return deg_pair
        if self.edge_feat_mode == 'mul_diff':
            return torch.cat([mul_feat, diff_feat], dim=-1)
        if self.edge_feat_mode == 'mul_signed_diff':
            return torch.cat([mul_feat, signed_diff_feat], dim=-1)
        if self.edge_feat_mode == 'concat':
            return concat_feat
        if self.edge_feat_mode == 'concat_diff':
            return torch.cat([concat_feat, diff_feat], dim=-1)
        if self.edge_feat_mode == 'mul_degree':
            return torch.cat([mul_feat, deg_pair], dim=-1)
        if self.edge_feat_mode == 'diff_degree':
            return torch.cat([diff_feat, deg_pair], dim=-1)
        if self.edge_feat_mode == 'mul_diff_degree':
            return torch.cat([mul_feat, diff_feat, deg_pair], dim=-1)
        if self.edge_feat_mode == 'mul_signed_diff_degree':
            return torch.cat([mul_feat, signed_diff_feat, deg_pair], dim=-1)
        raise ValueError(f"Unknown edge_feat_mode='{self.edge_feat_mode}'.")

    def compute_edge_summaries(self, h, edge_index, training=False):
        if edge_index.numel() == 0:
            zero = h.new_zeros(h.size())
            return zero, zero, None

        src, dst = edge_index
        deg = degree(dst, h.size(0)).to(device=h.device, dtype=h.dtype).clamp_min(1.0)
        deg_max = torch.log1p(deg).max().clamp_min(1.0)
        edge_feat = self.build_edge_feat(h[src], h[dst], deg[src], deg[dst], deg_max)
        edge_hidden = self.edge_pair_encoder(edge_feat)
        edge_hidden = F.dropout(edge_hidden, self.dropout, training=training)
        edge_gate = torch.sigmoid(self.edge_score_head(edge_hidden) / self.edge_score_temp)
        if edge_gate.dim() == 1:
            edge_gate = edge_gate.unsqueeze(-1)
        edge_gate = self.apply_energy_reliability_to_edge_gate(h, edge_index, edge_gate, training=training)

        norm = (deg[src].pow(-0.5) * deg[dst].pow(-0.5)).unsqueeze(-1)
        useful_weight = torch.nan_to_num(norm * edge_gate, nan=0.0, posinf=0.0, neginf=0.0)
        noise_weight = torch.nan_to_num(norm * (1.0 - edge_gate), nan=0.0, posinf=0.0, neginf=0.0)

        useful_summary = h.new_zeros(h.size())
        useful_summary.index_add_(0, dst, useful_weight * h[src])
        noise_summary = h.new_zeros(h.size())
        noise_summary.index_add_(0, dst, noise_weight * h[src])
        return self.edge_summary_norm(useful_summary), self.noise_summary_norm(noise_summary), edge_gate

    def _graph_cfam_energy(self, value):
        energy = value.pow(2)
        return energy / energy.mean(dim=-1, keepdim=True).clamp_min(1e-6)

    def _sample_reconstruction_edges(self, edge_index, num_nodes, training=False):
        if edge_index is None or edge_index.numel() == 0 or num_nodes <= 1:
            return None, None, None
        src, dst = edge_index
        num_edges = int(src.numel())
        max_edges = min(num_edges, self.energy_rsc_edge_sample)
        if num_edges > max_edges:
            if training:
                perm = torch.randperm(num_edges, device=edge_index.device)[:max_edges]
            else:
                perm = torch.linspace(0, num_edges - 1, steps=max_edges, device=edge_index.device).long()
            src = src.index_select(0, perm)
            dst = dst.index_select(0, perm)
        neg_dst = torch.randint(0, num_nodes, dst.shape, device=edge_index.device)
        neg_dst = torch.where(neg_dst == dst, (neg_dst + 1) % num_nodes, neg_dst)
        return src, dst, neg_dst

    def _normalize_score(self, values, default_value=0.5):
        if values.numel() == 0:
            return values
        v_min = values.min()
        v_max = values.max()
        if (v_max - v_min).abs() < 1e-4:
            return (values - values.detach()) + default_value
        return (values - v_min) / (v_max - v_min + 1e-8)

    def compute_energy_gate_reliability(self, h, edge_index, candidate_gate=None, training=False):
        sampled = self._sample_reconstruction_edges(edge_index, h.size(0), training=training)
        if sampled[0] is None:
            return None
        src, dst, neg_dst = sampled
        basis = h if candidate_gate is None else h * candidate_gate
        diag = F.softplus(self.edge_energy_diag.to(device=h.device, dtype=h.dtype)).view(1, -1)
        pos_dim = (basis[src] * basis[dst] * diag).mean(dim=0)
        neg_dim = (basis[src] * basis[neg_dst] * diag).mean(dim=0)
        margin = self._normalize_score(pos_dim - neg_dim, default_value=0.5)
        reliability = torch.sigmoid((margin - 0.5) / self.energy_rsc_reliability_temp)
        floor = self.energy_rsc_reliability_floor
        reliability = floor + (1.0 - floor) * reliability
        if self.energy_rsc_detach_reliability:
            reliability = reliability.detach()
        return reliability.clamp(0.0, 1.0)

    def apply_energy_reliability_to_edge_gate(self, h, edge_index, edge_gate, training=False):
        if not self.use_energy_rsc_gate or edge_gate is None:
            return edge_gate
        reliability = self.compute_energy_gate_reliability(h, edge_index, training=training)
        if reliability is None:
            return edge_gate
        rel = reliability.mean().view(1, 1) if edge_gate.size(-1) == 1 else reliability.view(1, -1)
        return (edge_gate * rel).clamp(0.0, 1.0)

    def apply_energy_rsc_gate(self, h, smooth, residual, smooth_energy, residual_energy, base_gate, edge_index, training=False):
        if not self.use_energy_rsc_gate:
            return base_gate
        reliability_1 = self.compute_energy_gate_reliability(
            smooth,
            edge_index,
            candidate_gate=base_gate,
            training=training,
        )
        if reliability_1 is None:
            return base_gate
        reliability_1_node = reliability_1.view(1, -1).expand_as(base_gate)
        shortcut_score = base_gate.detach() * (1.0 - reliability_1_node.detach())
        keep_mask = torch.ones_like(base_gate)
        k = min(base_gate.size(1), int(round(base_gate.size(1) * self.energy_rsc_top_frac)))
        if k > 0:
            shortcut_idx = shortcut_score.topk(k, dim=1).indices
            keep_mask = keep_mask.scatter(1, shortcut_idx, 0.0)

        challenge_smooth = smooth * keep_mask
        gate2_input = torch.cat(
            [h, challenge_smooth, residual, self._graph_cfam_energy(challenge_smooth), residual_energy],
            dim=-1,
        )
        gate2 = torch.sigmoid(self.energy_rsc_second_gate(gate2_input) / self.graph_cfam_gate_temp)
        reliability_2 = self.compute_energy_gate_reliability(
            smooth,
            edge_index,
            candidate_gate=gate2,
            training=training,
        )
        if reliability_2 is None:
            reliability_2 = reliability_1
        reliability_2_node = reliability_2.view(1, -1).expand_as(base_gate)
        final_gate = (
            reliability_1_node * base_gate
            + self.energy_rsc_second_weight * reliability_2_node * (1.0 - base_gate) * gate2
        ).clamp(0.0, 1.0)
        self._last_energy_gate_reliability_mean = reliability_1.mean().detach()
        self._last_energy_gate_rsc_mask_mean = (1.0 - keep_mask).mean().detach()
        self._last_energy_gate_second_mean = gate2.mean().detach()
        return final_gate

    def compute_energy_gate_reconstruction_loss(self, z, edge_index, training=False):
        zero = z.new_zeros(())
        if not self.use_energy_rsc_gate or self.lambda_energy_gate_rec <= 0.0:
            self._last_energy_gate_rec_loss = zero
            return zero
        sampled = self._sample_reconstruction_edges(edge_index, z.size(0), training=training)
        if sampled[0] is None:
            self._last_energy_gate_rec_loss = zero
            return zero
        src, dst, neg_dst = sampled
        diag = F.softplus(self.edge_energy_diag.to(device=z.device, dtype=z.dtype)).view(1, -1)
        scale = float(max(1, z.size(1))) ** 0.5
        pos_score = (z[src] * z[dst] * diag).sum(dim=-1) / scale
        neg_score = (z[src] * z[neg_dst] * diag).sum(dim=-1) / scale
        loss = F.softplus(-pos_score).mean() + F.softplus(neg_score).mean()
        self._last_energy_gate_rec_loss = loss
        return loss

    def graph_cfam_adapt(self, h, edge_index, training=False, local_blend=None, residual_blend=None):
        smooth, noise_summary, edge_gate = self.compute_edge_summaries(h, edge_index, training=training)
        residual = h - smooth
        smooth_energy = self._graph_cfam_energy(smooth)
        residual_energy = self._graph_cfam_energy(residual)
        gate_input = torch.cat([h, smooth, residual, smooth_energy, residual_energy], dim=-1)
        base_gate = torch.sigmoid(self.graph_cfam_gate(gate_input) / self.graph_cfam_gate_temp)
        gate = self.apply_energy_rsc_gate(
            h,
            smooth,
            residual,
            smooth_energy,
            residual_energy,
            base_gate,
            edge_index,
            training=training,
        )
        causal_local = gate * smooth
        domain_local = (1.0 - gate) * smooth + noise_summary
        if local_blend is None:
            local_blend = self.edge_blend
        if residual_blend is None:
            residual_blend = self.graph_cfam_residual_blend
        adapted = h + local_blend * causal_local + residual_blend * residual
        adapted = F.dropout(adapted, self.dropout, training=training)
        adapted = self.graph_cfam_norm(adapted)
        gate_loss = (gate.mean() - self.graph_cfam_gate_target).pow(2)
        return adapted, causal_local, domain_local, gate, edge_gate, gate_loss

    def fuse_node_edge_representation(self, h, edge_summary, training=False):
        useful_input = torch.cat([h, edge_summary, h * edge_summary], dim=-1)
        useful_delta = self.node_edge_fuser(useful_input)
        useful_delta = F.dropout(useful_delta, self.dropout, training=training)
        fused = h + self.edge_blend * useful_delta
        if self.use_node_edge_norm:
            return self.node_edge_norm(fused)
        return fused

    def encode_representation(self, x, edge_index, training=False):
        self._clear_diagnostics()
        x = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(self.input_proj(x))
        cfam_gate_loss = h.new_zeros(())
        cfam_gate_mean = h.new_zeros(())
        cfam_layers = 0
        shortcut_summary = h.new_zeros(h.size())

        if self.use_graph_cfam and self.use_pre_gnn_graph_cfam:
            h, _, shortcut_summary, cfam_gate, _, loss = self.graph_cfam_adapt(
                h,
                edge_index,
                training=training,
                local_blend=self.pre_graph_cfam_blend,
                residual_blend=self.pre_graph_cfam_residual_blend,
            )
            cfam_gate_loss = cfam_gate_loss + loss
            cfam_gate_mean = cfam_gate_mean + cfam_gate.mean()
            cfam_layers += 1

        num_backbone_layers = len(self.backbone_layers)
        for layer_idx, layer in enumerate(self.backbone_layers):
            h = F.dropout(h, self.dropout, training=training)
            h = self.act_fn(layer(h, edge_index))
            if self.use_graph_cfam:
                should_route = not (
                    self.layerwise_local_igm_skip_last
                    and layer_idx == num_backbone_layers - 1
                )
                if should_route:
                    h, _, shortcut_summary, cfam_gate, _, loss = self.graph_cfam_adapt(
                        h,
                        edge_index,
                        training=training,
                    )
                    cfam_gate_loss = cfam_gate_loss + loss
                    cfam_gate_mean = cfam_gate_mean + cfam_gate.mean()
                    cfam_layers += 1

        if self.use_graph_cfam and self.use_final_graph_cfam:
            z, _, shortcut_summary, cfam_gate, _, loss = self.graph_cfam_adapt(
                h,
                edge_index,
                training=training,
            )
            cfam_gate_loss = cfam_gate_loss + loss
            cfam_gate_mean = cfam_gate_mean + cfam_gate.mean()
            cfam_layers += 1
        else:
            edge_summary, noise_summary, _ = self.compute_edge_summaries(
                h,
                edge_index,
                training=training,
            )
            z = self.fuse_node_edge_representation(h, edge_summary, training=training)
            shortcut_summary = noise_summary

        if cfam_layers > 0:
            self._last_graph_cfam_gate_loss = cfam_gate_loss / float(cfam_layers)
            self._last_graph_cfam_gate_mean = (cfam_gate_mean / float(cfam_layers)).detach()
        else:
            self._last_graph_cfam_gate_loss = h.new_zeros(())
            self._last_graph_cfam_gate_mean = h.new_zeros(())
        self._last_graph_cfam_layers = cfam_layers

        z_mediator = F.dropout(z, self.dropout, training=training)
        if self.direct_z_spurious_mode == 'zero':
            z_spurious = z.new_zeros(z.size())
        elif self.direct_z_spurious_mode == 'z_adapter':
            z_spurious = self.spurious_norm(z + 0.1 * self.spurious_adapter(z))
        else:
            z_spurious = self.spurious_norm(shortcut_summary + 0.1 * self.spurious_adapter(shortcut_summary))
        z_spurious = F.dropout(z_spurious, self.dropout, training=training)
        mediator_logits = self.classifier(z_mediator)
        return z, z_mediator, z_spurious, mediator_logits

    def build_multi_ratio_spurious_contexts(self, z_spurious, training=False):
        if z_spurious is None or z_spurious.numel() == 0 or not self.multi_ratio_spurious_ratios:
            return None
        num_nodes, hidden_dim = z_spurious.shape
        source = z_spurious
        if self.multi_ratio_spurious_source == 'shuffle' and num_nodes > 1:
            if training:
                perm = torch.randperm(num_nodes, device=z_spurious.device)
            else:
                generator = torch.Generator(device='cpu')
                generator.manual_seed(self.context_sample_seed + num_nodes + len(self.multi_ratio_spurious_ratios))
                perm = torch.randperm(num_nodes, generator=generator).to(z_spurious.device)
            if torch.equal(perm, torch.arange(num_nodes, device=z_spurious.device)):
                perm = torch.roll(perm, shifts=1)
            source = z_spurious.index_select(0, perm)

        score = source.detach().abs()
        contexts = []
        for ratio in self.multi_ratio_spurious_ratios:
            if ratio <= 0.0:
                masked = source.new_zeros(source.size())
            elif ratio >= 1.0:
                masked = source
            else:
                keep_dim = min(hidden_dim, max(1, int(round(float(ratio) * hidden_dim))))
                top_idx = score.topk(keep_dim, dim=1).indices
                mask = source.new_zeros(num_nodes, hidden_dim)
                mask.scatter_(1, top_idx, 1.0)
                masked = source * mask
            contexts.append(F.normalize(masked, dim=1))
        return torch.stack(contexts, dim=1)

    def frontdoor_logits_from_contexts(self, z_mediator, contexts):
        base_logits = self.fd_classifier(z_mediator)
        if contexts is None or contexts.numel() == 0:
            return base_logits, None
        if contexts.dim() != 3 or contexts.size(0) != z_mediator.size(0):
            return base_logits, None
        num_contexts = contexts.size(1)
        mediator_expand = z_mediator.unsqueeze(1).expand(-1, num_contexts, -1)
        fused_input = torch.cat([mediator_expand, contexts], dim=-1)
        fused = self.fd_fuser(fused_input.reshape(-1, self.d * 2)).view(
            z_mediator.size(0),
            num_contexts,
            self.d,
        )
        fused = self.fd_norm(fused + mediator_expand)
        logits_stack = self.fd_classifier(fused.reshape(-1, self.d)).view(
            z_mediator.size(0),
            num_contexts,
            self.c,
        )
        return logits_stack.mean(dim=1), logits_stack

    def blend_logits(self, mediator_logits, fd_logits):
        return (1.0 - self.fd_blend) * mediator_logits + self.fd_blend * fd_logits

    def forward(self, x, edge_index, training=False):
        _, z_mediator, z_spurious, mediator_logits = self.encode_representation(
            x,
            edge_index,
            training=training,
        )
        contexts = self.build_multi_ratio_spurious_contexts(z_spurious, training=training)
        fd_logits, fd_stack = self.frontdoor_logits_from_contexts(z_mediator, contexts)
        logits = self.blend_logits(mediator_logits, fd_logits)
        if training:
            return logits, z_mediator, z_spurious, mediator_logits, fd_logits, fd_stack, contexts
        if self.eval_pred_mode == 'frontdoor':
            return fd_logits
        if self.eval_pred_mode == 'blend':
            return logits
        return mediator_logits

    def compute_supervised_loss(self, logits, y, criterion, args):
        if args.dataset in ('twitch', 'elliptic'):
            if y.dim() > 1 and y.size(1) == 1 and logits.size(1) > 1:
                target = F.one_hot(y.squeeze(1).long(), logits.size(1)).float()
            else:
                target = y.float()
            loss = criterion(logits, target)
            if loss.dim() > 1:
                loss = loss.mean(dim=1)
            return loss
        out = F.log_softmax(logits, dim=1)
        return criterion(out, y.squeeze(1).long())

    def compute_multi_ratio_losses(self, logits_stack, y, criterion, args):
        zero = logits_stack.new_zeros(()) if logits_stack is not None else self.edge_energy_diag.new_zeros(())
        if logits_stack is None or logits_stack.numel() == 0:
            return zero, zero, zero, 0
        num_nodes, num_ratios, num_classes = logits_stack.shape
        flat_logits = logits_stack.reshape(num_nodes * num_ratios, num_classes)
        flat_labels = y.unsqueeze(1).expand(-1, num_ratios, *y.shape[1:]).reshape(
            num_nodes * num_ratios,
            *y.shape[1:],
        )
        raw_loss = self.compute_supervised_loss(flat_logits, flat_labels, criterion, args)
        raw_loss = raw_loss.reshape(num_nodes, num_ratios)
        loss_mean = raw_loss.mean()
        loss_worst = raw_loss.max(dim=1).values.mean()
        mean_logits = logits_stack.mean(dim=1).detach()
        if num_classes == 1:
            target_prob = torch.sigmoid(mean_logits).unsqueeze(1).expand_as(logits_stack)
            loss_cons = F.binary_cross_entropy(torch.sigmoid(logits_stack), target_prob)
        else:
            target_prob = F.softmax(mean_logits, dim=-1).unsqueeze(1).expand_as(logits_stack)
            loss_cons = F.kl_div(
                F.log_softmax(flat_logits, dim=-1),
                target_prob.reshape(num_nodes * num_ratios, num_classes),
                reduction='batchmean',
            )
        return loss_mean, loss_worst, loss_cons, num_ratios

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y.to(data.x.device)
        train_idx = data.train_idx.to(device=x.device, dtype=torch.long)

        (
            logits_all,
            z_mediator_all,
            z_spurious_all,
            mediator_logits_all,
            fd_logits_all,
            fd_stack_all,
            contexts_all,
        ) = self.forward(x, edge_index, training=True)

        y_tr = y[train_idx]
        logits_tr = logits_all[train_idx]
        mediator_logits_tr = mediator_logits_all[train_idx]
        fd_logits_tr = fd_logits_all[train_idx]
        fd_stack_tr = fd_stack_all[train_idx] if fd_stack_all is not None else None
        z_mediator_tr = z_mediator_all[train_idx]

        loss_cls = self.compute_supervised_loss(logits_tr, y_tr, criterion, args).mean()
        loss_med = self.compute_supervised_loss(mediator_logits_tr, y_tr, criterion, args).mean()
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        loss_multi_ratio_fd, loss_multi_ratio_fd_worst, loss_multi_ratio_fd_cons, num_multi_ratio_contexts = (
            self.compute_multi_ratio_losses(fd_stack_tr, y_tr, criterion, args)
        )
        loss_graph_cfam_gate = self._last_graph_cfam_gate_loss
        graph_cfam_gate_mean = self._last_graph_cfam_gate_mean
        if loss_graph_cfam_gate is None:
            loss_graph_cfam_gate = loss_cls.new_zeros(())
            graph_cfam_gate_mean = loss_cls.new_zeros(())
        loss_energy_gate_rec = self.compute_energy_gate_reconstruction_loss(
            z_mediator_all,
            edge_index,
            training=True,
        )
        total_loss = (
            loss_cls
            + self.lambda_fd * loss_fd
            + self.lambda_graph_cfam_gate * loss_graph_cfam_gate
            + self.lambda_energy_gate_rec * loss_energy_gate_rec
            + self.lambda_multi_ratio_fd * loss_multi_ratio_fd
            + self.lambda_multi_ratio_fd_worst * loss_multi_ratio_fd_worst
            + self.lambda_multi_ratio_fd_cons * loss_multi_ratio_fd_cons
        )
        zero = loss_cls.new_zeros(())
        num_contexts = 0 if contexts_all is None else int(contexts_all.size(1))
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_med': loss_med,
            'loss_fd': loss_fd,
            'loss_graph_cfam_gate': loss_graph_cfam_gate,
            'loss_energy_gate_rec': loss_energy_gate_rec,
            'loss_multi_ratio_fd': loss_multi_ratio_fd,
            'loss_multi_ratio_fd_worst': loss_multi_ratio_fd_worst,
            'loss_multi_ratio_fd_cons': loss_multi_ratio_fd_cons,
            'graph_cfam_gate_mean': graph_cfam_gate_mean.detach(),
            'graph_cfam_layers': torch.tensor(float(self._last_graph_cfam_layers), device=x.device),
            'energy_gate_reliability_mean': (
                zero if self._last_energy_gate_reliability_mean is None else self._last_energy_gate_reliability_mean.detach()
            ),
            'energy_gate_rsc_mask_mean': (
                zero if self._last_energy_gate_rsc_mask_mean is None else self._last_energy_gate_rsc_mask_mean.detach()
            ),
            'energy_gate_second_mean': (
                zero if self._last_energy_gate_second_mean is None else self._last_energy_gate_second_mean.detach()
            ),
            'mediator_norm': z_mediator_tr.norm(dim=1).mean().detach(),
            'spurious_norm': z_spurious_all[train_idx].norm(dim=1).mean().detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_multi_ratio_contexts': torch.tensor(float(num_multi_ratio_contexts), device=x.device),
            'state_payload': None,
        }

    @torch.no_grad()
    def apply_state_update(self, state_payload):
        return

    def loss_compute(self, data, criterion, args):
        losses = self.compute_losses(data, criterion, args, update_state=True)
        return (
            losses['total_loss'],
            losses['loss_cls'].item(),
            0.0,
            0.0,
            (self.lambda_fd * losses['loss_fd']).item(),
        )
