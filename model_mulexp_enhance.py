import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_sparse import SparseTensor, matmul


def gcn_conv(x, edge_index):
    N = x.shape[0]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1.0 / d[col]).sqrt()
    d_norm_out = (1.0 / d[row]).sqrt()
    value = torch.ones_like(row, dtype=x.dtype) * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    return matmul(adj, x)


class GraphConvolutionBase(nn.Module):
    def __init__(self, in_features, out_features, residual=False):
        super(GraphConvolutionBase, self).__init__()
        self.residual = residual
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if self.residual:
            self.weight_r = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.residual:
            self.weight_r.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, x0=None):
        hi = gcn_conv(x, adj)
        output = torch.mm(hi, self.weight)
        if self.residual:
            output = output + torch.mm(x, self.weight_r)
        return output


class CaNetConv(nn.Module):
    def __init__(self, in_features, out_features, K, residual=True, backbone_type='gcn', variant=False, device=None):
        super(CaNetConv, self).__init__()
        self.backbone_type = backbone_type
        self.out_features = out_features
        self.residual = residual
        if backbone_type == 'gcn':
            self.weights = Parameter(torch.FloatTensor(K, in_features * 2, out_features))
        elif backbone_type == 'gat':
            self.leakyrelu = nn.LeakyReLU()
            self.weights = nn.Parameter(torch.zeros(K, in_features, out_features))
            self.a = nn.Parameter(torch.zeros(K, 2 * out_features, 1))
        else:
            raise NotImplementedError(f"Unsupported backbone_type='{backbone_type}'")
        self.K = K
        self.device = device
        self.variant = variant
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weights.data.uniform_(-stdv, stdv)
        if self.backbone_type == 'gat':
            nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def specialspmm(self, adj, spm, size, h):
        adj = SparseTensor(row=adj[0], col=adj[1], value=spm, sparse_sizes=size)
        return matmul(adj, h)

    def forward(self, x, adj, e, weights=None):
        if weights is None:
            weights = self.weights
        if self.backbone_type == 'gcn':
            if not self.variant:
                hi = gcn_conv(x, adj)
            else:
                adj_sp = torch.sparse_coo_tensor(
                    adj,
                    torch.ones(adj.shape[1], device=x.device, dtype=x.dtype),
                    size=(x.shape[0], x.shape[0]),
                ).to(x.device)
                hi = torch.sparse.mm(adj_sp, x)
            hi = torch.cat([hi, x], 1)
            hi = hi.unsqueeze(0).repeat(self.K, 1, 1)
            outputs = torch.matmul(hi, weights)
            outputs = outputs.transpose(1, 0)
        else:
            xi = x.unsqueeze(0).repeat(self.K, 1, 1)
            h = torch.matmul(xi, weights)
            N = x.size(0)
            adj_loops, _ = remove_self_loops(adj)
            adj_loops, _ = add_self_loops(adj_loops, num_nodes=N)
            edge_h = torch.cat((h[:, adj_loops[0, :], :], h[:, adj_loops[1, :], :]), dim=2)
            logits = self.leakyrelu(torch.matmul(edge_h, self.a)).squeeze(2)
            logits_max, _ = torch.max(logits, dim=1, keepdim=True)
            edge_e = torch.exp(logits - logits_max)

            outputs = []
            eps = 1e-8
            for k in range(self.K):
                edge_e_k = edge_e[k, :]
                e_expsum_k = self.specialspmm(
                    adj_loops,
                    edge_e_k,
                    torch.Size([N, N]),
                    torch.ones(N, 1, device=x.device, dtype=x.dtype),
                ) + eps
                hi_k = self.specialspmm(adj_loops, edge_e_k, torch.Size([N, N]), h[k])
                hi_k = torch.div(hi_k, e_expsum_k)
                outputs.append(hi_k)
            outputs = torch.stack(outputs, dim=1)

        es = e.unsqueeze(2).repeat(1, 1, self.out_features)
        output = torch.sum(torch.mul(es, outputs), dim=1)
        if self.residual:
            output = output + x
        return output


class NodeEnhancer(nn.Module):
    """
    Node enhancement module transplanted from graph_cfam_nego.

    It scores each edge from endpoint semantics, builds useful/noise neighbor
    summaries, then either fuses useful summaries into the node state or applies
    the optional Graph-CFAM local adaptation gate.
    """

    def __init__(self, hidden_dim, args):
        super().__init__()
        self.d = hidden_dim
        self.dropout = float(getattr(args, 'dropout', 0.0))
        self.edge_score_temp = max(1e-3, float(getattr(args, 'edge_score_temp', 2.0)))
        self.edge_blend = max(0.0, float(getattr(args, 'edge_blend', 0.2)))
        self.edge_feat_mode = getattr(args, 'edge_feat_mode', 'mul')
        self.edge_gate_mode = getattr(args, 'edge_gate_mode', 'vector')
        if self.edge_gate_mode not in ('scalar', 'vector'):
            self.edge_gate_mode = 'vector'

        edge_feat_dim = self._get_edge_feat_dim(self.edge_feat_mode)
        self.edge_pair_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        edge_gate_out_dim = 1 if self.edge_gate_mode == 'scalar' else self.d
        self.edge_score_head = nn.Linear(self.d, edge_gate_out_dim)
        self.edge_summary_norm = nn.LayerNorm(self.d)
        self.noise_summary_norm = nn.LayerNorm(self.d)
        self.node_edge_fuser = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.node_edge_norm = nn.LayerNorm(self.d)

        self.use_graph_cfam = bool(getattr(args, 'use_graph_cfam', False))
        self.graph_cfam_residual_blend = max(0.0, float(getattr(args, 'graph_cfam_residual_blend', 0.1)))
        self.graph_cfam_gate_temp = max(1e-3, float(getattr(args, 'graph_cfam_gate_temp', 1.0)))
        self.graph_cfam_gate_target = min(max(float(getattr(args, 'graph_cfam_gate_target', 0.5)), 0.0), 1.0)
        self.graph_cfam_gate = nn.Sequential(
            nn.Linear(self.d * 5, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.graph_cfam_norm = nn.LayerNorm(self.d)
        self.reset_parameters()

    def reset_parameters(self):
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

    def _reset_module_parameters(self, module):
        for submodule in module.modules():
            if submodule is module:
                continue
            if hasattr(submodule, 'reset_parameters'):
                submodule.reset_parameters()

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
        raise ValueError(
            f"Unknown edge_feat_mode='{mode}'. Use one of: mul, diff, signed_diff, "
            "degree, mul_diff, mul_signed_diff, concat, concat_diff, mul_degree, "
            "diff_degree, mul_diff_degree, mul_signed_diff_degree."
        )

    def build_edge_feat(self, h_src, h_dst, deg_src, deg_dst, deg_max):
        mode = self.edge_feat_mode
        mul_feat = h_src * h_dst
        signed_diff_feat = h_src - h_dst
        diff_feat = torch.abs(h_src - h_dst)
        concat_feat = torch.cat([h_src, h_dst], dim=-1)

        log_deg_src = torch.log1p(deg_src)
        log_deg_dst = torch.log1p(deg_dst)
        deg_pair = torch.maximum(log_deg_src, log_deg_dst) / deg_max.clamp_min(1.0)
        deg_pair = deg_pair.unsqueeze(-1)

        if mode == 'mul':
            return mul_feat
        if mode == 'diff':
            return diff_feat
        if mode == 'signed_diff':
            return signed_diff_feat
        if mode == 'degree':
            return deg_pair
        if mode == 'mul_diff':
            return torch.cat([mul_feat, diff_feat], dim=-1)
        if mode == 'mul_signed_diff':
            return torch.cat([mul_feat, signed_diff_feat], dim=-1)
        if mode == 'concat':
            return concat_feat
        if mode == 'concat_diff':
            return torch.cat([concat_feat, diff_feat], dim=-1)
        if mode == 'mul_degree':
            return torch.cat([mul_feat, deg_pair], dim=-1)
        if mode == 'diff_degree':
            return torch.cat([diff_feat, deg_pair], dim=-1)
        if mode == 'mul_diff_degree':
            return torch.cat([mul_feat, diff_feat, deg_pair], dim=-1)
        if mode == 'mul_signed_diff_degree':
            return torch.cat([mul_feat, signed_diff_feat, deg_pair], dim=-1)
        raise ValueError(f"Unknown edge_feat_mode='{mode}'")

    def compute_edge_summaries(self, h, edge_index, training=False):
        if edge_index.numel() == 0:
            zero = h.new_zeros(h.size())
            return zero, zero, None

        src, dst = edge_index
        deg = degree(dst, h.size(0)).to(device=h.device, dtype=h.dtype).clamp_min(1.0)
        deg_max = torch.log1p(deg).max().clamp_min(1.0)
        h_src = h[src]
        h_dst = h[dst]
        edge_feat = self.build_edge_feat(h_src, h_dst, deg[src], deg[dst], deg_max)
        edge_hidden = self.edge_pair_encoder(edge_feat)
        edge_hidden = F.dropout(edge_hidden, self.dropout, training=training)
        edge_logits = self.edge_score_head(edge_hidden) / self.edge_score_temp
        edge_gate = torch.sigmoid(edge_logits)
        if edge_gate.dim() == 1:
            edge_gate = edge_gate.unsqueeze(-1)

        norm = (deg[src].pow(-0.5) * deg[dst].pow(-0.5)).unsqueeze(-1)
        useful_weight = torch.nan_to_num(norm * edge_gate, nan=0.0, posinf=0.0, neginf=0.0)
        noise_weight = torch.nan_to_num(norm * (1.0 - edge_gate), nan=0.0, posinf=0.0, neginf=0.0)

        useful_summary = h.new_zeros(h.size())
        useful_summary.index_add_(0, dst, useful_weight * h_src)
        useful_summary = self.edge_summary_norm(useful_summary)

        noise_summary = h.new_zeros(h.size())
        noise_summary.index_add_(0, dst, noise_weight * h_src)
        noise_summary = self.noise_summary_norm(noise_summary)
        return useful_summary, noise_summary, edge_gate

    def fuse_node_edge_representation(self, h, edge_summary, training=False):
        useful_input = torch.cat([h, edge_summary, h * edge_summary], dim=-1)
        useful_delta = self.node_edge_fuser(useful_input)
        useful_delta = F.dropout(useful_delta, self.dropout, training=training)
        fused = h + self.edge_blend * useful_delta
        return self.node_edge_norm(fused)

    def _graph_cfam_energy(self, value):
        energy = value.pow(2)
        denom = energy.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return energy / denom

    def graph_cfam_adapt(self, h, edge_index, training=False):
        smooth, noise_summary, edge_gate = self.compute_edge_summaries(h, edge_index, training=training)
        residual = h - smooth
        smooth_energy = self._graph_cfam_energy(smooth)
        residual_energy = self._graph_cfam_energy(residual)
        gate_input = torch.cat([h, smooth, residual, smooth_energy, residual_energy], dim=-1)
        gate = torch.sigmoid(self.graph_cfam_gate(gate_input) / self.graph_cfam_gate_temp)

        causal_local = gate * smooth
        adapted = h + self.edge_blend * causal_local + self.graph_cfam_residual_blend * residual
        adapted = F.dropout(adapted, self.dropout, training=training)
        adapted = self.graph_cfam_norm(adapted)
        gate_loss = (gate.mean() - self.graph_cfam_gate_target).pow(2)
        return adapted, gate, edge_gate, gate_loss

    def forward(self, h, edge_index, training=False):
        if self.use_graph_cfam:
            return self.graph_cfam_adapt(h, edge_index, training=training)

        edge_summary, _, edge_gate = self.compute_edge_summaries(h, edge_index, training=training)
        enhanced = self.fuse_node_edge_representation(h, edge_summary, training=training)
        zero = h.new_zeros(())
        gate = torch.full_like(h, 0.5)
        return enhanced, gate, edge_gate, zero


class ExpertNet(nn.Module):
    """
    Single expert with the graph node-enhancement module inserted before the
    classifier.  The external interface stays compatible with model.py.
    """

    def __init__(self, d, c, args, device):
        super(ExpertNet, self).__init__()
        hidden_channels = args.hidden_channels
        backbone_type = getattr(args, 'backbone_type', getattr(args, 'backbone', 'gcn'))
        variant = getattr(args, 'variant', False)

        self.convs = nn.ModuleList()
        for _ in range(args.num_layers):
            self.convs.append(
                CaNetConv(
                    hidden_channels,
                    hidden_channels,
                    K=1,
                    backbone_type=backbone_type,
                    residual=True,
                    device=device,
                    variant=variant,
                )
            )
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d, hidden_channels))
        self.fcs.append(nn.Linear(hidden_channels, c))
        # The enhancer is shared across layers, matching the source model's
        # layer-wise reuse of the same edge semantic scorer and Graph-CFAM gate.
        self.enhancer = NodeEnhancer(hidden_channels, args)
        self.act_fn = nn.ReLU()
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.device = device
        self.lambda_enhance_sem = max(0.0, float(getattr(args, 'lambda_enhance_sem', 0.0)))
        self.lambda_graph_cfam_gate = max(0.0, float(getattr(args, 'lambda_graph_cfam_gate', 0.0)))
        self.enhance_sem_mode = getattr(args, 'enhance_sem_mode', 'cosine')
        if self.enhance_sem_mode not in ('cosine', 'mse'):
            self.enhance_sem_mode = 'cosine'
        self._last_enhance_sem = None
        self._last_graph_cfam_gate = None
        self._last_gate_mean = None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        self.enhancer.reset_parameters()
        self._last_enhance_sem = None
        self._last_graph_cfam_gate = None
        self._last_gate_mean = None

    def compute_enhance_semantic_loss(self, enhanced, anchor):
        if self.lambda_enhance_sem <= 0.0:
            return enhanced.new_zeros(())
        anchor = anchor.detach()
        if self.enhance_sem_mode == 'mse':
            return F.mse_loss(F.normalize(enhanced, dim=1), F.normalize(anchor, dim=1))
        return (1.0 - (F.normalize(enhanced, dim=1) * F.normalize(anchor, dim=1)).sum(dim=1)).mean()

    def forward(self, x, adj, training=None):
        if training is None:
            training = self.training
        x = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(self.fcs[0](x))
        e_dummy = torch.ones(x.shape[0], 1, device=self.device, dtype=h.dtype)

        enhance_sem_total = None
        graph_cfam_gate_total = None
        gate_mean_total = None
        enhance_layers = 0

        for con in self.convs:
            h = F.dropout(h, self.dropout, training=training)
            h = self.act_fn(con(h, adj, e_dummy))

            h_anchor = h
            h, cfam_gate, _, gate_loss = self.enhancer(h, adj, training=training)
            sem_loss = self.compute_enhance_semantic_loss(h, h_anchor)
            if enhance_sem_total is None:
                enhance_sem_total = sem_loss
                graph_cfam_gate_total = gate_loss
                gate_mean_total = cfam_gate.mean()
            else:
                enhance_sem_total = enhance_sem_total + sem_loss
                graph_cfam_gate_total = graph_cfam_gate_total + gate_loss
                gate_mean_total = gate_mean_total + cfam_gate.mean()
            enhance_layers += 1

        if enhance_layers > 0:
            self._last_enhance_sem = enhance_sem_total / float(enhance_layers)
            self._last_graph_cfam_gate = graph_cfam_gate_total / float(enhance_layers)
            self._last_gate_mean = (gate_mean_total / float(enhance_layers)).detach()
        else:
            zero = h.new_zeros(())
            self._last_enhance_sem = zero
            self._last_graph_cfam_gate = zero
            self._last_gate_mean = zero.detach()

        h = F.dropout(h, self.dropout, training=training)
        out = self.fcs[-1](h)
        return out

    def auxiliary_loss(self):
        device = self.fcs[-1].weight.device
        zero = self.fcs[-1].weight.new_zeros(())
        loss_enhance = zero if self._last_enhance_sem is None else self._last_enhance_sem
        loss_gate = zero if self._last_graph_cfam_gate is None else self._last_graph_cfam_gate
        return self.lambda_enhance_sem * loss_enhance + self.lambda_graph_cfam_gate * loss_gate


class CaNet(nn.Module):
    """
    Multi-expert CaNet with the graph node-enhancement module from
    model_gmm3_reviewed1_graph_cfam_nego.py fused into every expert.
    """

    def __init__(self, d, c, args, device):
        super(CaNet, self).__init__()
        self.num_experts = args.K
        self.expert_agg = getattr(args, 'expert_agg', 'mean')
        self.experts = nn.ModuleList([ExpertNet(d, c, args, device) for _ in range(self.num_experts)])
        gate_hidden = getattr(args, 'gate_hidden', args.hidden_channels)
        self.gate = nn.Sequential(
            nn.Linear(d, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, self.num_experts),
        )
        self.device = device
        self._last_loss_breakdown = {}

    def reset_parameters(self):
        for expert in self.experts:
            expert.reset_parameters()
        for module in self.gate:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self._last_loss_breakdown = {}

    def compute_expert_weights(self, x):
        gate_logits = self.gate(x)
        return torch.softmax(gate_logits, dim=-1)

    def forward(self, x, adj, idx=None, training=False):
        logits_list = [expert(x, adj, training=training) for expert in self.experts]
        logits_stack = torch.stack(logits_list, dim=0)

        if self.expert_agg == 'gate':
            expert_weights = self.compute_expert_weights(x)
            return torch.einsum('nk,knc->nc', expert_weights, logits_stack)
        return torch.mean(logits_stack, dim=0)

    def sup_loss_calc(self, y, pred, criterion, args):
        if args.dataset in ('twitch', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            target = y.squeeze(1)
            loss = criterion(pred, target)
        return loss

    def loss_compute(self, d, criterion, args):
        total_loss = 0.0
        unique_envs = torch.unique(d.env[d.train_idx])
        num_unique_envs = len(unique_envs)
        logits_list = [expert(d.x, d.edge_index, training=True) for expert in self.experts]
        alpha_other = args.lamda
        self_loss_total = d.x.new_zeros(())
        other_loss_total = d.x.new_zeros(())

        for i, expert_logits in enumerate(logits_list):
            env_id = unique_envs[i % num_unique_envs]
            env_self_mask = (d.env == env_id)[d.train_idx]
            train_idx_self = d.train_idx[env_self_mask]

            loss_self = d.x.new_zeros(())
            if len(train_idx_self) > 0:
                loss_self = self.sup_loss_calc(d.y[train_idx_self], expert_logits[train_idx_self], criterion, args)

            env_other_mask = (d.env != env_id)[d.train_idx]
            train_idx_other = d.train_idx[env_other_mask]

            loss_other = d.x.new_zeros(())
            if len(train_idx_other) > 0:
                if getattr(args, 'other_env_reduce', 'sample') == 'env':
                    other_env_losses = []
                    for other_env_id in unique_envs:
                        if other_env_id == env_id:
                            continue
                        env_k_mask = (d.env == other_env_id)[d.train_idx]
                        train_idx_k = d.train_idx[env_k_mask]
                        if len(train_idx_k) > 0:
                            loss_k = self.sup_loss_calc(d.y[train_idx_k], expert_logits[train_idx_k], criterion, args)
                            other_env_losses.append(loss_k)
                    if len(other_env_losses) > 0:
                        loss_other = torch.stack(other_env_losses).mean()
                else:
                    loss_other = self.sup_loss_calc(d.y[train_idx_other], expert_logits[train_idx_other], criterion, args)

            self_loss_total = self_loss_total + loss_self
            other_loss_total = other_loss_total + loss_other
            total_loss = total_loss + (loss_self + alpha_other * loss_other)

        base_loss = total_loss / self.num_experts
        aux_losses = torch.stack([expert.auxiliary_loss() for expert in self.experts]).mean()
        loss = base_loss + aux_losses
        self._last_loss_breakdown = {
            'self_env': (self_loss_total / self.num_experts).detach(),
            'other_env': (other_loss_total / self.num_experts).detach(),
            'enhance_aux': aux_losses.detach(),
        }
        return loss
