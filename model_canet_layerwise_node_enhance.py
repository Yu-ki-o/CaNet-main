import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops, add_self_loops, degree, add_remaining_self_loops
from data_utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
from torch_sparse import SparseTensor, matmul

def gcn_conv(x, edge_index):
    """
    对应 CaNet 中 mixture-of-expert GNN predictor 的基础图传播算子。
    这里先做一阶邻居聚合，再交给环境相关的专家权重进行加权组合。
    """
    N = x.shape[0]
    row, col = edge_index #这里是无向图，每条边出现两次
    d = degree(col, N).float() #计算节点的度数
    #计算每条边的权重，边连接两端节点度数大的边的权重反而小
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    value = torch.ones_like(row) * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    #节点吸收邻居节点信息生成新节点表示，等价于消息传递
    return matmul(adj, x) # [N, D]

class GraphConvolutionBase(nn.Module):
    """
    对应论文中的 environment estimator q_φ(E|G) 里的图结构编码器版本。
    当 `env_type == graph` 时，用图卷积把当前节点表示映射成环境 logits。
    """

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
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_r.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, x0):
        hi = gcn_conv(x, adj)
        output = torch.mm(hi, self.weight)
        if self.residual:
            output = output + torch.mm(x, self.weight_r)
        return output

class CaNetConv(nn.Module):
    """
    对应论文中的 mixture-of-expert GNN predictor p_θ(Y | G, E)。

    其核心思想是：
    1. 为 K 个潜在环境分别维护一组专家参数；
    2. 先对每个环境独立计算一份邻居传播结果；
    3. 再用环境估计器输出的环境分布 e 对 K 个专家输出做加权求和。
    """

    def __init__(self, in_features, out_features, K, residual=True, backbone_type='gcn', variant=False, device=None):
        super(CaNetConv, self).__init__()
        self.backbone_type = backbone_type
        self.out_features = out_features  
        self.residual = residual
        if backbone_type == 'gcn':
            self.weights = Parameter(torch.FloatTensor(K, in_features*2, out_features))
        elif backbone_type == 'gat':
            self.leakyrelu = nn.LeakyReLU()
            self.weights = nn.Parameter(torch.zeros(K, in_features, out_features))
            self.a = nn.Parameter(torch.zeros(K, 2 * out_features, 1))
        self.K = K
        self.device = device
        self.variant = variant
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weights.data.uniform_(-stdv, stdv)
        if self.backbone_type == 'gat':
            nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def specialspmm(self, adj, spm, size, h):
        adj = SparseTensor(row=adj[0], col=adj[1], value=spm, sparse_sizes=size)
        return matmul(adj, h)

    def forward(self, x, adj, e, weights=None):
        if weights == None:
            weights = self.weights
        if self.backbone_type == 'gcn':
            # GCN 骨干：先做标准图卷积邻居聚合，再和节点自身表示拼接，
            # 对应论文里“feature propagation units conditioned on environments”。
            if not self.variant:
                hi = gcn_conv(x, adj)
            else:#和gcn_conv差个归一化，高度数节点的影响会更大
                adj = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]).to(self.device), size=(x.shape[0],x.shape[0])).to(self.device)
                hi = torch.sparse.mm(adj, x)
            hi = torch.cat([hi, x], 1)
            hi = hi.unsqueeze(0).repeat(self.K, 1, 1)  # [K, N, D*2]
            outputs = torch.matmul(hi, weights) # [K, N, D]
            outputs = outputs.transpose(1, 0)  # [N, K, D]
        elif self.backbone_type == 'gat':
            # GAT 骨干：每个潜在环境对应一套独立注意力参数，
            # 先分别计算环境特定的 attention aggregation，再由 e 做环境加权。
            xi = x.unsqueeze(0).repeat(self.K, 1, 1)  # [K, N, D]
            h = torch.matmul(xi, weights) # [K, N, D]
            N = x.size()[0]
            adj, _ = remove_self_loops(adj)
            adj, _ = add_self_loops(adj, num_nodes=N)
            edge_h = torch.cat((h[:, adj[0, :], :], h[:, adj[1, :], :]), dim=2)  # [K, E, 2*D]
            logits = self.leakyrelu(torch.matmul(edge_h, self.a)).squeeze(2)
            logits_max , _ = torch.max(logits, dim=1, keepdim=True)
            edge_e = torch.exp(logits-logits_max)  # [K, E]

            outputs = []
            eps = 1e-8
            for k in range(self.K):
                edge_e_k = edge_e[k, :] # [E]
                e_expsum_k = self.specialspmm(
                    adj,
                    edge_e_k,
                    torch.Size([N, N]),
                    torch.ones(N, 1, device=x.device, dtype=x.dtype),
                ) + eps
                assert not torch.isnan(e_expsum_k).any()

                hi_k = self.specialspmm(adj, edge_e_k, torch.Size([N, N]), h[k])
                hi_k = torch.div(hi_k, e_expsum_k)  # [N, D]
                outputs.append(hi_k)
            outputs = torch.stack(outputs, dim=1) # [N, K, D]

        es = e.unsqueeze(2).repeat(1, 1, self.out_features)  # [N, K, D]
        # 对应论文里的 backdoor-adjustment 近似实现：
        # 用环境分布 e 对 K 个环境专家的输出进行加权组合。
        output = torch.sum(torch.mul(es, outputs), dim=1)  # [N, D]

        if self.residual:
            output = output + x

        return output


class LayerwiseNodeEnhancementMixin:
    """
    Edge-gated layer-wise node enhancement migrated from the reviewed model.
    """

    def _init_layerwise_node_enhancement(self, args):
        self.edge_score_temp = max(1e-3, float(getattr(args, 'edge_score_temp', 2.0)))
        self.edge_blend = max(0.0, float(getattr(args, 'edge_blend', 0.2)))
        self.use_layerwise_local_igm = bool(getattr(args, 'use_layerwise_local_igm', False))
        self.layerwise_local_igm_skip_last = bool(getattr(args, 'layerwise_local_igm_skip_last', True))
        self.layerwise_final_edge_fuse = bool(getattr(args, 'layerwise_final_edge_fuse', True))
        self.layerwise_gate_target = min(max(float(getattr(args, 'layerwise_gate_target', 0.5)), 0.0), 1.0)
        self.lambda_layerwise_gate = max(0.0, float(getattr(args, 'lambda_layerwise_gate', 0.0)))
        self.lambda_enhance_sem = float(getattr(args, 'lambda_enhance_sem', 0.0))
        self.enhance_sem_mode = getattr(args, 'enhance_sem_mode', 'cosine')
        if self.enhance_sem_mode not in ('cosine', 'mse'):
            self.enhance_sem_mode = 'cosine'
        self.use_graph_cfam = bool(getattr(args, 'use_graph_cfam', False))
        self.use_final_graph_cfam = bool(getattr(args, 'use_final_graph_cfam', True))
        self.graph_cfam_residual_blend = max(0.0, float(getattr(args, 'graph_cfam_residual_blend', 0.1)))
        self.use_pre_gnn_graph_cfam = bool(getattr(args, 'use_pre_gnn_graph_cfam', False))
        self.pre_graph_cfam_blend = max(0.0, float(getattr(args, 'pre_graph_cfam_blend', 0.1)))
        self.pre_graph_cfam_residual_blend = max(
            0.0,
            float(getattr(args, 'pre_graph_cfam_residual_blend', 0.0)),
        )
        self.graph_cfam_gate_temp = max(1e-3, float(getattr(args, 'graph_cfam_gate_temp', 1.0)))
        self.graph_cfam_gate_target = min(max(float(getattr(args, 'graph_cfam_gate_target', 0.5)), 0.0), 1.0)
        self.lambda_graph_cfam_gate = max(0.0, float(getattr(args, 'lambda_graph_cfam_gate', 0.0)))
        self.lambda_graph_delf = max(0.0, float(getattr(args, 'lambda_graph_delf', 0.0)))
        self.graph_delf_top_frac = min(max(float(getattr(args, 'graph_delf_top_frac', 0.2)), 0.0), 1.0)
        self.graph_delf_margin = float(getattr(args, 'graph_delf_margin', 0.2))
        self.graph_delf_shortcut_weight = max(0.0, float(getattr(args, 'graph_delf_shortcut_weight', 0.5)))
        self._last_layerwise_gate_loss = None
        self._last_layerwise_gate_mean = None
        self._last_layerwise_gate_layers = 0
        self._last_graph_cfam_gate_loss = None
        self._last_graph_cfam_gate_mean = None
        self._last_graph_cfam_layers = 0
        self._last_pre_enhance_repr = None
        self._last_enhanced_repr = None
        self._last_shortcut_summary = None

        self.edge_feat_mode = getattr(args, 'edge_feat_mode', 'mul')
        self.edge_gate_mode = getattr(args, 'edge_gate_mode', 'vector')
        if self.edge_gate_mode not in ('scalar', 'vector'):
            self.edge_gate_mode = 'vector'
        edge_feat_dim = self._get_edge_feat_dim(self.edge_feat_mode)
        edge_gate_out_dim = 1 if self.edge_gate_mode == 'scalar' else self.hidden_channels
        self.edge_pair_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )
        self.edge_score_head = nn.Linear(self.hidden_channels, edge_gate_out_dim)
        self.edge_summary_norm = nn.LayerNorm(self.hidden_channels)
        self.noise_summary_norm = nn.LayerNorm(self.hidden_channels)
        self.node_edge_fuser = nn.Sequential(
            nn.Linear(self.hidden_channels * 3, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )
        self.use_node_edge_norm = not bool(getattr(args, 'disable_node_edge_norm', False))
        self.node_edge_norm = nn.LayerNorm(self.hidden_channels)
        self.graph_cfam_gate = nn.Sequential(
            nn.Linear(self.hidden_channels * 5, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )
        self.graph_cfam_norm = nn.LayerNorm(self.hidden_channels)
        self._reset_layerwise_node_enhancement_parameters()

    def _reset_module_parameters(self, module):
        for sub_module in module.modules():
            if sub_module is module:
                continue
            if hasattr(sub_module, 'reset_parameters'):
                sub_module.reset_parameters()

    def _reset_layerwise_node_enhancement_parameters(self):
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
        self._last_layerwise_gate_loss = None
        self._last_layerwise_gate_mean = None
        self._last_layerwise_gate_layers = 0
        self._last_graph_cfam_gate_loss = None
        self._last_graph_cfam_gate_mean = None
        self._last_graph_cfam_layers = 0
        self._last_pre_enhance_repr = None
        self._last_enhanced_repr = None
        self._last_shortcut_summary = None

    def _get_edge_feat_dim(self, mode):
        if mode in ('mul', 'diff', 'signed_diff'):
            return self.hidden_channels
        if mode == 'degree':
            return 1
        if mode in ('mul_diff', 'mul_signed_diff', 'concat'):
            return 2 * self.hidden_channels
        if mode == 'concat_diff':
            return 3 * self.hidden_channels
        if mode in ('mul_degree', 'diff_degree'):
            return self.hidden_channels + 1
        if mode in ('mul_signed_diff_degree', 'mul_diff_degree'):
            return 2 * self.hidden_channels + 1
        raise ValueError(
            f"Unknown edge_feat_mode='{mode}'. Use one of: "
            "mul, diff, signed_diff, degree, mul_diff, mul_signed_diff, "
            "concat, concat_diff, mul_degree, diff_degree, "
            "mul_diff_degree, mul_signed_diff_degree."
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
        raise ValueError(f"Unknown edge_feat_mode='{mode}'.")

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

        norm = (deg[src].pow(-0.5) * deg[dst].pow(-0.5)).unsqueeze(-1)
        useful_weight = torch.nan_to_num(norm * edge_gate, nan=0.0, posinf=0.0, neginf=0.0)
        noise_weight = torch.nan_to_num(norm * (1.0 - edge_gate), nan=0.0, posinf=0.0, neginf=0.0)

        useful_summary = h.new_zeros(h.size())
        useful_summary.index_add_(0, dst, useful_weight * h[src])
        useful_summary = self.edge_summary_norm(useful_summary)

        noise_summary = h.new_zeros(h.size())
        noise_summary.index_add_(0, dst, noise_weight * h[src])
        noise_summary = self.noise_summary_norm(noise_summary)
        return useful_summary, noise_summary, edge_gate

    def fuse_node_edge_representation(self, h, edge_summary, noise_summary=None, training=False):
        useful_input = torch.cat([h, edge_summary, h * edge_summary], dim=-1)
        useful_delta = self.node_edge_fuser(useful_input)
        useful_delta = F.dropout(useful_delta, self.dropout, training=training)
        fused = h + self.edge_blend * useful_delta
        if self.use_node_edge_norm:
            return self.node_edge_norm(fused)
        return fused

    def _graph_cfam_energy(self, value):
        energy = value.pow(2)
        denom = energy.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return energy / denom

    def graph_cfam_adapt(self, h, edge_index, training=False, local_blend=None, residual_blend=None):
        smooth, noise_summary, edge_gate = self.compute_edge_summaries(
            h,
            edge_index,
            training=training,
        )
        residual = h - smooth
        smooth_energy = self._graph_cfam_energy(smooth)
        residual_energy = self._graph_cfam_energy(residual)
        gate_input = torch.cat([h, smooth, residual, smooth_energy, residual_energy], dim=-1)
        gate = torch.sigmoid(self.graph_cfam_gate(gate_input) / self.graph_cfam_gate_temp)

        causal_local = gate * smooth
        domain_local = (1.0 - gate) * smooth
        if noise_summary is not None:
            domain_local = domain_local + noise_summary
        if local_blend is None:
            local_blend = self.edge_blend
        if residual_blend is None:
            residual_blend = self.graph_cfam_residual_blend
        adapted = h + local_blend * causal_local + residual_blend * residual
        adapted = F.dropout(adapted, self.dropout, training=training)
        adapted = self.graph_cfam_norm(adapted)
        gate_loss = (gate.mean() - self.graph_cfam_gate_target).pow(2)
        return adapted, causal_local, domain_local, gate, edge_gate, gate_loss

    def _accumulate_layerwise_gate(self, edge_gate, gate_loss, gate_mean, gate_layers):
        if edge_gate is None:
            return gate_loss, gate_mean, gate_layers
        gate_mean = gate_mean + edge_gate.mean()
        if self.lambda_layerwise_gate > 0.0:
            gate_loss = gate_loss + (edge_gate.mean() - self.layerwise_gate_target).pow(2)
        return gate_loss, gate_mean, gate_layers + 1

    def _finalize_layerwise_gate_stats(self, gate_loss, gate_mean, gate_layers):
        if gate_layers > 0:
            gate_loss = gate_loss / float(gate_layers)
            gate_mean = gate_mean / float(gate_layers)
        self._last_layerwise_gate_loss = gate_loss
        self._last_layerwise_gate_mean = gate_mean.detach()
        self._last_layerwise_gate_layers = int(gate_layers)

    def _finalize_graph_cfam_gate_stats(self, gate_loss, gate_mean, gate_layers):
        if gate_layers > 0:
            gate_loss = gate_loss / float(gate_layers)
            gate_mean = gate_mean / float(gate_layers)
        self._last_graph_cfam_gate_loss = gate_loss
        self._last_graph_cfam_gate_mean = gate_mean.detach()
        self._last_graph_cfam_layers = int(gate_layers)

    def compute_enhance_semantic_loss(self, z, h_anchor, train_idx=None):
        if (
            self.lambda_enhance_sem <= 0.0
            or z is None
            or h_anchor is None
            or z.numel() == 0
            or h_anchor.numel() == 0
        ):
            ref = z if z is not None else self.edge_score_head.weight
            return ref.new_zeros(())

        if train_idx is not None and train_idx.numel() > 0:
            z_view = z[train_idx]
            anchor = h_anchor[train_idx].detach()
        else:
            z_view = z
            anchor = h_anchor.detach()

        if self.enhance_sem_mode == 'mse':
            return F.mse_loss(F.normalize(z_view, dim=1), F.normalize(anchor, dim=1))
        return (1.0 - (F.normalize(z_view, dim=1) * F.normalize(anchor, dim=1)).sum(dim=1)).mean()

    def layerwise_node_enhance(self, h, edge_index, training=False):
        edge_summary, noise_summary, edge_gate = self.compute_edge_summaries(
            h,
            edge_index,
            training=training,
        )
        h = self.fuse_node_edge_representation(
            h,
            edge_summary,
            noise_summary=noise_summary,
            training=training,
        )
        return h, edge_gate

    def _flat_class_labels(self, y):
        if y.dim() > 1 and y.size(1) > 1:
            return y.argmax(dim=1).long()
        return y.squeeze().long()

    def compute_graph_delf_loss(self, z, z_shortcut, logits, y, train_idx, criterion, args):
        zero = logits.new_zeros(())
        if (
            self.lambda_graph_delf <= 0.0
            or train_idx is None
            or train_idx.numel() <= 1
            or z is None
            or z_shortcut is None
            or z.numel() == 0
            or z_shortcut.numel() == 0
        ):
            return zero

        train_idx = train_idx.to(device=logits.device, dtype=torch.long)
        y = y.to(logits.device)
        with torch.no_grad():
            raw_loss = self.per_node_supervised_loss(logits[train_idx], y[train_idx], args)
            hard_score = self._normalize_score(raw_loss.detach(), default_value=0.5)
            shortcut_score = self._normalize_score(z_shortcut[train_idx].norm(dim=1).detach(), default_value=0.5)
            ambiguous_score = hard_score + shortcut_score
            top_k = max(1, int(round(float(train_idx.numel()) * self.graph_delf_top_frac)))
            top_k = min(top_k, int(train_idx.numel()))
            ambiguous_pos = ambiguous_score.topk(top_k).indices
            ambiguous_mask = torch.zeros(train_idx.numel(), device=train_idx.device, dtype=torch.bool)
            ambiguous_mask[ambiguous_pos] = True

        labels_flat = self._flat_class_labels(y)
        labels_train = labels_flat[train_idx]
        z_train = z[train_idx]
        shortcut_train = z_shortcut[train_idx].detach()

        losses = []
        for cls in labels_train.unique().tolist():
            cls = int(cls)
            class_mask = labels_train == cls
            class_amb_mask = class_mask & ambiguous_mask
            if class_mask.sum() <= 1 or class_amb_mask.sum() == 0:
                continue
            stable_mask = class_mask & (~ambiguous_mask)
            if stable_mask.sum() == 0:
                stable_mask = class_mask
            causal_proto = z_train[stable_mask].mean(dim=0).detach()
            shortcut_proto = shortcut_train[class_mask].mean(dim=0).detach()
            z_amb = z_train[class_amb_mask]
            causal_align = 1.0 - F.cosine_similarity(
                F.normalize(z_amb, dim=1),
                F.normalize(causal_proto.unsqueeze(0), dim=1).expand_as(z_amb),
                dim=1,
            )
            shortcut_align = F.cosine_similarity(
                F.normalize(z_amb, dim=1),
                F.normalize(shortcut_proto.unsqueeze(0), dim=1).expand_as(z_amb),
                dim=1,
            )
            shortcut_push = F.relu(shortcut_align - self.graph_delf_margin)
            losses.append(causal_align.mean() + self.graph_delf_shortcut_weight * shortcut_push.mean())

        if not losses:
            return zero
        return torch.stack(losses).mean()

    def per_node_supervised_loss(self, logits, y, args):
        if args.dataset in ('twitch', 'elliptic'):
            if y.shape[1] == 1 and logits.shape[1] > 1:
                true_label = F.one_hot(y.squeeze().long(), logits.shape[1]).float()
            else:
                true_label = y.float()
            raw_loss = F.binary_cross_entropy_with_logits(logits, true_label, reduction='none')
            if raw_loss.dim() > 1:
                raw_loss = raw_loss.mean(dim=1)
            return raw_loss
        return F.cross_entropy(logits, y.squeeze(1), reduction='none')

    def _normalize_score(self, values, default_value=0.5):
        if values.numel() == 0:
            return values
        v_min = values.min()
        v_max = values.max()
        if (v_max - v_min).abs() < 1e-4:
            return (values - values.detach()) + default_value
        return (values - v_min) / (v_max - v_min + 1e-8)


class CaNet(LayerwiseNodeEnhancementMixin, nn.Module):
    """
    CaNet 主体对应论文中的两个核心部分：
    1. environment estimator：逐层推断节点所属的潜在环境分布；
    2. mixture-of-expert GNN predictor：在环境条件下做图传播并预测标签。

    整体训练目标是学习 p_θ(Y | do(G)) 的可计算近似，
    通过环境估计 + 环境条件专家传播来削弱环境混杂偏差。
    """
    def __init__(self, d, c, args, device):
        nn.Module.__init__(self)
        self.hidden_channels = args.hidden_channels
        # 论文中的 feature propagation units：每层都是一个环境条件专家卷积。
        self.convs = nn.ModuleList()
        for _ in range(args.num_layers):
            self.convs.append(CaNetConv(args.hidden_channels, args.hidden_channels, args.K, backbone_type=args.backbone_type, residual=True, device=device, variant=args.variant))
        # 两层 MLP：输入映射 + 最终标签预测头。
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d, args.hidden_channels))
        self.fcs.append(nn.Linear(args.hidden_channels, c))
        # 论文中的 environment estimator q_φ(E|G)。
        # node: 仅基于节点表示推断环境；
        # graph: 结合图结构信息推断环境。
        self.env_enc = nn.ModuleList()
        for _ in range(args.num_layers):
            if args.env_type == 'node':
                self.env_enc.append(nn.Linear(args.hidden_channels, args.K))
            elif args.env_type == 'graph':
                self.env_enc.append(GraphConvolutionBase(args.hidden_channels, args.K, residual=True))
            else:
                raise NotImplementedError
        self.act_fn = nn.ReLU()
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.tau = args.tau
        self.env_type = args.env_type
        self.device = device
        self._init_layerwise_node_enhancement(args)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        for enc in self.env_enc:
            enc.reset_parameters()
        self._reset_layerwise_node_enhancement_parameters()

    def encode_representation(self, x, adj, training=False):
        self.training = training
        # 输入编码：把原始节点特征映射到隐空间，作为环境估计器和专家传播器的共同输入。
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x))
        h0 = h.clone()

        reg = 0
        layerwise_gate_loss = h.new_zeros(())
        layerwise_gate_mean = h.new_zeros(())
        layerwise_gate_layers = 0
        graph_cfam_gate_loss = h.new_zeros(())
        graph_cfam_gate_mean = h.new_zeros(())
        graph_cfam_layers = 0
        graph_cfam_domain_states = []

        if self.use_pre_gnn_graph_cfam:
            h, _, domain_summary, cfam_gate, edge_gate, cfam_gate_loss = self.graph_cfam_adapt(
                h,
                adj,
                training=self.training,
                local_blend=self.pre_graph_cfam_blend,
                residual_blend=self.pre_graph_cfam_residual_blend,
            )
            graph_cfam_gate_loss = graph_cfam_gate_loss + cfam_gate_loss
            graph_cfam_gate_mean = graph_cfam_gate_mean + cfam_gate.mean()
            graph_cfam_layers += 1
            graph_cfam_domain_states.append(domain_summary)
            layerwise_gate_loss, layerwise_gate_mean, layerwise_gate_layers = (
                self._accumulate_layerwise_gate(
                    edge_gate,
                    layerwise_gate_loss,
                    layerwise_gate_mean,
                    layerwise_gate_layers,
                )
            )

        for i,con in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            should_enhance = (
                (self.use_graph_cfam or self.use_layerwise_local_igm)
                and not (
                    self.layerwise_local_igm_skip_last
                    and i == self.num_layers - 1
                )
            )
            if should_enhance:
                if self.use_graph_cfam:
                    # First enhance the current layer input, then let CaNet infer
                    # environments and route experts from the enhanced node states.
                    h, _, domain_summary, cfam_gate, edge_gate, cfam_gate_loss = self.graph_cfam_adapt(
                        h,
                        adj,
                        training=self.training,
                    )
                    graph_cfam_gate_loss = graph_cfam_gate_loss + cfam_gate_loss
                    graph_cfam_gate_mean = graph_cfam_gate_mean + cfam_gate.mean()
                    graph_cfam_layers += 1
                    graph_cfam_domain_states.append(domain_summary)
                    layerwise_gate_loss, layerwise_gate_mean, layerwise_gate_layers = (
                        self._accumulate_layerwise_gate(
                            edge_gate,
                            layerwise_gate_loss,
                            layerwise_gate_mean,
                            layerwise_gate_layers,
                        )
                    )
                else:
                    # First enhance the current layer input, then let CaNet infer
                    # environments and route experts from the enhanced node states.
                    h, edge_gate = self.layerwise_node_enhance(h, adj, training=self.training)
                    layerwise_gate_loss, layerwise_gate_mean, layerwise_gate_layers = (
                        self._accumulate_layerwise_gate(
                            edge_gate,
                            layerwise_gate_loss,
                            layerwise_gate_mean,
                            layerwise_gate_layers,
                        )
                    )
            if self.training:
                # 逐层环境推断：得到环境 logits，再通过 Gumbel-Softmax 近似采样环境分布 e。
                if self.env_type == 'node':
                    logit = self.env_enc[i](h)
                else:
                    logit = self.env_enc[i](h, adj, h0)
                e = F.gumbel_softmax(logit, tau=self.tau, dim=-1)
                # 对应论文 ELBO / regularization 项中的环境分布约束。
                reg += self.reg_loss(e, logit)
            else:
                # 测试时使用 soft assignment，而不是采样。
                if self.env_type == 'node':
                    e = F.softmax(self.env_enc[i](h), dim=-1)
                else:
                    e = F.softmax(self.env_enc[i](h, adj, h0), dim=-1)
            # 环境条件专家传播：由 e 选择并组合 K 个环境专家的输出。
            h = self.act_fn(con(h, adj, e))

        h_pre_enhance = h
        shortcut_summary = None
        if self.use_graph_cfam:
            if self.use_final_graph_cfam:
                h, _, domain_summary, cfam_gate, edge_gate, cfam_gate_loss = self.graph_cfam_adapt(
                    h,
                    adj,
                    training=self.training,
                )
                graph_cfam_gate_loss = graph_cfam_gate_loss + cfam_gate_loss
                graph_cfam_gate_mean = graph_cfam_gate_mean + cfam_gate.mean()
                graph_cfam_layers += 1
                graph_cfam_domain_states.append(domain_summary)
                layerwise_gate_loss, layerwise_gate_mean, layerwise_gate_layers = (
                    self._accumulate_layerwise_gate(
                        edge_gate,
                        layerwise_gate_loss,
                        layerwise_gate_mean,
                        layerwise_gate_layers,
                    )
                )
                shortcut_summary = domain_summary
            elif graph_cfam_domain_states:
                shortcut_summary = torch.stack(graph_cfam_domain_states, dim=0).mean(dim=0)
            else:
                _, shortcut_summary, _ = self.compute_edge_summaries(h, adj, training=self.training)
        elif self.use_layerwise_local_igm:
            if self.layerwise_final_edge_fuse:
                h, edge_gate = self.layerwise_node_enhance(h, adj, training=self.training)
                _, shortcut_summary, _ = self.compute_edge_summaries(h_pre_enhance, adj, training=self.training)
                layerwise_gate_loss, layerwise_gate_mean, layerwise_gate_layers = (
                    self._accumulate_layerwise_gate(
                        edge_gate,
                        layerwise_gate_loss,
                        layerwise_gate_mean,
                        layerwise_gate_layers,
                    )
                )
            elif self.use_node_edge_norm:
                _, shortcut_summary, _ = self.compute_edge_summaries(h, adj, training=self.training)
                h = self.node_edge_norm(h)
            else:
                _, shortcut_summary, _ = self.compute_edge_summaries(h, adj, training=self.training)
        else:
            _, shortcut_summary, _ = self.compute_edge_summaries(h, adj, training=self.training)

        self._finalize_layerwise_gate_stats(
            layerwise_gate_loss,
            layerwise_gate_mean,
            layerwise_gate_layers,
        )
        self._finalize_graph_cfam_gate_stats(
            graph_cfam_gate_loss,
            graph_cfam_gate_mean,
            graph_cfam_layers,
        )
        self._last_pre_enhance_repr = h_pre_enhance
        self._last_enhanced_repr = h
        self._last_shortcut_summary = shortcut_summary

        return h, reg / self.num_layers

    def forward(self, x, adj, idx=None, training=False):
        h, reg = self.encode_representation(x, adj, training=training)
        h = F.dropout(h, self.dropout, training=self.training)
        out = self.fcs[-1](h)
        if self.training:
            return out, reg
        else:
            return out

    def reg_loss(self, z, logit, logit_0 = None):
        # 对应论文中 environment estimator 的 regularization term，
        # 作用是避免环境分布塌缩到单一环境。
        log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, logit.size(1))
        return torch.mean(torch.sum(
            torch.mul(z, log_pi), dim=1))

    def sup_loss_calc(self, y, pred, criterion, args):
        # 对应论文中的 predictive term，即对目标标签 Y 的监督学习项。
        if args.dataset in ('twitch', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss

    def loss_compute(self, d, criterion, args):
        # 论文训练目标的可执行版本：
        # 监督预测损失 + 环境分布正则项（由 lamda 控制强度）。
        logits, reg_loss = self.forward(d.x, d.edge_index, idx=d.train_idx, training=True)
        sup_loss = self.sup_loss_calc(d.y[d.train_idx], logits[d.train_idx], criterion, args)
        zero = sup_loss.new_zeros(())
        loss_layerwise_gate = (
            zero if self._last_layerwise_gate_loss is None else self._last_layerwise_gate_loss
        )
        loss_graph_cfam_gate = (
            zero if self._last_graph_cfam_gate_loss is None else self._last_graph_cfam_gate_loss
        )
        loss_enhance_sem = self.compute_enhance_semantic_loss(
            self._last_enhanced_repr,
            self._last_pre_enhance_repr,
            d.train_idx,
        )
        loss_graph_delf = self.compute_graph_delf_loss(
            self._last_enhanced_repr,
            self._last_shortcut_summary,
            logits,
            d.y,
            d.train_idx,
            criterion,
            args,
        )
        loss = (
            sup_loss
            + args.lamda * reg_loss
            + self.lambda_layerwise_gate * loss_layerwise_gate
            + self.lambda_graph_cfam_gate * loss_graph_cfam_gate
            + self.lambda_graph_delf * loss_graph_delf
            + self.lambda_enhance_sem * loss_enhance_sem
        )
        self._last_loss_breakdown = {
            'total_loss': loss.detach(),
            'sup_loss': sup_loss.detach(),
            'reg_loss': reg_loss.detach(),
            'loss_layerwise_gate': loss_layerwise_gate.detach(),
            'loss_graph_cfam_gate': loss_graph_cfam_gate.detach(),
            'loss_graph_delf': loss_graph_delf.detach(),
            'layerwise_gate_mean': (
                zero if self._last_layerwise_gate_mean is None else self._last_layerwise_gate_mean.detach()
            ),
            'layerwise_gate_layers': torch.tensor(
                float(self._last_layerwise_gate_layers),
                device=sup_loss.device,
            ),
            'graph_cfam_gate_mean': (
                zero if self._last_graph_cfam_gate_mean is None else self._last_graph_cfam_gate_mean.detach()
            ),
            'graph_cfam_layers': torch.tensor(
                float(self._last_graph_cfam_layers),
                device=sup_loss.device,
            ),
            'loss_enhance_sem': loss_enhance_sem.detach(),
        }
        return loss
