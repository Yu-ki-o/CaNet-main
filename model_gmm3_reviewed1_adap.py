import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, softmax
from torch_sparse import SparseTensor, matmul


def gcn_backbone_conv(x, edge_index):
    num_nodes = x.size(0)
    row, col = edge_index
    deg = degree(col, num_nodes).float()
    deg_in = (1.0 / deg[col].clamp_min(1.0)).sqrt()
    deg_out = (1.0 / deg[row].clamp_min(1.0)).sqrt()
    value = torch.ones_like(row, dtype=x.dtype) * deg_in * deg_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
    return matmul(adj, x)


class FrontDoorBackboneLayer(nn.Module):
    def __init__(self, in_features, out_features, backbone_type='gcn', residual=True, variant=False):
        super().__init__()
        self.backbone_type = backbone_type
        self.out_features = out_features
        self.residual = residual and in_features == out_features
        self.variant = variant

        if backbone_type == 'gcn':
            self.weight = Parameter(torch.FloatTensor(in_features * 2, out_features))
        elif backbone_type == 'gat':
            self.leakyrelu = nn.LeakyReLU()
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
            self.att = Parameter(torch.FloatTensor(2 * out_features, 1))
        else:
            raise NotImplementedError("backbone_type must be 'gcn' or 'gat'.")
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.out_features ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        if self.backbone_type == 'gat':
            nn.init.xavier_uniform_(self.att.data, gain=1.414)

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
            adj = SparseTensor(
                row=att_edge_index[0],
                col=att_edge_index[1],
                value=alpha,
                sparse_sizes=(num_nodes, num_nodes),
            )
            out = matmul(adj, h)

        if self.residual:
            out = out + x
        return out


class GraphFrontDoorAdapter(nn.Module):
    """
    Clean adapter variant:
    1) GNN backbone.
    2) Local edge-aware node enhancement.
    3) Optional ego-out same-class virtual-edge enhancement during training.
    4) CIPT-style causal/spurious adapters.
    5) Front-door adjustment over cached/training spurious contexts.
    6) Bidirectional counterfactual role constraints:
       replacing spurious contexts should keep predictions stable, while
       replacing mediator contexts should move predictions to donor labels.

    The learned DAG, counterfactual branch, GMM contexts, global mixer,
    bi-smoothing, and their auxiliary constraints are intentionally absent.
    """

    def __init__(self, d_in, c, args, device):
        super().__init__()
        self.device = device
        self.d = int(args.hidden_channels)
        self.c = int(c)
        self.num_layers = max(1, int(getattr(args, 'num_layers', 2)))
        self.backbone_type = getattr(args, 'backbone_type', 'gcn')
        self.variant = getattr(args, 'variant', False)
        self.dropout = float(getattr(args, 'dropout', 0.0))

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

        self.edge_feat_mode = getattr(args, 'edge_feat_mode', 'mul')
        edge_feat_dim = self._get_edge_feat_dim(self.edge_feat_mode)
        self.edge_score_temp = max(1e-3, float(getattr(args, 'edge_score_temp', 2.0)))
        self.edge_blend = max(0.0, float(getattr(args, 'edge_blend', 0.2)))
        self.virtual_blend = max(0.0, float(getattr(args, 'virtual_blend', 0.2)))
        self.use_virtual_node_enhance = bool(getattr(args, 'use_virtual_node_enhance', True))
        self.virtual_k = max(0, int(getattr(args, 'virtual_k', getattr(args, 'K', 3))))
        self.virtual_sample_pool = max(
            self.virtual_k,
            int(getattr(args, 'virtual_sample_pool', max(1, self.virtual_k * 4))),
        )
        self.virtual_score_temp = max(1e-3, float(getattr(args, 'virtual_score_temp', 1.0)))
        self.virtual_diff_bias = float(getattr(args, 'virtual_diff_bias', 1.0))
        # Always exclude the full GNN receptive field for virtual edges:
        # an L-layer GNN samples only outside the L-hop ego graph.
        self.virtual_exclude_hops = self.num_layers

        self.edge_pair_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.edge_score_head = nn.Linear(self.d, 1)
        self.virtual_pair_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.virtual_score_head = nn.Linear(self.d, 1)
        self.local_summary_norm = nn.LayerNorm(self.d)
        self.virtual_summary_norm = nn.LayerNorm(self.d)
        self.node_aug_fuser = nn.Sequential(
            nn.Linear(self.d * 5, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.node_aug_norm = nn.LayerNorm(self.d)

        adapter_dropout = self.dropout
        bottleneck_dim = max(16, self.d // 2)
        self.causal_adapter = nn.Sequential(
            nn.Linear(self.d, self.d * 2),
            nn.LayerNorm(self.d * 2),
            nn.GELU(),
            nn.Dropout(p=adapter_dropout),
            nn.Linear(self.d * 2, self.d),
        )
        self.spurious_adapter = nn.Sequential(
            nn.Linear(self.d, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(p=adapter_dropout),
            nn.Linear(bottleneck_dim, self.d),
        )
        self.causal_norm = nn.LayerNorm(self.d)
        self.spurious_norm = nn.LayerNorm(self.d)

        self.classifier = nn.Linear(self.d, self.c)
        self.fd_classifier = nn.Linear(self.d, self.c)
        self.fd_fuser = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.fd_norm = nn.LayerNorm(self.d)
        self.env_num = max(0, int(getattr(args, 'train_env_num', getattr(args, 'env_num', 0))))
        if self.env_num > 1:
            self.env_classifier = nn.Sequential(
                nn.Linear(self.d, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.GELU(),
                nn.Dropout(p=adapter_dropout),
                nn.Linear(bottleneck_dim, self.env_num),
            )
            self.causal_env_classifier = nn.Linear(self.d, self.env_num)
        else:
            self.env_classifier = None
            self.causal_env_classifier = None
        self.fd_blend = float(getattr(args, 'fd_blend', 0.5))
        self.eval_pred_mode = getattr(args, 'eval_pred_mode', 'blend')
        if self.eval_pred_mode not in ('blend', 'mediator', 'frontdoor'):
            self.eval_pred_mode = 'blend'

        self.lambda_med = float(getattr(args, 'lambda_med', 0.25))
        self.lambda_fd = float(getattr(args, 'lambda_fd', 0.5))
        self.lambda_spu = float(getattr(args, 'lambda_spu', 0.05))
        self.lambda_ind = float(getattr(args, 'lambda_ind', 0.05))
        self.lambda_env_swap = float(getattr(args, 'lambda_env_swap', 0.1))
        self.lambda_causal_swap = float(getattr(args, 'lambda_causal_swap', 0.1))
        self.lambda_env_pred = float(getattr(args, 'lambda_env_pred', 0.05 if self.env_num > 1 else 0.0))
        self.lambda_causal_env = float(getattr(args, 'lambda_causal_env', 0.0))

        self.register_buffer('context_bank', torch.zeros(self.c, self.d))
        self.register_buffer('context_valid', torch.zeros(self.c, dtype=torch.bool))
        self._ego_cache_key = None
        self._ego_exclusion_cache = None
        self._virtual_candidate_key = None
        self._virtual_candidate_cache = None
        self._prepared_virtual_edge_indices = None
        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for layer in self.backbone_layers:
            layer.reset_parameters()
        self._reset_module_parameters(self.edge_pair_encoder)
        self.edge_score_head.reset_parameters()
        self._reset_module_parameters(self.virtual_pair_encoder)
        self.virtual_score_head.reset_parameters()
        self.local_summary_norm.reset_parameters()
        self.virtual_summary_norm.reset_parameters()
        self._reset_module_parameters(self.node_aug_fuser)
        nn.init.zeros_(self.node_aug_fuser[-1].weight)
        nn.init.zeros_(self.node_aug_fuser[-1].bias)
        self.node_aug_norm.reset_parameters()
        self._reset_module_parameters(self.causal_adapter)
        self._reset_module_parameters(self.spurious_adapter)
        self.causal_norm.reset_parameters()
        self.spurious_norm.reset_parameters()
        self.classifier.reset_parameters()
        self.fd_classifier.reset_parameters()
        self._reset_module_parameters(self.fd_fuser)
        self.fd_norm.reset_parameters()
        if self.env_classifier is not None:
            self._reset_module_parameters(self.env_classifier)
        if self.causal_env_classifier is not None:
            self.causal_env_classifier.reset_parameters()
        self.context_bank.zero_()
        self.context_valid.zero_()
        self._ego_cache_key = None
        self._ego_exclusion_cache = None
        self._virtual_candidate_key = None
        self._virtual_candidate_cache = None
        self._prepared_virtual_edge_indices = None

    def _reset_module_parameters(self, module):
        for sub_module in module.modules():
            if sub_module is module:
                continue
            if hasattr(sub_module, 'reset_parameters'):
                sub_module.reset_parameters()

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
        mode = self.edge_feat_mode
        mul_feat = h_src * h_dst
        signed_diff_feat = h_src - h_dst
        diff_feat = torch.abs(signed_diff_feat)
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

    def _pair_distance_score(self, h_src, h_dst, deg_src=None, deg_dst=None, deg_max=None):
        mode = self.edge_feat_mode
        diff_score = torch.abs(h_src - h_dst).mean(dim=-1)
        if 'diff' in mode:
            return diff_score
        if mode == 'degree' and deg_src is not None and deg_dst is not None and deg_max is not None:
            return torch.abs(torch.log1p(deg_src) - torch.log1p(deg_dst)) / deg_max.clamp_min(1.0)
        sim = F.cosine_similarity(h_src, h_dst, dim=-1)
        return 1.0 - sim

    def compute_local_summary(self, h, edge_index):
        if edge_index.numel() == 0:
            return h.new_zeros(h.size()), None
        src, dst = edge_index
        deg = degree(dst, h.size(0)).to(device=h.device, dtype=h.dtype).clamp_min(1.0)
        deg_max = torch.log1p(deg).max().clamp_min(1.0)
        edge_feat = self.build_edge_feat(h[src], h[dst], deg[src], deg[dst], deg_max)
        edge_hidden = F.dropout(self.edge_pair_encoder(edge_feat), self.dropout, training=self.training)
        logits = self.edge_score_head(edge_hidden).squeeze(-1) / self.edge_score_temp
        gate = torch.sigmoid(logits)
        norm = deg[src].pow(-0.5) * deg[dst].pow(-0.5)
        weight = torch.nan_to_num(norm * gate, nan=0.0, posinf=0.0, neginf=0.0)
        summary = h.new_zeros(h.size())
        summary.index_add_(0, dst, weight.unsqueeze(-1) * h[src])
        return self.local_summary_norm(summary), gate

    def _build_virtual_candidate_cache(self, edge_index, num_nodes, labels, train_idx):
        labels_flat = labels.squeeze().long()
        train_idx_cpu = train_idx.detach().cpu()
        train_cpu = tuple(int(v) for v in train_idx_cpu.tolist())
        train_labels_cpu = tuple(int(v) for v in labels_flat[train_idx].detach().cpu().tolist())
        key = (
            int(num_nodes),
            int(edge_index.size(1)),
            self.virtual_exclude_hops,
            train_cpu,
            train_labels_cpu,
        )
        if self._virtual_candidate_key == key and self._virtual_candidate_cache is not None:
            return self._virtual_candidate_cache

        class_to_nodes = {}
        for node, label in zip(train_cpu, train_labels_cpu):
            class_to_nodes.setdefault(label, []).append(node)

        train_count = len(train_cpu)
        train_pos = torch.arange(train_count, device=edge_index.device, dtype=torch.long)
        train_nodes = train_idx.to(edge_index.device)
        reach = SparseTensor(
            row=train_pos,
            col=train_nodes,
            value=torch.ones(train_count, device=edge_index.device),
            sparse_sizes=(train_count, num_nodes),
        )
        visited = reach

        if self.virtual_exclude_hops > 0 and edge_index.numel() > 0:
            src, dst = edge_index
            undirected_row = torch.cat([src, dst], dim=0)
            undirected_col = torch.cat([dst, src], dim=0)
            adj = SparseTensor(
                row=undirected_row,
                col=undirected_col,
                value=torch.ones(undirected_row.numel(), device=edge_index.device),
                sparse_sizes=(num_nodes, num_nodes),
            ).coalesce()

            for _ in range(self.virtual_exclude_hops):
                reach = matmul(reach, adj)
                reach = reach.set_value(
                    torch.ones(reach.nnz(), device=edge_index.device),
                    layout='coo',
                ).coalesce()
                visited = (visited + reach).coalesce()
                visited = visited.set_value(
                    torch.ones(visited.nnz(), device=edge_index.device),
                    layout='coo',
                ).coalesce()

        visited_rows, visited_cols, _ = visited.coo()
        ego_sets = [set() for _ in range(train_count)]
        for row, col in zip(
            visited_rows.detach().cpu().tolist(),
            visited_cols.detach().cpu().tolist(),
        ):
            ego_sets[int(row)].add(int(col))

        candidate_cache = {}
        for pos, node in enumerate(train_cpu):
            label = int(labels_flat[node].detach().cpu().item())
            ego_nodes = ego_sets[pos]
            candidates = [v for v in class_to_nodes.get(label, []) if v not in ego_nodes]
            if candidates:
                candidate_cache[node] = torch.tensor(candidates, dtype=torch.long)

        self._virtual_candidate_key = key
        self._virtual_candidate_cache = candidate_cache
        return candidate_cache

    def sample_virtual_edge_index(self, edge_index, num_nodes, labels=None, train_idx=None, device=None):
        if (
            not self.training
            or not self.use_virtual_node_enhance
            or self.virtual_k <= 0
            or labels is None
            or train_idx is None
            or train_idx.numel() == 0
        ):
            return None

        if device is None:
            device = edge_index.device
        train_idx = train_idx.to(device)
        candidate_cache = self._build_virtual_candidate_cache(
            edge_index,
            num_nodes,
            labels,
            train_idx,
        )

        src_edges = []
        dst_edges = []
        for dst_node in train_idx.tolist():
            candidates_cpu = candidate_cache.get(int(dst_node))
            if candidates_cpu is None or candidates_cpu.numel() == 0:
                continue
            # The expensive "same label and ego-out" candidate set is cached.
            # Each layer only samples K random nodes from that CPU cache, then
            # transfers the final compact edge_index once.
            k = min(self.virtual_k, int(candidates_cpu.numel()))
            draw_pos = torch.randperm(candidates_cpu.numel())[:k]
            src_edges.append(candidates_cpu[draw_pos])
            dst_edges.append(torch.full((k,), int(dst_node), dtype=torch.long))

        if not src_edges:
            return None
        return torch.stack([torch.cat(src_edges), torch.cat(dst_edges)], dim=0).to(device)

    def sample_layer_virtual_edges(self, edge_index, num_nodes, labels=None, train_idx=None):
        if not self.training or not self.use_virtual_node_enhance or self.virtual_k <= 0:
            return [None for _ in range(self.num_layers)]
        return [
            self.sample_virtual_edge_index(
                edge_index,
                num_nodes,
                labels=labels,
                train_idx=train_idx,
                device=edge_index.device,
            )
            for _ in range(self.num_layers)
        ]

    @torch.no_grad()
    def prepare_virtual_edges(self, edge_index, num_nodes, labels=None, train_idx=None):
        self._prepared_virtual_edge_indices = self.sample_layer_virtual_edges(
            edge_index,
            num_nodes,
            labels=labels,
            train_idx=train_idx,
        )
        return self._prepared_virtual_edge_indices

    def compute_virtual_summary(self, h, edge_index, virtual_edge_index=None):
        if virtual_edge_index is None or virtual_edge_index.numel() == 0:
            return h.new_zeros(h.size()), None, 0

        src, dst = virtual_edge_index
        deg = degree(edge_index[1], h.size(0)).to(device=h.device, dtype=h.dtype).clamp_min(1.0)
        deg_max = torch.log1p(deg).max().clamp_min(1.0)
        edge_feat = self.build_edge_feat(h[src], h[dst], deg[src], deg[dst], deg_max)
        edge_hidden = F.dropout(self.virtual_pair_encoder(edge_feat), self.dropout, training=self.training)
        learned = self.virtual_score_head(edge_hidden).squeeze(-1)
        prior = self._pair_distance_score(h[src], h[dst], deg[src], deg[dst], deg_max)
        logits = (learned + self.virtual_diff_bias * prior) / self.virtual_score_temp
        alpha = softmax(logits, dst, num_nodes=h.size(0))
        summary = h.new_zeros(h.size())
        summary.index_add_(0, dst, alpha.unsqueeze(-1) * h[src])
        return self.virtual_summary_norm(summary), alpha, int(src.numel())

    def fuse_node_representation(self, h, local_summary, virtual_summary):
        local_summary = self.edge_blend * local_summary
        virtual_summary = self.virtual_blend * virtual_summary
        fuse_input = torch.cat(
            [h, local_summary, virtual_summary, h * local_summary, h * virtual_summary],
            dim=-1,
        )
        delta = self.node_aug_fuser(fuse_input)
        delta = F.dropout(delta, self.dropout, training=self.training)
        return self.node_aug_norm(h + delta)

    def encode_representation(self, x, edge_index, labels=None, train_idx=None, virtual_edge_indices=None):
        h = F.dropout(x, self.dropout, training=self.training)
        h = F.relu(self.input_proj(h))
        h_pre_enhance = h
        local_gate_means = []
        virtual_alpha_means = []
        num_virtual_edges = 0
        if virtual_edge_indices is None:
            virtual_edge_indices = self.sample_layer_virtual_edges(
                edge_index,
                x.size(0),
                labels=labels,
                train_idx=train_idx,
            )
        for layer_idx, layer in enumerate(self.backbone_layers):
            h = F.dropout(h, self.dropout, training=self.training)
            h = F.relu(layer(h, edge_index))
            h_pre_enhance = h

            local_summary, local_gate = self.compute_local_summary(h, edge_index)
            virtual_edge_index = None
            if layer_idx < len(virtual_edge_indices):
                virtual_edge_index = virtual_edge_indices[layer_idx]
            virtual_summary, virtual_alpha, virtual_edges_l = self.compute_virtual_summary(
                h,
                edge_index,
                virtual_edge_index=virtual_edge_index,
            )
            h = self.fuse_node_representation(h, local_summary, virtual_summary)
            if local_gate is not None:
                local_gate_means.append(local_gate.mean())
            if virtual_alpha is not None:
                virtual_alpha_means.append(virtual_alpha.mean())
            num_virtual_edges += int(virtual_edges_l)

        z = h

        z_mediator = self.causal_norm(self.causal_adapter(z))
        z_spurious = self.spurious_norm(self.spurious_adapter(z))
        z_mediator = F.dropout(z_mediator, self.dropout, training=self.training)
        z_spurious = F.dropout(z_spurious, self.dropout, training=self.training)
        mediator_logits = self.classifier(z_mediator)
        return {
            'z': z,
            'z_mediator': z_mediator,
            'z_spurious': z_spurious,
            'mediator_logits': mediator_logits,
            'h_pre_enhance': h_pre_enhance,
            'local_gate': None if not local_gate_means else torch.stack(local_gate_means).mean(),
            'virtual_alpha': None if not virtual_alpha_means else torch.stack(virtual_alpha_means).mean(),
            'num_virtual_edges': num_virtual_edges,
        }

    def build_spurious_contexts(self, z_spurious, labels):
        if z_spurious is None or z_spurious.numel() == 0:
            return None
        labels_flat = labels.squeeze().long()
        contexts = []
        for cls in labels_flat.unique().tolist():
            mask = labels_flat == int(cls)
            if mask.sum() > 0:
                contexts.append(z_spurious[mask].mean(dim=0))
        if not contexts:
            return None
        return torch.stack(contexts, dim=0)

    def frontdoor_logits_from_contexts(self, z_mediator, contexts):
        base_logits = self.fd_classifier(z_mediator)
        if contexts is None or contexts.numel() == 0:
            return base_logits
        contexts = contexts.to(device=z_mediator.device, dtype=z_mediator.dtype)
        num_contexts = contexts.size(0)
        mediator_expand = z_mediator.unsqueeze(1).expand(-1, num_contexts, -1)
        context_expand = contexts.unsqueeze(0).expand(z_mediator.size(0), -1, -1)
        fuse_input = torch.cat(
            [mediator_expand, context_expand, mediator_expand * context_expand],
            dim=-1,
        )
        fused = self.fd_fuser(fuse_input.reshape(-1, self.d * 3)).view(
            z_mediator.size(0),
            num_contexts,
            self.d,
        )
        fused = self.fd_norm(fused + mediator_expand)
        logits = self.fd_classifier(fused.reshape(-1, self.d)).view(
            z_mediator.size(0),
            num_contexts,
            self.c,
        )
        return logits.mean(dim=1)

    def frontdoor_logits_from_pairs(self, z_mediator, z_spurious):
        fuse_input = torch.cat([z_mediator, z_spurious, z_mediator * z_spurious], dim=-1)
        fused = self.fd_fuser(fuse_input)
        fused = self.fd_norm(fused + z_mediator)
        return self.fd_classifier(fused)

    def blend_logits(self, mediator_logits, fd_logits):
        return (1.0 - self.fd_blend) * mediator_logits + self.fd_blend * fd_logits

    def forward(self, x, edge_index, training=False, labels=None, train_idx=None):
        self.train(training) if training else self.eval()
        enc = self.encode_representation(x, edge_index, labels=labels, train_idx=train_idx)
        contexts = self.context_bank[self.context_valid]
        fd_logits = self.frontdoor_logits_from_contexts(enc['z_mediator'], contexts)
        logits = self.blend_logits(enc['mediator_logits'], fd_logits)
        if self.eval_pred_mode == 'mediator':
            return enc['mediator_logits']
        if self.eval_pred_mode == 'frontdoor':
            return fd_logits
        return logits

    def compute_supervised_loss(self, logits, y, criterion, args):
        if args.dataset in ('twitch', 'elliptic'):
            if y.shape[1] == 1 and logits.shape[1] > 1:
                true_label = F.one_hot(y.squeeze().long(), logits.shape[1]).float()
            else:
                true_label = y.float()
            loss = criterion(logits, true_label)
            if loss.dim() > 1:
                loss = loss.mean(dim=1)
            return loss
        return criterion(logits, y.squeeze().long())

    def compute_uniform_loss(self, logits):
        if logits.size(-1) <= 1:
            return logits.new_zeros(())
        log_probs = F.log_softmax(logits, dim=-1)
        uniform = torch.full_like(log_probs, 1.0 / logits.size(-1))
        return F.kl_div(log_probs, uniform, reduction='batchmean')

    def compute_independence_loss(self, z_mediator, z_spurious):
        if z_mediator.numel() == 0:
            return z_mediator.new_zeros(())
        med = F.normalize(z_mediator, dim=1)
        spu = F.normalize(z_spurious, dim=1)
        return 0.5 * (med * spu).sum(dim=1).pow(2).mean()

    def sample_different_label_indices(self, labels):
        labels_flat = labels.squeeze().long()
        unique_labels = labels_flat.unique()
        if labels_flat.numel() == 0 or unique_labels.numel() <= 1:
            return None

        all_pos = torch.arange(labels_flat.size(0), device=labels_flat.device)
        donor_idx = torch.empty_like(all_pos)
        donor_labels = unique_labels.roll(1)
        for label, donor_label in zip(unique_labels.tolist(), donor_labels.tolist()):
            target_mask = labels_flat == int(label)
            donor_pool = all_pos[labels_flat == int(donor_label)]
            if donor_pool.numel() == 0:
                return None
            draw = torch.randint(donor_pool.numel(), (int(target_mask.sum().item()),), device=labels_flat.device)
            donor_idx[target_mask] = donor_pool[draw]
        return donor_idx

    def compute_env_prediction_loss(self, z_spurious, env_labels):
        if self.env_classifier is None or env_labels is None or env_labels.numel() == 0:
            return z_spurious.new_zeros(())

        env_flat = env_labels.squeeze().long()
        valid = (env_flat >= 0) & (env_flat < self.env_num)
        if valid.sum() == 0:
            return z_spurious.new_zeros(())
        env_logits = self.env_classifier(z_spurious[valid])
        return F.cross_entropy(env_logits, env_flat[valid])

    def compute_causal_env_uniform_loss(self, z_mediator):
        if self.causal_env_classifier is None or z_mediator.numel() == 0:
            return z_mediator.new_zeros(())
        return self.compute_uniform_loss(self.causal_env_classifier(z_mediator))

    def compute_env_swap_loss(self, z_mediator, z_spurious):
        if z_mediator.size(0) <= 1:
            return z_mediator.new_zeros(())

        perm = torch.randperm(z_spurious.size(0), device=z_spurious.device)
        if torch.equal(perm, torch.arange(z_spurious.size(0), device=z_spurious.device)):
            perm = perm.roll(1)

        base_logits = self.frontdoor_logits_from_pairs(z_mediator, z_spurious)
        swap_logits = self.frontdoor_logits_from_pairs(z_mediator, z_spurious[perm])
        base_prob = F.softmax(base_logits.detach(), dim=-1)
        swap_log_prob = F.log_softmax(swap_logits, dim=-1)
        return F.kl_div(swap_log_prob, base_prob, reduction='batchmean')

    def compute_causal_swap_loss(self, z_mediator, z_spurious, labels, criterion, args):
        donor_idx = self.sample_different_label_indices(labels)
        if donor_idx is None:
            return z_mediator.new_zeros(())

        donor_mediator = z_mediator[donor_idx]
        donor_labels = labels[donor_idx]
        swap_logits = self.frontdoor_logits_from_pairs(donor_mediator, z_spurious)
        return self.compute_supervised_loss(swap_logits, donor_labels, criterion, args).mean()

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx
        virtual_edge_indices = self._prepared_virtual_edge_indices
        if virtual_edge_indices is None:
            virtual_edge_indices = self.sample_layer_virtual_edges(
                edge_index,
                x.size(0),
                labels=y,
                train_idx=train_idx,
            )
        enc = self.encode_representation(
            x,
            edge_index,
            labels=y,
            train_idx=train_idx,
            virtual_edge_indices=virtual_edge_indices,
        )

        y_tr = y[train_idx]
        med_tr = enc['z_mediator'][train_idx]
        spu_tr = enc['z_spurious'][train_idx]
        med_logits_tr = enc['mediator_logits'][train_idx]
        contexts = self.build_spurious_contexts(spu_tr.detach(), y_tr)
        fd_logits_tr = self.frontdoor_logits_from_contexts(med_tr, contexts)
        final_logits_tr = self.blend_logits(med_logits_tr, fd_logits_tr)

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_med = self.compute_supervised_loss(med_logits_tr, y_tr, criterion, args).mean()
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        spu_logits = F.linear(
            spu_tr,
            self.classifier.weight.detach(),
            self.classifier.bias.detach() if self.classifier.bias is not None else None,
        )
        loss_spu = self.compute_uniform_loss(spu_logits)
        loss_ind = self.compute_independence_loss(med_tr, spu_tr)
        env_tr = data.env[train_idx] if hasattr(data, 'env') else None
        loss_env_swap = self.compute_env_swap_loss(med_tr, spu_tr)
        loss_causal_swap = self.compute_causal_swap_loss(med_tr, spu_tr, y_tr, criterion, args)
        loss_env_pred = self.compute_env_prediction_loss(spu_tr, env_tr)
        loss_causal_env = self.compute_causal_env_uniform_loss(med_tr)
        total_loss = (
            loss_cls
            + self.lambda_med * loss_med
            + self.lambda_fd * loss_fd
            + self.lambda_spu * loss_spu
            + self.lambda_ind * loss_ind
            + self.lambda_env_swap * loss_env_swap
            + self.lambda_causal_swap * loss_causal_swap
            + self.lambda_env_pred * loss_env_pred
            + self.lambda_causal_env * loss_causal_env
        )

        zero = loss_cls.new_zeros(())
        state_payload = None
        if update_state:
            state_payload = {
                'spu_tr': spu_tr.detach(),
                'labels_tr': y_tr.detach(),
            }

        local_gate = enc['local_gate']
        virtual_alpha = enc['virtual_alpha']
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_med': loss_med,
            'loss_fd': loss_fd,
            'loss_spu': loss_spu,
            'loss_ind': loss_ind,
            'loss_env_swap': loss_env_swap,
            'loss_causal_swap': loss_causal_swap,
            'loss_env_pred': loss_env_pred,
            'loss_causal_env': loss_causal_env,
            'local_gate_mean': zero if local_gate is None else local_gate.mean().detach(),
            'mediator_gate_mean': med_tr.detach().abs().mean(),
            'causal_score_mean': med_tr.detach().abs().mean(),
            'pollution_score_mean': spu_tr.detach().abs().mean(),
            'num_contexts': torch.tensor(0.0 if contexts is None else float(contexts.size(0)), device=x.device),
            'num_virtual_edges': torch.tensor(float(enc['num_virtual_edges']), device=x.device),
            'virtual_alpha_mean': zero if virtual_alpha is None else virtual_alpha.mean().detach(),
            'state_payload': state_payload,
        }

    @torch.no_grad()
    def apply_state_update(self, state_payload):
        if state_payload is None:
            return
        contexts = self.build_spurious_contexts(state_payload['spu_tr'], state_payload['labels_tr'])
        self.context_bank.zero_()
        self.context_valid.zero_()
        if contexts is None:
            return
        labels_flat = state_payload['labels_tr'].squeeze().long()
        classes = labels_flat.unique().tolist()
        for ctx, cls in zip(contexts, classes):
            cls = int(cls)
            if 0 <= cls < self.c:
                self.context_bank[cls] = ctx
                self.context_valid[cls] = True

    def loss_compute(self, data, criterion, args):
        losses = self.compute_losses(data, criterion, args, update_state=True)
        return (
            losses['total_loss'],
            losses['loss_cls'].item(),
            (self.lambda_ind * losses['loss_ind']).item(),
            0.0,
            (self.lambda_fd * losses['loss_fd']).item(),
        )


# Backward-compatible alias for older training scripts.
GraphFrontDoorDAG = GraphFrontDoorAdapter
