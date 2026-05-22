import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, softmax
from torch_sparse import SparseTensor, matmul


def build_gcn_norm_adj(edge_index, num_nodes, dtype=torch.float32):
    """Build the normalized GCN adjacency once for a fixed graph."""
    row, col = edge_index
    deg = degree(col, num_nodes).to(device=edge_index.device, dtype=dtype).clamp_min(1.0)
    deg_in = deg[col].pow(-0.5)
    deg_out = deg[row].pow(-0.5)
    value = torch.nan_to_num(deg_in * deg_out, nan=0.0, posinf=0.0, neginf=0.0)
    return SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)).coalesce()


def gcn_backbone_conv(x, edge_index, norm_adj=None):
    if norm_adj is None:
        norm_adj = build_gcn_norm_adj(edge_index, x.size(0), dtype=x.dtype)
    return matmul(norm_adj, x)


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

    def forward(self, x, edge_index, gcn_adj=None, variant_adj=None, gat_edge_index=None):
        if self.backbone_type == 'gcn':
            if self.variant:
                if variant_adj is None:
                    variant_adj = torch.sparse_coo_tensor(
                        edge_index,
                        torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype),
                        size=(x.size(0), x.size(0)),
                    ).coalesce()
                h_neigh = torch.sparse.mm(variant_adj, x)
            else:
                h_neigh = gcn_backbone_conv(x, edge_index, norm_adj=gcn_adj)
            out = torch.matmul(torch.cat([h_neigh, x], dim=1), self.weight)
        else:
            h = torch.matmul(x, self.weight)
            num_nodes = x.size(0)
            if gat_edge_index is None:
                att_edge_index, _ = remove_self_loops(edge_index)
                att_edge_index, _ = add_self_loops(att_edge_index, num_nodes=num_nodes)
            else:
                att_edge_index = gat_edge_index
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
    Dual-classifier adapter variant without ego-out enhancement:
    1) GNN backbone.
    2) Local edge-aware node enhancement.
    3) A shortcut/bias classifier restricted to local shortcut views.
    4) A main causal/front-door classifier trained cooperatively with the
       shortcut classifier through stop-gradient Product-of-Experts losses.

    Ego-out virtual edges are disabled by default in this variant.
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
        self.shortcut_num_layers = max(0, int(getattr(args, 'shortcut_num_layers', 1)))
        self.shortcut_view_mode = getattr(args, 'shortcut_view_mode', 'neighbor_only')
        if self.shortcut_view_mode not in ('neighbor_only', 'structure_only', 'raw_shallow'):
            self.shortcut_view_mode = 'neighbor_only'
        self.shortcut_input_proj = nn.Linear(d_in, self.d)
        self.shortcut_layers = nn.ModuleList([
            FrontDoorBackboneLayer(
                self.d,
                self.d,
                backbone_type=self.backbone_type,
                residual=True,
                variant=self.variant,
            )
            for _ in range(self.shortcut_num_layers)
        ])
        self.shortcut_degree_proj = nn.Linear(1, self.d)
        self.shortcut_norm = nn.LayerNorm(self.d)

        self.edge_feat_mode = getattr(args, 'edge_feat_mode', 'mul')
        edge_feat_dim = self._get_edge_feat_dim(self.edge_feat_mode)
        self.edge_score_temp = max(1e-3, float(getattr(args, 'edge_score_temp', 2.0)))
        self.edge_blend = max(0.0, float(getattr(args, 'edge_blend', 0.2)))
        self.virtual_blend = max(0.0, float(getattr(args, 'virtual_blend', 0.0)))
        self.use_virtual_node_enhance = bool(getattr(args, 'use_virtual_node_enhance', False))
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

        requested_dag_latent_dim = int(getattr(args, 'dag_latent_dim', min(16, self.d)))
        self.dag_latent_dim = max(1, min(requested_dag_latent_dim, self.d))
        self.node_dag_proj = nn.Sequential(
            nn.Linear(self.d, self.dag_latent_dim),
            nn.LayerNorm(self.dag_latent_dim),
            nn.Tanh(),
        )
        self.shortcut_dag_proj = nn.Sequential(
            nn.Linear(self.d, self.dag_latent_dim),
            nn.LayerNorm(self.dag_latent_dim),
            nn.Tanh(),
        )
        self.dag_gate_expander = nn.Linear(self.dag_latent_dim, self.d)
        self.node_var_dim = self.dag_latent_dim
        self.shortcut_var_dim = self.dag_latent_dim
        self.label_var_dim = self.c
        self.dag_var_dim = self.node_var_dim + self.shortcut_var_dim + self.label_var_dim
        self.node_var_slice = slice(0, self.node_var_dim)
        self.shortcut_var_slice = slice(self.node_var_dim, self.node_var_dim + self.shortcut_var_dim)
        self.label_var_slice = slice(self.node_var_dim + self.shortcut_var_dim, self.dag_var_dim)
        self.non_label_var_dim = self.node_var_dim + self.shortcut_var_dim
        self.A_feat = Parameter(torch.zeros(self.dag_var_dim, self.dag_var_dim))
        self.gate_base = Parameter(torch.zeros(self.dag_latent_dim))
        self.dag_label_bias = Parameter(torch.zeros(self.c))

        adapter_dropout = self.dropout
        bottleneck_dim = max(16, self.d // 2)
        shortcut_bottleneck_arg = int(getattr(args, 'shortcut_bottleneck_dim', 0))
        shortcut_bottleneck_dim = max(8, shortcut_bottleneck_arg if shortcut_bottleneck_arg > 0 else max(16, self.d // 4))
        self.bias_backbone_grad = min(1.0, max(0.0, float(getattr(args, 'bias_backbone_grad', 0.1))))
        self.ctx_grad_ratio = min(1.0, max(0.0, float(getattr(args, 'ctx_grad_ratio', 0.0))))
        self.coop_bias_scale = float(getattr(args, 'coop_bias_scale', 1.0))
        self.coop_main_scale = float(getattr(args, 'coop_main_scale', 0.5))
        self.lambda_bias_coop = float(getattr(args, 'lambda_bias_coop', 0.1))
        self.main_reweight_mode = getattr(args, 'main_reweight_mode', 'bias_failure')
        if self.main_reweight_mode not in ('none', 'bias_failure'):
            self.main_reweight_mode = 'bias_failure'
        self.main_reweight_power = max(0.0, float(getattr(args, 'main_reweight_power', 1.0)))
        self.main_reweight_floor = min(1.0, max(0.0, float(getattr(args, 'main_reweight_floor', 0.05))))
        self.mediator_temp = max(1e-3, float(getattr(args, 'mediator_temp', 8.0)))
        self.mediator_threshold = float(getattr(args, 'mediator_threshold', 0.5))
        self.pollution_coeff = float(getattr(args, 'pollution_coeff', 1.0))
        self.causal_support_coeff = float(getattr(args, 'causal_support_coeff', 0.5))
        self.lambda_l1 = float(getattr(args, 'lambda_l1', 1e-5))
        self.lambda_dag = float(getattr(args, 'lambda_dag', 0.05))
        self.lambda_dag_label = float(getattr(args, 'lambda_dag_label', 0.05))
        self.lambda_gate = float(getattr(args, 'lambda_gate', 0.0))
        self.lambda_bias_no_causal = float(getattr(args, 'lambda_bias_no_causal', 0.1))
        self.lambda_bias_spu_align = float(getattr(args, 'lambda_bias_spu_align', 0.05))
        self.lambda_spu_rec = float(getattr(args, 'lambda_spu_rec', 0.05))
        self.dag_gate_blend = min(1.0, max(0.0, float(getattr(args, 'dag_gate_blend', 0.5))))
        self.current_dag_gate_blend = self.dag_gate_blend
        self.causal_adapter = nn.Sequential(
            nn.Linear(self.d, self.d * 2),
            nn.LayerNorm(self.d * 2),
            nn.GELU(),
            nn.Dropout(p=adapter_dropout),
            nn.Linear(self.d * 2, self.d),
        )
        self.spurious_adapter = nn.Sequential(
            nn.Linear(self.d * 5, shortcut_bottleneck_dim),
            nn.LayerNorm(shortcut_bottleneck_dim),
            nn.GELU(),
            nn.Dropout(p=adapter_dropout),
            nn.Linear(shortcut_bottleneck_dim, self.d),
        )
        self.bias_latent_projector = nn.Linear(self.d, self.dag_latent_dim)
        self.spu_reconstructor = nn.Sequential(
            nn.Linear(self.dag_latent_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.causal_norm = nn.LayerNorm(self.d)
        self.spurious_norm = nn.LayerNorm(self.d)

        self.classifier = nn.Linear(self.d, self.c)
        self.spurious_classifier = nn.Linear(self.d, self.c)
        self.fd_classifier = nn.Linear(self.d, self.c)
        self.fd_fuser = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.fd_norm = nn.LayerNorm(self.d)
        self.fd_blend = float(getattr(args, 'fd_blend', 0.5))
        self.current_fd_blend = self.fd_blend
        self.eval_pred_mode = getattr(args, 'eval_pred_mode', 'blend')
        if self.eval_pred_mode not in ('blend', 'mediator', 'frontdoor'):
            self.eval_pred_mode = 'blend'

        self.lambda_med = float(getattr(args, 'lambda_med', 0.25))
        self.lambda_fd = float(getattr(args, 'lambda_fd', 0.5))
        self.lambda_spu = float(getattr(args, 'lambda_spu', 0.05))
        self.lambda_ind = float(getattr(args, 'lambda_ind', 0.05))

        self.register_buffer('context_bank', torch.zeros(self.c, self.d))
        self.register_buffer('context_valid', torch.zeros(self.c, dtype=torch.bool))
        self.register_buffer('dag_allowed_mask', self.build_dag_allowed_mask())
        self._graph_cache_key = None
        self._cached_degree = None
        self._cached_deg_max = None
        self._cached_gcn_adj = None
        self._cached_variant_adj = None
        self._cached_gat_edge_index = None
        self._virtual_candidate_key = None
        self._virtual_candidate_cache = None
        self._prepared_virtual_edge_indices = None
        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for layer in self.backbone_layers:
            layer.reset_parameters()
        self.shortcut_input_proj.reset_parameters()
        for layer in self.shortcut_layers:
            layer.reset_parameters()
        self.shortcut_degree_proj.reset_parameters()
        self.shortcut_norm.reset_parameters()
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
        self._reset_module_parameters(self.node_dag_proj)
        self._reset_module_parameters(self.shortcut_dag_proj)
        self.dag_gate_expander.reset_parameters()
        nn.init.xavier_uniform_(self.dag_gate_expander.weight, gain=0.1)
        nn.init.zeros_(self.dag_gate_expander.bias)
        nn.init.uniform_(self.A_feat, -0.01, 0.01)
        nn.init.zeros_(self.gate_base)
        nn.init.zeros_(self.dag_label_bias)
        self._reset_module_parameters(self.causal_adapter)
        self._reset_module_parameters(self.spurious_adapter)
        self.bias_latent_projector.reset_parameters()
        self._reset_module_parameters(self.spu_reconstructor)
        self.causal_norm.reset_parameters()
        self.spurious_norm.reset_parameters()
        self.classifier.reset_parameters()
        self.spurious_classifier.reset_parameters()
        self.fd_classifier.reset_parameters()
        self._reset_module_parameters(self.fd_fuser)
        self.fd_norm.reset_parameters()
        self.current_fd_blend = self.fd_blend
        self.current_dag_gate_blend = self.dag_gate_blend
        self.context_bank.zero_()
        self.context_valid.zero_()
        # Keep graph/candidate caches across parameter resets: they depend on the
        # fixed graph/labels, not on learnable weights. This avoids rebuilding
        # expensive ego-out candidate pools for every run.
        self._prepared_virtual_edge_indices = None

    def _reset_module_parameters(self, module):
        for sub_module in module.modules():
            if sub_module is module:
                continue
            if hasattr(sub_module, 'reset_parameters'):
                sub_module.reset_parameters()

    def build_dag_allowed_mask(self):
        allowed = torch.ones(self.dag_var_dim, self.dag_var_dim, dtype=torch.bool)
        allowed.fill_diagonal_(False)
        allowed[self.label_var_slice, :] = False
        return allowed

    def get_masked_A(self):
        return self.A_feat * self.dag_allowed_mask.to(
            device=self.A_feat.device,
            dtype=self.A_feat.dtype,
        )

    def _normalize_score(self, values, default_value=0.5):
        if values.numel() == 0:
            return values
        v_min = values.min()
        v_max = values.max()
        if (v_max - v_min).abs() < 1e-4:
            return (values - values.detach()) + default_value
        return (values - v_min) / (v_max - v_min + 1e-8)

    def get_causal_effect_and_mask(self):
        A = self.get_masked_A()
        A_sq = A * A
        dag_total = torch.matrix_exp(A_sq)
        eye = torch.eye(self.dag_var_dim, device=A.device, dtype=A.dtype)
        dag_flow = dag_total - eye

        node_to_label = dag_flow[self.node_var_slice, self.label_var_slice]
        label_effect = self._normalize_score(node_to_label.mean(dim=1), default_value=0.5)

        node_flow = dag_flow[self.node_var_slice, self.node_var_slice]
        shortcut_to_node = dag_flow[self.shortcut_var_slice, self.node_var_slice]
        shortcut_pollution = self._normalize_score(
            shortcut_to_node.mean(dim=0),
            default_value=0.0,
        )
        causal_support = self._normalize_score(
            node_flow.mean(dim=0),
            default_value=0.0,
        )
        base_score = torch.sigmoid(self.gate_base)
        causal_score = self._normalize_score(
            base_score + label_effect + self.causal_support_coeff * causal_support,
            default_value=0.5,
        )
        pollution_score = self._normalize_score(shortcut_pollution, default_value=0.0)
        mediator_logit = causal_score - self.pollution_coeff * pollution_score - self.mediator_threshold
        hidden_logit = self.dag_gate_expander(mediator_logit.unsqueeze(0)).squeeze(0)
        mediator_gate = torch.sigmoid(self.mediator_temp * hidden_logit)
        return causal_score, pollution_score, mediator_gate, dag_total

    def dag_regularization_loss(self, mediator_gate, dag_total):
        A = self.get_masked_A()
        A_sq = A * A
        h_A = torch.trace(torch.matrix_exp(A_sq)) - self.dag_var_dim
        h_A_clipped = torch.clamp(h_A, min=-10.0, max=10.0)
        loss_dag = 0.5 * (h_A_clipped ** 2) + self.lambda_l1 * torch.norm(A, p=1)
        if self.lambda_gate > 0.0:
            loss_dag = loss_dag + self.lambda_gate * mediator_gate.mean()
        label_flow = dag_total[self.node_var_slice, self.label_var_slice].mean(dim=1)
        flow_score = self.dag_gate_expander(
            self._normalize_score(label_flow, default_value=0.5).unsqueeze(0)
        ).squeeze(0)
        flow_score = self._normalize_score(flow_score, default_value=0.5)
        return loss_dag + 0.1 * F.mse_loss(mediator_gate, flow_score.detach())

    def dag_label_loss(self, dag_vars, labels, train_idx, criterion, args):
        if dag_vars.numel() == 0 or train_idx.numel() == 0:
            return self.A_feat.new_zeros(())
        label_A = self.get_masked_A()[:self.non_label_var_dim, self.label_var_slice]
        label_logits = torch.matmul(dag_vars, label_A) + self.dag_label_bias
        return self.compute_supervised_loss(
            label_logits[train_idx],
            labels[train_idx],
            criterion,
            args,
        ).mean()

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

    def _degree_pair_feature(self, deg_src, deg_dst, deg_max):
        log_deg_src = torch.log1p(deg_src)
        log_deg_dst = torch.log1p(deg_dst)
        return (torch.maximum(log_deg_src, log_deg_dst) / deg_max.clamp_min(1.0)).unsqueeze(-1)

    def build_edge_feat(self, h_src, h_dst, deg_src, deg_dst, deg_max):
        # Compute only the feature branch requested by edge_feat_mode.  The
        # previous version built mul, diff, concat and degree features every
        # time, even for the default mode='mul'.
        mode = self.edge_feat_mode
        if mode == 'mul':
            return h_src * h_dst
        if mode == 'diff':
            return torch.abs(h_src - h_dst)
        if mode == 'signed_diff':
            return h_src - h_dst
        if mode == 'degree':
            return self._degree_pair_feature(deg_src, deg_dst, deg_max)
        if mode == 'mul_diff':
            signed_diff = h_src - h_dst
            return torch.cat([h_src * h_dst, torch.abs(signed_diff)], dim=-1)
        if mode == 'mul_signed_diff':
            return torch.cat([h_src * h_dst, h_src - h_dst], dim=-1)
        if mode == 'concat':
            return torch.cat([h_src, h_dst], dim=-1)
        if mode == 'concat_diff':
            signed_diff = h_src - h_dst
            return torch.cat([h_src, h_dst, torch.abs(signed_diff)], dim=-1)
        if mode == 'mul_degree':
            return torch.cat([h_src * h_dst, self._degree_pair_feature(deg_src, deg_dst, deg_max)], dim=-1)
        if mode == 'diff_degree':
            return torch.cat([torch.abs(h_src - h_dst), self._degree_pair_feature(deg_src, deg_dst, deg_max)], dim=-1)
        if mode == 'mul_diff_degree':
            signed_diff = h_src - h_dst
            return torch.cat([
                h_src * h_dst,
                torch.abs(signed_diff),
                self._degree_pair_feature(deg_src, deg_dst, deg_max),
            ], dim=-1)
        if mode == 'mul_signed_diff_degree':
            return torch.cat([
                h_src * h_dst,
                h_src - h_dst,
                self._degree_pair_feature(deg_src, deg_dst, deg_max),
            ], dim=-1)
        raise ValueError(f"Unknown edge_feat_mode='{mode}'.")

    def _pair_distance_score(self, h_src, h_dst, deg_src=None, deg_dst=None, deg_max=None):
        mode = self.edge_feat_mode
        if 'diff' in mode:
            return torch.abs(h_src - h_dst).mean(dim=-1)
        if mode == 'degree' and deg_src is not None and deg_dst is not None and deg_max is not None:
            return torch.abs(torch.log1p(deg_src) - torch.log1p(deg_dst)) / deg_max.clamp_min(1.0)
        sim = F.cosine_similarity(h_src, h_dst, dim=-1)
        return 1.0 - sim

    @torch.no_grad()
    def prepare_graph_cache(self, edge_index, num_nodes):
        device_key = str(edge_index.device)
        key = (int(num_nodes), int(edge_index.size(1)), int(edge_index.data_ptr()), device_key, self.backbone_type, bool(self.variant))
        if self._graph_cache_key == key:
            return

        if edge_index.numel() == 0:
            deg = torch.ones(num_nodes, device=edge_index.device, dtype=torch.float32)
        else:
            deg = degree(edge_index[1], num_nodes).to(device=edge_index.device, dtype=torch.float32).clamp_min(1.0)
        self._cached_degree = deg
        self._cached_deg_max = torch.log1p(deg).max().clamp_min(1.0)

        self._cached_gcn_adj = None
        self._cached_variant_adj = None
        self._cached_gat_edge_index = None
        if self.backbone_type == 'gcn' and not self.variant:
            self._cached_gcn_adj = build_gcn_norm_adj(edge_index, num_nodes, dtype=torch.float32)
        elif self.backbone_type == 'gcn' and self.variant:
            self._cached_variant_adj = torch.sparse_coo_tensor(
                edge_index,
                torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32),
                size=(num_nodes, num_nodes),
            ).coalesce()
        elif self.backbone_type == 'gat':
            att_edge_index, _ = remove_self_loops(edge_index)
            self._cached_gat_edge_index, _ = add_self_loops(att_edge_index, num_nodes=num_nodes)

        self._graph_cache_key = key

    def _get_cached_degree(self, h, edge_index):
        self.prepare_graph_cache(edge_index, h.size(0))
        deg = self._cached_degree.to(device=h.device, dtype=h.dtype)
        deg_max = self._cached_deg_max.to(device=h.device, dtype=h.dtype)
        return deg, deg_max

    def compute_local_summary(self, h, edge_index):
        if self.edge_blend <= 0.0 or edge_index.numel() == 0:
            return h.new_zeros(h.size()), None
        src, dst = edge_index
        deg, deg_max = self._get_cached_degree(h, edge_index)
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
        pool_size = max(1, int(self.virtual_sample_pool))
        key = (
            int(num_nodes),
            int(edge_index.size(1)),
            int(edge_index.data_ptr()),
            str(edge_index.device),
            self.virtual_exclude_hops,
            pool_size,
            train_cpu,
            train_labels_cpu,
        )
        if self._virtual_candidate_key == key and self._virtual_candidate_cache is not None:
            return self._virtual_candidate_cache

        class_to_nodes = {}
        for node, label in zip(train_cpu, train_labels_cpu):
            class_to_nodes.setdefault(int(label), []).append(int(node))

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

        # Build a small fixed candidate pool per train node once.  Sampling then
        # becomes a vectorized GPU gather instead of a per-layer Python loop over
        # all train nodes.
        src_pools = []
        dst_nodes = []
        lengths = []
        for pos, node in enumerate(train_cpu):
            label = int(train_labels_cpu[pos])
            ego_nodes = ego_sets[pos]
            candidates = [v for v in class_to_nodes.get(label, []) if v not in ego_nodes]
            if not candidates:
                continue
            cand = torch.tensor(candidates, dtype=torch.long)
            if cand.numel() > pool_size:
                cand = cand[torch.randperm(cand.numel())[:pool_size]]
            length = int(cand.numel())
            if length < pool_size:
                pad = cand[torch.randint(length, (pool_size - length,), dtype=torch.long)]
                cand = torch.cat([cand, pad], dim=0)
            src_pools.append(cand)
            dst_nodes.append(int(node))
            lengths.append(min(length, pool_size))

        if not src_pools:
            cache = {
                'src_pool': torch.empty((0, pool_size), dtype=torch.long, device=edge_index.device),
                'dst_nodes': torch.empty((0,), dtype=torch.long, device=edge_index.device),
                'lengths': torch.empty((0,), dtype=torch.long, device=edge_index.device),
            }
        else:
            cache = {
                'src_pool': torch.stack(src_pools, dim=0).to(edge_index.device, non_blocking=True),
                'dst_nodes': torch.tensor(dst_nodes, dtype=torch.long, device=edge_index.device),
                'lengths': torch.tensor(lengths, dtype=torch.long, device=edge_index.device),
            }

        self._virtual_candidate_key = key
        self._virtual_candidate_cache = cache
        return cache

    def sample_virtual_edge_index(self, edge_index, num_nodes, labels=None, train_idx=None, device=None):
        if (
            not self.training
            or not self.use_virtual_node_enhance
            or self.virtual_blend <= 0.0
            or self.virtual_k <= 0
            or labels is None
            or train_idx is None
            or train_idx.numel() == 0
        ):
            return None

        if device is None:
            device = edge_index.device
        train_idx = train_idx.to(device)
        cache = self._build_virtual_candidate_cache(edge_index, num_nodes, labels, train_idx)
        src_pool = cache['src_pool']
        dst_nodes = cache['dst_nodes']
        lengths = cache['lengths']
        active_count = int(dst_nodes.numel())
        if active_count == 0:
            return None

        k = int(self.virtual_k)
        max_draw = torch.minimum(lengths, torch.full_like(lengths, k))
        draw_mask = torch.arange(k, device=device).unsqueeze(0) < max_draw.unsqueeze(1)
        random_pos = torch.floor(torch.rand(active_count, k, device=device) * lengths.clamp_min(1).unsqueeze(1).float()).long()
        src = src_pool.gather(1, random_pos)
        dst = dst_nodes.unsqueeze(1).expand(-1, k)
        src = src[draw_mask]
        dst = dst[draw_mask]
        if src.numel() == 0:
            return None
        return torch.stack([src, dst], dim=0)

    def sample_layer_virtual_edges(self, edge_index, num_nodes, labels=None, train_idx=None):
        if not self.training or not self.use_virtual_node_enhance or self.virtual_blend <= 0.0 or self.virtual_k <= 0:
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
        self.prepare_graph_cache(edge_index, num_nodes)
        self._prepared_virtual_edge_indices = self.sample_layer_virtual_edges(
            edge_index,
            num_nodes,
            labels=labels,
            train_idx=train_idx,
        )
        return self._prepared_virtual_edge_indices

    def compute_virtual_summary(self, h, edge_index, virtual_edge_index=None):
        if self.virtual_blend <= 0.0 or virtual_edge_index is None or virtual_edge_index.numel() == 0:
            return h.new_zeros(h.size()), None, 0

        src, dst = virtual_edge_index
        deg, deg_max = self._get_cached_degree(h, edge_index)
        h_src = h[src]
        h_dst = h[dst]
        edge_feat = self.build_edge_feat(h_src, h_dst, deg[src], deg[dst], deg_max)
        edge_hidden = F.dropout(self.virtual_pair_encoder(edge_feat), self.dropout, training=self.training)
        learned = self.virtual_score_head(edge_hidden).squeeze(-1)
        prior = self._pair_distance_score(h_src, h_dst, deg[src], deg[dst], deg_max)
        logits = (learned + self.virtual_diff_bias * prior) / self.virtual_score_temp
        alpha = softmax(logits, dst, num_nodes=h.size(0))
        summary = h.new_zeros(h.size())
        summary.index_add_(0, dst, alpha.unsqueeze(-1) * h_src)
        return self.virtual_summary_norm(summary), alpha, int(src.numel())

    def build_shortcut_view(self, x, edge_index):
        # The bias classifier gets its own shallow/raw GNN view instead of the
        # final main representation.  This keeps it strong on local shortcuts
        # while reducing its access to deeper causal semantics.
        h_raw = F.dropout(x, self.dropout, training=self.training)
        h_raw = F.relu(self.shortcut_input_proj(h_raw))
        # Start the shortcut path from one-hop neighbor aggregation.  This
        # removes direct access to the node's own semantic features in the
        # default mode, making the bias classifier focus on local homophily and
        # topology-driven shortcuts.
        h_short = gcn_backbone_conv(h_raw, edge_index, norm_adj=self._cached_gcn_adj)
        for layer in self.shortcut_layers:
            h_short = F.dropout(h_short, self.dropout, training=self.training)
            h_short = F.relu(layer(
                h_short,
                edge_index,
                gcn_adj=self._cached_gcn_adj,
                variant_adj=self._cached_variant_adj,
                gat_edge_index=self._cached_gat_edge_index,
            ))
        h_short = self.shortcut_norm(h_short)
        deg, deg_max = self._get_cached_degree(h_short, edge_index)
        deg_feat = (torch.log1p(deg) / deg_max.clamp_min(1.0)).unsqueeze(-1)
        deg_embed = F.relu(self.shortcut_degree_proj(deg_feat.to(dtype=h_short.dtype)))
        if self.shortcut_view_mode == 'structure_only':
            h_self_view = h_raw.new_zeros(h_raw.size())
            h_short_view = gcn_backbone_conv(deg_embed, edge_index)
        elif self.shortcut_view_mode == 'neighbor_only':
            h_self_view = h_raw.new_zeros(h_raw.size())
            h_short_view = h_short
        else:
            h_self_view = h_raw
            h_short_view = h_short
        return torch.cat([
            h_self_view,
            h_short_view,
            h_self_view * h_short_view,
            torch.abs(h_self_view - h_short_view),
            deg_embed,
        ], dim=-1)

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
        self.prepare_graph_cache(edge_index, x.size(0))
        h = F.dropout(x, self.dropout, training=self.training)
        h = F.relu(self.input_proj(h))
        h_pre_enhance = h
        last_shortcut_view = None
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
            h = F.relu(layer(
                h,
                edge_index,
                gcn_adj=self._cached_gcn_adj,
                variant_adj=self._cached_variant_adj,
                gat_edge_index=self._cached_gat_edge_index,
            ))
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

        last_shortcut_view = self.build_shortcut_view(x, edge_index)
        if last_shortcut_view is None:
            last_shortcut_view = z.new_zeros(z.size(0), self.d * 5)
        # The shortcut encoder is separate from the main GNN, so the bias loss
        # can train it fully without leaking shortcut gradients into z_mediator.
        shortcut_for_bias = last_shortcut_view

        shortcut_base = self.spurious_norm(self.spurious_adapter(shortcut_for_bias))
        node_latent = self.node_dag_proj(z)
        shortcut_latent = self.shortcut_dag_proj(shortcut_base)
        dag_vars = torch.cat([node_latent, shortcut_latent], dim=-1)
        causal_score, pollution_score, mediator_gate, dag_total = self.get_causal_effect_and_mask()
        spurious_gate = 1.0 - mediator_gate

        gate_blend = float(getattr(self, 'current_dag_gate_blend', self.dag_gate_blend))
        mediator_mask = (1.0 - gate_blend) + gate_blend * mediator_gate
        spurious_mask = (1.0 - gate_blend) + gate_blend * spurious_gate

        z_mediator = self.causal_norm(self.causal_adapter(z)) * mediator_mask.unsqueeze(0)
        # Bias branch reads the DAG spurious gate as a teacher signal.  Detach
        # prevents the shortcut classifier from redefining causal dimensions as
        # spurious simply because they predict labels well.
        z_spurious = shortcut_base * spurious_mask.detach().unsqueeze(0)
        z_spurious_causal_view = shortcut_base * mediator_gate.detach().unsqueeze(0)
        z_mediator = F.dropout(z_mediator, self.dropout, training=self.training)
        z_spurious = F.dropout(z_spurious, self.dropout, training=self.training)
        mediator_logits = self.classifier(z_mediator)
        spurious_logits = self.spurious_classifier(z_spurious)
        spurious_causal_logits = self.spurious_classifier(z_spurious_causal_view)
        return {
            'z': z,
            'z_mediator': z_mediator,
            'z_spurious': z_spurious,
            'z_spurious_causal_view': z_spurious_causal_view,
            'shortcut_base': shortcut_base,
            'shortcut_latent': shortcut_latent,
            'dag_vars': dag_vars,
            'mediator_gate': mediator_gate,
            'spurious_gate': spurious_gate,
            'causal_score': causal_score,
            'pollution_score': pollution_score,
            'dag_total': dag_total,
            'mediator_logits': mediator_logits,
            'spurious_logits': spurious_logits,
            'spurious_causal_logits': spurious_causal_logits,
            'shortcut_view': last_shortcut_view,
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

    def blend_logits(self, mediator_logits, fd_logits):
        blend = float(getattr(self, 'current_fd_blend', self.fd_blend))
        return (1.0 - blend) * mediator_logits + blend * fd_logits

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

    @torch.no_grad()
    def compute_bias_failure_weight(self, bias_logits, y, args):
        if self.main_reweight_mode == 'none':
            return bias_logits.new_ones(bias_logits.size(0))

        if args.dataset in ('twitch', 'elliptic'):
            if y.shape[1] == 1 and bias_logits.shape[1] > 1:
                true_label = F.one_hot(y.squeeze().long(), bias_logits.shape[1]).float()
            else:
                true_label = y.float()
            probs = torch.sigmoid(bias_logits)
            bias_conf = (probs * true_label + (1.0 - probs) * (1.0 - true_label)).mean(dim=1)
        else:
            probs = F.softmax(bias_logits, dim=-1)
            bias_conf = probs.gather(1, y.squeeze().long().view(-1, 1)).squeeze(1)

        failure = (1.0 - bias_conf).clamp(0.0, 1.0)
        if self.main_reweight_power != 1.0:
            failure = failure.pow(self.main_reweight_power)
        return self.main_reweight_floor + (1.0 - self.main_reweight_floor) * failure

    def reduce_main_loss(self, per_sample_loss, bias_logits, y, args):
        if self.main_reweight_mode == 'none':
            return per_sample_loss.mean()
        weight = self.compute_bias_failure_weight(bias_logits, y, args).to(
            device=per_sample_loss.device,
            dtype=per_sample_loss.dtype,
        )
        return (per_sample_loss * weight).sum() / weight.sum().clamp_min(1e-6)

    def compute_uniform_loss(self, logits):
        if logits.size(-1) == 1:
            target = torch.full_like(logits, 0.5)
            return F.binary_cross_entropy_with_logits(logits, target)
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
        spu_causal_logits_tr = enc['spurious_causal_logits'][train_idx]
        shortcut_base_tr = enc['shortcut_base'][train_idx]
        shortcut_latent_tr = enc['shortcut_latent'][train_idx]
        med_logits_tr = enc['mediator_logits'][train_idx]
        spu_logits_tr = enc['spurious_logits'][train_idx]

        if self.ctx_grad_ratio >= 1.0:
            ctx_spu = spu_tr
        elif self.ctx_grad_ratio <= 0.0:
            ctx_spu = spu_tr.detach()
        else:
            ctx_spu = self.ctx_grad_ratio * spu_tr + (1.0 - self.ctx_grad_ratio) * spu_tr.detach()
        contexts = self.build_spurious_contexts(ctx_spu, y_tr)
        fd_logits_tr = self.frontdoor_logits_from_contexts(med_tr, contexts)
        final_logits_tr = self.blend_logits(med_logits_tr, fd_logits_tr)

        bias_detached = self.coop_bias_scale * spu_logits_tr.detach()
        main_detached = self.coop_main_scale * final_logits_tr.detach()

        # Cooperative stop-gradient Product-of-Experts losses.  The main/front-door
        # branch receives gradients through final/fd logits while conditioning on
        # what the shortcut classifier already explains.  The shortcut classifier
        # is still trained to classify from local-shortcut views.
        loss_cls_vec = self.compute_supervised_loss(final_logits_tr + bias_detached, y_tr, criterion, args)
        loss_cls = self.reduce_main_loss(loss_cls_vec, spu_logits_tr.detach(), y_tr, args)
        loss_med = self.compute_supervised_loss(med_logits_tr, y_tr, criterion, args).mean()
        loss_fd_vec = self.compute_supervised_loss(fd_logits_tr + bias_detached, y_tr, criterion, args)
        loss_fd = self.reduce_main_loss(loss_fd_vec, spu_logits_tr.detach(), y_tr, args)
        loss_spu = self.compute_supervised_loss(spu_logits_tr, y_tr, criterion, args).mean()
        loss_bias_coop = self.compute_supervised_loss(spu_logits_tr + main_detached, y_tr, criterion, args).mean()
        loss_bias_no_causal = self.compute_uniform_loss(spu_causal_logits_tr)
        bias_latent = self.bias_latent_projector(spu_tr)
        loss_bias_spu_align = F.mse_loss(
            F.normalize(bias_latent, dim=1),
            F.normalize(shortcut_latent_tr.detach(), dim=1),
        )
        spu_recon = self.spu_reconstructor(shortcut_latent_tr)
        loss_spu_rec = F.mse_loss(
            F.normalize(spu_recon, dim=1),
            F.normalize(shortcut_base_tr.detach(), dim=1),
        )
        loss_dag = self.dag_regularization_loss(enc['mediator_gate'], enc['dag_total'])
        loss_dag_label = self.dag_label_loss(enc['dag_vars'], y, train_idx, criterion, args)
        loss_ind = self.compute_independence_loss(med_tr, spu_tr)
        total_loss = (
            loss_cls
            + self.lambda_med * loss_med
            + self.lambda_fd * loss_fd
            + self.lambda_spu * loss_spu
            + self.lambda_bias_coop * loss_bias_coop
            + self.lambda_bias_no_causal * loss_bias_no_causal
            + self.lambda_bias_spu_align * loss_bias_spu_align
            + self.lambda_spu_rec * loss_spu_rec
            + self.lambda_dag * loss_dag
            + self.lambda_dag_label * loss_dag_label
            + self.lambda_ind * loss_ind
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
            'loss_bias_coop': loss_bias_coop,
            'loss_bias_no_causal': loss_bias_no_causal,
            'loss_bias_spu_align': loss_bias_spu_align,
            'loss_spu_rec': loss_spu_rec,
            'loss_dag': loss_dag,
            'loss_dag_label': loss_dag_label,
            'local_gate_mean': zero if local_gate is None else local_gate.mean().detach(),
            'mediator_gate_mean': enc['mediator_gate'].detach().mean(),
            'spurious_gate_mean': enc['spurious_gate'].detach().mean(),
            'causal_score_mean': enc['causal_score'].detach().mean(),
            'pollution_score_mean': enc['pollution_score'].detach().mean(),
            'bias_logit_abs_mean': spu_logits_tr.detach().abs().mean(),
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
