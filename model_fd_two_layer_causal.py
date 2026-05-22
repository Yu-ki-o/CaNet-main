import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import degree, softmax
from torch_sparse import SparseTensor, matmul


def gcn_backbone_conv(x, edge_index):
    """CaNet-style normalized propagation kept for compatibility."""
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros_like(x)
    num_nodes = x.size(0)
    row, col = edge_index
    deg = degree(col, num_nodes).float().clamp_min(1.0)
    deg_in = (1.0 / deg[col]).sqrt()
    deg_out = (1.0 / deg[row]).sqrt()
    value = torch.ones_like(row, dtype=x.dtype) * deg_in * deg_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
    return matmul(adj, x)


class SignedSoftCutLayer(nn.Module):
    """GNN update layer fed by signed soft-cut neighbor summaries."""

    def __init__(self, in_features, out_features, residual=True):
        super().__init__()
        self.out_features = out_features
        self.residual = residual and in_features == out_features
        self.weight = Parameter(torch.FloatTensor(in_features * 2, out_features))
        self.bias = Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.out_features ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, x, signed_summary):
        out = torch.matmul(torch.cat([signed_summary, x], dim=-1), self.weight) + self.bias
        if self.residual:
            out = out + x
        return out


class FrontDoorContextMixer(nn.Module):
    """CIPT-style mixer: keep mediator fixed and inject selected context."""

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
            module.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, mediator, context):
        q = self.query_proj(mediator)
        k = self.key_proj(context)
        v = self.value_proj(context)
        score = (q * k).sum(dim=-1, keepdim=True) / max(float(mediator.size(-1)) ** 0.5, 1e-6)
        attn = torch.sigmoid(score)
        delta = self.out_proj(attn * v)
        delta = F.dropout(delta, self.dropout, training=self.training)
        return self.norm(mediator + delta)


class DAGAwareLatentMixer(nn.Module):
    """
    Token-level front-door mixer with the same DAG mask used by the reviewed
    GMM3 model. Token order is [mediator, spurious, context, label_query].
    """

    def __init__(self, hidden_dim, num_heads=1, num_layers=2, dropout=0.0):
        super().__init__()
        if hidden_dim % max(1, int(num_heads)) != 0:
            num_heads = 1
        self.hidden_dim = hidden_dim
        self.num_heads = max(1, int(num_heads))
        self.num_layers = max(1, int(num_layers))
        self.label_query = Parameter(torch.zeros(1, 1, hidden_dim))

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim,
                self.num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(self.num_layers)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(self.num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(self.num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(self.num_layers)
        ])

        allowed = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 1, 1],
            ],
            dtype=torch.bool,
        )
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
        label_query = self.label_query.expand(z_mediator.size(0), -1, -1)
        tokens = torch.stack([z_mediator, z_spurious, context], dim=1)
        tokens = torch.cat([tokens, label_query], dim=1)
        blocked = self.blocked_attn_mask.to(tokens.device)
        attn_mask = tokens.new_zeros(blocked.shape)
        attn_mask = attn_mask.masked_fill(blocked, -1e9)

        for attn, attn_norm, ffn, ffn_norm in zip(
            self.attn_layers,
            self.attn_norms,
            self.ffn_layers,
            self.ffn_norms,
        ):
            attn_out, _ = attn(tokens, tokens, tokens, attn_mask=attn_mask, need_weights=False)
            tokens = attn_norm(tokens + attn_out)
            tokens = ffn_norm(tokens + ffn(tokens))

        return tokens[:, 3, :]


class GraphSoftCutCIPT(nn.Module):
    """
    DAG-free signed-operator two-layer GNN.

    Main changes compared with the earlier soft gate version:
    1. The relation operator "--" is dataset-selectable via relation_mode.
    2. f1 and f2 both map a relation expression to a signed interval
       [-score_scale, score_scale]. Positive scores enhance a neighbor and
       negative scores suppress/subtract a neighbor.
    3. Layer 1 aggregates with f1. Layer 2 computes f2 and then applies a
       base-dominant signed SoftCut(f1_anchor, f2_dynamic) so a v--t relation
       is not reopened only because v absorbed causal information from u.
    4. Environment context is selected from a deterministic spurious/context
       bank by solving f1(x_origin, context) ~= a_i. There is no GMM sampling.
    """

    def __init__(self, d_in, c, args, device):
        super().__init__()
        self.device = device
        self.d = int(getattr(args, 'hidden_channels', 64))
        self.c = int(c)
        self.num_envs = max(1, int(getattr(args, 'train_env_num', 1)))
        self.num_layers = max(1, int(getattr(args, 'num_layers', 2)))
        self.dropout = float(getattr(args, 'dropout', 0.0))
        self.act_fn = nn.ReLU()

        self.input_proj = nn.Linear(d_in, self.d)
        self.origin_norm = nn.LayerNorm(self.d)
        self.backbone_layers = nn.ModuleList([
            SignedSoftCutLayer(self.d, self.d, residual=True)
            for _ in range(self.num_layers)
        ])

        requested_relation = getattr(args, 'relation_mode', None)
        if requested_relation is None:
            requested_relation = getattr(args, 'edge_feat_mode', 'signed_hadamard')
        # Backward compatible aliases.
        alias = {
            'hadamard': 'signed_hadamard',
            'mul': 'signed_hadamard',
            'stable_qk': 'signed_qk',
            'concat_diff': 'signed_concat_diff',
            'diff': 'signed_diff',
            'degree': 'signed_concat_diff_degree',
        }
        self.relation_mode = alias.get(requested_relation, requested_relation)
        valid_modes = {
            'signed_hadamard',
            'signed_qk',
            'signed_concat_diff',
            'signed_concat_diff_degree',
            'signed_cosine_hadamard',
            'signed_diff',
        }
        if self.relation_mode not in valid_modes:
            self.relation_mode = 'signed_hadamard'

        self.score_scale = max(1e-6, float(getattr(args, 'score_scale', 1.0)))
        self.edge_score_temp = max(1e-3, float(getattr(args, 'edge_score_temp', 1.0)))
        self.edge_blend = max(0.0, float(getattr(args, 'edge_blend', 0.2)))
        self.softcut_lambda = min(max(float(getattr(args, 'softcut_lambda', 0.1)), 0.0), 1.0)
        self.softcut_epsilon = max(0.0, float(getattr(args, 'softcut_epsilon', 0.05)))
        self.softcut_sign_temp = max(1e-3, float(getattr(args, 'softcut_sign_temp', 5.0)))
        self.softcut_strength_temp = max(1e-3, float(getattr(args, 'softcut_strength_temp', 5.0)))
        self.softcut_margin = max(0.0, float(getattr(args, 'softcut_margin', 0.0)))
        self.softcut_env_scale = max(0.0, float(getattr(args, 'softcut_env_scale', 0.5)))
        self.operator_margin = max(0.0, float(getattr(args, 'operator_margin', 0.1)))
        self.twohop_sample_size = max(0, int(getattr(args, 'twohop_sample_size', 512)))
        self.use_node_enhance = bool(getattr(args, 'use_node_enhance', True))
        self.use_dag_module = bool(getattr(args, 'use_dag_module', False))
        self.use_dag_mixer = bool(getattr(args, 'use_dag_mixer', True))

        # Relation projections used by the relation expression "--".
        self.rel_src_proj = nn.Linear(self.d, self.d, bias=False)
        self.rel_dst_proj = nn.Linear(self.d, self.d, bias=False)
        self.rel_context_proj = nn.Linear(self.d, self.d, bias=False)
        relation_dim = self._get_relation_dim(self.relation_mode)
        self.f1 = nn.Sequential(
            nn.Linear(relation_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, 1),
        )
        self.f2 = nn.Sequential(
            nn.Linear(relation_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, 1),
        )
        # f3 is kept as an alias-style solver score, but context matching uses f1
        # as requested: solve f1(x_origin, context) ~= a_i.
        self.f3 = nn.Sequential(
            nn.Linear(relation_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, 1),
        )

        self.value_proj_l1 = nn.Linear(self.d, self.d, bias=False)
        self.value_proj_l2 = nn.Linear(self.d, self.d, bias=False)
        self.env_value_proj = nn.Linear(self.d, self.d, bias=False)
        self.edge_summary_norm = nn.LayerNorm(self.d)
        self.noise_summary_norm = nn.LayerNorm(self.d)
        self.edge_context_norm = nn.LayerNorm(self.d)

        requested_dag_latent_dim = int(getattr(args, 'dag_latent_dim', min(16, self.d)))
        self.dag_latent_dim = max(1, min(requested_dag_latent_dim, self.d))
        self.node_dag_proj = nn.Sequential(
            nn.Linear(self.d, self.dag_latent_dim),
            nn.LayerNorm(self.dag_latent_dim),
            nn.Tanh(),
        )
        self.edge_dag_proj = nn.Sequential(
            nn.Linear(self.d, self.dag_latent_dim),
            nn.LayerNorm(self.dag_latent_dim),
            nn.Tanh(),
        )
        self.dag_gate_expander = nn.Linear(self.dag_latent_dim, self.d)
        self.node_var_dim = self.dag_latent_dim
        self.edge_var_dim = self.dag_latent_dim
        self.label_var_dim = self.c
        self.dag_var_dim = self.node_var_dim + self.edge_var_dim + self.label_var_dim
        self.node_var_slice = slice(0, self.node_var_dim)
        self.edge_var_slice = slice(self.node_var_dim, self.node_var_dim + self.edge_var_dim)
        self.label_var_slice = slice(self.node_var_dim + self.edge_var_dim, self.dag_var_dim)
        self.non_label_var_dim = self.node_var_dim + self.edge_var_dim
        self.A_feat = Parameter(torch.zeros(self.dag_var_dim, self.dag_var_dim))
        self.gate_base = Parameter(torch.zeros(self.dag_latent_dim))
        self.dag_label_bias = Parameter(torch.zeros(self.c))
        self.mediator_temp = float(getattr(args, 'mediator_temp', 8.0))
        self.mediator_threshold = float(getattr(args, 'mediator_threshold', 0.5))
        self.low_temp = float(getattr(args, 'low_temp', 8.0))
        self.low_threshold = float(getattr(args, 'low_threshold', 0.35))
        self.pollution_coeff = float(getattr(args, 'pollution_coeff', 1.0))
        self.edge_pollution_coeff = float(getattr(args, 'edge_pollution_coeff', 0.5))
        self.causal_support_coeff = float(getattr(args, 'causal_support_coeff', 0.5))

        self.node_edge_fuser = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.node_edge_norm = nn.LayerNorm(self.d)
        self.node_noise_fuser = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.node_noise_gate = nn.Linear(self.d * 3, 1)
        self.noise_subtract_alpha = max(0.0, float(getattr(args, 'noise_subtract_alpha', 0.05)))
        self.noise_gate_temp = max(1e-3, float(getattr(args, 'noise_gate_temp', 1.0)))
        self.use_neighbor_denoise = bool(getattr(args, 'use_neighbor_denoise', False))

        self.mediator_norm = nn.LayerNorm(self.d)
        self.spurious_norm = nn.LayerNorm(self.d)
        self.spurious_adapter = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.classifier = nn.Linear(self.d, self.c)
        self.fd_classifier = nn.Linear(self.d, self.c)
        self.spurious_classifier = nn.Linear(self.d, self.c)
        self.env_classifier = nn.Linear(self.d, self.num_envs)

        self.context_mixer = FrontDoorContextMixer(self.d, dropout=self.dropout)
        self.dag_mixer = DAGAwareLatentMixer(
            self.d,
            num_heads=getattr(args, 'dag_mixer_heads', 1),
            num_layers=getattr(args, 'dag_mixer_layers', 2),
            dropout=self.dropout,
        )
        self.fd_norm = nn.LayerNorm(self.d)
        self.fd_blend = float(getattr(args, 'fd_blend', 0.5))
        self.fd_sample_k = max(0, int(getattr(args, 'K', 0)))
        self.use_node_conditioned_context = bool(getattr(args, 'use_node_conditioned_context', True))
        self.node_context_topk = max(0, int(getattr(args, 'node_context_topk', self.fd_sample_k)))
        self.context_bank_type = getattr(args, 'context_bank_type', 'prototype')
        if self.context_bank_type not in ('prototype', 'memory', 'prototype_memory'):
            self.context_bank_type = 'prototype'
        self.context_memory_size = max(0, int(getattr(args, 'context_memory_size', 512)))
        self.context_detach = bool(getattr(args, 'context_detach', True))
        self.pseudo_env_balance = float(getattr(args, 'pseudo_env_balance', 1.0))

        self.lambda_med = float(getattr(args, 'lambda_med', 1.0))
        self.lambda_spu = float(getattr(args, 'lambda_spu', 2.0))
        self.lambda_ind = float(getattr(args, 'lambda_ind', 5.0))
        self.lambda_fd = float(getattr(args, 'lambda_fd', 0.5))
        self.lambda_env = float(getattr(args, 'lambda_env', 0.05))
        self.lambda_operator = float(getattr(args, 'lambda_operator', 0.1))
        self.lambda_spu_env = float(getattr(args, 'lambda_spu_env', 0.05))
        self.lambda_var = float(getattr(args, 'lambda_var', 0.0))
        self.lambda_context = float(getattr(args, 'lambda_context', 0.0))
        self.lambda_bismooth = float(getattr(args, 'lambda_bismooth', 0.0))
        self.lambda_bismooth_cls = float(getattr(args, 'lambda_bismooth_cls', 0.0))
        self.lambda_layerwise_gate = float(getattr(args, 'lambda_layerwise_gate', 0.0))
        self.lambda_l1 = float(getattr(args, 'lambda_l1', 1e-5))
        self.lambda_dag = float(getattr(args, 'lambda_dag', 0.0))
        self.lambda_dag_label = float(getattr(args, 'lambda_dag_label', 0.0))
        self.lambda_gate = float(getattr(args, 'lambda_gate', 0.0))

        self._last_operator_loss = None
        self._last_context_solve_loss = None
        self._last_l1_signed_mean = None
        self._last_l2_signed_mean = None
        self._last_softcut_signed_mean = None
        self._last_env_gate_mean = None
        self._last_target_mean = None
        self._last_context_match_mean = None
        self._last_layer_count = None
        self._last_num_bank_contexts = 0
        self.register_buffer('dag_allowed_mask', self.build_dag_allowed_mask())

        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.origin_norm.reset_parameters()
        for layer in self.backbone_layers:
            layer.reset_parameters()
        for module in (self.rel_src_proj, self.rel_dst_proj, self.rel_context_proj,
                       self.value_proj_l1, self.value_proj_l2, self.env_value_proj):
            nn.init.xavier_uniform_(module.weight)
        self._reset_module_parameters(self.f1)
        self._reset_module_parameters(self.f2)
        self._reset_module_parameters(self.f3)
        self.edge_summary_norm.reset_parameters()
        self.noise_summary_norm.reset_parameters()
        self.edge_context_norm.reset_parameters()
        self._reset_module_parameters(self.node_dag_proj)
        self._reset_module_parameters(self.edge_dag_proj)
        self.dag_gate_expander.reset_parameters()
        nn.init.xavier_uniform_(self.dag_gate_expander.weight, gain=0.1)
        nn.init.zeros_(self.dag_gate_expander.bias)
        nn.init.uniform_(self.A_feat, -0.01, 0.01)
        nn.init.zeros_(self.gate_base)
        nn.init.zeros_(self.dag_label_bias)
        self._reset_module_parameters(self.node_edge_fuser)
        nn.init.zeros_(self.node_edge_fuser[-1].weight)
        nn.init.zeros_(self.node_edge_fuser[-1].bias)
        self.node_edge_norm.reset_parameters()
        self._reset_module_parameters(self.node_noise_fuser)
        nn.init.zeros_(self.node_noise_fuser[-1].weight)
        nn.init.zeros_(self.node_noise_fuser[-1].bias)
        self.node_noise_gate.reset_parameters()
        nn.init.zeros_(self.node_noise_gate.weight)
        nn.init.zeros_(self.node_noise_gate.bias)
        self.mediator_norm.reset_parameters()
        self.spurious_norm.reset_parameters()
        self._reset_module_parameters(self.spurious_adapter)
        self.classifier.reset_parameters()
        self.fd_classifier.reset_parameters()
        self.spurious_classifier.reset_parameters()
        self.env_classifier.reset_parameters()
        self.context_mixer.reset_parameters()
        self.dag_mixer.reset_parameters()
        self.fd_norm.reset_parameters()
        self._last_operator_loss = None
        self._last_context_solve_loss = None
        self._last_l1_signed_mean = None
        self._last_l2_signed_mean = None
        self._last_softcut_signed_mean = None
        self._last_env_gate_mean = None
        self._last_target_mean = None
        self._last_context_match_mean = None
        self._last_layer_count = None
        self._last_num_bank_contexts = 0

    def _reset_module_parameters(self, module):
        for sub_module in module.modules():
            if sub_module is module:
                continue
            if hasattr(sub_module, 'reset_parameters'):
                sub_module.reset_parameters()

    def _get_relation_dim(self, mode):
        if mode in ('signed_hadamard', 'signed_qk', 'signed_diff'):
            return self.d
        if mode == 'signed_cosine_hadamard':
            return self.d + 1
        if mode == 'signed_concat_diff':
            return 4 * self.d
        if mode == 'signed_concat_diff_degree':
            return 4 * self.d + 2
        return self.d

    def build_dag_allowed_mask(self):
        allowed = torch.ones(self.dag_var_dim, self.dag_var_dim, dtype=torch.bool)
        allowed.fill_diagonal_(False)
        allowed[self.label_var_slice, :] = False
        return allowed

    def get_masked_A(self):
        return self.A_feat * self.dag_allowed_mask.to(self.A_feat.device, dtype=self.A_feat.dtype)

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
        C_tot = torch.matrix_exp(A_sq)
        eye = torch.eye(self.dag_var_dim, device=C_tot.device, dtype=C_tot.dtype)
        C_flow = C_tot - eye

        node_to_label = C_flow[self.node_var_slice, self.label_var_slice]
        label_effect = self._normalize_score(node_to_label.mean(dim=1), default_value=0.5)

        node_flow = C_flow[self.node_var_slice, self.node_var_slice]
        edge_to_node = C_flow[self.edge_var_slice, self.node_var_slice]
        edge_incoming = edge_to_node.mean(dim=0) if edge_to_node.numel() > 0 else torch.zeros_like(label_effect)

        causal_weight = torch.sigmoid(self.low_temp * (label_effect - self.low_threshold))
        low_weight = 1.0 - causal_weight
        causal_weight_norm = causal_weight / causal_weight.sum().clamp_min(1e-8)
        low_weight_norm = low_weight / low_weight.sum().clamp_min(1e-8)

        causal_support = torch.matmul(node_flow.t(), causal_weight_norm)
        causal_support = self._normalize_score(causal_support, default_value=0.0)
        low_effect_incoming = torch.matmul(node_flow.t(), low_weight_norm)
        pollution_score = low_effect_incoming + self.edge_pollution_coeff * edge_incoming
        pollution_score = self._normalize_score(pollution_score, default_value=0.0)

        base_score = torch.sigmoid(self.gate_base)
        causal_score = self._normalize_score(
            base_score + label_effect + self.causal_support_coeff * causal_support,
            default_value=0.5,
        )
        mediator_logit = causal_score - self.pollution_coeff * pollution_score - self.mediator_threshold
        hidden_mediator_logit = self.dag_gate_expander(mediator_logit.unsqueeze(0)).squeeze(0)
        mediator_gate = torch.sigmoid(self.mediator_temp * hidden_mediator_logit)
        return causal_score, pollution_score, mediator_gate, C_tot

    def build_relation(self, src_repr, dst_repr, src_deg=None, dst_deg=None):
        src = self.rel_src_proj(src_repr)
        dst = self.rel_dst_proj(dst_repr)
        mode = self.relation_mode
        if mode in ('signed_hadamard', 'signed_qk'):
            return dst * src
        if mode == 'signed_diff':
            return torch.abs(dst - src)
        if mode == 'signed_cosine_hadamard':
            cos = F.cosine_similarity(src, dst, dim=-1, eps=1e-8).unsqueeze(-1)
            return torch.cat([dst * src, cos], dim=-1)
        if mode == 'signed_concat_diff':
            return torch.cat([src, dst, torch.abs(src - dst), src * dst], dim=-1)
        if mode == 'signed_concat_diff_degree':
            if src_deg is None:
                src_deg = torch.ones(src.size(0), device=src.device, dtype=src.dtype)
            if dst_deg is None:
                dst_deg = torch.ones(dst.size(0), device=dst.device, dtype=dst.dtype)
            src_deg = torch.log1p(src_deg).unsqueeze(-1)
            dst_deg = torch.log1p(dst_deg).unsqueeze(-1)
            deg_norm = torch.maximum(src_deg, dst_deg).clamp_min(1.0)
            deg_feat = torch.cat([src_deg / deg_norm, dst_deg / deg_norm], dim=-1)
            return torch.cat([src, dst, torch.abs(src - dst), src * dst, deg_feat], dim=-1)
        return dst * src

    def signed_score(self, relation, which='f1'):
        if which == 'f2':
            raw = self.f2(relation).squeeze(-1)
        elif which == 'f3':
            raw = self.f3(relation).squeeze(-1)
        else:
            raw = self.f1(relation).squeeze(-1)
        return self.score_scale * torch.tanh(raw / self.edge_score_temp)

    def pair_score(self, src_repr, dst_repr, which='f1', src_deg=None, dst_deg=None):
        relation = self.build_relation(src_repr, dst_repr, src_deg=src_deg, dst_deg=dst_deg)
        return self.signed_score(relation, which=which)

    def softcut_signed(self, base_score, dynamic_score):
        same_sign = torch.sigmoid(self.softcut_sign_temp * base_score * dynamic_score)
        strong_base = torch.sigmoid(self.softcut_strength_temp * (base_score.abs() - self.softcut_epsilon))
        allow = same_sign * strong_base
        routed = base_score + self.softcut_lambda * allow * (dynamic_score - base_score)
        routed = routed.clamp(-self.score_scale, self.score_scale)
        env_gate = (1.0 - allow) * torch.relu((dynamic_score - base_score).abs() - self.softcut_margin)
        env_gate = self.softcut_env_scale * env_gate
        return routed, env_gate, allow

    def compute_edge_summaries(self, h, edge_index, training=False, anchor=None, layer_idx=0):
        if edge_index is None or edge_index.numel() == 0:
            zero = h.new_zeros(h.size())
            scalar = h.new_zeros(())
            stats = {
                'l1_mean': scalar.detach(), 'l2_mean': scalar.detach(),
                'softcut_mean': scalar.detach(), 'env_mean': scalar.detach(),
                'operator_loss': scalar,
            }
            return zero, zero, stats

        src, dst = edge_index
        num_nodes = h.size(0)
        deg_dst_all = degree(dst, num_nodes).to(device=h.device, dtype=h.dtype).clamp_min(1.0)
        deg_src_all = degree(src, num_nodes).to(device=h.device, dtype=h.dtype).clamp_min(1.0)
        norm = deg_dst_all[dst].pow(-0.5) * deg_src_all[src].pow(-0.5)
        if anchor is None:
            anchor = h.detach()

        h_src, h_dst = h[src], h[dst]
        a_src, a_dst = anchor[src], anchor[dst]
        base_relation = self.build_relation(a_src, a_dst, deg_src_all[src], deg_dst_all[dst])
        base_score = self.signed_score(base_relation, which='f1')

        if layer_idx == 0:
            dynamic_score = base_score
            routed_score = base_score
            env_gate = torch.zeros_like(base_score)
        else:
            dynamic_relation = self.build_relation(h_src, h_dst, deg_src_all[src], deg_dst_all[dst])
            dynamic_score = self.signed_score(dynamic_relation, which='f2')
            routed_score, env_gate, _ = self.softcut_signed(base_score.detach(), dynamic_score)

        value = self.value_proj_l1(h_src) if layer_idx == 0 else self.value_proj_l2(h_src)
        signed_weight = torch.nan_to_num(norm * routed_score, nan=0.0, posinf=0.0, neginf=0.0)
        useful_summary = h.new_zeros(h.size())
        useful_summary.index_add_(0, dst, signed_weight.unsqueeze(-1) * value)
        useful_summary = self.edge_summary_norm(useful_summary)

        noise_weight = torch.nan_to_num(norm * env_gate, nan=0.0, posinf=0.0, neginf=0.0)
        noise_value = self.env_value_proj(h_src)
        noise_summary = h.new_zeros(h.size())
        noise_summary.index_add_(0, dst, noise_weight.unsqueeze(-1) * noise_value)
        noise_summary = self.noise_summary_norm(noise_summary)

        # Edge-level proxy operator loss: when f2 tries to move far from f1,
        # keep the routed value closer to the original f1 anchor.
        operator_loss = ((routed_score - base_score.detach()).pow(2) * env_gate.detach()).mean()
        stats = {
            'l1_mean': base_score.mean().detach(),
            'l2_mean': dynamic_score.mean().detach(),
            'softcut_mean': routed_score.mean().detach(),
            'env_mean': env_gate.mean().detach(),
            'operator_loss': operator_loss,
        }
        return useful_summary, noise_summary, stats

    def fuse_node_edge_representation(self, h, useful_summary, noise_summary=None, training=False):
        fused = h
        if self.use_node_enhance and self.edge_blend > 0.0:
            useful_input = torch.cat([h, useful_summary, h * useful_summary], dim=-1)
            useful_delta = self.node_edge_fuser(useful_input)
            useful_delta = F.dropout(useful_delta, self.dropout, training=training)
            fused = h + self.edge_blend * useful_delta
        if self.use_neighbor_denoise and self.noise_subtract_alpha > 0.0 and noise_summary is not None:
            noise_input = torch.cat([h, noise_summary, h * noise_summary], dim=-1)
            noise_delta = self.node_noise_fuser(noise_input)
            noise_delta = F.dropout(noise_delta, self.dropout, training=training)
            gate_input = torch.cat([h, noise_summary, torch.abs(h - noise_summary)], dim=-1)
            noise_gate = torch.sigmoid(self.node_noise_gate(gate_input) / self.noise_gate_temp)
            fused = fused - self.noise_subtract_alpha * noise_gate * noise_delta
        if self.use_node_enhance or self.use_neighbor_denoise:
            return self.node_edge_norm(fused)
        return fused

    def encode_representation(self, x, edge_index, training=False):
        h_anchor = self.act_fn(self.input_proj(x))
        z_origin = self.origin_norm(h_anchor)
        h = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(self.input_proj(h))

        spurious_accum = None
        useful_accum = None
        op_losses, l1_means, l2_means, soft_means, env_means = [], [], [], [], []
        for layer_idx, layer in enumerate(self.backbone_layers):
            h = F.dropout(h, self.dropout, training=training)
            useful_summary, noise_summary, stats = self.compute_edge_summaries(
                h, edge_index, training=training, anchor=z_origin, layer_idx=layer_idx
            )
            h = self.act_fn(layer(h, useful_summary))
            h = self.fuse_node_edge_representation(h, useful_summary, noise_summary, training=training)
            useful_accum = useful_summary if useful_accum is None else useful_accum + useful_summary
            spurious_accum = noise_summary if spurious_accum is None else spurious_accum + noise_summary
            op_losses.append(stats['operator_loss'])
            l1_means.append(stats['l1_mean'])
            l2_means.append(stats['l2_mean'])
            soft_means.append(stats['softcut_mean'])
            env_means.append(stats['env_mean'])

        if spurious_accum is None:
            spurious_accum = h.new_zeros(h.size())
        if useful_accum is None:
            useful_accum = h.new_zeros(h.size())

        z = self.mediator_norm(h)
        if self.use_dag_module:
            edge_summary = self.edge_summary_norm(useful_accum)
            node_latent = self.node_dag_proj(z)
            edge_latent = self.edge_dag_proj(edge_summary)
            dag_vars = torch.cat([node_latent, edge_latent], dim=-1)
            causal_score, pollution_score, mediator_gate, dag_total = self.get_causal_effect_and_mask()
            z_mediator = z * mediator_gate.unsqueeze(0)
            z_spurious = z * (1.0 - mediator_gate).unsqueeze(0)
            z_mediator = F.dropout(z_mediator, self.dropout, training=training)
            z_spurious = F.dropout(self.spurious_norm(z_spurious), self.dropout, training=training)
        else:
            edge_summary = self.edge_summary_norm(useful_accum)
            dag_vars = z.new_zeros(z.size(0), self.non_label_var_dim)
            causal_score = z.new_ones(self.dag_latent_dim)
            pollution_score = z.new_zeros(self.dag_latent_dim)
            mediator_gate = z.new_ones(self.d)
            dag_total = torch.eye(self.dag_var_dim, device=z.device, dtype=z.dtype)
            z_mediator = F.dropout(z, self.dropout, training=training)
            z_spurious = z.new_zeros(z.size())

        mediator_logits = self.classifier(z_mediator)
        spurious_logits = self.spurious_classifier(z_spurious)

        if op_losses:
            self._last_operator_loss = torch.stack(op_losses).mean()
            self._last_l1_signed_mean = torch.stack(l1_means).mean()
            self._last_l2_signed_mean = torch.stack(l2_means).mean()
            self._last_softcut_signed_mean = torch.stack(soft_means).mean()
            self._last_env_gate_mean = torch.stack(env_means).mean()
            self._last_layer_count = z_mediator.new_tensor(float(len(op_losses)))
        else:
            zero = z_mediator.new_zeros(())
            self._last_operator_loss = zero
            self._last_l1_signed_mean = zero.detach()
            self._last_l2_signed_mean = zero.detach()
            self._last_softcut_signed_mean = zero.detach()
            self._last_env_gate_mean = zero.detach()
            self._last_layer_count = zero.detach()

        return (
            z_origin,
            edge_summary,
            dag_vars,
            z_mediator,
            z_spurious,
            mediator_logits,
            spurious_logits,
            mediator_gate,
            causal_score,
            pollution_score,
            dag_total,
        )

    def compute_pseudo_env_probs(self, z_spurious):
        if self.num_envs <= 1 or z_spurious.numel() == 0:
            return z_spurious.new_ones(z_spurious.size(0), 1)
        return F.softmax(self.env_classifier(z_spurious), dim=-1)

    def get_prototype_contexts(self, z_spurious, env_probs=None):
        if z_spurious is None or z_spurious.numel() == 0:
            return None
        if env_probs is None:
            env_probs = self.compute_pseudo_env_probs(z_spurious)
        env_probs = env_probs.detach().clamp_min(0.0)
        contexts = []
        values = z_spurious.detach() if self.context_detach else z_spurious
        for env_idx in range(env_probs.size(1)):
            weight = env_probs[:, env_idx]
            mass = weight.sum()
            if mass > 1e-6:
                context_vec = (weight.unsqueeze(-1) * values).sum(dim=0) / mass.clamp_min(1e-6)
                contexts.append(F.normalize(context_vec, dim=0))
        if not contexts:
            return None
        return torch.stack(contexts, dim=0)

    def get_memory_contexts(self, z_spurious, train_idx=None, training=False):
        if z_spurious is None or z_spurious.numel() == 0:
            return None
        values = z_spurious.detach() if self.context_detach else z_spurious
        if train_idx is not None and train_idx.numel() > 0:
            values = values[train_idx]
        if values.size(0) == 0:
            return None
        max_size = self.context_memory_size
        if max_size > 0 and values.size(0) > max_size:
            if training:
                perm = torch.randperm(values.size(0), device=values.device)[:max_size]
            else:
                step = max(1, values.size(0) // max_size)
                perm = torch.arange(0, values.size(0), step, device=values.device)[:max_size]
            values = values.index_select(0, perm)
        return F.normalize(values, dim=1)

    def get_context_bank(self, z_spurious, env_probs=None, train_idx=None, training=False):
        banks = []
        if self.context_bank_type in ('prototype', 'prototype_memory'):
            proto = self.get_prototype_contexts(z_spurious, env_probs)
            if proto is not None and proto.numel() > 0:
                banks.append(proto)
        if self.context_bank_type in ('memory', 'prototype_memory'):
            mem = self.get_memory_contexts(z_spurious, train_idx=train_idx, training=training)
            if mem is not None and mem.numel() > 0:
                banks.append(mem)
        if not banks:
            self._last_num_bank_contexts = 0
            return None
        bank = torch.cat(banks, dim=0)
        self._last_num_bank_contexts = int(bank.size(0))
        return bank

    def compute_node_relation_target(self, z_origin, edge_index):
        """
        Compute a_i from the first-layer signed relation between each node and
        its original neighbors. No manual fallback is used. Isolated nodes use
        their self relation f1(u, u), which is still obtained by the operator.
        """
        num_nodes = z_origin.size(0)
        if edge_index is None or edge_index.numel() == 0:
            return self.pair_score(z_origin, z_origin, which='f1')
        src, dst = edge_index
        edge_score = self.pair_score(z_origin[src], z_origin[dst], which='f1')
        weight = softmax(edge_score.abs(), dst, num_nodes=num_nodes)
        target = z_origin.new_zeros(num_nodes)
        mass = z_origin.new_zeros(num_nodes)
        target.index_add_(0, dst, weight * edge_score)
        mass.index_add_(0, dst, weight)
        self_score = self.pair_score(z_origin, z_origin, which='f1')
        has_neighbor = mass > 1e-8
        target = torch.where(has_neighbor, target / mass.clamp_min(1e-8), self_score)
        return target.clamp(-self.score_scale, self.score_scale)

    def solve_contexts_from_bank(self, z_origin, context_bank, target_scores):
        """
        Select K contexts from the bank by solving
            f1(z_origin_i, context) ~= a_i.
        The context target a_i is computed by compute_node_relation_target.
        """
        if (not self.use_node_conditioned_context or context_bank is None or
                context_bank.numel() == 0 or z_origin.numel() == 0):
            self._last_context_solve_loss = z_origin.new_zeros(())
            return context_bank
        if context_bank.dim() == 3:
            self._last_context_solve_loss = z_origin.new_zeros(())
            return context_bank
        num_contexts = context_bank.size(0)
        topk = self.node_context_topk
        if topk <= 0:
            topk = self.fd_sample_k if self.fd_sample_k > 0 else min(1, num_contexts)
        topk = min(max(1, topk), num_contexts)

        # Pair all nodes with all candidate contexts. Candidate bank is kept small
        # by prototypes or context_memory_size.
        node_expand = z_origin.unsqueeze(1).expand(-1, num_contexts, -1).reshape(-1, self.d)
        ctx_expand = context_bank.unsqueeze(0).expand(z_origin.size(0), -1, -1).reshape(-1, self.d)
        scores = self.pair_score(node_expand, ctx_expand, which='f1').view(z_origin.size(0), num_contexts)
        distance = torch.abs(scores - target_scores.unsqueeze(1))
        indices = torch.topk(-distance, k=topk, dim=1).indices
        selected = context_bank.index_select(0, indices.reshape(-1)).view(z_origin.size(0), topk, self.d)
        selected_scores = scores.gather(1, indices)
        self._last_context_solve_loss = (selected_scores - target_scores.unsqueeze(1)).pow(2).mean()
        self._last_target_mean = target_scores.mean().detach()
        self._last_context_match_mean = selected_scores.mean().detach()
        return selected

    def frontdoor_logits_from_contexts(self, z_mediator, contexts, z_spurious=None):
        base_logits = self.fd_classifier(z_mediator)
        if contexts is None or contexts.size(0) == 0:
            return base_logits, None
        if contexts.dim() == 3:
            num_contexts = contexts.size(1)
            med = z_mediator.unsqueeze(1).expand(-1, num_contexts, -1)
            ctx = contexts
        else:
            num_contexts = contexts.size(0)
            med = z_mediator.unsqueeze(1).expand(-1, num_contexts, -1)
            ctx = contexts.unsqueeze(0).expand(z_mediator.size(0), -1, -1)
        if self.use_dag_module and self.use_dag_mixer:
            if z_spurious is None:
                z_spurious = torch.zeros_like(z_mediator)
            spu = z_spurious.unsqueeze(1).expand(-1, num_contexts, -1)
            fused = self.dag_mixer(
                med.reshape(-1, self.d),
                spu.reshape(-1, self.d),
                ctx.reshape(-1, self.d),
            )
            fused = self.fd_norm(fused + med.reshape(-1, self.d))
        else:
            fused = self.context_mixer(med.reshape(-1, self.d), ctx.reshape(-1, self.d))
        fused = fused.view(z_mediator.size(0), num_contexts, self.d)
        logits_stack = self.fd_classifier(fused.reshape(-1, self.d)).view(z_mediator.size(0), num_contexts, self.c)
        return logits_stack.mean(dim=1), logits_stack

    def blend_logits(self, med_logits, fd_logits):
        if fd_logits is None:
            return med_logits
        return (1.0 - self.fd_blend) * med_logits + self.fd_blend * fd_logits

    def forward(self, x, edge_index, training=False, train_idx=None):
        (
            z_origin,
            edge_summary,
            dag_vars,
            z_mediator,
            z_spurious,
            mediator_logits,
            spurious_logits,
            mediator_gate,
            causal_score,
            pollution_score,
            dag_total,
        ) = self.encode_representation(x, edge_index, training=training)
        if self.use_dag_module:
            env_probs = self.compute_pseudo_env_probs(z_spurious)
            context_bank = self.get_context_bank(z_spurious, env_probs, train_idx=train_idx, training=training)
            target_scores = self.compute_node_relation_target(z_origin, edge_index)
            contexts = self.solve_contexts_from_bank(z_origin, context_bank, target_scores)
            fd_logits, fd_stack = self.frontdoor_logits_from_contexts(
                z_mediator,
                contexts,
                z_spurious=z_spurious,
            )
            logits = self.blend_logits(mediator_logits, fd_logits)
        else:
            env_probs = z_mediator.new_ones(z_mediator.size(0), 1)
            contexts = None
            target_scores = z_mediator.new_zeros(z_mediator.size(0))
            fd_logits = None
            fd_stack = None
            logits = mediator_logits
            self._last_num_bank_contexts = 0
            self._last_context_solve_loss = z_mediator.new_zeros(())
            self._last_target_mean = target_scores.mean().detach()
            self._last_context_match_mean = z_mediator.new_zeros(()).detach()
        if training:
            return (
                logits, z_origin, edge_summary, dag_vars, z_mediator,
                z_spurious, mediator_logits, spurious_logits, fd_logits,
                fd_stack, env_probs, contexts, target_scores, mediator_gate,
                causal_score, pollution_score, dag_total,
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

    def compute_uniform_loss(self, logits):
        if logits.size(-1) <= 1:
            return logits.pow(2).mean()
        log_probs = F.log_softmax(logits, dim=-1)
        uniform = torch.full_like(log_probs, 1.0 / logits.size(-1))
        return F.kl_div(log_probs, uniform, reduction='batchmean')

    def compute_env_uniform_loss(self, logits):
        return self.compute_uniform_loss(logits)

    def compute_independence_loss(self, z_mediator, z_spurious):
        if z_mediator.numel() == 0:
            return z_mediator.new_zeros(())
        z_med = F.normalize(z_mediator, dim=1)
        z_spu = F.normalize(z_spurious, dim=1)
        corr = (z_med * z_spu).sum(dim=1)
        cosine_loss = 0.5 * (corr ** 2).mean()
        med_center = z_mediator - z_mediator.mean(dim=0, keepdim=True)
        spu_center = z_spurious - z_spurious.mean(dim=0, keepdim=True)
        denom = max(1, z_mediator.size(0) - 1)
        cov = torch.matmul(med_center.transpose(0, 1), spu_center) / float(denom)
        return cosine_loss + cov.pow(2).mean()

    def compute_frontdoor_variance_loss(self, logits_stack):
        if logits_stack is None or logits_stack.size(1) <= 1:
            return self.classifier.weight.new_zeros(())
        probs = torch.softmax(logits_stack, dim=-1)
        return probs.var(dim=1, unbiased=False).mean()

    def compute_pseudo_env_loss(self, env_logits):
        if env_logits.size(-1) <= 1:
            return env_logits.new_zeros(())
        probs = F.softmax(env_logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        mean_probs = probs.mean(dim=0)
        uniform = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
        balance = F.kl_div(mean_probs.clamp_min(1e-8).log(), uniform, reduction='sum')
        return entropy + self.pseudo_env_balance * balance

    def _sample_twohop(self, edge_index, num_nodes, sample_size, device):
        if edge_index is None or edge_index.numel() == 0 or sample_size <= 0:
            return None
        src, dst = edge_index
        num_edges = src.numel()
        if num_edges == 0:
            return None
        order = torch.argsort(src)
        src_sorted = src[order]
        # Sample first edge u->v.
        idx1 = torch.randint(num_edges, (sample_size,), device=device)
        u = src[idx1]
        v = dst[idx1]
        left = torch.searchsorted(src_sorted, v, right=False)
        right = torch.searchsorted(src_sorted, v, right=True)
        valid = right > left
        if valid.sum() == 0:
            return None
        u = u[valid]
        v = v[valid]
        left = left[valid]
        right = right[valid]
        span = (right - left).clamp_min(1)
        offset = torch.floor(torch.rand(span.size(0), device=device) * span.float()).long()
        idx2_sorted = left + offset
        idx2 = order[idx2_sorted]
        t = dst[idx2]
        return u, v, t

    def compute_twohop_operator_loss(self, z_origin, z_mediator, edge_index):
        sample = self._sample_twohop(edge_index, z_origin.size(0), self.twohop_sample_size, z_origin.device)
        if sample is None:
            return z_origin.new_zeros(())
        u, v, t = sample
        a = self.pair_score(z_origin[u], z_origin[v], which='f1')
        b = self.pair_score(z_origin[v], z_origin[t], which='f1')
        c = self.pair_score(z_mediator[v], z_mediator[t], which='f2')
        c_tilde, env_gate, _ = self.softcut_signed(b.detach(), c)
        v_bar = z_mediator[v] + c_tilde.unsqueeze(-1) * z_mediator[t]
        r = self.pair_score(z_mediator[u], v_bar, which='f2')
        mse = (r - a.detach()).pow(2)
        hinge = torch.relu((r - a.detach()).abs() - (r - b.detach()).abs() + self.operator_margin)
        # Emphasize cases where the second layer tries to reopen or flip v--t.
        reopen = torch.relu((c - b.detach()).abs() - self.softcut_epsilon)
        sign_flip = torch.sigmoid(-self.softcut_sign_temp * b.detach() * c)
        weight = (0.5 + reopen.detach() + sign_flip.detach()).clamp_max(2.0)
        return (weight * (mse + hinge)).mean()

    def dag_regularization_loss(self, mediator_gate, dag_total):
        A = self.get_masked_A()
        A_sq = A * A
        h_A = torch.trace(torch.matrix_exp(A_sq)) - self.dag_var_dim
        h_A_clipped = torch.clamp(h_A, -10.0, 10.0)
        loss_dag = 0.5 * (h_A_clipped ** 2) + self.lambda_l1 * torch.norm(A, p=1)
        if self.lambda_gate > 0.0:
            loss_dag = loss_dag + self.lambda_gate * mediator_gate.mean()
        label_flow = dag_total[self.node_var_slice, self.label_var_slice].mean(dim=1)
        flow_score = self.dag_gate_expander(
            self._normalize_score(label_flow, default_value=0.5).unsqueeze(0)
        ).squeeze(0)
        loss_dag = loss_dag + 0.1 * F.mse_loss(mediator_gate, torch.sigmoid(flow_score).detach())
        return loss_dag

    def dag_label_loss(self, dag_vars, labels, train_idx, criterion, args):
        if dag_vars.numel() == 0 or train_idx.numel() == 0:
            return self.A_feat.new_zeros(())
        A = self.get_masked_A()
        label_A = A[:self.non_label_var_dim, self.label_var_slice]
        label_logits = torch.matmul(dag_vars, label_A) + self.dag_label_bias
        return self.compute_supervised_loss(
            label_logits[train_idx],
            labels[train_idx],
            criterion,
            args,
        ).mean()

    @torch.no_grad()
    def apply_state_update(self, state_payload):
        # No GMM/EMA state in the bank-based version. Kept for main compatibility.
        return

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx
        (
            logits_all, z_origin_all, edge_summary_all, dag_vars_all,
            z_mediator_all, z_spurious_all, mediator_logits_all,
            spurious_logits_all, fd_logits_all, fd_stack_all, env_probs_all,
            contexts, target_scores_all, mediator_gate, causal_score,
            pollution_score, dag_total,
        ) = self.forward(x, edge_index, training=True, train_idx=train_idx)

        y_tr = y[train_idx]
        logits_tr = logits_all[train_idx]
        med_tr = z_mediator_all[train_idx]
        spu_tr = z_spurious_all[train_idx]
        mediator_logits_tr = mediator_logits_all[train_idx]
        spurious_logits_tr = spurious_logits_all[train_idx]
        fd_logits_tr = fd_logits_all[train_idx] if fd_logits_all is not None else None
        fd_stack_tr = fd_stack_all[train_idx] if fd_stack_all is not None else None

        loss_cls = self.compute_supervised_loss(logits_tr, y_tr, criterion, args).mean()
        loss_med = self.compute_supervised_loss(mediator_logits_tr, y_tr, criterion, args).mean()
        if self.use_dag_module and fd_logits_tr is not None:
            loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        else:
            loss_fd = med_tr.new_zeros(())
        loss_spu = self.compute_uniform_loss(spurious_logits_tr) if self.use_dag_module else med_tr.new_zeros(())
        loss_ind = self.compute_independence_loss(med_tr, spu_tr) if self.use_dag_module else med_tr.new_zeros(())
        loss_var = self.compute_frontdoor_variance_loss(fd_stack_tr) if self.use_dag_module else med_tr.new_zeros(())
        if self.use_dag_module and self.num_envs > 1:
            loss_env_med = self.compute_env_uniform_loss(self.env_classifier(med_tr))
            loss_spu_env = self.compute_pseudo_env_loss(self.env_classifier(spu_tr))
        else:
            loss_env_med = med_tr.new_zeros(())
            loss_spu_env = med_tr.new_zeros(())

        edge_operator_loss = self._last_operator_loss if self._last_operator_loss is not None else med_tr.new_zeros(())
        twohop_loss = self.compute_twohop_operator_loss(z_origin_all, z_mediator_all, edge_index)
        loss_operator = edge_operator_loss + twohop_loss
        loss_context = (
            self._last_context_solve_loss
            if self.use_dag_module and self._last_context_solve_loss is not None
            else med_tr.new_zeros(())
        )
        loss_dag = self.dag_regularization_loss(mediator_gate, dag_total) if self.use_dag_module else med_tr.new_zeros(())
        loss_dag_label = (
            self.dag_label_loss(dag_vars_all, y, train_idx, criterion, args)
            if self.use_dag_module
            else med_tr.new_zeros(())
        )

        total_loss = (
            loss_cls
            + self.lambda_med * loss_med
            + self.lambda_fd * loss_fd
            + self.lambda_spu * loss_spu
            + self.lambda_ind * loss_ind
            + self.lambda_env * loss_env_med
            + self.lambda_spu_env * loss_spu_env
            + self.lambda_var * loss_var
            + self.lambda_operator * loss_operator
            + self.lambda_context * loss_context
            + self.lambda_dag * loss_dag
            + self.lambda_dag_label * loss_dag_label
        )

        if contexts is None:
            num_contexts = 0
        elif contexts.dim() == 3:
            num_contexts = int(contexts.size(1))
        else:
            num_contexts = int(contexts.size(0))
        zero = med_tr.new_zeros(())
        l1_mean = self._last_l1_signed_mean if self._last_l1_signed_mean is not None else zero.detach()
        l2_mean = self._last_l2_signed_mean if self._last_l2_signed_mean is not None else zero.detach()
        soft_mean = self._last_softcut_signed_mean if self._last_softcut_signed_mean is not None else zero.detach()
        env_mean = self._last_env_gate_mean if self._last_env_gate_mean is not None else zero.detach()
        target_mean = self._last_target_mean if self._last_target_mean is not None else target_scores_all.mean().detach()
        match_mean = self._last_context_match_mean if self._last_context_match_mean is not None else zero.detach()
        layer_count = self._last_layer_count if self._last_layer_count is not None else zero.detach()

        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_med': loss_med,
            'loss_fd': loss_fd,
            'loss_spu': loss_spu,
            'loss_ind': loss_ind,
            'loss_env_med': loss_env_med,
            'loss_spu_env': loss_spu_env,
            'loss_var': loss_var,
            'loss_operator': loss_operator,
            'loss_context': loss_context,
            'loss_fd_aug': zero,
            'loss_dag': loss_dag,
            'loss_dag_label': loss_dag_label,
            'loss_sem': zero,
            'loss_degree': zero,
            'loss_spu_y': zero,
            'loss_inv': zero,
            'loss_global_env': zero,
            'loss_bismooth': zero,
            'loss_bismooth_cls': zero,
            'loss_layerwise_gate': zero,
            'bismooth_valid_ratio': zero.detach(),
            'layerwise_gate_mean': soft_mean.detach(),
            'layerwise_gate_layers': layer_count.detach(),
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'causal_score_mean': causal_score.mean().detach(),
            'pollution_score_mean': pollution_score.mean().detach(),
            'softcut_inv_gate_mean': soft_mean.detach(),
            'softcut_env_gate_mean': env_mean.detach(),
            'softcut_base_gate_mean': l1_mean.detach(),
            'softcut_dyn_gate_mean': l2_mean.detach(),
            'signed_f1_mean': l1_mean.detach(),
            'signed_f2_mean': l2_mean.detach(),
            'context_target_mean': target_mean.detach(),
            'context_match_mean': match_mean.detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_bank_contexts': torch.tensor(float(self._last_num_bank_contexts), device=x.device),
            'num_mixed_contexts': torch.tensor(0.0, device=x.device),
            'num_gmm_contexts': torch.tensor(0.0, device=x.device),
            'num_global_contexts': torch.tensor(0.0, device=x.device),
            'counterexample_penalty_mean': zero.detach(),
            'counterexample_penalty_batch_mean': zero.detach(),
            'state_payload': None,
        }

    def loss_compute(self, data, criterion, args):
        losses = self.compute_losses(data, criterion, args, update_state=True)
        return (
            losses['total_loss'],
            losses['loss_cls'].item(),
            (self.lambda_ind * losses['loss_ind']).item(),
            (self.lambda_operator * losses['loss_operator']).item(),
            (self.lambda_fd * losses['loss_fd']).item(),
        )


# Backward-compatible alias for older scripts/eval utilities.
GraphFrontDoorDAG = GraphSoftCutCIPT
