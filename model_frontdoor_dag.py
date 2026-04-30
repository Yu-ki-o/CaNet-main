import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, softmax
from torch_sparse import SparseTensor, matmul


def gcn_backbone_conv(x, edge_index):
    """
    CaNet-style GCN propagation used as the front-door encoder backbone.

    This keeps the same normalization rule as the original CaNet codebase,
    so switching `backbone_type` changes the actual graph encoder instead of
    leaving the front-door model fixed to GraphSAGE.
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
    Single front-door encoder layer with a CaNet-style backbone choice.

    - `gcn`: normalized graph propagation + self-feature concatenation
    - `gat`: single-head attention aggregation with the same flavor as CaNet
    """

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
            raise NotImplementedError(
                f"Front-door DAG backbone_type='{backbone_type}' is not implemented. "
                "Use 'gcn' or 'gat' to match the CaNet-style backbone."
            )

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


class DAGAwareLatentMixer(nn.Module):
    """
    Small token-level DAG mixer for the front-door path.

    Token order is [mediator, spurious, context, label_query]. The attention
    mask allows spurious information to reach the label only through the
    context token, matching the intended front-door path:

        mediator -> label
        spurious -> context -> label
        spurious -/-> label
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
                [1, 0, 0, 0],  # mediator only preserves itself
                [0, 1, 0, 0],  # spurious only preserves itself
                [0, 1, 1, 0],  # context may absorb spurious information
                [1, 0, 1, 1],  # label sees mediator and context, not spurious
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


class GraphFrontDoorDAG(nn.Module):
    """
    Front-door graph model with a DAG over node and edge-summary variables.

    Main idea:
    1) Encode nodes with the GNN backbone.
    2) Build a NodeIGM-style edge semantic summary from endpoint hidden states.
    3) Learn a DAG A over [node hidden dims, edge-summary dims, label dims].
    4) Use DAG-derived label effects and incoming pollution to construct M.
    5) Split node representations into mediator/spurious parts.
    6) Keep the front-door aggregation path by averaging predictions over
       environment-specific spurious contexts.

    Compared with the previous prototype-reconstruction DAG, this version:
    - avoids feeding raw node degree as a feature shortcut,
    - lets edge/neighbor semantics guide mediator discovery through the DAG,
    - learns the DAG jointly with label sufficiency, environment invariance,
      spurious-environment predictability, and front-door consistency.
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

        self.classifier = nn.Linear(self.d, c)
        self.fd_classifier = nn.Linear(self.d, c)
        self.env_classifier = nn.Linear(self.d, self.num_envs)

        self.edge_score_temp = max(1e-3, float(getattr(args, 'edge_score_temp', 1.0)))
        self.edge_pair_encoder = nn.Sequential(
            nn.Linear(self.d * 4, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.edge_score_head = nn.Linear(self.d, 1)
        self.edge_message_head = nn.Linear(self.d, self.d)
        self.edge_summary_norm = nn.LayerNorm(self.d)
        self.node_edge_fuser = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.node_edge_norm = nn.LayerNorm(self.d)

        # DAG variable order: [node hidden dims, edge-summary dims, label dims].
        self.node_var_dim = self.d
        self.edge_var_dim = self.d
        self.label_var_dim = self.c
        self.dag_var_dim = self.node_var_dim + self.edge_var_dim + self.label_var_dim
        self.node_var_slice = slice(0, self.node_var_dim)
        self.edge_var_slice = slice(self.node_var_dim, self.node_var_dim + self.edge_var_dim)
        self.label_var_slice = slice(self.node_var_dim + self.edge_var_dim, self.dag_var_dim)
        self.non_label_var_dim = self.node_var_dim + self.edge_var_dim

        self.A_feat = Parameter(torch.zeros(self.dag_var_dim, self.dag_var_dim))
        # Learnable base score per hidden dimension; DAG structure refines it.
        self.gate_base = Parameter(torch.zeros(self.d))
        self.sem_reconstructor = nn.Sequential(
            nn.Linear(self.non_label_var_dim, self.non_label_var_dim),
            nn.ReLU(),
            nn.Linear(self.non_label_var_dim, self.non_label_var_dim),
        )
        self.dag_label_bias = Parameter(torch.zeros(self.c))

        # Front-door fusion: mediator + context -> intervened representation.
        self.fd_fuser = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.fd_norm = nn.LayerNorm(self.d)
        self.use_dag_mixer = getattr(args, 'use_dag_mixer', True)
        self.dag_mixer = DAGAwareLatentMixer(
            self.d,
            num_heads=getattr(args, 'dag_mixer_heads', 1),
            num_layers=getattr(args, 'dag_mixer_layers', 2),
            dropout=getattr(args, 'dropout', 0.0),
        )

        self.dropout = getattr(args, 'dropout', 0.0)
        self.gamma = getattr(args, 'gamma', 0.99)
        self.fd_blend = getattr(args, 'fd_blend', 0.5)
        self.fd_sample_k = max(0, int(getattr(args, 'K', 0)))
        self.context_sample_seed = int(getattr(args, 'seed', 0))
        self.proto_aug_k = max(0, int(getattr(args, 'proto_aug_k', 0)))
        self.proto_mix_alpha = max(1e-3, float(getattr(args, 'proto_mix_alpha', 1.0)))

        self.lambda_l1 = getattr(args, 'lambda_l1', 1e-5)
        self.lambda_dag = getattr(args, 'lambda_dag', 0.1)
        self.lambda_med = getattr(args, 'lambda_med', 0.5)
        self.lambda_spu = getattr(args, 'lambda_spu', 0.1)
        self.lambda_fd = getattr(args, 'lambda_fd', 0.5)
        self.lambda_fd_aug = getattr(args, 'lambda_fd_aug', 0.5)
        self.lambda_var = getattr(args, 'lambda_var', 0.05)
        self.lambda_ind = getattr(args, 'lambda_ind', 0.1)
        self.lambda_env = getattr(args, 'lambda_env', 0.1)
        self.lambda_inv = getattr(args, 'lambda_inv', 0.1)
        self.lambda_gate = getattr(args, 'lambda_gate', 0.0)
        self.lambda_sem = getattr(args, 'lambda_sem', 0.05)
        self.lambda_dag_degree = getattr(args, 'lambda_dag_degree', 0.0)

        self.mediator_temp = getattr(args, 'mediator_temp', 8.0)
        self.mediator_threshold = getattr(args, 'mediator_threshold', 0.5)
        self.low_temp = getattr(args, 'low_temp', 8.0)
        self.low_threshold = getattr(args, 'low_threshold', 0.35)
        self.pollution_coeff = getattr(args, 'pollution_coeff', 1.0)
        self.edge_pollution_coeff = getattr(args, 'edge_pollution_coeff', 0.5)

        # Only keep env-level spurious prototypes for front-door contexts.
        self.register_buffer('proto_spu_env', torch.zeros(self.num_envs, self.d))
        self.register_buffer('proto_spu_env_valid', torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer('dag_allowed_mask', self.build_dag_allowed_mask())
        self._last_node_degree_signal = None

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
        self.edge_message_head.reset_parameters()
        self.edge_summary_norm.reset_parameters()
        self._reset_module_parameters(self.node_edge_fuser)
        nn.init.zeros_(self.node_edge_fuser[-1].weight)
        nn.init.zeros_(self.node_edge_fuser[-1].bias)
        self.node_edge_norm.reset_parameters()
        for module in self.fd_fuser:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.fd_norm.reset_parameters()
        self.dag_mixer.reset_parameters()
        nn.init.uniform_(self.A_feat, -0.01, 0.01)
        nn.init.zeros_(self.gate_base)
        self._reset_module_parameters(self.sem_reconstructor)
        nn.init.zeros_(self.sem_reconstructor[-1].weight)
        nn.init.zeros_(self.sem_reconstructor[-1].bias)
        nn.init.zeros_(self.dag_label_bias)
        self.proto_spu_env.zero_()
        self.proto_spu_env_valid.zero_()
        self._last_node_degree_signal = None

    def _reset_module_parameters(self, module):
        for sub_module in module.modules():
            if sub_module is module:
                continue
            if hasattr(sub_module, 'reset_parameters'):
                sub_module.reset_parameters()

    def build_dag_allowed_mask(self):
        allowed = torch.ones(self.dag_var_dim, self.dag_var_dim, dtype=torch.bool)
        allowed.fill_diagonal_(False)
        # Label dimensions are supervised sink nodes: features may point to labels,
        # but labels must not become parents of hidden/edge variables.
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
        """
        Use the node/edge/label DAG to produce mediator scores.

        - label_effect: total effect from a hidden dimension to label sinks.
        - pollution_score: incoming structural pressure from other variables,
          especially edge-summary dimensions that may encode environmental
          shortcuts.
        - gate_base: lets the task losses directly refine mediator selection,
          while the DAG acts as a structured prior.
        """
        A = self.get_masked_A()
        A_sq = A * A
        C_tot = torch.matrix_exp(A_sq)
        eye = torch.eye(self.dag_var_dim, device=C_tot.device, dtype=C_tot.dtype)
        C_flow = C_tot - eye

        node_to_label = C_flow[self.node_var_slice, self.label_var_slice]
        label_effect = self._normalize_score(node_to_label.mean(dim=1), default_value=0.5)

        non_label_flow = C_flow[:self.non_label_var_dim, :self.non_label_var_dim]
        incoming_score = non_label_flow[:, self.node_var_slice].mean(dim=0)
        edge_incoming = C_flow[self.edge_var_slice, self.node_var_slice].mean(dim=0)

        low_weight = torch.sigmoid(self.low_temp * (self.low_threshold - label_effect))
        low_weight = low_weight / low_weight.sum().clamp_min(1e-8)
        node_symmetric_flow = 0.5 * (
            C_flow[self.node_var_slice, self.node_var_slice]
            + C_flow[self.node_var_slice, self.node_var_slice].t()
        )
        low_score_coupling = torch.matmul(node_symmetric_flow, low_weight)
        pollution_score = incoming_score + self.edge_pollution_coeff * edge_incoming + low_score_coupling
        pollution_score = self._normalize_score(pollution_score, default_value=0.0)

        base_score = torch.sigmoid(self.gate_base)
        causal_score = self._normalize_score(base_score + label_effect, default_value=0.5)
        mediator_logit = causal_score - self.pollution_coeff * pollution_score - self.mediator_threshold
        mediator_gate = torch.sigmoid(self.mediator_temp * mediator_logit)
        return causal_score, pollution_score, mediator_gate, C_tot

    def compute_edge_semantic_summary(self, h, edge_index, training=False):
        if edge_index.numel() == 0:
            self._last_node_degree_signal = h.new_zeros(h.size(0))
            return h.new_zeros(h.size())

        src, dst = edge_index
        edge_feat = torch.cat(
            [
                h[src],
                h[dst],
                h[src] * h[dst],
                torch.abs(h[src] - h[dst]),
            ],
            dim=-1,
        )
        edge_hidden = self.edge_pair_encoder(edge_feat)
        edge_hidden = F.dropout(edge_hidden, self.dropout, training=training)
        edge_logits = self.edge_score_head(edge_hidden).squeeze(-1) / self.edge_score_temp
        edge_alpha = softmax(edge_logits, dst, num_nodes=h.size(0))
        edge_msg = self.edge_message_head(edge_hidden)

        edge_summary = h.new_zeros(h.size())
        edge_summary.index_add_(0, dst, edge_alpha.unsqueeze(-1) * edge_msg)
        edge_summary = self.edge_summary_norm(edge_summary)

        deg = degree(dst, h.size(0)).to(device=h.device, dtype=h.dtype)
        deg = torch.log1p(deg)
        deg = deg / deg.max().clamp_min(1.0)
        self._last_node_degree_signal = deg.detach()
        return edge_summary

    def fuse_node_edge_representation(self, h, edge_summary, training=False):
        fuse_input = torch.cat([h, edge_summary, h * edge_summary], dim=-1)
        edge_delta = self.node_edge_fuser(fuse_input)
        edge_delta = F.dropout(edge_delta, self.dropout, training=training)
        return self.node_edge_norm(h + edge_delta)

    def encode_representation(self, x, edge_index, training=False):
        x = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(self.input_proj(x))
        for layer in self.backbone_layers:
            h = F.dropout(h, self.dropout, training=training)
            h = self.act_fn(layer(h, edge_index))

        edge_summary = self.compute_edge_semantic_summary(h, edge_index, training=training)
        z = self.fuse_node_edge_representation(h, edge_summary, training=training)
        dag_vars = torch.cat([z, edge_summary], dim=-1)
        causal_score, pollution_score, mediator_gate, dag_total = self.get_causal_effect_and_mask()
        z_mediator = F.dropout(z * mediator_gate.unsqueeze(0), self.dropout, training=training)
        z_spurious = F.dropout(z * (1.0 - mediator_gate).unsqueeze(0), self.dropout, training=training)
        mediator_logits = self.classifier(z_mediator)
        return (
            z,
            edge_summary,
            dag_vars,
            z_mediator,
            z_spurious,
            mediator_logits,
            mediator_gate,
            causal_score,
            pollution_score,
            dag_total,
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
        """
        Approximate the front-door intervention with K diverse contexts.

        During training we randomly sample K environments to mimic the paper's
        stochastic diversity augmentation. During evaluation we keep the subset
        deterministic so validation / test metrics stay stable across calls.
        """
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

    def augment_frontdoor_contexts(self, contexts, training=False):
        if (
            not training
            or self.proto_aug_k <= 0
            or contexts is None
            or contexts.size(0) < 2
        ):
            return contexts, 0

        contexts = contexts.detach()
        num_contexts = contexts.size(0)
        mix_count = self.proto_aug_k

        idx_a = torch.randint(num_contexts, (mix_count,), device=contexts.device)
        idx_b = torch.randint(num_contexts - 1, (mix_count,), device=contexts.device)
        idx_b = idx_b + (idx_b >= idx_a).long()

        beta_dist = torch.distributions.Beta(
            contexts.new_tensor(self.proto_mix_alpha),
            contexts.new_tensor(self.proto_mix_alpha),
        )
        mix_weight = beta_dist.sample((mix_count, 1)).to(contexts.device)
        mixed_contexts = (
            mix_weight * contexts.index_select(0, idx_a)
            + (1.0 - mix_weight) * contexts.index_select(0, idx_b)
        )
        mixed_contexts = F.normalize(mixed_contexts, dim=1)
        return torch.cat([contexts, mixed_contexts.detach()], dim=0), mix_count

    def frontdoor_logits_from_contexts(self, z_mediator, z_spurious, contexts):
        base_logits = self.fd_classifier(z_mediator)
        if contexts is None or contexts.size(0) == 0:
            return base_logits, None

        num_contexts = contexts.size(0)
        mediator_expand = z_mediator.unsqueeze(1).expand(-1, num_contexts, -1)
        spurious_expand = z_spurious.unsqueeze(1).expand(-1, num_contexts, -1)
        context_expand = contexts.unsqueeze(0).expand(z_mediator.size(0), -1, -1)

        if self.use_dag_mixer:
            fused = self.dag_mixer(
                mediator_expand.reshape(-1, self.d),
                spurious_expand.reshape(-1, self.d),
                context_expand.reshape(-1, self.d),
            ).view(z_mediator.size(0), num_contexts, self.d)
            fused = self.fd_norm(fused + mediator_expand)
        else:
            fused_input = torch.cat([mediator_expand, context_expand], dim=-1)
            fused = self.fd_fuser(fused_input.reshape(-1, self.d * 2)).view(z_mediator.size(0), num_contexts, self.d)
            fused = self.fd_norm(fused + mediator_expand)

        logits_stack = self.fd_classifier(fused.reshape(-1, self.d)).view(z_mediator.size(0), num_contexts, self.c)
        fd_logits = logits_stack.mean(dim=1)
        return fd_logits, logits_stack

    def blend_logits(self, med_logits, fd_logits):
        if fd_logits is None:
            return med_logits
        return (1.0 - self.fd_blend) * med_logits + self.fd_blend * fd_logits

    def forward(self, x, edge_index, training=False):
        (
            z,
            edge_summary,
            dag_vars,
            z_mediator,
            z_spurious,
            mediator_logits,
            mediator_gate,
            causal_score,
            pollution_score,
            dag_total,
        ) = self.encode_representation(x, edge_index, training=training)

        contexts = self.sample_frontdoor_contexts(
            self.get_frontdoor_contexts(),
            training=training,
        )
        fd_logits, fd_stack = self.frontdoor_logits_from_contexts(z_mediator, z_spurious, contexts)
        logits = self.blend_logits(mediator_logits, fd_logits)

        if training:
            return (
                logits,
                z,
                edge_summary,
                dag_vars,
                z_mediator,
                z_spurious,
                mediator_gate,
                causal_score,
                pollution_score,
                dag_total,
                mediator_logits,
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

    def compute_uniform_loss(self, logits):
        if logits.size(-1) <= 1:
            return logits.new_zeros(())
        log_probs = F.log_softmax(logits, dim=-1)
        uniform = torch.full_like(log_probs, 1.0 / logits.size(-1))
        return F.kl_div(log_probs, uniform, reduction='batchmean')

    def compute_env_uniform_loss(self, logits):
        if logits.size(-1) <= 1:
            return logits.new_zeros(())
        log_probs = F.log_softmax(logits, dim=-1)
        uniform = torch.full_like(log_probs, 1.0 / logits.size(-1))
        return F.kl_div(log_probs, uniform, reduction='batchmean')

    def compute_independence_loss(self, z_mediator, z_spurious):
        if z_mediator.numel() == 0:
            return self.A_feat.new_zeros(())
        z_med = F.normalize(z_mediator, dim=1)
        z_spu = F.normalize(z_spurious, dim=1)
        corr = (z_med * z_spu).sum(dim=1)
        return 0.5 * (corr ** 2).mean()

    def compute_frontdoor_variance_loss(self, logits_stack):
        if logits_stack is None or logits_stack.size(1) <= 1:
            return self.A_feat.new_zeros(())
        probs = torch.softmax(logits_stack, dim=-1)
        return probs.var(dim=1, unbiased=False).mean()

    def compute_context_supervised_loss(self, logits_stack, y, criterion, args):
        if logits_stack is None or logits_stack.size(1) == 0:
            return self.A_feat.new_zeros(())
        num_nodes, num_contexts, _ = logits_stack.shape
        logits_flat = logits_stack.reshape(num_nodes * num_contexts, self.c)
        y_flat = y.repeat_interleave(num_contexts, dim=0)
        return self.compute_supervised_loss(logits_flat, y_flat, criterion, args).mean()

    def compute_env_invariance_loss(self, logits, envs):
        if envs is None or envs.numel() == 0 or self.num_envs <= 1:
            return self.A_feat.new_zeros(())
        probs = torch.softmax(logits, dim=-1)
        global_mean = probs.mean(dim=0)
        env_values = envs.squeeze().long()
        loss = probs.new_zeros(())
        count = 0
        for env_idx in range(self.num_envs):
            mask = env_values == env_idx
            if mask.any():
                loss = loss + F.mse_loss(probs[mask].mean(dim=0), global_mean)
                count += 1
        if count == 0:
            return probs.new_zeros(())
        return loss / count

    def dag_regularization_loss(self, mediator_gate, dag_total):
        A = self.get_masked_A()
        A_sq = A * A
        h_A = torch.trace(torch.matrix_exp(A_sq)) - self.dag_var_dim
        h_A_clipped = torch.clamp(h_A, min=-10.0, max=10.0)
        loss_dag = 0.5 * (h_A_clipped ** 2) + self.lambda_l1 * torch.norm(A, p=1)

        # Optional soft sparsity on mediator gate to avoid trivial all-ones masks.
        if self.lambda_gate > 0.0:
            loss_dag = loss_dag + self.lambda_gate * mediator_gate.mean()

        # Mild consistency term: dimensions with strong total effect on labels
        # should align with selected mediators.
        label_flow = dag_total[self.node_var_slice, self.label_var_slice].mean(dim=1)
        flow_score = self._normalize_score(label_flow, default_value=0.5)
        loss_dag = loss_dag + 0.1 * F.mse_loss(mediator_gate, flow_score.detach())
        return loss_dag

    def dag_semantic_loss(self, dag_vars, labels, train_idx, criterion, args):
        if dag_vars.numel() == 0 or train_idx.numel() == 0:
            return self.A_feat.new_zeros(())

        A = self.get_masked_A()
        A_non_label = A[:self.non_label_var_dim, :self.non_label_var_dim]
        parent_signal = torch.matmul(dag_vars, A_non_label)
        recon = self.sem_reconstructor(parent_signal)
        recon_loss = F.mse_loss(
            F.normalize(recon[train_idx], dim=0),
            F.normalize(dag_vars[train_idx].detach(), dim=0),
        )

        label_A = A[:self.non_label_var_dim, self.label_var_slice]
        label_logits = torch.matmul(dag_vars, label_A) + self.dag_label_bias
        label_loss = self.compute_supervised_loss(
            label_logits[train_idx],
            labels[train_idx],
            criterion,
            args,
        ).mean()
        return recon_loss + label_loss

    def compute_dag_degree_loss(self, z_mediator, train_idx):
        if (
            self._last_node_degree_signal is None
            or z_mediator.numel() == 0
            or train_idx.numel() == 0
        ):
            return self.A_feat.new_zeros(())
        med_strength = z_mediator.norm(dim=1)
        degree_signal = self._last_node_degree_signal.to(device=z_mediator.device, dtype=z_mediator.dtype)
        med_strength = med_strength[train_idx]
        degree_signal = degree_signal[train_idx]
        med_strength = (med_strength - med_strength.mean()) / med_strength.std(unbiased=False).clamp_min(1e-4)
        degree_signal = (degree_signal - degree_signal.mean()) / degree_signal.std(unbiased=False).clamp_min(1e-4)
        return (med_strength * degree_signal).mean().pow(2)

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

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx
        envs = data.env if hasattr(data, 'env') else None

        (
            _,
            _,
            _,
            dag_vars_all,
            z_mediator_all,
            z_spurious_all,
            mediator_gate,
            causal_score,
            pollution_score,
            dag_total,
            mediator_logits_all,
            _,
            _,
        ) = self.forward(x, edge_index, training=True)

        y_tr = y[train_idx]
        med_tr = z_mediator_all[train_idx]
        spu_tr = z_spurious_all[train_idx]
        env_tr = envs[train_idx] if envs is not None else None
        mediator_logits_tr = mediator_logits_all[train_idx]

        contexts = self.sample_frontdoor_contexts(
            self.get_frontdoor_contexts(spu_tr, env_tr),
            training=True,
        )
        num_base_contexts = 0 if contexts is None else int(contexts.size(0))
        contexts, num_mixed_contexts = self.augment_frontdoor_contexts(
            contexts,
            training=True,
        )
        fd_logits_tr, fd_stack_tr = self.frontdoor_logits_from_contexts(med_tr, spu_tr, contexts)
        final_logits_tr = self.blend_logits(mediator_logits_tr, fd_logits_tr)

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_med = self.compute_supervised_loss(mediator_logits_tr, y_tr, criterion, args).mean()
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        fd_aug_stack_tr = None
        if fd_stack_tr is not None and num_mixed_contexts > 0:
            fd_aug_stack_tr = fd_stack_tr[:, num_base_contexts:, :]
        loss_fd_aug = self.compute_context_supervised_loss(fd_aug_stack_tr, y_tr, criterion, args)
        loss_var = self.compute_frontdoor_variance_loss(fd_stack_tr)
        loss_ind = self.compute_independence_loss(med_tr, spu_tr)
        loss_dag = self.dag_regularization_loss(mediator_gate, dag_total)
        loss_sem = self.dag_semantic_loss(dag_vars_all, y, train_idx, criterion, args)
        loss_degree = self.compute_dag_degree_loss(z_mediator_all, train_idx)

        # Environment-related losses: mediator should be invariant, spurious should capture env.
        if env_tr is not None and env_tr.numel() > 0 and self.num_envs > 1:
            env_targets = env_tr.squeeze().long()
            env_logits_med = self.env_classifier(med_tr)
            env_logits_spu = self.env_classifier(spu_tr)
            loss_env_med = self.compute_env_uniform_loss(env_logits_med)
            loss_spu = F.cross_entropy(env_logits_spu, env_targets)
            loss_inv = self.compute_env_invariance_loss(final_logits_tr, env_tr)
        else:
            env_logits_med = None
            env_logits_spu = None
            loss_env_med = self.A_feat.new_zeros(())
            # fallback: keep spurious branch uninformative wrt class if env labels absent.
            loss_spu = self.compute_uniform_loss(self.classifier(spu_tr))
            loss_inv = self.A_feat.new_zeros(())

        total_loss = (
            loss_cls
            + self.lambda_med * loss_med
            + self.lambda_fd * loss_fd
            + self.lambda_fd_aug * loss_fd_aug
            + self.lambda_var * loss_var
            + self.lambda_ind * loss_ind
            + self.lambda_dag * loss_dag
            + self.lambda_spu * loss_spu
            + self.lambda_env * loss_env_med
            + self.lambda_inv * loss_inv
            + self.lambda_sem * loss_sem
            + self.lambda_dag_degree * loss_degree
        )

        state_payload = None
        if update_state:
            state_payload = {
                'spu_tr': spu_tr.detach(),
                'env_tr': env_tr.detach() if env_tr is not None else None,
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
            'loss_sem': loss_sem,
            'loss_degree': loss_degree,
            'loss_spu': loss_spu,
            'loss_env_med': loss_env_med,
            'loss_inv': loss_inv,
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'causal_score_mean': causal_score.mean().detach(),
            'pollution_score_mean': pollution_score.mean().detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_mixed_contexts': torch.tensor(float(num_mixed_contexts), device=x.device),
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
