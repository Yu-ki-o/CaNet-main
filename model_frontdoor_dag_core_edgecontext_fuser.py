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


class GlobalLinearAttention(nn.Module):
    """
    MLEI-style global linear attention used as the non-local diffusion channel.

    It avoids materializing an N x N attention matrix, so it can be used on
    Arxiv-scale full-batch node classification while still giving every node a
    graph-level information path.
    """

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
    """
    Lightweight AdvDIFFormer-S inspired mixer.

    Each step combines non-local linear attention C with beta-scaled local
    topology propagation V, then projects [z0, Pz0, ..., P^K z0] back to
    hidden_dim. The local operator can be raw GCN propagation or the model's
    learned edge-gated neighbor aggregation.
    """

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
            if local_fn is None:
                h_local = gcn_backbone_conv(h, edge_index)
            else:
                h_local = local_fn(h, edge_index, training=training)
            h = h_global + self.beta * h_local
            h = F.dropout(h, self.dropout, training=training)
            states.append(h)
        mixed = self.proj(torch.cat(states, dim=-1))
        return self.norm(mixed)


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

    DAG-Core variant:
    - avoids feeding raw node degree as a feature shortcut,
    - lets edge/neighbor semantics guide mediator discovery through the DAG,
    - learns the DAG with a compact objective: supervised task loss,
      front-door consistency, acyclicity/sparsity, DAG-to-label supervision,
      spurious pseudo-environment discovery, and mediator env-invariance.
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
                self.global_encoder = GlobalLinearAttention(self.d, dropout=getattr(args, 'dropout', 0.0))
            elif self.global_info_mode == 'advective':
                self.global_encoder = AdvectiveGlobalMixer(
                    self.d,
                    beta=self.global_beta,
                    steps=self.global_steps,
                    dropout=getattr(args, 'dropout', 0.0),
                )
            else:
                raise ValueError(
                    f"Unsupported global_info_mode='{self.global_info_mode}'. "
                    "Use 'linear' or 'advective'."
                )
            self.global_fuse_norm = nn.LayerNorm(self.d)
            self.global_context_proj = nn.Sequential(
                nn.Linear(self.d, self.d),
                nn.ReLU(),
                nn.Linear(self.d, self.d),
            )
            self.global_context_norm = nn.LayerNorm(self.d)
        else:
            self.global_encoder = None
            self.global_fuse_norm = None
            self.global_context_proj = None
            self.global_context_norm = None

        self.classifier = nn.Linear(self.d, c)
        self.fd_classifier = nn.Linear(self.d, c)
        self.env_classifier = nn.Linear(self.d, self.num_envs)

        self.edge_score_temp = max(1e-3, float(getattr(args, 'edge_score_temp', 2.0)))
        self.edge_blend = max(0.0, float(getattr(args, 'edge_blend', 0.2)))
        self.edge_feat_mode = getattr(args, 'edge_feat_mode', 'mul_diff')
        self.dag_input_mode = getattr(args, 'dag_input_mode', 'node_only')
        if self.dag_input_mode not in ('node_only', 'edge_only', 'node_edge'):
            raise ValueError(
                f"Unknown dag_input_mode='{self.dag_input_mode}'. Use one of: "
                "node_only, edge_only, node_edge."
            )
        edge_feat_dim = self._get_edge_feat_dim(self.edge_feat_mode)
        self.edge_pair_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.edge_score_head = nn.Linear(self.d, 1)
        # Optional structural-spurious context path. Low-relevance edges are
        # summarized as an environment factor and fused into z_spurious before
        # pseudo-environment prediction and GMM/front-door context sampling.
        self.use_edge_spu_context = bool(getattr(args, 'use_edge_spu_context', False))
        self.edge_spu_threshold = float(getattr(args, 'edge_spu_threshold', 0.35))
        self.edge_spu_temp = max(1e-3, float(getattr(args, 'edge_spu_temp', 8.0)))
        self.edge_spu_context_alpha = max(0.0, float(getattr(args, 'edge_spu_context_alpha', 0.3)))
        self.edge_spu_msg_mode = getattr(args, 'edge_spu_msg_mode', 'residual')
        if self.edge_spu_msg_mode not in ('residual', 'neighbor'):
            raise ValueError(
                f"Unknown edge_spu_msg_mode='{self.edge_spu_msg_mode}'. "
                "Use 'residual' or 'neighbor'."
            )
        self.edge_spu_norm = nn.LayerNorm(self.d)
        self.edge_spu_context_fuser = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.edge_spu_context_norm = nn.LayerNorm(self.d)
        self.edge_summary_norm = nn.LayerNorm(self.d)
        self.node_edge_fuser = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.node_edge_norm = nn.LayerNorm(self.d)

        # DAG variable order: [node bottleneck dims, edge bottleneck dims, label dims].
        # The DAG is deliberately learned in a compact latent space instead of
        # the full hidden channel space, then expanded back to a hidden-dim gate.
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

        # DAG variable order is controlled by dag_input_mode.
        # - node_only: DAG sees only compact factors of z, where z already fuses edge_summary.
        # - edge_only: DAG sees only compact factors of edge_summary, but prediction still uses z.
        # - node_edge: original version, DAG sees both node and edge-summary factors.
        self.node_var_dim = self.dag_latent_dim
        self.edge_var_dim = self.dag_latent_dim if self.dag_input_mode == 'node_edge' else 0
        self.label_var_dim = self.c
        self.dag_var_dim = self.node_var_dim + self.edge_var_dim + self.label_var_dim
        self.node_var_slice = slice(0, self.node_var_dim)
        self.edge_var_slice = slice(self.node_var_dim, self.node_var_dim + self.edge_var_dim)
        self.label_var_slice = slice(self.node_var_dim + self.edge_var_dim, self.dag_var_dim)
        self.non_label_var_dim = self.node_var_dim + self.edge_var_dim

        self.A_feat = Parameter(torch.zeros(self.dag_var_dim, self.dag_var_dim))
        # Learnable base score per bottleneck dimension; DAG structure refines it.
        self.gate_base = Parameter(torch.zeros(self.dag_latent_dim))
        self.sem_reconstructor = nn.Sequential(
            nn.Linear(self.non_label_var_dim, self.non_label_var_dim),
            nn.ReLU(),
            nn.Linear(self.non_label_var_dim, self.non_label_var_dim),
        )
        self.dag_label_bias = Parameter(torch.zeros(self.c))
        self.pseudo_env_emb = Parameter(torch.zeros(self.num_envs, self.d))
        self.spurious_label_head = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.c),
        )

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
        self.use_spu_gmm = bool(getattr(args, 'use_spu_gmm', True))
        requested_gmm_sample_k = int(getattr(args, 'gmm_sample_k', 0))
        if requested_gmm_sample_k <= 0:
            requested_gmm_sample_k = self.fd_sample_k
        self.gmm_sample_k = max(0, requested_gmm_sample_k)
        self.gmm_min_var = max(1e-6, float(getattr(args, 'gmm_min_var', 1e-4)))
        self.gmm_max_std = max(0.0, float(getattr(args, 'gmm_max_std', 1.0)))
        self.global_context_weight = max(0.0, float(getattr(args, 'global_context_weight', 1.0)))
        self.global_context_detach = bool(getattr(args, 'global_context_detach', True))

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
        self.lambda_global_env = getattr(args, 'lambda_global_env', 0.0)
        self.lambda_gate = getattr(args, 'lambda_gate', 0.0)
        # DAG-Core uses only direct DAG-to-label supervision; the old
        # semantic reconstruction loss is deliberately disabled.
        self.lambda_dag_label = getattr(args, 'lambda_dag_label', getattr(args, 'lambda_sem', 0.05))
        self.lambda_sem = getattr(args, 'lambda_sem', 0.0)
        self.lambda_dag_degree = getattr(args, 'lambda_dag_degree', 0.0)
        self.lambda_spu_y = getattr(args, 'lambda_spu_y', 0.0)
        self.pseudo_env_balance = getattr(args, 'pseudo_env_balance', 1.0)
        self.edge_env_momentum = getattr(args, 'edge_env_momentum', 0.9)

        self.mediator_temp = getattr(args, 'mediator_temp', 8.0)
        self.mediator_threshold = getattr(args, 'mediator_threshold', 0.5)
        self.low_temp = getattr(args, 'low_temp', 8.0)
        self.low_threshold = getattr(args, 'low_threshold', 0.35)
        self.pollution_coeff = getattr(args, 'pollution_coeff', 1.0)
        self.edge_pollution_coeff = getattr(args, 'edge_pollution_coeff', 0.5)
        self.causal_support_coeff = getattr(args, 'causal_support_coeff', 0.5)
        self.counterexample_coeff = max(0.0, float(getattr(args, 'counterexample_coeff', 0.0)))
        self.counterexample_top_frac = min(
            max(float(getattr(args, 'counterexample_top_frac', 0.2)), 0.0),
            1.0,
        )
        self.counterexample_momentum = min(
            max(float(getattr(args, 'counterexample_momentum', 0.9)), 0.0),
            1.0,
        )

        self.register_buffer('gmm_spu_mean', torch.zeros(self.num_envs, self.d))
        self.register_buffer('gmm_spu_var', torch.ones(self.num_envs, self.d))
        self.register_buffer('gmm_spu_valid', torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer('dag_allowed_mask', self.build_dag_allowed_mask())
        self.register_buffer('edge_env_sensitivity', torch.zeros(self.edge_var_dim))
        self.register_buffer('counterexample_penalty', torch.zeros(self.dag_latent_dim))
        self._last_node_degree_signal = None

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
        self.classifier.reset_parameters()
        self.fd_classifier.reset_parameters()
        self.env_classifier.reset_parameters()
        self._reset_module_parameters(self.edge_pair_encoder)
        self.edge_score_head.reset_parameters()
        self.edge_summary_norm.reset_parameters()
        self.edge_spu_norm.reset_parameters()
        self._reset_module_parameters(self.edge_spu_context_fuser)
        nn.init.zeros_(self.edge_spu_context_fuser[-1].weight)
        nn.init.zeros_(self.edge_spu_context_fuser[-1].bias)
        self.edge_spu_context_norm.reset_parameters()
        self._reset_module_parameters(self.node_edge_fuser)
        nn.init.zeros_(self.node_edge_fuser[-1].weight)
        nn.init.zeros_(self.node_edge_fuser[-1].bias)
        self.node_edge_norm.reset_parameters()
        self._reset_module_parameters(self.node_dag_proj)
        self._reset_module_parameters(self.edge_dag_proj)
        self.dag_gate_expander.reset_parameters()
        nn.init.xavier_uniform_(self.dag_gate_expander.weight, gain=0.1)
        nn.init.zeros_(self.dag_gate_expander.bias)
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
        nn.init.normal_(self.pseudo_env_emb, std=0.02)
        self._reset_module_parameters(self.spurious_label_head)
        self.gmm_spu_mean.zero_()
        self.gmm_spu_var.fill_(1.0)
        self.gmm_spu_valid.zero_()
        self.edge_env_sensitivity.zero_()
        self.counterexample_penalty.zero_()
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
        counterexample_penalty = self.counterexample_penalty.to(
            device=C_flow.device,
            dtype=C_flow.dtype,
        )
        if self.counterexample_coeff > 0.0 and counterexample_penalty.sum() > 1e-8:
            counterexample_penalty = self._normalize_score(counterexample_penalty, default_value=0.0)
        else:
            counterexample_penalty = torch.zeros_like(label_effect)

        node_flow = C_flow[self.node_var_slice, self.node_var_slice]
        if self.edge_var_dim > 0:
            edge_to_node = C_flow[self.edge_var_slice, self.node_var_slice]
            edge_sensitivity = self.edge_env_sensitivity.to(device=C_flow.device, dtype=C_flow.dtype)
            if edge_sensitivity.numel() > 0 and edge_sensitivity.sum() > 1e-6:
                edge_weight = edge_sensitivity / edge_sensitivity.sum().clamp_min(1e-8)
                edge_incoming = torch.matmul(edge_to_node.t(), edge_weight)
            else:
                edge_incoming = torch.zeros(self.node_var_dim, device=C_flow.device, dtype=C_flow.dtype)
        else:
            edge_incoming = torch.zeros(self.node_var_dim, device=C_flow.device, dtype=C_flow.dtype)

        causal_weight = torch.sigmoid(self.low_temp * (label_effect - self.low_threshold))
        low_weight = 1.0 - causal_weight
        causal_weight_norm = causal_weight / causal_weight.sum().clamp_min(1e-8)
        low_weight_norm = low_weight / low_weight.sum().clamp_min(1e-8)

        # Incoming support from other high-effect node dimensions should not
        # disqualify a mediator; it often indicates a causal module/chain.
        causal_support = torch.matmul(node_flow.t(), causal_weight_norm)
        causal_support = self._normalize_score(causal_support, default_value=0.0)

        # Incoming pressure from low-effect node dimensions or edge-summary
        # variables is treated as pollution because it is more likely to carry
        # environmental/structural shortcuts.
        low_effect_incoming = torch.matmul(node_flow.t(), low_weight_norm)
        pollution_score = low_effect_incoming + self.edge_pollution_coeff * edge_incoming
        pollution_score = self._normalize_score(pollution_score, default_value=0.0)

        base_score = torch.sigmoid(self.gate_base)
        causal_score = self._normalize_score(
            base_score
            + label_effect
            + self.causal_support_coeff * causal_support
            - self.counterexample_coeff * counterexample_penalty,
            default_value=0.5,
        )
        robust_pollution = pollution_score + self.counterexample_coeff * counterexample_penalty
        robust_pollution = self._normalize_score(robust_pollution, default_value=0.0)
        mediator_logit = causal_score - self.pollution_coeff * robust_pollution - self.mediator_threshold
        hidden_mediator_logit = self.dag_gate_expander(mediator_logit.unsqueeze(0)).squeeze(0)
        mediator_gate = torch.sigmoid(self.mediator_temp * hidden_mediator_logit)
        return causal_score, pollution_score, mediator_gate, C_tot

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
        raise ValueError(
            f"Unknown edge_feat_mode='{mode}'. Use one of: "
            "mul, diff, degree, mul_diff, mul_degree, diff_degree, mul_diff_degree."
        )

    def build_edge_feat(self, h_src, h_dst, deg_src, deg_dst, deg_max):
        mode = self.edge_feat_mode
        mul_feat = h_src * h_dst
        diff_feat = torch.abs(h_src - h_dst)

        log_deg_src = torch.log1p(deg_src)
        log_deg_dst = torch.log1p(deg_dst)
        deg_pair = torch.maximum(log_deg_src, log_deg_dst) / deg_max.clamp_min(1.0)
        deg_pair = deg_pair.unsqueeze(-1)

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
        raise ValueError(
            f"Unknown edge_feat_mode='{mode}'. Use one of: "
            "mul, diff, degree, mul_diff, mul_degree, diff_degree, mul_diff_degree."
        )

    def compute_edge_summaries(self, h, edge_index, training=False):
        if edge_index.numel() == 0:
            zeros = h.new_zeros(h.size())
            return zeros, zeros

        src, dst = edge_index
        deg = degree(dst, h.size(0)).to(device=h.device, dtype=h.dtype).clamp_min(1.0)
        deg_max = torch.log1p(deg).max().clamp_min(1.0)

        h_src = h[src]
        h_dst = h[dst]
        edge_feat = self.build_edge_feat(
            h_src,
            h_dst,
            deg[src],
            deg[dst],
            deg_max,
        )
        edge_hidden = self.edge_pair_encoder(edge_feat)
        edge_hidden = F.dropout(edge_hidden, self.dropout, training=training)
        edge_logits = self.edge_score_head(edge_hidden).squeeze(-1) / self.edge_score_temp
        edge_gate = torch.sigmoid(edge_logits)

        norm = deg[src].pow(-0.5) * deg[dst].pow(-0.5)
        useful_weight = torch.nan_to_num(norm * edge_gate, nan=0.0, posinf=0.0, neginf=0.0)

        # Useful/reliable neighbor summary: original relation-aware propagation.
        edge_summary = h.new_zeros(h.size())
        edge_summary.index_add_(0, dst, useful_weight.unsqueeze(-1) * h_src)
        edge_summary = self.edge_summary_norm(edge_summary)

        # Structural-spurious summary: clearly low-relevance edges become an
        # environment factor for front-door context sampling.
        edge_spu_gate = torch.sigmoid(
            self.edge_spu_temp * (self.edge_spu_threshold - edge_gate)
        )
        edge_spu_weight = torch.nan_to_num(
            norm * edge_spu_gate,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if self.edge_spu_msg_mode == 'neighbor':
            edge_spu_msg = h_src
        else:
            # Default: use the neighbor-induced offset relative to the target
            # node, which is a cleaner structural-spurious signal.
            edge_spu_msg = h_src - h_dst

        edge_spu_summary = h.new_zeros(h.size())
        edge_spu_summary.index_add_(0, dst, edge_spu_weight.unsqueeze(-1) * edge_spu_msg)
        edge_spu_summary = self.edge_spu_norm(edge_spu_summary)
        return edge_summary, edge_spu_summary

    def compute_edge_semantic_summary(self, h, edge_index, training=False):
        # Backward-compatible local propagation function used by the global
        # advective mixer. It returns only the useful relation-aware summary.
        edge_summary, _ = self.compute_edge_summaries(h, edge_index, training=training)
        return edge_summary

    def fuse_node_edge_representation(self, h, edge_summary, training=False):
        fuse_input = torch.cat([h, edge_summary, h * edge_summary], dim=-1)
        edge_delta = self.node_edge_fuser(fuse_input)
        edge_delta = F.dropout(edge_delta, self.dropout, training=training)
        return self.node_edge_norm(h + self.edge_blend * edge_delta)

    def fuse_edge_spurious_context(self, z_spurious, edge_spu_summary, training=False):
        if (
            not self.use_edge_spu_context
            or self.edge_spu_context_alpha <= 0.0
            or edge_spu_summary is None
        ):
            return z_spurious
        context_input = torch.cat([z_spurious, edge_spu_summary], dim=-1)
        context_delta = self.edge_spu_context_fuser(context_input)
        context_delta = F.dropout(context_delta, self.dropout, training=training)
        return self.edge_spu_context_norm(
            z_spurious + self.edge_spu_context_alpha * context_delta
        )

    def encode_representation(self, x, edge_index, training=False):
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
                local_fn = None
                if self.global_local_source == 'edge':
                    local_fn = self.compute_edge_semantic_summary
                elif self.global_local_source != 'gcn':
                    raise ValueError(
                        f"Unsupported global_local_source='{self.global_local_source}'. "
                        "Use 'edge' or 'gcn'."
                    )
                h_global = self.global_encoder(h, edge_index, training=training, local_fn=local_fn)
            h_global_context = self.global_context_norm(self.global_context_proj(h_global))
            if self.use_global_info:
                h = h + self.global_alpha * self.global_fuse_norm(h_global)

        edge_summary, edge_spu_summary = self.compute_edge_summaries(h, edge_index, training=training)
        z = self.fuse_node_edge_representation(h, edge_summary, training=training)

        # Avoid redundant DAG inputs by default: z already contains edge_summary.
        # node_only: DAG operates over compact latent factors of edge-refined z.
        # edge_only: DAG is driven by the neighborhood/edge summary only, while prediction still uses z.
        # node_edge: original formulation with both z-factors and edge_summary-factors.
        node_latent = self.node_dag_proj(z)
        edge_latent = self.edge_dag_proj(edge_summary)
        if self.dag_input_mode == 'node_only':
            dag_vars = node_latent
        elif self.dag_input_mode == 'edge_only':
            dag_vars = edge_latent
        else:
            dag_vars = torch.cat([node_latent, edge_latent], dim=-1)

        causal_score, pollution_score, mediator_gate, dag_total = self.get_causal_effect_and_mask()
        z_mediator = F.dropout(z * mediator_gate.unsqueeze(0), self.dropout, training=training)
        z_spurious = F.dropout(z * (1.0 - mediator_gate).unsqueeze(0), self.dropout, training=training)
        z_env_context = self.fuse_edge_spurious_context(
            z_spurious,
            edge_spu_summary,
            training=training,
        )
        mediator_logits = self.classifier(z_mediator)
        return (
            z,
            edge_summary,
            edge_spu_summary,
            dag_vars,
            z_mediator,
            z_spurious,
            z_env_context,
            mediator_logits,
            mediator_gate,
            causal_score,
            pollution_score,
            dag_total,
            h_global_context,
        )

    def compute_pseudo_env_probs(self, z_spurious):
        if self.num_envs <= 1 or z_spurious.numel() == 0:
            return z_spurious.new_ones(z_spurious.size(0), 1)
        return F.softmax(self.env_classifier(z_spurious), dim=-1)

    def get_frontdoor_contexts(self, z_spurious=None, env_probs=None):
        context_map = {}
        if z_spurious is not None and z_spurious.numel() > 0:
            if env_probs is None:
                env_probs = self.compute_pseudo_env_probs(z_spurious).detach()
            env_probs = env_probs.detach()
            for env_idx in range(self.num_envs):
                weight = env_probs[:, env_idx]
                mass = weight.sum()
                if mass > 1e-6:
                    context_vec = (weight.unsqueeze(-1) * z_spurious).sum(dim=0) / mass.clamp_min(1e-6)
                    context_map[int(env_idx)] = F.normalize(context_vec.detach(), dim=0)

        if not context_map:
            return None
        ordered = [context_map[idx] for idx in sorted(context_map.keys())]
        return torch.stack(ordered, dim=0)

    def get_global_contexts(self, h_global=None, env_probs=None):
        if (
            not self.use_global_contexts
            or h_global is None
            or h_global.numel() == 0
            or self.global_context_weight <= 0.0
        ):
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
        contexts = torch.stack(contexts, dim=0)
        return self.global_context_weight * contexts

    def merge_frontdoor_contexts(self, *context_sets):
        contexts = [ctx for ctx in context_sets if ctx is not None and ctx.numel() > 0]
        if not contexts:
            return None
        return torch.cat(contexts, dim=0)

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

        sample_k = self.gmm_sample_k
        if self.fd_sample_k > 0:
            sample_k = min(sample_k, self.fd_sample_k)
        if sample_k <= 0:
            return None

        if training:
            env_indices = valid_envs[torch.randint(valid_envs.numel(), (sample_k,), device=valid_envs.device)]
        else:
            repeat = (sample_k + valid_envs.numel() - 1) // valid_envs.numel()
            env_indices = valid_envs.repeat(repeat)[:sample_k]

        mean = means.index_select(0, env_indices)
        var = vars_.index_select(0, env_indices).clamp_min(self.gmm_min_var)
        std = var.sqrt()
        if self.gmm_max_std > 0.0:
            std = std.clamp_max(self.gmm_max_std)

        if training:
            noise = torch.randn_like(mean)
        else:
            generator = torch.Generator(device=mean.device)
            generator.manual_seed(self.context_sample_seed + sample_k + int(valid_envs.numel()))
            noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)

        contexts = mean + noise * std
        return F.normalize(contexts, dim=1)

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
            edge_spu_summary,
            dag_vars,
            z_mediator,
            z_spurious,
            z_env_context,
            mediator_logits,
            mediator_gate,
            causal_score,
            pollution_score,
            dag_total,
            h_global_context,
        ) = self.encode_representation(x, edge_index, training=training)

        gmm_contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(training=training),
            training=training,
        )
        contexts = self.merge_frontdoor_contexts(
            gmm_contexts,
            self.get_global_contexts(h_global_context),
        )
        fd_logits, fd_stack = self.frontdoor_logits_from_contexts(z_mediator, z_env_context, contexts)
        logits = self.blend_logits(mediator_logits, fd_logits)

        if training:
            return (
                logits,
                z,
                edge_summary,
                edge_spu_summary,
                dag_vars,
                z_mediator,
                z_spurious,
                z_env_context,
                mediator_gate,
                causal_score,
                pollution_score,
                dag_total,
                mediator_logits,
                fd_logits,
                fd_stack,
                h_global_context,
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

    def compute_pseudo_env_loss(self, env_logits):
        if env_logits.size(-1) <= 1:
            return env_logits.new_zeros(())
        probs = F.softmax(env_logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        mean_probs = probs.mean(dim=0)
        uniform = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
        balance = F.kl_div(mean_probs.clamp_min(1e-8).log(), uniform, reduction='sum')
        return entropy + self.pseudo_env_balance * balance

    def compute_pseudo_env_invariance_loss(self, logits, env_probs):
        if env_probs is None or env_probs.numel() == 0 or env_probs.size(-1) <= 1:
            return self.A_feat.new_zeros(())
        probs = torch.softmax(logits, dim=-1)
        global_mean = probs.mean(dim=0)
        weights = env_probs.detach()
        loss = probs.new_zeros(())
        count = 0
        for env_idx in range(weights.size(1)):
            weight = weights[:, env_idx]
            mass = weight.sum()
            if mass > 1e-6:
                env_mean = (weight.unsqueeze(-1) * probs).sum(dim=0) / mass.clamp_min(1e-6)
                loss = loss + F.mse_loss(env_mean, global_mean)
                count += 1
        if count == 0:
            return probs.new_zeros(())
        return loss / count

    def compute_global_env_consistency_loss(self, h_global, z_spurious, env_probs_spu=None):
        if (
            self.lambda_global_env <= 0.0
            or h_global is None
            or h_global.numel() == 0
            or z_spurious is None
            or z_spurious.numel() == 0
            or self.num_envs <= 1
        ):
            return self.A_feat.new_zeros(())

        global_probs = self.compute_pseudo_env_probs(h_global)
        if env_probs_spu is None:
            local_probs = self.compute_pseudo_env_probs(z_spurious)
        else:
            local_probs = env_probs_spu

        global_log = global_probs.clamp_min(1e-8).log()
        local_log = local_probs.clamp_min(1e-8).log()
        loss_g_to_l = F.kl_div(global_log, local_probs.detach(), reduction='batchmean')
        loss_l_to_g = F.kl_div(local_log, global_probs.detach(), reduction='batchmean')
        return 0.5 * (loss_g_to_l + loss_l_to_g)

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
        flow_score = self.dag_gate_expander(
            self._normalize_score(label_flow, default_value=0.5).unsqueeze(0)
        ).squeeze(0)
        flow_score = self._normalize_score(flow_score, default_value=0.5)
        loss_dag = loss_dag + 0.1 * F.mse_loss(mediator_gate, flow_score.detach())
        return loss_dag

    def dag_label_loss(self, dag_vars, labels, train_idx, criterion, args):
        """
        DAG-Core supervision for A_feat.

        We keep only the direct node/edge latent -> label sink prediction path.
        The previous parent-signal reconstruction term is removed because it can
        encourage the learned DAG to fit statistical correlations among latent
        variables rather than label-relevant causal strength.
        """
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

    def compute_counterexample_penalty(self, dag_vars, labels, train_idx, criterion, args):
        """
        Estimate per-node-latent hard-case instability.

        A dimension is penalized when its single-dimension label predictor has
        much larger loss on the hardest training samples than on the average
        training sample. This demotes features that are usually predictive but
        fail sharply on counterexamples, a common signature of spurious context.
        """
        if (
            self.counterexample_coeff <= 0.0
            or self.counterexample_top_frac <= 0.0
            or dag_vars.numel() == 0
            or train_idx.numel() == 0
        ):
            return self.counterexample_penalty.new_zeros(self.counterexample_penalty.size())

        node_latent = dag_vars[train_idx, self.node_var_slice].detach()
        labels_tr = labels[train_idx]
        if node_latent.numel() == 0:
            return self.counterexample_penalty.new_zeros(self.counterexample_penalty.size())

        A = self.get_masked_A().detach()
        label_A = A[self.node_var_slice, self.label_var_slice]
        logits = node_latent.unsqueeze(-1) * label_A.unsqueeze(0)
        logits = logits + self.dag_label_bias.detach().view(1, 1, -1)

        num_nodes, num_dims, _ = logits.shape
        logits_flat = logits.reshape(num_nodes * num_dims, self.c)
        labels_flat = labels_tr.repeat_interleave(num_dims, dim=0)
        losses = self.compute_supervised_loss(logits_flat, labels_flat, criterion, args)
        losses = losses.reshape(num_nodes, num_dims)

        mean_loss = losses.mean(dim=0)
        hard_k = max(1, int(round(num_nodes * self.counterexample_top_frac)))
        hard_k = min(hard_k, num_nodes)
        hard_loss = losses.topk(hard_k, dim=0).values.mean(dim=0)
        penalty = (hard_loss - mean_loss).clamp_min(0.0)
        return self._normalize_score(penalty, default_value=0.0).detach()

    def dag_semantic_loss(self, dag_vars, labels, train_idx, criterion, args):
        # Deprecated in DAG-Core. Kept for backward-compatible logging/calls.
        return self.A_feat.new_zeros(())

    def compute_dag_degree_loss(self, z_mediator, train_idx):
        return self.A_feat.new_zeros(())

    def compute_spurious_label_loss(self, z_spurious, env_probs, labels, criterion, args):
        if z_spurious.numel() == 0:
            return self.A_feat.new_zeros(())
        if env_probs is None or env_probs.numel() == 0:
            env_emb = torch.zeros_like(z_spurious)
        else:
            env_emb = torch.matmul(env_probs.detach(), self.pseudo_env_emb)
        logits = self.spurious_label_head(torch.cat([z_spurious, env_emb], dim=-1))
        return self.compute_supervised_loss(logits, labels, criterion, args).mean()

    def compute_edge_env_sensitivity(self, edge_latent, env_probs):
        if (
            edge_latent is None
            or edge_latent.numel() == 0
            or env_probs is None
            or env_probs.numel() == 0
            or env_probs.size(-1) <= 1
        ):
            return self.edge_env_sensitivity.new_zeros(self.edge_env_sensitivity.size())

        weights = env_probs.detach()
        edge_values = edge_latent.detach()
        global_mean = edge_values.mean(dim=0)
        sensitivity = edge_values.new_zeros(edge_values.size(1))
        count = 0
        for env_idx in range(weights.size(1)):
            weight = weights[:, env_idx]
            mass = weight.sum()
            if mass > 1e-6:
                env_mean = (weight.unsqueeze(-1) * edge_values).sum(dim=0) / mass.clamp_min(1e-6)
                sensitivity = sensitivity + (env_mean - global_mean).pow(2)
                count += 1
        if count > 0:
            sensitivity = sensitivity / count
        return self._normalize_score(sensitivity, default_value=0.0)

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

    def update_edge_env_sensitivity(self, edge_latent, env_probs=None):
        if edge_latent is None or env_probs is None:
            return
        sensitivity = self.compute_edge_env_sensitivity(edge_latent, env_probs)
        momentum = min(max(float(self.edge_env_momentum), 0.0), 1.0)
        self.edge_env_sensitivity.mul_(momentum).add_((1.0 - momentum) * sensitivity)

    def update_counterexample_penalty(self, penalty=None):
        if penalty is None or self.counterexample_coeff <= 0.0:
            return
        penalty = penalty.to(device=self.counterexample_penalty.device, dtype=self.counterexample_penalty.dtype)
        momentum = self.counterexample_momentum
        self.counterexample_penalty.mul_(momentum).add_((1.0 - momentum) * penalty)

    @torch.no_grad()
    def apply_state_update(self, state_payload):
        if state_payload is None:
            return
        self.update_spurious_gmm(
            state_payload['spu_tr'],
            state_payload['env_probs_tr'],
        )
        self.update_edge_env_sensitivity(
            state_payload['edge_latent_tr'],
            state_payload['env_probs_tr'],
        )
        self.update_counterexample_penalty(
            state_payload.get('counterexample_penalty'),
        )

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx

        (
            _,
            _,
            _,
            edge_spu_all,
            dag_vars_all,
            z_mediator_all,
            z_spurious_all,
            z_env_context_all,
            mediator_gate,
            causal_score,
            pollution_score,
            dag_total,
            mediator_logits_all,
            _,
            _,
            h_global_all,
        ) = self.forward(x, edge_index, training=True)

        y_tr = y[train_idx]
        med_tr = z_mediator_all[train_idx]
        spu_tr = z_spurious_all[train_idx]
        env_context_tr = z_env_context_all[train_idx]
        dag_vars_tr = dag_vars_all[train_idx]
        edge_latent_tr = dag_vars_tr[:, self.edge_var_slice]
        mediator_logits_tr = mediator_logits_all[train_idx]
        env_logits_spu = self.env_classifier(env_context_tr)
        env_probs_spu = F.softmax(env_logits_spu, dim=-1) if self.num_envs > 1 else None

        gmm_contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(env_context_tr, env_probs_spu, training=True),
            training=True,
        )
        global_contexts = self.get_global_contexts(
            h_global_all,
            self.compute_pseudo_env_probs(h_global_all) if h_global_all is not None else None,
        )
        contexts = self.merge_frontdoor_contexts(gmm_contexts, global_contexts)
        num_gmm_contexts = 0 if gmm_contexts is None else int(gmm_contexts.size(0))
        num_global_contexts = 0 if global_contexts is None else int(global_contexts.size(0))
        num_mixed_contexts = 0
        fd_logits_tr, fd_stack_tr = self.frontdoor_logits_from_contexts(med_tr, env_context_tr, contexts)
        final_logits_tr = self.blend_logits(mediator_logits_tr, fd_logits_tr)

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        loss_dag = self.dag_regularization_loss(mediator_gate, dag_total)
        loss_dag_label = self.dag_label_loss(dag_vars_all, y, train_idx, criterion, args)
        loss_global_env = self.compute_global_env_consistency_loss(
            h_global_all,
            z_env_context_all,
            self.compute_pseudo_env_probs(z_env_context_all) if self.num_envs > 1 else None,
        )
        current_counterexample_penalty = self.compute_counterexample_penalty(
            dag_vars_all,
            y,
            train_idx,
            criterion,
            args,
        )

        # DAG-Core keeps only the essential pseudo-environment constraints:
        # spurious features should form label-free contexts, and mediator
        # features should be invariant to those pseudo-contexts.
        if self.num_envs > 1:
            env_logits_med = self.env_classifier(med_tr)
            loss_env_med = self.compute_env_uniform_loss(env_logits_med)
            loss_spu = self.compute_pseudo_env_loss(env_logits_spu)
        else:
            env_logits_med = None
            env_logits_spu = None
            loss_env_med = self.A_feat.new_zeros(())
            loss_spu = self.compute_uniform_loss(self.classifier(spu_tr))

        zero = self.A_feat.new_zeros(())
        loss_med = zero
        loss_fd_aug = zero
        loss_var = zero
        loss_ind = zero
        loss_sem = zero
        loss_degree = zero
        loss_spu_y = zero
        loss_inv = zero

        total_loss = (
            loss_cls
            + self.lambda_fd * loss_fd
            + self.lambda_dag * loss_dag
            + self.lambda_dag_label * loss_dag_label
            + self.lambda_spu * loss_spu
            + self.lambda_env * loss_env_med
            + self.lambda_global_env * loss_global_env
        )

        state_payload = None
        if update_state:
            state_payload = {
                # Keep this legacy key name for compatibility. When
                # use_edge_spu_context is enabled, it stores the fused
                # node-spurious + structural-spurious environment context.
                'spu_tr': env_context_tr.detach(),
                'env_probs_tr': env_probs_spu.detach() if env_probs_spu is not None else None,
                'edge_latent_tr': edge_latent_tr.detach(),
                'counterexample_penalty': current_counterexample_penalty.detach(),
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
            'loss_dag_label': loss_dag_label,
            'loss_sem': loss_sem,
            'loss_degree': loss_degree,
            'loss_spu_y': loss_spu_y,
            'loss_spu': loss_spu,
            'loss_env_med': loss_env_med,
            'loss_inv': loss_inv,
            'loss_global_env': loss_global_env,
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'causal_score_mean': causal_score.mean().detach(),
            'pollution_score_mean': pollution_score.mean().detach(),
            'edge_spu_context_norm': edge_spu_all[train_idx].norm(dim=1).mean().detach(),
            'env_context_norm': env_context_tr.norm(dim=1).mean().detach(),
            'counterexample_penalty_mean': self.counterexample_penalty.mean().detach(),
            'counterexample_penalty_batch_mean': current_counterexample_penalty.mean().detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_mixed_contexts': torch.tensor(float(num_mixed_contexts), device=x.device),
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
