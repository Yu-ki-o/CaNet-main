import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, degree, negative_sampling, remove_self_loops, softmax
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
            # Per-destination softmax is more stable than subtracting a single
            # graph-level maximum, especially when attention is combined with
            # edge/front-door regularizers.
            alpha = softmax(logits, att_edge_index[1], num_nodes=num_nodes)
            alpha = F.dropout(alpha, p=0.0, training=self.training)
            out = self.specialspmm(att_edge_index, alpha, torch.Size([num_nodes, num_nodes]), h)

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
        # Layer-wise Local-IGM routing. When enabled, the same edge gate used
        # by the final NodeIGM-style summary is applied between GNN layers.
        # This routes one-hop messages at each layer; after L layers, the
        # filtered representation has implicitly filtered up to L-hop context.
        self.use_layerwise_local_igm = bool(getattr(args, 'use_layerwise_local_igm', False))
        self.layerwise_local_igm_skip_last = bool(getattr(args, 'layerwise_local_igm_skip_last', True))
        self.layerwise_final_edge_fuse = bool(getattr(args, 'layerwise_final_edge_fuse', True))
        self.layerwise_gate_target = min(
            max(float(getattr(args, 'layerwise_gate_target', 0.5)), 0.0),
            1.0,
        )
        self.lambda_layerwise_gate = max(0.0, float(getattr(args, 'lambda_layerwise_gate', 0.0)))
        self._last_layerwise_gate_loss = None
        self._last_layerwise_gate_mean = None
        self._last_layerwise_gate_layers = 0
        # Optional low-relevance neighbor denoising. The edge gate still builds
        # a useful neighbor summary with g_uv, while (1-g_uv) builds a
        # low-relevance neighbor summary that can be softly subtracted from h.
        self.use_neighbor_denoise = bool(getattr(args, 'use_neighbor_denoise', False))
        self.noise_subtract_alpha = max(0.0, float(getattr(args, 'noise_subtract_alpha', 0.1)))
        self.noise_gate_temp = max(1e-3, float(getattr(args, 'noise_gate_temp', 1.0)))
        self.edge_feat_mode = getattr(args, 'edge_feat_mode', 'mul')
        edge_feat_dim = self._get_edge_feat_dim(self.edge_feat_mode)
        self.edge_pair_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.edge_score_head = nn.Linear(self.d, 1)
        self.edge_summary_norm = nn.LayerNorm(self.d)
        self.node_edge_fuser = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.node_edge_norm = nn.LayerNorm(self.d)
        self.noise_summary_norm = nn.LayerNorm(self.d)
        self.node_noise_fuser = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.node_noise_gate = nn.Linear(self.d * 3, 1)

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

        self.node_var_dim = self.dag_latent_dim
        self.edge_var_dim = self.dag_latent_dim
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
        requested_gmm_sample_k = int(getattr(args, 'gmm_sample_k', 3))
        if requested_gmm_sample_k <= 0:
            requested_gmm_sample_k = self.fd_sample_k
        self.gmm_sample_k = max(0, requested_gmm_sample_k)
        self.gmm_min_var = max(1e-6, float(getattr(args, 'gmm_min_var', 1e-4)))
        self.gmm_max_std = max(0.0, float(getattr(args, 'gmm_max_std', 1.0)))
        self.global_context_weight = max(0.0, float(getattr(args, 'global_context_weight', 1.0)))
        self.global_context_detach = bool(getattr(args, 'global_context_detach', True))
        self.use_layerwise_spurious_contexts = bool(getattr(args, 'use_layerwise_spurious_contexts', False))
        self.layerwise_spurious_context_weight = max(
            0.0,
            float(getattr(args, 'layerwise_spurious_context_weight', 1.0)),
        )
        self.layerwise_spurious_context_detach = bool(
            getattr(args, 'layerwise_spurious_context_detach', True),
        )

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

        # CGRL-aligned Energy-Based Reconstruction (EBR) applied to the
        # pre-DAG node representation z.  This adds a reverse constraint
        # z -> Q_phi(G | z), so the representation entering the DAG must still
        # explain stable local graph structure instead of only fitting labels.
        self.lambda_ebr = max(0.0, float(getattr(args, 'lambda_ebr', 0.05)))
        self.ebr_temperature = max(1e-3, float(getattr(args, 'ebr_temperature', 1.0)))
        self.ebr_neg_ratio = max(0.0, float(getattr(args, 'ebr_neg_ratio', 1.0)))
        self.ebr_max_edges = int(getattr(args, 'ebr_max_edges', 200000))
        self.ebr_bce_weight = max(0.0, float(getattr(args, 'ebr_bce_weight', 1.0)))
        self.ebr_kl_weight = max(0.0, float(getattr(args, 'ebr_kl_weight', 1.0)))
        self.ebr_prior_detach = bool(getattr(args, 'ebr_prior_detach', True))
        self.ebr_use_edge_prior = bool(getattr(args, 'ebr_use_edge_prior', True))
        self.ebr_remove_self_loops = bool(getattr(args, 'ebr_remove_self_loops', True))
        self.ebr_weight = Parameter(torch.empty(self.d, self.d))
        self._last_ebr_bce_loss = None
        self._last_ebr_kl_loss = None
        self._last_ebr_pos_score = None
        self._last_ebr_neg_score = None
        self._last_ebr_num_pos = 0
        self._last_ebr_num_neg = 0

        # Local node-aware bi-smoothing consistency. This keeps the overall
        # front-door framework unchanged: the randomized local subgraph is
        # only used to regularize the mediator M, not to replace the
        # front-door prediction path.
        self.use_local_bismooth = bool(getattr(args, 'use_local_bismooth', False))
        self.lambda_bismooth = float(getattr(args, 'lambda_bismooth', 0.0))
        self.lambda_bismooth_cls = float(getattr(args, 'lambda_bismooth_cls', 0.0))
        self.bismooth_edge_drop = min(max(float(getattr(args, 'bismooth_edge_drop', 0.1)), 0.0), 1.0)
        self.bismooth_node_drop = min(max(float(getattr(args, 'bismooth_node_drop', 0.05)), 0.0), 1.0)
        self.bismooth_samples = max(1, int(getattr(args, 'bismooth_samples', 1)))
        self.bismooth_consistency = getattr(args, 'bismooth_consistency', 'cosine')
        self.bismooth_keep_train_nodes = bool(getattr(args, 'bismooth_keep_train_nodes', True))
        self.bismooth_singleton = getattr(args, 'bismooth_singleton', 'exclude')
        if self.bismooth_singleton not in ('include', 'exclude'):
            self.bismooth_singleton = 'exclude'
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
        self.register_buffer('edge_env_sensitivity', torch.zeros(self.dag_latent_dim))
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
        nn.init.xavier_uniform_(self.ebr_weight)
        self._reset_module_parameters(self.edge_pair_encoder)
        self.edge_score_head.reset_parameters()
        self.edge_summary_norm.reset_parameters()
        self._reset_module_parameters(self.node_edge_fuser)
        nn.init.zeros_(self.node_edge_fuser[-1].weight)
        nn.init.zeros_(self.node_edge_fuser[-1].bias)
        self.node_edge_norm.reset_parameters()
        self.noise_summary_norm.reset_parameters()
        self._reset_module_parameters(self.node_noise_fuser)
        # Start from the original model behavior: the denoising branch initially
        # subtracts exactly zero, then learns only if useful.
        nn.init.zeros_(self.node_noise_fuser[-1].weight)
        nn.init.zeros_(self.node_noise_fuser[-1].bias)
        self.node_noise_gate.reset_parameters()
        nn.init.zeros_(self.node_noise_gate.weight)
        nn.init.zeros_(self.node_noise_gate.bias)
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
        self._last_layerwise_gate_loss = None
        self._last_layerwise_gate_mean = None
        self._last_layerwise_gate_layers = 0
        self._last_ebr_bce_loss = None
        self._last_ebr_kl_loss = None
        self._last_ebr_pos_score = None
        self._last_ebr_neg_score = None
        self._last_ebr_num_pos = 0
        self._last_ebr_num_neg = 0

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
        edge_to_node = C_flow[self.edge_var_slice, self.node_var_slice]
        edge_sensitivity = self.edge_env_sensitivity.to(device=C_flow.device, dtype=C_flow.dtype)
        if edge_sensitivity.sum() > 1e-6:
            edge_weight = edge_sensitivity / edge_sensitivity.sum().clamp_min(1e-8)
            edge_incoming = torch.matmul(edge_to_node.t(), edge_weight)
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
        if mode in ('mul_diff', 'concat'):
            return 2 * self.d
        if mode == 'concat_diff':
            return 3 * self.d
        if mode in ('mul_degree', 'diff_degree'):
            return self.d + 1
        if mode == 'mul_diff_degree':
            return 2 * self.d + 1
        raise ValueError(
            f"Unknown edge_feat_mode='{mode}'. Use one of: "
            "mul, diff, degree, mul_diff, concat, concat_diff, "
            "mul_degree, diff_degree, mul_diff_degree."
        )

    def build_edge_feat(self, h_src, h_dst, deg_src, deg_dst, deg_max):
        mode = self.edge_feat_mode
        mul_feat = h_src * h_dst
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
        if mode == 'degree':
            return deg_pair
        if mode == 'mul_diff':
            return torch.cat([mul_feat, diff_feat], dim=-1)
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
        raise ValueError(
            f"Unknown edge_feat_mode='{mode}'. Use one of: "
            "mul, diff, degree, mul_diff, concat, concat_diff, "
            "mul_degree, diff_degree, mul_diff_degree."
        )

    def compute_edge_summaries(self, h, edge_index, training=False):
        """
        Build both useful and low-relevance neighbor summaries.

        useful_summary_v = sum_u norm_uv * g_uv       * h_u
        noise_summary_v  = sum_u norm_uv * (1 - g_uv) * h_u

        The existing edge feature modes (mul/diff/degree/...) define g_uv.
        The low-relevance summary is not used unless --use_neighbor_denoise is set.
        """
        if edge_index.numel() == 0:
            zero = h.new_zeros(h.size())
            return zero, zero, None

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
        noise_weight = torch.nan_to_num(norm * (1.0 - edge_gate), nan=0.0, posinf=0.0, neginf=0.0)

        useful_summary = h.new_zeros(h.size())
        useful_summary.index_add_(0, dst, useful_weight.unsqueeze(-1) * h_src)
        useful_summary = self.edge_summary_norm(useful_summary)

        noise_summary = h.new_zeros(h.size())
        noise_summary.index_add_(0, dst, noise_weight.unsqueeze(-1) * h_src)
        noise_summary = self.noise_summary_norm(noise_summary)
        return useful_summary, noise_summary, edge_gate

    def compute_edge_semantic_summary(self, h, edge_index, training=False):
        # Backward-compatible local propagation function used by the global
        # advective mixer. It returns only the useful relation-aware summary.
        useful_summary, _, _ = self.compute_edge_summaries(h, edge_index, training=training)
        return useful_summary

    def fuse_node_edge_representation(self, h, edge_summary, noise_summary=None, training=False):
        useful_input = torch.cat([h, edge_summary, h * edge_summary], dim=-1)
        useful_delta = self.node_edge_fuser(useful_input)
        useful_delta = F.dropout(useful_delta, self.dropout, training=training)

        fused = h + self.edge_blend * useful_delta

        if (
            self.use_neighbor_denoise
            and self.noise_subtract_alpha > 0.0
            and noise_summary is not None
        ):
            # Estimate the environment/noise component contributed by low-gate
            # neighbors and subtract it softly. The node-level gate prevents
            # hard removal of heterophilic but useful neighbors.
            noise_input = torch.cat([h, noise_summary, h * noise_summary], dim=-1)
            noise_delta = self.node_noise_fuser(noise_input)
            noise_delta = F.dropout(noise_delta, self.dropout, training=training)

            gate_input = torch.cat([h, noise_summary, torch.abs(h - noise_summary)], dim=-1)
            noise_gate = torch.sigmoid(self.node_noise_gate(gate_input) / self.noise_gate_temp)
            fused = fused - self.noise_subtract_alpha * noise_gate * noise_delta

        return self.node_edge_norm(fused)

    def encode_representation(self, x, edge_index, training=False):
        x = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(self.input_proj(x))
        layerwise_states = []
        layerwise_gate_loss = h.new_zeros(())
        layerwise_gate_mean = h.new_zeros(())
        layerwise_gate_layers = 0
        num_backbone_layers = len(self.backbone_layers)

        for layer_idx, layer in enumerate(self.backbone_layers):
            h = F.dropout(h, self.dropout, training=training)
            h = self.act_fn(layer(h, edge_index))

            # Layer-wise Local-IGM message routing: after each ordinary GNN
            # propagation, re-score the original one-hop edges using current
            # node states, add useful messages, and optionally subtract a
            # softly gated low-relevance context branch. Because h at layer l
            # already contains l-hop information, this implicitly filters
            # multi-hop information without materializing A^2/A^3 ego graphs.
            if self.use_layerwise_local_igm:
                should_route = not (
                    self.layerwise_local_igm_skip_last
                    and layer_idx == num_backbone_layers - 1
                )
                if should_route:
                    edge_summary_l, noise_summary_l, edge_gate_l = self.compute_edge_summaries(
                        h,
                        edge_index,
                        training=training,
                    )
                    h = self.fuse_node_edge_representation(
                        h,
                        edge_summary_l,
                        noise_summary=noise_summary_l,
                        training=training,
                    )
                    if edge_gate_l is not None:
                        layerwise_gate_mean = layerwise_gate_mean + edge_gate_l.mean()
                        if self.lambda_layerwise_gate > 0.0:
                            layerwise_gate_loss = layerwise_gate_loss + (
                                edge_gate_l.mean() - self.layerwise_gate_target
                            ).pow(2)
                        layerwise_gate_layers += 1

            if self.use_layerwise_spurious_contexts:
                layerwise_states.append(h)

        if layerwise_gate_layers > 0:
            layerwise_gate_mean = layerwise_gate_mean / float(layerwise_gate_layers)
            layerwise_gate_loss = layerwise_gate_loss / float(layerwise_gate_layers)
        self._last_layerwise_gate_mean = layerwise_gate_mean.detach()
        self._last_layerwise_gate_loss = layerwise_gate_loss
        self._last_layerwise_gate_layers = int(layerwise_gate_layers)

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

        edge_summary, noise_summary, final_edge_gate = self.compute_edge_summaries(h, edge_index, training=training)
        if self.use_layerwise_local_igm and not self.layerwise_final_edge_fuse:
            z = self.node_edge_norm(h)
        else:
            z = self.fuse_node_edge_representation(
                h,
                edge_summary,
                noise_summary=noise_summary,
                training=training,
            )
        node_latent = self.node_dag_proj(z)
        edge_latent = self.edge_dag_proj(edge_summary)
        dag_vars = torch.cat([node_latent, edge_latent], dim=-1)
        causal_score, pollution_score, mediator_gate, dag_total = self.get_causal_effect_and_mask()
        z_mediator = F.dropout(z * mediator_gate.unsqueeze(0), self.dropout, training=training)
        z_spurious = F.dropout(z * (1.0 - mediator_gate).unsqueeze(0), self.dropout, training=training)
        layerwise_spurious = None
        if self.use_layerwise_spurious_contexts and layerwise_states:
            spurious_gate = (1.0 - mediator_gate).view(1, 1, -1)
            layerwise_spurious = torch.stack(layerwise_states, dim=0) * spurious_gate
        mediator_logits = self.classifier(z_mediator)
        return (
            z,
            edge_summary,
            final_edge_gate,
            dag_vars,
            z_mediator,
            z_spurious,
            mediator_logits,
            mediator_gate,
            causal_score,
            pollution_score,
            dag_total,
            h_global_context,
            layerwise_spurious,
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

    def get_layerwise_spurious_contexts(self, layerwise_spurious=None, env_probs=None):
        if (
            not self.use_layerwise_spurious_contexts
            or layerwise_spurious is None
            or layerwise_spurious.numel() == 0
            or self.layerwise_spurious_context_weight <= 0.0
        ):
            return None

        if layerwise_spurious.dim() == 2:
            layerwise_spurious = layerwise_spurious.unsqueeze(0)
        if layerwise_spurious.dim() != 3:
            return None

        num_layers, num_nodes, _ = layerwise_spurious.shape
        values = (
            layerwise_spurious.detach()
            if self.layerwise_spurious_context_detach
            else layerwise_spurious
        )

        if env_probs is None:
            flat_values = values.reshape(num_layers * num_nodes, self.d)
            flat_probs = self.compute_pseudo_env_probs(flat_values)
            env_probs = flat_probs.view(num_layers, num_nodes, -1)
        else:
            env_probs = env_probs.detach().clamp_min(0.0)
            if env_probs.dim() == 2:
                env_probs = env_probs.unsqueeze(0).expand(num_layers, -1, -1)
            elif env_probs.dim() != 3:
                return None

        if env_probs.size(0) != num_layers or env_probs.size(1) != num_nodes:
            return None

        contexts = []
        for layer_idx in range(num_layers):
            layer_values = values[layer_idx]
            layer_probs = env_probs[layer_idx]
            for env_idx in range(layer_probs.size(1)):
                weight = layer_probs[:, env_idx]
                mass = weight.sum()
                if mass > 1e-6:
                    context_vec = (
                        weight.unsqueeze(-1) * layer_values
                    ).sum(dim=0) / mass.clamp_min(1e-6)
                    contexts.append(F.normalize(context_vec, dim=0))

        if not contexts:
            return None
        contexts = torch.stack(contexts, dim=0)
        return self.layerwise_spurious_context_weight * contexts

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
            final_edge_gate,
            dag_vars,
            z_mediator,
            z_spurious,
            mediator_logits,
            mediator_gate,
            causal_score,
            pollution_score,
            dag_total,
            h_global_context,
            layerwise_spurious,
        ) = self.encode_representation(x, edge_index, training=training)

        gmm_contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(training=training),
            training=training,
        )
        layerwise_contexts = self.get_layerwise_spurious_contexts(
            layerwise_spurious,
            self.compute_pseudo_env_probs(z_spurious) if self.num_envs > 1 else None,
        )
        contexts = self.merge_frontdoor_contexts(
            gmm_contexts,
            self.get_global_contexts(h_global_context),
            layerwise_contexts,
        )
        fd_logits, fd_stack = self.frontdoor_logits_from_contexts(z_mediator, z_spurious, contexts)
        logits = self.blend_logits(mediator_logits, fd_logits)

        if training:
            return (
                logits,
                z,
                edge_summary,
                final_edge_gate,
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
                h_global_context,
                layerwise_spurious,
            )
        return logits

    def _ebr_edge_logits(self, h, edge_index):
        """
        Bilinear edge score used by the CGRL-style energy function:
            E(u, v) = - h_v^T W h_u.
        The returned value is -E(u, v), so larger logits mean lower energy and
        a higher reconstructed edge probability.
        """
        if edge_index is None or edge_index.numel() == 0:
            return h.new_zeros((0,))
        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]
        projected_dst = torch.matmul(h_dst, self.ebr_weight)
        logits = (projected_dst * h_src).sum(dim=-1)
        return logits / self.ebr_temperature

    def _uniform_edge_prior(self, dst, num_nodes):
        deg_in = degree(dst, num_nodes).to(device=dst.device, dtype=torch.float32).clamp_min(1.0)
        return (1.0 / deg_in[dst]).clamp_min(1e-12)

    def compute_energy_reconstruction_loss(self, h, edge_index, final_edge_gate=None, training=False):
        """
        CGRL-aligned Energy-Based Reconstruction (EBR) on the representation
        immediately before the DAG module.

        We construct Q_phi(u, v | h) with an energy score on h and normalize it
        over observed incoming neighbours of each destination node, matching the
        paper's edge-posterior form.  A local KL term aligns this posterior to a
        graph prior P_theta(G) from the current encoder edge gate when available
        (or a uniform observed-neighbour prior otherwise).  A lightweight
        positive/negative NCE-BCE term is added as a practical reconstruction
        signal so the energy model also distinguishes observed edges from
        sampled non-edges.
        """
        zero = h.new_zeros(())
        self._last_ebr_bce_loss = zero.detach()
        self._last_ebr_kl_loss = zero.detach()
        self._last_ebr_pos_score = zero.detach()
        self._last_ebr_neg_score = zero.detach()
        self._last_ebr_num_pos = 0
        self._last_ebr_num_neg = 0

        if self.lambda_ebr <= 0.0 or edge_index is None or edge_index.numel() == 0:
            return zero

        num_nodes = h.size(0)
        src, dst = edge_index
        if self.ebr_remove_self_loops:
            keep = src != dst
            if keep.sum() == 0:
                return zero
            pos_edge_index = edge_index[:, keep]
            pos_prior_gate = final_edge_gate[keep] if final_edge_gate is not None else None
        else:
            pos_edge_index = edge_index
            pos_prior_gate = final_edge_gate

        num_pos_total = pos_edge_index.size(1)
        if num_pos_total == 0:
            return zero

        max_edges = self.ebr_max_edges
        if max_edges > 0 and num_pos_total > max_edges:
            if training:
                perm = torch.randperm(num_pos_total, device=pos_edge_index.device)[:max_edges]
            else:
                perm = torch.arange(max_edges, device=pos_edge_index.device)
            pos_edge_index = pos_edge_index[:, perm]
            if pos_prior_gate is not None:
                pos_prior_gate = pos_prior_gate[perm]

        pos_src, pos_dst = pos_edge_index
        pos_logits = self._ebr_edge_logits(h, pos_edge_index)
        self._last_ebr_num_pos = int(pos_edge_index.size(1))
        self._last_ebr_pos_score = pos_logits.detach().mean() if pos_logits.numel() > 0 else zero.detach()

        # Q_phi(u, v | h): energy posterior over observed incoming neighbours.
        q_edge = softmax(pos_logits, pos_dst, num_nodes=num_nodes).clamp_min(1e-12)
        # Re-normalize exactly per destination to avoid numeric drift.
        q_norm = h.new_zeros(num_nodes)
        q_norm.index_add_(0, pos_dst, q_edge)
        q_edge = q_edge / q_norm[pos_dst].clamp_min(1e-12)

        if self.ebr_use_edge_prior and pos_prior_gate is not None:
            prior_logits = torch.log(pos_prior_gate.clamp_min(1e-6))
            p_edge = softmax(prior_logits, pos_dst, num_nodes=num_nodes).clamp_min(1e-12)
            p_norm = h.new_zeros(num_nodes)
            p_norm.index_add_(0, pos_dst, p_edge)
            p_edge = p_edge / p_norm[pos_dst].clamp_min(1e-12)
            if self.ebr_prior_detach:
                p_edge = p_edge.detach()
        else:
            p_edge = self._uniform_edge_prior(pos_dst, num_nodes).to(device=h.device, dtype=h.dtype)

        active_dst = torch.unique(pos_dst).numel()
        kl_loss = (q_edge * (q_edge.log() - p_edge.log())).sum() / max(1, int(active_dst))
        self._last_ebr_kl_loss = kl_loss.detach()

        bce_loss = zero
        num_neg = int(round(pos_edge_index.size(1) * self.ebr_neg_ratio))
        if self.ebr_bce_weight > 0.0 and num_neg > 0:
            neg_edge_index = negative_sampling(
                edge_index=edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_neg,
                method='sparse',
            )
            if neg_edge_index is not None and neg_edge_index.numel() > 0:
                neg_logits = self._ebr_edge_logits(h, neg_edge_index)
                self._last_ebr_num_neg = int(neg_edge_index.size(1))
                self._last_ebr_neg_score = neg_logits.detach().mean() if neg_logits.numel() > 0 else zero.detach()
                logits = torch.cat([pos_logits, neg_logits], dim=0)
                labels = torch.cat([
                    torch.ones_like(pos_logits),
                    torch.zeros_like(neg_logits),
                ], dim=0)
                bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        self._last_ebr_bce_loss = bce_loss.detach()

        return self.ebr_kl_weight * kl_loss + self.ebr_bce_weight * bce_loss

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

    def sample_bismooth_edge_index(self, edge_index, num_nodes, train_idx=None):
        """
        Node-aware bi-smoothing randomization used as a training regularizer.

        It follows the official node-aware bi-smoothing spirit: randomly delete
        existing edges and randomly delete nodes by removing all incident edges.
        The node feature matrix shape is unchanged.  When singleton='exclude',
        nodes that become isolated in the smoothed graph are excluded from the
        bi-smoothing consistency/classification loss, matching the practical
        node-aware-exclude idea without turning this model into a certificate.
        """
        if edge_index is None or edge_index.numel() == 0:
            valid_nodes = torch.ones(num_nodes, device=edge_index.device, dtype=torch.bool)
            return edge_index, valid_nodes

        src, dst = edge_index
        device = edge_index.device

        node_keep = torch.ones(num_nodes, device=device, dtype=torch.bool)
        if self.bismooth_node_drop > 0.0:
            node_keep = torch.rand(num_nodes, device=device) >= self.bismooth_node_drop
            if self.bismooth_keep_train_nodes and train_idx is not None and train_idx.numel() > 0:
                node_keep[train_idx] = True
            keep = node_keep[src] & node_keep[dst]
        else:
            keep = torch.ones(src.size(0), device=device, dtype=torch.bool)

        if self.bismooth_edge_drop > 0.0:
            edge_keep = torch.rand(src.size(0), device=device) >= self.bismooth_edge_drop
            keep = keep & edge_keep

        if keep.sum() == 0:
            smooth_edge_index = edge_index
            valid_nodes = torch.ones(num_nodes, device=device, dtype=torch.bool)
            return smooth_edge_index, valid_nodes

        smooth_edge_index = edge_index[:, keep]
        if self.bismooth_singleton == 'exclude':
            smooth_src, smooth_dst = smooth_edge_index
            deg_in = degree(smooth_dst, num_nodes).to(device=device)
            deg_out = degree(smooth_src, num_nodes).to(device=device)
            valid_nodes = (deg_in + deg_out) > 0
            # Nodes explicitly dropped by node smoothing should not contribute
            # to the local-invariance regularizer even if directed bookkeeping
            # accidentally leaves a residual degree.
            valid_nodes = valid_nodes & node_keep
        else:
            valid_nodes = node_keep
        return smooth_edge_index, valid_nodes

    def compute_local_bismooth_loss(
        self,
        x,
        edge_index,
        train_idx,
        y,
        z_mediator_clean,
        criterion,
        args,
    ):
        """
        Local bi-smoothed mediator consistency.

        The randomized graph is not used as the final prediction graph. It is
        used only to enforce that the front-door mediator representation M is
        stable under distribution-internal local structural perturbations. This
        keeps the framework as front-door adjustment:

            stable local structure -> mediator M
            unstable structure     -> spurious/context C
            M + sampled C          -> front-door prediction
        """
        if (
            not self.use_local_bismooth
            or self.lambda_bismooth <= 0.0
            or train_idx is None
            or train_idx.numel() == 0
        ):
            zero = z_mediator_clean.new_zeros(())
            return zero, zero, zero

        loss_cons = z_mediator_clean.new_zeros(())
        loss_cls = z_mediator_clean.new_zeros(())
        valid_ratio_sum = z_mediator_clean.new_zeros(())
        used_samples = 0
        samples = max(1, int(self.bismooth_samples))
        clean_ref = z_mediator_clean.detach()

        for _ in range(samples):
            smooth_edge_index, valid_nodes = self.sample_bismooth_edge_index(
                edge_index,
                x.size(0),
                train_idx=train_idx,
            )
            train_valid = valid_nodes[train_idx]
            if train_valid.sum() == 0:
                continue

            (
                _,
                _,
                _,
                _,
                z_mediator_smooth,
                _,
                mediator_logits_smooth,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = self.encode_representation(x, smooth_edge_index, training=True)

            idx = train_idx[train_valid]
            clean_train = clean_ref[idx]
            smooth_train = z_mediator_smooth[idx]
            if self.bismooth_consistency == 'mse':
                loss_cons = loss_cons + F.mse_loss(smooth_train, clean_train)
            else:
                clean_norm = F.normalize(clean_train, dim=1)
                smooth_norm = F.normalize(smooth_train, dim=1)
                loss_cons = loss_cons + (1.0 - (smooth_norm * clean_norm).sum(dim=1)).mean()

            if self.lambda_bismooth_cls > 0.0:
                loss_cls = loss_cls + self.compute_supervised_loss(
                    mediator_logits_smooth[idx],
                    y[idx],
                    criterion,
                    args,
                ).mean()
            valid_ratio_sum = valid_ratio_sum + train_valid.float().mean()
            used_samples += 1

        if used_samples == 0:
            zero = z_mediator_clean.new_zeros(())
            return zero, zero, zero
        denom = float(used_samples)
        return loss_cons / denom, loss_cls / denom, valid_ratio_sum / denom

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx

        (
            _,
            z_all,
            edge_summary_all,
            final_edge_gate_all,
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
            h_global_all,
            layerwise_spurious_all,
        ) = self.forward(x, edge_index, training=True)

        y_tr = y[train_idx]
        med_tr = z_mediator_all[train_idx]
        spu_tr = z_spurious_all[train_idx]
        dag_vars_tr = dag_vars_all[train_idx]
        edge_latent_tr = dag_vars_tr[:, self.edge_var_slice]
        mediator_logits_tr = mediator_logits_all[train_idx]
        env_logits_spu = self.env_classifier(spu_tr)
        env_probs_spu = F.softmax(env_logits_spu, dim=-1) if self.num_envs > 1 else None

        gmm_contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(spu_tr, env_probs_spu, training=True),
            training=True,
        )
        global_contexts = self.get_global_contexts(
            h_global_all,
            self.compute_pseudo_env_probs(h_global_all) if h_global_all is not None else None,
        )
        layerwise_contexts = None
        if layerwise_spurious_all is not None:
            layerwise_contexts = self.get_layerwise_spurious_contexts(
                layerwise_spurious_all[:, train_idx, :],
                env_probs_spu,
            )
        contexts = self.merge_frontdoor_contexts(gmm_contexts, global_contexts, layerwise_contexts)
        num_gmm_contexts = 0 if gmm_contexts is None else int(gmm_contexts.size(0))
        num_global_contexts = 0 if global_contexts is None else int(global_contexts.size(0))
        num_layerwise_contexts = 0 if layerwise_contexts is None else int(layerwise_contexts.size(0))
        num_mixed_contexts = 0
        fd_logits_tr, fd_stack_tr = self.frontdoor_logits_from_contexts(med_tr, spu_tr, contexts)
        final_logits_tr = self.blend_logits(mediator_logits_tr, fd_logits_tr)

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        loss_dag = self.dag_regularization_loss(mediator_gate, dag_total)
        loss_dag_label = self.dag_label_loss(dag_vars_all, y, train_idx, criterion, args)
        loss_global_env = self.compute_global_env_consistency_loss(
            h_global_all,
            z_spurious_all,
            self.compute_pseudo_env_probs(z_spurious_all) if self.num_envs > 1 else None,
        )
        loss_ebr = self.compute_energy_reconstruction_loss(
            z_all,
            edge_index,
            final_edge_gate=final_edge_gate_all,
            training=True,
        )
        current_counterexample_penalty = self.compute_counterexample_penalty(
            dag_vars_all,
            y,
            train_idx,
            criterion,
            args,
        )
        loss_bismooth, loss_bismooth_cls, bismooth_valid_ratio = self.compute_local_bismooth_loss(
            x,
            edge_index,
            train_idx,
            y,
            z_mediator_all,
            criterion,
            args,
        )
        if self._last_layerwise_gate_loss is None:
            loss_layerwise_gate = self.A_feat.new_zeros(())
            layerwise_gate_mean = self.A_feat.new_zeros(())
        else:
            loss_layerwise_gate = self._last_layerwise_gate_loss
            layerwise_gate_mean = self._last_layerwise_gate_mean

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
            + self.lambda_ebr * loss_ebr
            + self.lambda_bismooth * loss_bismooth
            + self.lambda_bismooth_cls * loss_bismooth_cls
            + self.lambda_layerwise_gate * loss_layerwise_gate
        )

        state_payload = None
        if update_state:
            state_payload = {
                'spu_tr': spu_tr.detach(),
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
            'loss_ebr': loss_ebr,
            'loss_ebr_bce': self._last_ebr_bce_loss if self._last_ebr_bce_loss is not None else zero.detach(),
            'loss_ebr_kl': self._last_ebr_kl_loss if self._last_ebr_kl_loss is not None else zero.detach(),
            'ebr_pos_score': self._last_ebr_pos_score if self._last_ebr_pos_score is not None else zero.detach(),
            'ebr_neg_score': self._last_ebr_neg_score if self._last_ebr_neg_score is not None else zero.detach(),
            'ebr_num_pos': torch.tensor(float(self._last_ebr_num_pos), device=x.device),
            'ebr_num_neg': torch.tensor(float(self._last_ebr_num_neg), device=x.device),
            'loss_bismooth': loss_bismooth,
            'loss_bismooth_cls': loss_bismooth_cls,
            'loss_layerwise_gate': loss_layerwise_gate,
            'bismooth_valid_ratio': bismooth_valid_ratio.detach(),
            'layerwise_gate_mean': layerwise_gate_mean.detach(),
            'layerwise_gate_layers': torch.tensor(float(self._last_layerwise_gate_layers), device=x.device),
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'causal_score_mean': causal_score.mean().detach(),
            'pollution_score_mean': pollution_score.mean().detach(),
            'counterexample_penalty_mean': self.counterexample_penalty.mean().detach(),
            'counterexample_penalty_batch_mean': current_counterexample_penalty.mean().detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_mixed_contexts': torch.tensor(float(num_mixed_contexts), device=x.device),
            'num_gmm_contexts': torch.tensor(float(num_gmm_contexts), device=x.device),
            'num_global_contexts': torch.tensor(float(num_global_contexts), device=x.device),
            'num_layerwise_contexts': torch.tensor(float(num_layerwise_contexts), device=x.device),
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
