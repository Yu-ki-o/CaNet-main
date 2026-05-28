import math
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


class LatentDiffusionDenoiser(nn.Module):
    """
    Small DDPM-style denoiser for node latent states.

    It trains by predicting Gaussian noise added to z.  The predicted clean
    latent is used as an optional residual blend, so the main graph encoder
    remains intact when diffusion_blend is zero.
    """

    def __init__(self, hidden_dim, steps=20, beta_start=1e-4, beta_end=2e-2, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.steps = max(1, int(steps))
        betas = torch.linspace(float(beta_start), float(beta_end), self.steps).clamp(1e-6, 0.999)
        alphas = 1.0 - betas
        self.register_buffer('alpha_bars', torch.cumprod(alphas, dim=0))
        self.time_embed = nn.Embedding(self.steps, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.time_embed.reset_parameters()
        for module in self.net:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.norm.reset_parameters()

    def _predict_noise(self, z_t, t):
        t_emb = self.time_embed(t)
        return self.net(torch.cat([z_t, t_emb], dim=-1))

    def forward(self, z, training=False, blend=0.0):
        zero = z.new_zeros(())
        if z.numel() == 0:
            return z, zero

        if training:
            t = torch.randint(self.steps, (z.size(0),), device=z.device)
            alpha_bar = self.alpha_bars.to(device=z.device, dtype=z.dtype).index_select(0, t).unsqueeze(-1)
            noise = torch.randn_like(z)
            z_t = alpha_bar.sqrt() * z + (1.0 - alpha_bar).sqrt() * noise
            pred_noise = self._predict_noise(z_t, t)
            loss = F.mse_loss(pred_noise, noise)
        else:
            t = torch.zeros(z.size(0), device=z.device, dtype=torch.long)
            alpha_bar = self.alpha_bars.to(device=z.device, dtype=z.dtype).index_select(0, t).unsqueeze(-1)
            z_t = z
            pred_noise = self._predict_noise(z_t, t)
            loss = zero

        z0_hat = (z_t - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp_min(1e-6)
        z0_hat = self.norm(z0_hat)
        if blend <= 0.0:
            return z, loss
        return z + float(blend) * (z0_hat - z), loss


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
        # Class-family neighbor modeling. Same-class edges keep dimensions
        # with high similarity; different-class edges keep dimensions with
        # high difference.  Within each family, dimension-wise neighbor
        # variance estimates uncertainty: stable dimensions are treated as
        # causal, highly varying dimensions as environment/noise.
        self.use_class_neighbor_uncertainty = bool(
            getattr(args, 'use_class_neighbor_uncertainty', False)
        )
        self.class_neighbor_label_source = getattr(args, 'class_neighbor_label_source', 'pred')
        if self.class_neighbor_label_source not in ('pred', 'detach_pred', 'train_label', 'label'):
            self.class_neighbor_label_source = 'pred'
        self.class_neighbor_uncertainty_temp = max(
            1e-3,
            float(getattr(args, 'class_neighbor_uncertainty_temp', 1.0)),
        )
        self.class_neighbor_uncertainty_blend = min(
            max(float(getattr(args, 'class_neighbor_uncertainty_blend', 1.0)), 0.0),
            1.0,
        )
        self.class_neighbor_min_gate = min(
            max(float(getattr(args, 'class_neighbor_min_gate', 0.05)), 0.0),
            1.0,
        )
        self.class_neighbor_same_threshold = min(
            max(float(getattr(args, 'class_neighbor_same_threshold', 0.5)), 0.0),
            1.0,
        )
        self.class_neighbor_test_agg = getattr(args, 'class_neighbor_test_agg', 'same_only')
        if self.class_neighbor_test_agg not in ('same_only', 'all_masked'):
            self.class_neighbor_test_agg = 'same_only'
        self.same_family_edge_pair_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.diff_family_edge_pair_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.same_family_edge_score_head = nn.Linear(self.d, edge_gate_out_dim)
        self.diff_family_edge_score_head = nn.Linear(self.d, edge_gate_out_dim)
        self.same_family_diff_energy = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.diff_family_diff_energy = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.family_shared_mask_head = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.same_family_mask_delta = nn.Linear(self.d, self.d)
        self.diff_family_mask_delta = nn.Linear(self.d, self.d)
        self.same_family_pred_head = nn.Linear(self.d, self.c)
        self.diff_family_pred_head = nn.Linear(self.d, self.c)
        self.lambda_class_neighbor_uncert = max(
            0.0,
            float(getattr(args, 'lambda_class_neighbor_uncert', 0.0)),
        )
        self.class_neighbor_ce_weight = max(
            0.0,
            float(getattr(args, 'class_neighbor_ce_weight', 1.0)),
        )
        self.class_neighbor_energy_weight = max(
            0.0,
            float(getattr(args, 'class_neighbor_energy_weight', 0.1)),
        )
        self.class_neighbor_var_weight = max(
            0.0,
            float(getattr(args, 'class_neighbor_var_weight', 0.1)),
        )
        self.edge_summary_norm = nn.LayerNorm(self.d)
        self.node_edge_fuser = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.node_edge_norm = nn.LayerNorm(self.d)

        # Graph-CFAM: graph causal feature adaptation inspired by CDDGNet.
        # It decomposes each layer representation into a local-smooth component
        # and a graph high-pass residual, then uses residual/energy cues to
        # re-attend the smooth component dimension-wise.  The causal-local part
        # is fused into the node state, while the complementary domain-local
        # part is kept as the edge/DAG pollution summary.
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
        self.use_energy_cfam = bool(getattr(args, 'use_energy_cfam', False))
        self.use_energy_node_gate = bool(getattr(args, 'use_energy_node_gate', False))
        self.use_energy_edge_split = bool(getattr(args, 'use_energy_edge_split', False))
        self.energy_detach = bool(getattr(args, 'energy_detach', True))
        self.energy_prop_steps = max(0, int(getattr(args, 'energy_prop_steps', 2)))
        self.energy_prop_gamma = min(max(float(getattr(args, 'energy_prop_gamma', 0.7)), 0.0), 1.0)
        self.energy_cfam_bias_scale = max(0.0, float(getattr(args, 'energy_cfam_bias_scale', 1.0)))
        self.energy_node_gate_scale = max(0.0, float(getattr(args, 'energy_node_gate_scale', 1.0)))
        self.energy_edge_threshold = min(max(float(getattr(args, 'energy_edge_threshold', 0.5)), 0.0), 1.0)
        self.energy_edge_consistency_weight = min(
            max(float(getattr(args, 'energy_edge_consistency_weight', 0.5)), 0.0),
            1.0,
        )
        self.energy_min_causal_edge_ratio = min(
            max(float(getattr(args, 'energy_min_causal_edge_ratio', 0.05)), 0.0),
            1.0,
        )
        self.energy_max_causal_edge_ratio = min(
            max(float(getattr(args, 'energy_max_causal_edge_ratio', 1.0)), 0.0),
            1.0,
        )
        self.lambda_energy_reg = max(0.0, float(getattr(args, 'lambda_energy_reg', 0.0)))
        self.energy_reg_norm_weight = max(0.0, float(getattr(args, 'energy_reg_norm_weight', 1.0)))
        self.energy_reg_mass_weight = max(0.0, float(getattr(args, 'energy_reg_mass_weight', 1.0)))
        self.energy_reg_mean_weight = max(0.0, float(getattr(args, 'energy_reg_mean_weight', 0.0)))
        self.graph_cfam_gate = nn.Sequential(
            nn.Linear(self.d * 5, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.energy_cfam_gate_bias = nn.Sequential(
            nn.Linear(3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.energy_node_gate = nn.Sequential(
            nn.Linear(3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, 1),
        )
        self.graph_cfam_norm = nn.LayerNorm(self.d)
        self._last_graph_cfam_gate_loss = None
        self._last_graph_cfam_gate_mean = None
        self._last_graph_cfam_layers = 0
        self._last_energy_raw_mean = None
        self._last_energy_prop_mean = None
        self._last_energy_delta_mean = None
        self._last_energy_node_gate_mean = None
        self._last_energy_edge_score_mean = None
        self._last_energy_causal_edge_ratio = None
        self._last_energy_num_causal_edges = None
        self._last_energy_num_env_edges = None

        self.noise_summary_norm = nn.LayerNorm(self.d)

        # CIPT-style decomposition: keep node enhancement and front-door
        # adjustment decoupled, and use two adapters as a temporary
        # causal/spurious splitter before the front-door path.
        bottleneck_dim = max(16, self.d // 2)
        adapter_dropout = float(getattr(args, 'dropout', 0.0))
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

        # Compatibility diagnostics: keep compact node/edge projections so the
        # existing logging/state-update code can run, but do not use them to
        # choose the mediator while DAG is disabled.
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

        # DAG-free main path: use the enhanced node representation z directly
        # as the causal mediator.  The old DAG/ICA split modules are kept only
        # as compatibility fields for checkpoints/logging; they are not used by
        # the forward or loss path.
        self.use_enhanced_as_causal = True
        self.use_ica_split = False
        requested_ica_dim = int(getattr(args, 'ica_components', self.dag_latent_dim))
        self.ica_dim = max(1, min(requested_ica_dim, self.d))
        self.ica_proj = nn.Linear(self.d, self.ica_dim, bias=False)
        self.ica_norm = nn.LayerNorm(self.ica_dim)
        self.ica_component_gate = Parameter(torch.zeros(self.ica_dim))
        self.ica_causal_proj = nn.Linear(self.ica_dim, self.d)
        self.ica_spurious_proj = nn.Linear(self.ica_dim, self.d)
        self.ica_gate_temp = max(1e-3, float(getattr(args, 'ica_gate_temp', 1.0)))
        self.ica_gate_target = min(max(float(getattr(args, 'ica_gate_target', 0.5)), 0.0), 1.0)
        self.lambda_ica_cov = float(getattr(args, 'lambda_ica_cov', 0.0))
        self.lambda_ica_ng = float(getattr(args, 'lambda_ica_ng', 0.0))
        self.lambda_ica_gate = float(getattr(args, 'lambda_ica_gate', 0.0))
        self.lambda_ica_entropy = float(getattr(args, 'lambda_ica_entropy', 0.0))
        self._last_ica_cov_loss = None
        self._last_ica_ng_loss = None
        self._last_ica_gate_loss = None
        self._last_ica_entropy_loss = None
        self._last_ica_gate_mean = None

        # Front-door fusion: mediator + context -> intervened representation.
        self.fd_fuser = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.fd_norm = nn.LayerNorm(self.d)
        self.use_dag_mixer = False
        self.dag_mixer = None

        self.dropout = getattr(args, 'dropout', 0.0)
        self.gamma = getattr(args, 'gamma', 0.99)
        self.fd_blend = getattr(args, 'fd_blend', 0.5)
        self.eval_pred_mode = getattr(args, 'eval_pred_mode', 'blend')
        if self.eval_pred_mode not in ('blend', 'mediator', 'frontdoor'):
            self.eval_pred_mode = 'blend'
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

        # NeGo-lite negative-inference environment sampler.
        # It augments the front-door context bank with class-conditioned
        # extra-class environment answers, rather than replacing the mediator
        # or DAG split.
        self.use_nego_prompt = bool(getattr(args, 'use_nego_prompt', False))
        self.use_nego_context = bool(getattr(args, 'use_nego_context', self.use_nego_prompt))
        self.lambda_nego = float(getattr(args, 'lambda_nego', 0.0))
        self.nego_temp = max(1e-3, float(getattr(args, 'nego_temp', 0.2)))
        self.nego_context_weight = max(0.0, float(getattr(args, 'nego_context_weight', 1.0)))
        self.nego_momentum = min(max(float(getattr(args, 'nego_momentum', 0.9)), 0.0), 1.0)
        self.nego_detach_source = bool(getattr(args, 'nego_detach_source', True))
        self.nego_source = getattr(args, 'nego_source', 'spurious')
        if self.nego_source not in ('spurious', 'mediator', 'z'):
            self.nego_source = 'spurious'
        self.nego_context_mode = getattr(args, 'nego_context_mode', 'class_mean')
        if self.nego_context_mode not in ('class_mean', 'sample_mix'):
            self.nego_context_mode = 'class_mean'
        self.nego_mix_k = max(1, int(getattr(args, 'nego_mix_k', 3)))
        self.nego_mix_alpha = max(1e-3, float(getattr(args, 'nego_mix_alpha', 0.5)))
        self.fd_context_source = getattr(args, 'fd_context_source', 'mixed')
        if self.fd_context_source not in ('mixed', 'nego_only'):
            self.fd_context_source = 'mixed'
        self.nego_prompts = Parameter(torch.zeros(self.c, self.d))
        self.nego_prompt_decoder = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Dropout(p=adapter_dropout),
            nn.Linear(self.d, self.d),
        )
        self.nego_prompt_norm = nn.LayerNorm(self.d)
        self._last_nego_context_mean = None

        # Class-conditioned fine-grained environment prototypes.
        # Enhanced node representations z provide the prototype content, while
        # a detached assignment head separates each class into K soft modes.
        self.class_proto_k = max(1, int(getattr(args, 'class_proto_k', 3)))
        self.class_proto_temp = max(1e-3, float(getattr(args, 'class_proto_temp', 1.0)))
        self.class_proto_momentum = min(
            max(float(getattr(args, 'class_proto_momentum', 0.9)), 0.0),
            1.0,
        )
        self.class_proto_min_mass = max(1e-6, float(getattr(args, 'class_proto_min_mass', 1e-3)))
        self.class_proto_detach_assign = bool(getattr(args, 'class_proto_detach_assign', True))
        self.lambda_class_proto_var = float(getattr(args, 'lambda_class_proto_var', 0.0))
        self.lambda_class_proto_pos = float(getattr(args, 'lambda_class_proto_pos', 0.0))
        self.lambda_class_proto_neg = float(getattr(args, 'lambda_class_proto_neg', 0.0))
        self.lambda_class_proto_balance = float(getattr(args, 'lambda_class_proto_balance', 0.0))
        self.class_proto_assign_head = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Dropout(p=adapter_dropout),
            nn.Linear(self.d, self.class_proto_k),
        )

        self.lambda_l1 = 0.0
        self.lambda_dag = 0.0
        self.lambda_med = getattr(args, 'lambda_med', 0.5)
        self.lambda_spu = 0.0
        self.lambda_role = float(getattr(args, 'lambda_role', 0.0))
        self.role_med_y_weight = max(0.0, float(getattr(args, 'role_med_y_weight', 1.0)))
        self.role_spu_y_weight = max(0.0, float(getattr(args, 'role_spu_y_weight', 1.0)))
        self.role_spu_env_weight = max(0.0, float(getattr(args, 'role_spu_env_weight', 1.0)))
        self.role_med_env_weight = max(0.0, float(getattr(args, 'role_med_env_weight', 1.0)))
        self.lambda_fd = getattr(args, 'lambda_fd', 0.5)
        self.lambda_cf = float(getattr(args, 'lambda_cf', 0.0))
        self.cf_target = getattr(args, 'cf_target', 'mediator')
        if self.cf_target not in ('mediator', 'spurious', 'both'):
            self.cf_target = 'mediator'
        self.cf_mode = getattr(args, 'cf_mode', 'shuffle')
        if self.cf_mode.startswith('spurious_'):
            self.cf_mode = self.cf_mode[len('spurious_'):]
        if self.cf_mode not in ('shuffle', 'noise', 'zero'):
            self.cf_mode = 'shuffle'
        self.cf_samples = max(1, int(getattr(args, 'cf_samples', 1)))
        self.cf_consistency = getattr(args, 'cf_consistency', 'cauvq')
        if self.cf_consistency not in ('cauvq', 'kl', 'mse'):
            self.cf_consistency = 'cauvq'
        self.cf_temp = max(1e-3, float(getattr(args, 'cf_temp', 1.0)))
        self.cf_beta = max(0.0, float(getattr(args, 'cf_beta', 1.0)))
        self.cf_noise_std = max(0.0, float(getattr(args, 'cf_noise_std', 1.0)))
        self.lambda_fd_aug = getattr(args, 'lambda_fd_aug', 0.5)
        self.lambda_var = getattr(args, 'lambda_var', 0.0)
        self.lambda_ind = getattr(args, 'lambda_ind', 0.1)
        self.lambda_env = 0.0
        self.lambda_inv = getattr(args, 'lambda_inv', 0.1)
        self.lambda_global_env = getattr(args, 'lambda_global_env', 0.0)
        # Retired experimental CNS/noise branch.  Kept as zero-valued fields so
        # old checkpoints or scripts that inspect these attributes do not fail.
        self.use_complement_noise_smoothing = False
        self.lambda_cns = 0.0
        self.lambda_cns_cons = 0.0
        self.direct_z_spurious_mode = getattr(args, 'direct_z_spurious_mode', 'zero')
        if self.direct_z_spurious_mode not in ('shortcut', 'zero', 'z_adapter'):
            self.direct_z_spurious_mode = 'zero'
        self.lambda_enhance_sem = float(getattr(args, 'lambda_enhance_sem', 0.0))
        self.enhance_sem_mode = getattr(args, 'enhance_sem_mode', 'cosine')
        if self.enhance_sem_mode not in ('cosine', 'mse'):
            self.enhance_sem_mode = 'cosine'
        self.lambda_latent_diffusion = max(0.0, float(getattr(args, 'lambda_latent_diffusion', 0.0)))
        self.diffusion_blend = max(0.0, float(getattr(args, 'diffusion_blend', 0.0)))
        self.use_latent_diffusion = (
            self.lambda_latent_diffusion > 0.0
            or self.diffusion_blend > 0.0
            or bool(getattr(args, 'use_latent_diffusion', False))
        )
        if self.use_latent_diffusion:
            self.latent_diffusion = LatentDiffusionDenoiser(
                self.d,
                steps=getattr(args, 'diffusion_steps', 20),
                beta_start=getattr(args, 'diffusion_beta_start', 1e-4),
                beta_end=getattr(args, 'diffusion_beta_end', 2e-2),
                dropout=getattr(args, 'dropout', 0.0),
            )
        else:
            self.latent_diffusion = None
        self._last_diffusion_loss = None
        self.lambda_entropy_dro = max(0.0, float(getattr(args, 'lambda_entropy_dro', 0.0)))
        self.dro_entropy_beta = max(1e-6, float(getattr(args, 'dro_entropy_beta', 1.0)))
        self.dro_num_groups = max(1, int(getattr(args, 'dro_num_groups', 4)))
        self.dro_group_by = getattr(args, 'dro_group_by', 'degree_label')
        if self.dro_group_by not in ('degree', 'label', 'degree_label', 'none'):
            self.dro_group_by = 'degree_label'
        self._last_dro_entropy = None
        self._last_dro_max_weight = None
        self.lambda_gate = 0.0
        self.lambda_dag_label = 0.0
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
        self.dag_ablate_label_effect = bool(getattr(args, 'dag_ablate_label_effect', False))
        self.dag_ablate_causal_support = bool(getattr(args, 'dag_ablate_causal_support', False))
        self.dag_ablate_pollution = bool(getattr(args, 'dag_ablate_pollution', False))
        self.dag_ablate_acyclic_loss = bool(getattr(args, 'dag_ablate_acyclic_loss', False))
        self.dag_ablate_flow_consistency = bool(getattr(args, 'dag_ablate_flow_consistency', False))

        self.register_buffer('gmm_spu_mean', torch.zeros(self.num_envs, self.d))
        self.register_buffer('gmm_spu_var', torch.ones(self.num_envs, self.d))
        self.register_buffer('gmm_spu_valid', torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer('dag_allowed_mask', self.build_dag_allowed_mask())
        self.register_buffer('edge_env_sensitivity', torch.zeros(self.dag_latent_dim))
        self.register_buffer('counterexample_penalty', torch.zeros(self.dag_latent_dim))
        self.register_buffer('nego_context_bank', torch.zeros(self.c, self.d))
        self.register_buffer('nego_context_valid', torch.zeros(self.c, dtype=torch.bool))
        self.register_buffer('class_env_proto_bank', torch.zeros(self.c, self.class_proto_k, self.d))
        self.register_buffer('class_env_proto_valid', torch.zeros(self.c, self.class_proto_k, dtype=torch.bool))
        self.register_buffer('class_env_proto_usage', torch.zeros(self.c, self.class_proto_k))
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
        self._reset_module_parameters(self.same_family_edge_pair_encoder)
        self._reset_module_parameters(self.diff_family_edge_pair_encoder)
        self.same_family_edge_score_head.reset_parameters()
        self.diff_family_edge_score_head.reset_parameters()
        self._reset_module_parameters(self.same_family_diff_energy)
        self._reset_module_parameters(self.diff_family_diff_energy)
        self._reset_module_parameters(self.family_shared_mask_head)
        self.same_family_mask_delta.reset_parameters()
        self.diff_family_mask_delta.reset_parameters()
        self.same_family_pred_head.reset_parameters()
        self.diff_family_pred_head.reset_parameters()
        self.edge_summary_norm.reset_parameters()
        self._reset_module_parameters(self.node_edge_fuser)
        nn.init.zeros_(self.node_edge_fuser[-1].weight)
        nn.init.zeros_(self.node_edge_fuser[-1].bias)
        self.node_edge_norm.reset_parameters()
        self._reset_module_parameters(self.graph_cfam_gate)
        nn.init.zeros_(self.graph_cfam_gate[-1].weight)
        nn.init.zeros_(self.graph_cfam_gate[-1].bias)
        self._reset_module_parameters(self.energy_cfam_gate_bias)
        nn.init.zeros_(self.energy_cfam_gate_bias[-1].weight)
        nn.init.zeros_(self.energy_cfam_gate_bias[-1].bias)
        self._reset_module_parameters(self.energy_node_gate)
        nn.init.zeros_(self.energy_node_gate[-1].weight)
        nn.init.zeros_(self.energy_node_gate[-1].bias)
        self.graph_cfam_norm.reset_parameters()
        self._last_graph_cfam_gate_loss = None
        self._last_graph_cfam_gate_mean = None
        self._last_graph_cfam_layers = 0
        self._last_energy_raw_mean = None
        self._last_energy_prop_mean = None
        self._last_energy_delta_mean = None
        self._last_energy_node_gate_mean = None
        self._last_energy_edge_score_mean = None
        self._last_energy_causal_edge_ratio = None
        self._last_energy_num_causal_edges = None
        self._last_energy_num_env_edges = None
        self._last_same_family_edge_ratio = None
        self._last_explicit_family_edge_ratio = None
        self._last_family_energy_conf_mean = None
        self._last_family_diff_energy_mean = None
        self._last_family_mask_mean = None
        self._last_class_neighbor_uncert_loss = None
        self._last_same_family_uncertainty_mean = None
        self._last_diff_family_uncertainty_mean = None
        self._last_class_neighbor_causal_gate_mean = None
        self.noise_summary_norm.reset_parameters()
        if self.latent_diffusion is not None:
            self.latent_diffusion.reset_parameters()
        self._last_diffusion_loss = None
        self._last_dro_entropy = None
        self._last_dro_max_weight = None
        self._reset_module_parameters(self.node_dag_proj)
        self._reset_module_parameters(self.edge_dag_proj)
        self.dag_gate_expander.reset_parameters()
        nn.init.xavier_uniform_(self.dag_gate_expander.weight, gain=0.1)
        nn.init.zeros_(self.dag_gate_expander.bias)
        for module in self.fd_fuser:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.fd_norm.reset_parameters()
        if self.dag_mixer is not None:
            self.dag_mixer.reset_parameters()
        nn.init.normal_(self.nego_prompts, mean=0.0, std=0.02)
        self._reset_module_parameters(self.nego_prompt_decoder)
        self.nego_prompt_norm.reset_parameters()
        self._reset_module_parameters(self.class_proto_assign_head)
        nn.init.uniform_(self.A_feat, -0.01, 0.01)
        nn.init.zeros_(self.gate_base)
        self._reset_module_parameters(self.sem_reconstructor)
        nn.init.zeros_(self.sem_reconstructor[-1].weight)
        nn.init.zeros_(self.sem_reconstructor[-1].bias)
        nn.init.zeros_(self.dag_label_bias)
        nn.init.normal_(self.pseudo_env_emb, std=0.02)
        self._reset_module_parameters(self.spurious_label_head)
        self.ica_proj.reset_parameters()
        self.ica_norm.reset_parameters()
        nn.init.zeros_(self.ica_component_gate)
        self.ica_causal_proj.reset_parameters()
        self.ica_spurious_proj.reset_parameters()
        self.gmm_spu_mean.zero_()
        self.gmm_spu_var.fill_(1.0)
        self.gmm_spu_valid.zero_()
        self.nego_context_bank.zero_()
        self.nego_context_valid.zero_()
        self.class_env_proto_bank.zero_()
        self.class_env_proto_valid.zero_()
        self.class_env_proto_usage.zero_()
        self._last_nego_context_mean = None
        self.edge_env_sensitivity.zero_()
        self.counterexample_penalty.zero_()
        self._last_node_degree_signal = None
        self._last_layerwise_gate_loss = None
        self._last_layerwise_gate_mean = None
        self._last_layerwise_gate_layers = 0
        self._last_ica_cov_loss = None
        self._last_ica_ng_loss = None
        self._last_ica_gate_loss = None
        self._last_ica_entropy_loss = None
        self._last_ica_gate_mean = None

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
        if self.dag_ablate_label_effect:
            label_effect = C_flow.new_full((self.node_var_dim,), 0.5)
        else:
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
        if self.dag_ablate_causal_support:
            causal_support = torch.zeros_like(causal_support)

        # Incoming pressure from low-effect node dimensions or edge-summary
        # variables is treated as pollution because it is more likely to carry
        # environmental/structural shortcuts.
        low_effect_incoming = torch.matmul(node_flow.t(), low_weight_norm)
        pollution_score = low_effect_incoming + self.edge_pollution_coeff * edge_incoming
        pollution_score = self._normalize_score(pollution_score, default_value=0.0)
        if self.dag_ablate_pollution:
            pollution_score = torch.zeros_like(pollution_score)

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

    def compute_ica_split(self, z):
        s = self.ica_norm(self.ica_proj(z))
        gate = torch.sigmoid(self.ica_component_gate / self.ica_gate_temp)
        s_causal = s * gate.unsqueeze(0)
        s_spurious = s * (1.0 - gate).unsqueeze(0)
        z_mediator = self.causal_norm(self.ica_causal_proj(s_causal))
        z_spurious = self.spurious_norm(self.ica_spurious_proj(s_spurious))
        return s, gate, s_causal, s_spurious, z_mediator, z_spurious

    def compute_ica_regularization(self, s, gate):
        if s.numel() == 0:
            zero = self.ica_component_gate.new_zeros(())
            return zero, zero, zero, zero

        s_centered = s - s.mean(dim=0, keepdim=True)
        s_std = s_centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        s_norm = s_centered / s_std
        n = max(1, int(s_norm.size(0)))
        cov = torch.matmul(s_norm.t(), s_norm) / float(n)
        offdiag = cov - torch.diag_embed(torch.diagonal(cov))
        loss_cov = offdiag.pow(2).mean()

        # A positive differentiable proxy for non-Gaussianity.  Higher contrast
        # gets a smaller loss, avoiding a negative unbounded auxiliary term.
        contrast = torch.log(torch.cosh(s_norm.clamp(-5.0, 5.0))).mean()
        loss_ng = 1.0 / contrast.clamp_min(1e-6)
        loss_gate = (gate.mean() - self.ica_gate_target).pow(2)
        gate_clamped = gate.clamp(1e-6, 1.0 - 1e-6)
        loss_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        ).mean()
        return loss_cov, loss_ng, loss_gate, loss_entropy

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
        if mode == 'mul_signed_diff_degree':
            return 2 * self.d + 1
        if mode == 'mul_diff_degree':
            return 2 * self.d + 1
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
        raise ValueError(
            f"Unknown edge_feat_mode='{mode}'. Use one of: "
            "mul, diff, signed_diff, degree, mul_diff, mul_signed_diff, "
            "concat, concat_diff, mul_degree, diff_degree, "
            "mul_diff_degree, mul_signed_diff_degree."
        )

    def build_family_edge_feat(self, h_src, h_dst, deg_src, deg_dst, deg_max, same_mask):
        """
        Edge feature construction for same/different class families.

        Same-family edges use the requested edge feature mode directly.  For
        different-family edges, the modes that rely on multiplication replace
        the leading similarity signal with an absolute-difference signal, so
        `mul` keeps the original "similar dimensions" meaning for same-class
        neighbors while becoming "different dimensions" for heterophilic
        neighbors.
        """
        same_feat = self.build_edge_feat(h_src, h_dst, deg_src, deg_dst, deg_max)
        if not self.use_class_neighbor_uncertainty:
            return same_feat

        diff_family_feat = self.build_difference_edge_feat(h_src, h_dst, deg_src, deg_dst, deg_max)
        return torch.where(same_mask.unsqueeze(-1), same_feat, diff_family_feat)

    def build_difference_edge_feat(self, h_src, h_dst, deg_src, deg_dst, deg_max):
        mode = self.edge_feat_mode
        mul_feat = h_src * h_dst
        signed_diff_feat = h_src - h_dst
        diff_feat = torch.abs(signed_diff_feat)

        log_deg_src = torch.log1p(deg_src)
        log_deg_dst = torch.log1p(deg_dst)
        deg_pair = torch.maximum(log_deg_src, log_deg_dst) / deg_max.clamp_min(1.0)
        deg_pair = deg_pair.unsqueeze(-1)

        if mode == 'mul':
            diff_family_feat = diff_feat
        elif mode == 'mul_degree':
            diff_family_feat = torch.cat([diff_feat, deg_pair], dim=-1)
        elif mode == 'mul_diff':
            diff_family_feat = torch.cat([diff_feat, mul_feat], dim=-1)
        elif mode == 'mul_signed_diff':
            diff_family_feat = torch.cat([diff_feat, signed_diff_feat], dim=-1)
        elif mode == 'mul_diff_degree':
            diff_family_feat = torch.cat([diff_feat, mul_feat, deg_pair], dim=-1)
        elif mode == 'mul_signed_diff_degree':
            diff_family_feat = torch.cat([diff_feat, signed_diff_feat, deg_pair], dim=-1)
        else:
            diff_family_feat = self.build_edge_feat(h_src, h_dst, deg_src, deg_dst, deg_max)
        return diff_family_feat

    def infer_neighbor_family_labels(self, h, labels=None, label_mask=None):
        logits = self.classifier(h)
        if self.class_neighbor_label_source in ('detach_pred', 'train_label', 'label'):
            logits = logits.detach()
        if logits.size(-1) == 1:
            pred_labels = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
        else:
            pred_labels = logits.argmax(dim=-1).long()

        explicit_mask = torch.zeros(h.size(0), device=h.device, dtype=torch.bool)
        if self.class_neighbor_label_source in ('train_label', 'label') and labels is not None:
            label_values = self._flat_class_labels(labels).to(device=h.device)
            valid_label = (label_values >= 0) & (label_values < self.c)
            if self.class_neighbor_label_source == 'train_label':
                if label_mask is None:
                    explicit_mask = valid_label
                else:
                    explicit_mask = label_mask.to(device=h.device, dtype=torch.bool) & valid_label
            else:
                explicit_mask = valid_label
            pred_labels = torch.where(explicit_mask, label_values.long(), pred_labels)
        return pred_labels, explicit_mask

    def infer_edge_same_by_feature(self, h_src, h_dst):
        mode = self.edge_feat_mode
        diff_mag = torch.abs(h_src - h_dst).mean(dim=-1)
        diff_same_score = 1.0 - self._normalize_score(diff_mag, default_value=0.5)
        cosine_score = 0.5 * (F.cosine_similarity(h_src, h_dst, dim=-1) + 1.0)
        mul_score = self._normalize_score((h_src * h_dst).mean(dim=-1), default_value=0.5)

        if mode in ('diff', 'signed_diff', 'diff_degree'):
            same_score = diff_same_score
        elif mode in (
            'mul',
            'mul_degree',
            'mul_diff',
            'mul_signed_diff',
            'mul_diff_degree',
            'mul_signed_diff_degree',
        ):
            same_score = 0.5 * (mul_score + cosine_score)
        else:
            same_score = cosine_score
        return same_score >= self.class_neighbor_same_threshold, same_score

    def aggregate_family_uncertainty(self, evidence, dst, family_mask, num_nodes):
        zero = evidence.new_zeros(num_nodes, evidence.size(1))
        if evidence.numel() == 0 or not bool(family_mask.any()):
            uncertainty = evidence.new_full((num_nodes, evidence.size(1)), 0.5)
            valid = torch.zeros(num_nodes, device=evidence.device, dtype=torch.bool)
            return uncertainty, valid

        family_evidence = evidence * family_mask.to(evidence.dtype).unsqueeze(-1)
        mass = evidence.new_zeros(num_nodes, 1)
        mass.index_add_(0, dst, family_mask.to(evidence.dtype).unsqueeze(-1))

        mean = zero.clone()
        mean.index_add_(0, dst, family_evidence)
        mean = mean / mass.clamp_min(1.0)

        second = zero.clone()
        second.index_add_(0, dst, family_evidence.pow(2))
        second = second / mass.clamp_min(1.0)
        var = (second - mean.pow(2)).clamp_min(0.0)

        valid = mass.squeeze(-1) > 0.0
        uncertainty = self._normalize_score(var.reshape(-1), default_value=0.5).view_as(var)
        uncertainty = torch.where(valid.unsqueeze(-1), uncertainty, uncertainty.new_full(uncertainty.shape, 0.5))
        return uncertainty, valid

    def aggregate_family_scalar_variance(self, values, dst, family_mask, num_nodes):
        if values.numel() == 0 or not bool(family_mask.any()):
            return values.new_full((num_nodes,), 0.5), torch.zeros(num_nodes, device=values.device, dtype=torch.bool)

        weight = family_mask.to(values.dtype)
        mass = values.new_zeros(num_nodes)
        mass.index_add_(0, dst, weight)
        mean = values.new_zeros(num_nodes)
        mean.index_add_(0, dst, values * weight)
        mean = mean / mass.clamp_min(1.0)
        second = values.new_zeros(num_nodes)
        second.index_add_(0, dst, values.pow(2) * weight)
        second = second / mass.clamp_min(1.0)
        var = (second - mean.pow(2)).clamp_min(0.0)
        valid = mass > 0.0
        var = self._normalize_score(var, default_value=0.5)
        var = torch.where(valid, var, var.new_full(var.shape, 0.5))
        return var, valid

    def compute_family_difference_energy_confidence(self, diff_info, family_evidence, same_mask):
        same_energy = self.same_family_diff_energy(diff_info).pow(2)
        diff_energy = self.diff_family_diff_energy(diff_info).pow(2)
        diff_energy = torch.where(same_mask.unsqueeze(-1), same_energy, diff_energy)

        energy_score = self._normalize_score(diff_energy.reshape(-1), default_value=0.5).view_as(diff_energy)
        evidence_score = self._normalize_score(
            family_evidence.abs().reshape(-1),
            default_value=0.5,
        ).view_as(family_evidence)
        confidence = torch.sigmoid(
            (evidence_score - energy_score) / self.class_neighbor_uncertainty_temp
        )
        return confidence.clamp_min(self.class_neighbor_min_gate), energy_score

    def compute_class_family_causal_gate(
        self,
        h_src,
        h_dst,
        dst,
        same_mask,
        edge_gate,
        num_nodes,
        family_evidence,
        diff_info,
        energy_confidence=None,
        energy_score=None,
    ):
        if not self.use_class_neighbor_uncertainty:
            return edge_gate, None

        same_uncertainty, same_valid = self.aggregate_family_uncertainty(
            diff_info,
            dst,
            same_mask,
            num_nodes,
        )
        diff_uncertainty, diff_valid = self.aggregate_family_uncertainty(
            diff_info,
            dst,
            ~same_mask,
            num_nodes,
        )

        same_conf = torch.sigmoid((0.5 - same_uncertainty) / self.class_neighbor_uncertainty_temp)
        diff_conf = torch.sigmoid((0.5 - diff_uncertainty) / self.class_neighbor_uncertainty_temp)
        same_conf = torch.where(same_valid.unsqueeze(-1), same_conf, same_conf.new_full(same_conf.shape, 0.5))
        diff_conf = torch.where(diff_valid.unsqueeze(-1), diff_conf, diff_conf.new_full(diff_conf.shape, 0.5))

        family_conf = torch.where(same_mask.unsqueeze(-1), same_conf[dst], diff_conf[dst])
        family_conf = family_conf.clamp_min(self.class_neighbor_min_gate)
        if edge_gate.size(1) == 1:
            edge_gate = edge_gate.expand(-1, self.d)
        blend = self.class_neighbor_uncertainty_blend
        if energy_confidence is not None:
            family_conf = family_conf * energy_confidence
        causal_gate = edge_gate * ((1.0 - blend) + blend * family_conf)
        causal_gate = causal_gate.clamp(0.0, 1.0)

        stats = {
            'same_ratio': same_mask.to(edge_gate.dtype).mean().detach(),
            'energy_conf_mean': energy_confidence.detach().mean() if energy_confidence is not None else edge_gate.new_zeros(()),
            'diff_energy_mean': energy_score.detach().mean() if energy_score is not None else edge_gate.new_zeros(()),
            'same_uncertainty_mean': same_uncertainty[same_valid].mean().detach() if same_valid.any() else edge_gate.new_zeros(()),
            'diff_uncertainty_mean': diff_uncertainty[diff_valid].mean().detach() if diff_valid.any() else edge_gate.new_zeros(()),
            'causal_gate_mean': causal_gate.detach().mean(),
        }
        return causal_gate, stats

    def compute_class_neighbor_uncertainty_loss(
        self,
        masked_relation,
        edge_logits,
        edge_energy,
        dst,
        same_mask,
        active_mask,
        num_nodes,
        labels=None,
    ):
        zero = masked_relation.new_zeros(())
        if not self.use_class_neighbor_uncertainty or masked_relation.numel() == 0:
            return zero

        active_mask = active_mask.to(device=masked_relation.device, dtype=torch.bool)
        if not active_mask.any():
            return zero

        loss_ce = zero
        if labels is not None and self.class_neighbor_ce_weight > 0.0:
            flat_labels = self._flat_class_labels(labels).to(device=masked_relation.device)
            target = flat_labels[dst]
            valid_target = active_mask & (target >= 0) & (target < self.c)
            if valid_target.any():
                logits_valid = edge_logits[valid_target]
                target_valid = target[valid_target]
                if logits_valid.size(-1) == 1:
                    loss_ce = F.binary_cross_entropy_with_logits(
                        logits_valid.squeeze(-1),
                        target_valid.to(dtype=logits_valid.dtype),
                    )
                else:
                    loss_ce = F.cross_entropy(logits_valid, target_valid.long())

        loss_energy = zero
        if self.class_neighbor_energy_weight > 0.0:
            energy_score = self._normalize_score(edge_energy[active_mask], default_value=0.5)
            loss_energy = energy_score.mean()

        loss_var = zero
        if self.class_neighbor_var_weight > 0.0:
            same_active = same_mask & active_mask
            diff_active = (~same_mask) & active_mask
            same_rel_uncert, same_rel_valid = self.aggregate_family_uncertainty(
                masked_relation,
                dst,
                same_active,
                num_nodes,
            )
            diff_rel_uncert, diff_rel_valid = self.aggregate_family_uncertainty(
                masked_relation,
                dst,
                diff_active,
                num_nodes,
            )
            same_energy_var, same_energy_valid = self.aggregate_family_scalar_variance(
                edge_energy,
                dst,
                same_active,
                num_nodes,
            )
            diff_energy_var, diff_energy_valid = self.aggregate_family_scalar_variance(
                edge_energy,
                dst,
                diff_active,
                num_nodes,
            )
            var_terms = []
            if same_rel_valid.any():
                var_terms.append(same_rel_uncert[same_rel_valid].mean())
            if diff_rel_valid.any():
                var_terms.append(diff_rel_uncert[diff_rel_valid].mean())
            if same_energy_valid.any():
                var_terms.append(same_energy_var[same_energy_valid].mean())
            if diff_energy_valid.any():
                var_terms.append(diff_energy_var[diff_energy_valid].mean())
            if var_terms:
                loss_var = torch.stack(var_terms).mean()

        return (
            self.class_neighbor_ce_weight * loss_ce
            + self.class_neighbor_energy_weight * loss_energy
            + self.class_neighbor_var_weight * loss_var
        )

    def compute_edge_summaries(self, h, edge_index, training=False, labels=None, label_mask=None):
        """
        Build both useful and low-relevance neighbor summaries.

        useful_summary_v = sum_u norm_uv * g_uv       * h_u
        noise_summary_v  = sum_u norm_uv * (1 - g_uv) * h_u

        The existing edge feature modes (mul/diff/degree/...) define g_uv.
        When edge_gate_mode='scalar', g_uv is one score per edge. When
        edge_gate_mode='vector', g_uv is a per-dimension edge gate so each
        hidden channel can keep or reject a neighbor independently.  The
        complementary low-gate summary is now used as a shortcut/spurious
        source instead of being subtracted from h.
        """
        if edge_index.numel() == 0:
            zero = h.new_zeros(h.size())
            scalar_zero = h.new_zeros(())
            self._last_same_family_edge_ratio = scalar_zero
            self._last_explicit_family_edge_ratio = scalar_zero
            self._last_family_energy_conf_mean = scalar_zero
            self._last_family_diff_energy_mean = scalar_zero
            self._last_family_mask_mean = scalar_zero
            self._last_class_neighbor_uncert_loss = scalar_zero
            self._last_same_family_uncertainty_mean = scalar_zero
            self._last_diff_family_uncertainty_mean = scalar_zero
            self._last_class_neighbor_causal_gate_mean = scalar_zero
            return zero, zero, None

        src, dst = edge_index
        deg = degree(dst, h.size(0)).to(device=h.device, dtype=h.dtype).clamp_min(1.0)
        deg_max = torch.log1p(deg).max().clamp_min(1.0)

        h_src = h[src]
        h_dst = h[dst]
        if self.use_class_neighbor_uncertainty:
            family_labels, explicit_node_mask = self.infer_neighbor_family_labels(
                h,
                labels=labels,
                label_mask=label_mask,
            )
            explicit_edge_mask = explicit_node_mask[src] & explicit_node_mask[dst]
            label_same_mask = family_labels[src] == family_labels[dst]
            feature_same_mask, feature_same_score = self.infer_edge_same_by_feature(h_src, h_dst)
            same_mask = torch.where(explicit_edge_mask, label_same_mask, feature_same_mask)
            same_edge_feat = self.build_edge_feat(
                h_src,
                h_dst,
                deg[src],
                deg[dst],
                deg_max,
            )
            diff_edge_feat = self.build_difference_edge_feat(
                h_src,
                h_dst,
                deg[src],
                deg[dst],
                deg_max,
            )
            same_hidden = self.same_family_edge_pair_encoder(same_edge_feat)
            diff_hidden = self.diff_family_edge_pair_encoder(diff_edge_feat)
            same_hidden = F.dropout(same_hidden, self.dropout, training=training)
            diff_hidden = F.dropout(diff_hidden, self.dropout, training=training)
            edge_hidden = torch.where(same_mask.unsqueeze(-1), same_hidden, diff_hidden)

            shared_mask = torch.sigmoid(self.family_shared_mask_head(edge_hidden))
            same_causal_mask = shared_mask * torch.sigmoid(self.same_family_mask_delta(same_hidden))
            diff_causal_mask = shared_mask * torch.sigmoid(self.diff_family_mask_delta(diff_hidden))
            causal_mask = torch.where(same_mask.unsqueeze(-1), same_causal_mask, diff_causal_mask)

            family_evidence = torch.where(same_mask.unsqueeze(-1), same_hidden, diff_hidden)
            diff_info = torch.abs(h_src - h_dst)
            energy_confidence, diff_energy_score = self.compute_family_difference_energy_confidence(
                diff_info,
                family_evidence,
                same_mask,
            )
            masked_same = same_causal_mask * same_hidden
            masked_diff = diff_causal_mask * diff_hidden
            same_pred_logits = self.same_family_pred_head(masked_same)
            diff_pred_logits = self.diff_family_pred_head(masked_diff)
            family_pred_logits = torch.where(same_mask.unsqueeze(-1), same_pred_logits, diff_pred_logits)
            family_energy = self.compute_logit_energy(family_pred_logits)

            same_logits = self.same_family_edge_score_head(same_hidden)
            diff_logits = self.diff_family_edge_score_head(diff_hidden)
            edge_logits = torch.where(same_mask.unsqueeze(-1), same_logits, diff_logits)
        else:
            same_mask = None
            explicit_edge_mask = None
            causal_mask = None
            family_evidence = None
            energy_confidence = None
            diff_energy_score = None
            family_pred_logits = None
            family_energy = None
            edge_feat = self.build_edge_feat(
                h_src,
                h_dst,
                deg[src],
                deg[dst],
                deg_max,
            )
            edge_hidden = self.edge_pair_encoder(edge_feat)
            edge_hidden = F.dropout(edge_hidden, self.dropout, training=training)
            edge_logits = self.edge_score_head(edge_hidden)
        edge_logits = edge_logits / self.edge_score_temp
        edge_gate = torch.sigmoid(edge_logits)
        if edge_gate.dim() == 1:
            edge_gate = edge_gate.unsqueeze(-1)
        if self.use_class_neighbor_uncertainty:
            edge_gate, family_stats = self.compute_class_family_causal_gate(
                h_src,
                h_dst,
                dst,
                same_mask,
                edge_gate,
                h.size(0),
                family_evidence,
                torch.abs(h_src - h_dst),
                energy_confidence=energy_confidence,
                energy_score=diff_energy_score,
            )
            edge_gate = edge_gate * causal_mask
            if self.training or self.class_neighbor_test_agg == 'same_only':
                aggregate_edge_mask = same_mask
            else:
                aggregate_edge_mask = torch.ones_like(same_mask, dtype=torch.bool)
            aggregation_mask = aggregate_edge_mask.unsqueeze(-1).to(dtype=edge_gate.dtype)
            edge_gate = edge_gate * aggregation_mask
            if explicit_edge_mask.any():
                active_mask = explicit_edge_mask
            elif label_mask is not None:
                active_mask = label_mask.to(device=h.device, dtype=torch.bool)[dst]
            else:
                active_mask = torch.zeros_like(same_mask, dtype=torch.bool)
            self._last_class_neighbor_uncert_loss = self.compute_class_neighbor_uncertainty_loss(
                causal_mask * family_evidence,
                family_pred_logits,
                family_energy,
                dst,
                same_mask,
                active_mask,
                h.size(0),
                labels=labels,
            )
            self._last_same_family_edge_ratio = family_stats['same_ratio']
            self._last_explicit_family_edge_ratio = explicit_edge_mask.to(h.dtype).mean().detach()
            self._last_family_energy_conf_mean = family_stats['energy_conf_mean']
            self._last_family_diff_energy_mean = family_stats['diff_energy_mean']
            self._last_family_mask_mean = causal_mask.detach().mean()
            self._last_same_family_uncertainty_mean = family_stats['same_uncertainty_mean']
            self._last_diff_family_uncertainty_mean = family_stats['diff_uncertainty_mean']
            self._last_class_neighbor_causal_gate_mean = family_stats['causal_gate_mean']
        else:
            zero = h.new_zeros(())
            aggregation_mask = edge_gate.new_ones(edge_gate.shape)
            self._last_same_family_edge_ratio = zero
            self._last_explicit_family_edge_ratio = zero
            self._last_family_energy_conf_mean = zero
            self._last_family_diff_energy_mean = zero
            self._last_family_mask_mean = zero
            self._last_class_neighbor_uncert_loss = zero
            self._last_same_family_uncertainty_mean = zero
            self._last_diff_family_uncertainty_mean = zero
            self._last_class_neighbor_causal_gate_mean = edge_gate.detach().mean()

        norm = (deg[src].pow(-0.5) * deg[dst].pow(-0.5)).unsqueeze(-1)
        useful_weight = torch.nan_to_num(norm * edge_gate, nan=0.0, posinf=0.0, neginf=0.0)
        noise_weight = torch.nan_to_num(
            norm * (1.0 - edge_gate) * aggregation_mask,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        useful_summary = h.new_zeros(h.size())
        useful_summary.index_add_(0, dst, useful_weight * h_src)
        useful_summary = self.edge_summary_norm(useful_summary)

        noise_summary = h.new_zeros(h.size())
        noise_summary.index_add_(0, dst, noise_weight * h_src)
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
        return self.node_edge_norm(fused)

    def apply_latent_diffusion(self, z, training=False):
        if self.latent_diffusion is None:
            zero = z.new_zeros(())
            self._last_diffusion_loss = zero
            return z, zero
        z_denoised, loss_diffusion = self.latent_diffusion(
            z,
            training=training,
            blend=self.diffusion_blend,
        )
        self._last_diffusion_loss = loss_diffusion
        return z_denoised, loss_diffusion

    def _graph_cfam_energy(self, value):
        # Row-wise energy spectralization.  Squared activations highlight
        # dimensions that dominate local smooth/residual components, and the
        # normalization removes pure scale effects.
        energy = value.pow(2)
        denom = energy.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return energy / denom

    def compute_logit_energy(self, logits):
        if logits is None or logits.numel() == 0:
            return None
        if logits.size(-1) == 1:
            return -F.softplus(logits.squeeze(-1))
        return -torch.logsumexp(logits, dim=-1)

    def propagate_scalar_signal(self, signal, edge_index):
        if signal is None or signal.numel() == 0 or edge_index is None or edge_index.numel() == 0:
            return signal
        return gcn_backbone_conv(signal.view(-1, 1), edge_index).view(-1)

    def compute_structure_energy_uncertainty(self, logits, edge_index):
        energy_raw = self.compute_logit_energy(logits)
        if energy_raw is None:
            return None, None, None

        energy_prop = energy_raw
        for _ in range(self.energy_prop_steps):
            energy_neigh = self.propagate_scalar_signal(energy_prop, edge_index)
            energy_prop = self.energy_prop_gamma * energy_prop + (1.0 - self.energy_prop_gamma) * energy_neigh

        uncertainty = self._normalize_score(energy_prop, default_value=0.5)
        uncertainty_neigh = self._normalize_score(
            self.propagate_scalar_signal(uncertainty, edge_index),
            default_value=0.5,
        )
        uncertainty_delta = (uncertainty - uncertainty_neigh).abs()
        if self.energy_detach:
            uncertainty = uncertainty.detach()
            uncertainty_neigh = uncertainty_neigh.detach()
            uncertainty_delta = uncertainty_delta.detach()

        self._last_energy_raw_mean = self._normalize_score(energy_raw.detach(), default_value=0.5).mean()
        self._last_energy_prop_mean = uncertainty.detach().mean()
        self._last_energy_delta_mean = uncertainty_delta.detach().mean()
        return uncertainty, uncertainty_neigh, uncertainty_delta

    def build_energy_cfam_bias(self, uncertainty, uncertainty_neigh, uncertainty_delta):
        if (
            not self.use_energy_cfam
            or self.energy_cfam_bias_scale <= 0.0
            or uncertainty is None
            or uncertainty_neigh is None
            or uncertainty_delta is None
        ):
            return None
        features = torch.stack([uncertainty, uncertainty_neigh, uncertainty_delta], dim=-1)
        return self.energy_cfam_bias_scale * self.energy_cfam_gate_bias(features)

    def build_energy_node_gate(self, uncertainty, uncertainty_neigh, uncertainty_delta):
        if (
            not self.use_energy_node_gate
            or self.energy_node_gate_scale <= 0.0
            or uncertainty is None
            or uncertainty_neigh is None
            or uncertainty_delta is None
        ):
            return None
        features = torch.stack([uncertainty, uncertainty_neigh, uncertainty_delta], dim=-1)
        gate = torch.sigmoid(self.energy_node_gate(features) * self.energy_node_gate_scale)
        self._last_energy_node_gate_mean = gate.detach().mean()
        return gate

    def _topk_edge_ratio_mask(self, scores, mask, min_ratio, max_ratio):
        num_edges = scores.numel()
        if num_edges == 0:
            return mask

        min_keep = int(math.ceil(max(0.0, min_ratio) * num_edges))
        max_keep = int(math.ceil(max(0.0, min(max_ratio, 1.0)) * num_edges))
        max_keep = max(max_keep, min_keep, 1)

        if int(mask.sum().item()) < min_keep:
            k = min(max(min_keep, 1), num_edges)
            top_idx = torch.topk(scores, k=k, largest=True).indices
            add_mask = torch.zeros_like(mask)
            add_mask[top_idx] = True
            mask = mask | add_mask

        if max_ratio < 1.0 and int(mask.sum().item()) > max_keep:
            selected_idx = mask.nonzero(as_tuple=False).view(-1)
            selected_scores = scores[selected_idx]
            keep_local = torch.topk(selected_scores, k=max_keep, largest=True).indices
            new_mask = torch.zeros_like(mask)
            new_mask[selected_idx[keep_local]] = True
            mask = new_mask
        return mask

    def split_edges_by_energy_uncertainty(self, edge_index, uncertainty):
        if (
            not self.use_energy_edge_split
            or uncertainty is None
            or edge_index is None
            or edge_index.numel() == 0
        ):
            return None

        src, dst = edge_index
        u_src = uncertainty[src]
        u_dst = uncertainty[dst]
        confidence = 1.0 - 0.5 * (u_src + u_dst)
        consistency = 1.0 - (u_src - u_dst).abs()
        edge_scores = confidence * (
            (1.0 - self.energy_edge_consistency_weight)
            + self.energy_edge_consistency_weight * consistency
        )
        edge_scores = edge_scores.clamp(1e-6, 1.0 - 1e-6)

        causal_mask = edge_scores > self.energy_edge_threshold
        causal_mask = self._topk_edge_ratio_mask(
            edge_scores,
            causal_mask,
            self.energy_min_causal_edge_ratio,
            self.energy_max_causal_edge_ratio,
        )
        if int(causal_mask.sum().item()) == 0:
            causal_mask = torch.ones_like(causal_mask, dtype=torch.bool)
        env_mask = ~causal_mask

        self._last_energy_edge_score_mean = edge_scores.detach().mean()
        self._last_energy_causal_edge_ratio = causal_mask.float().mean().detach()
        self._last_energy_num_causal_edges = edge_scores.new_tensor(float(causal_mask.sum().item()))
        self._last_energy_num_env_edges = edge_scores.new_tensor(float(env_mask.sum().item()))

        return {
            'causal_edge_index': edge_index[:, causal_mask],
            'env_edge_index': edge_index[:, env_mask],
            'edge_scores': edge_scores,
            'causal_mask': causal_mask,
            'env_mask': env_mask,
        }

    def graph_cfam_adapt(
        self,
        h,
        edge_index,
        training=False,
        local_blend=None,
        residual_blend=None,
        uncertainty=None,
        uncertainty_neigh=None,
        uncertainty_delta=None,
        labels=None,
        label_mask=None,
    ):
        """
        Graph-CFAM local decoupling.

        smooth: relation-aware local low-pass information from neighbors.
        residual: node high-pass residual h - smooth.
        gate: dimension-wise re-attention.  High gate dimensions are treated as
        causal/stable local information; low gate dimensions are kept as
        domain/shortcut information for DAG/front-door contexts.
        """
        smooth, noise_summary, edge_gate = self.compute_edge_summaries(
            h,
            edge_index,
            training=training,
            labels=labels,
            label_mask=label_mask,
        )
        residual = h - smooth
        smooth_energy = self._graph_cfam_energy(smooth)
        residual_energy = self._graph_cfam_energy(residual)
        gate_input = torch.cat([h, smooth, residual, smooth_energy, residual_energy], dim=-1)
        gate_logits = self.graph_cfam_gate(gate_input)
        energy_bias = self.build_energy_cfam_bias(
            uncertainty,
            uncertainty_neigh,
            uncertainty_delta,
        )
        if energy_bias is not None:
            gate_logits = gate_logits + energy_bias
        gate = torch.sigmoid(gate_logits / self.graph_cfam_gate_temp)

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

    def _flat_class_labels(self, y):
        if y.dim() > 1 and y.size(1) > 1:
            return y.argmax(dim=1).long()
        return y.squeeze().long()

    def compute_graph_delf_loss(
        self,
        z_mediator,
        z_shortcut,
        final_logits,
        y,
        train_idx,
        criterion,
        args,
    ):
        """
        Graph-DELF: a graph analogue of decoupling-effect supervision.

        Ambiguous nodes are those that are hard for the final classifier and/or
        have strong local shortcut energy.  For each class, their mediator
        representation is pulled toward the stable class prototype and pushed
        away from the local shortcut prototype.  This directly supervises the
        local decoupling effect instead of only optimizing CE.
        """
        zero = z_mediator.new_zeros(())
        if (
            self.lambda_graph_delf <= 0.0
            or train_idx is None
            or train_idx.numel() <= 1
            or z_mediator.numel() == 0
            or z_shortcut is None
            or z_shortcut.numel() == 0
        ):
            return zero

        device = z_mediator.device
        train_idx = train_idx.to(device=device, dtype=torch.long)
        y = y.to(device)
        z_shortcut = z_shortcut.to(device)
        final_logits = final_logits.to(device)

        y_tr = y[train_idx]
        with torch.no_grad():
            raw_loss = self.compute_supervised_loss(
                final_logits,
                y_tr,
                criterion,
                args,
            )
            if raw_loss.dim() > 1:
                raw_loss = raw_loss.mean(dim=1)
            shortcut_energy = z_shortcut[train_idx].norm(dim=1)
            hard_score = self._normalize_score(raw_loss.detach(), default_value=0.5)
            shortcut_score = self._normalize_score(shortcut_energy.detach(), default_value=0.5)
            ambiguous_score = hard_score + shortcut_score
            top_k = max(1, int(round(float(train_idx.numel()) * self.graph_delf_top_frac)))
            top_k = min(top_k, int(train_idx.numel()))
            ambiguous_pos = ambiguous_score.topk(top_k).indices
            ambiguous_mask = torch.zeros(train_idx.numel(), device=train_idx.device, dtype=torch.bool)
            ambiguous_mask[ambiguous_pos] = True

        labels_flat = self._flat_class_labels(y)
        labels_train = labels_flat[train_idx]
        med_train = z_mediator[train_idx]
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
            causal_proto = med_train[stable_mask].mean(dim=0).detach()
            shortcut_proto = shortcut_train[class_mask].mean(dim=0).detach()
            med_amb = med_train[class_amb_mask]

            causal_align = 1.0 - F.cosine_similarity(
                F.normalize(med_amb, dim=1),
                F.normalize(causal_proto.unsqueeze(0), dim=1).expand_as(med_amb),
                dim=1,
            )
            shortcut_align = F.cosine_similarity(
                F.normalize(med_amb, dim=1),
                F.normalize(shortcut_proto.unsqueeze(0), dim=1).expand_as(med_amb),
                dim=1,
            )
            shortcut_push = F.relu(shortcut_align - self.graph_delf_margin)
            losses.append(causal_align.mean() + self.graph_delf_shortcut_weight * shortcut_push.mean())

        if not losses:
            return zero
        return torch.stack(losses).mean()

    def encode_representation(self, x, edge_index, training=False, labels=None, label_mask=None):
        x = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(self.input_proj(x))
        layerwise_states = []
        layerwise_gate_loss = h.new_zeros(())
        layerwise_gate_mean = h.new_zeros(())
        layerwise_gate_layers = 0
        graph_cfam_gate_loss = h.new_zeros(())
        graph_cfam_gate_mean = h.new_zeros(())
        graph_cfam_layers = 0
        num_backbone_layers = len(self.backbone_layers)

        # Pre-message-passing CFAM: filter/enhance projected node states before
        # the first GNN aggregation mixes potentially shortcut-heavy neighbours.
        if self.use_pre_gnn_graph_cfam:
            h, edge_summary_pre, domain_summary_pre, cfam_gate_pre, edge_gate_pre, cfam_gate_loss_pre = self.graph_cfam_adapt(
                h,
                edge_index,
                training=training,
                local_blend=self.pre_graph_cfam_blend,
                residual_blend=self.pre_graph_cfam_residual_blend,
                labels=labels,
                label_mask=label_mask,
            )
            graph_cfam_gate_mean = graph_cfam_gate_mean + cfam_gate_pre.mean()
            graph_cfam_gate_loss = graph_cfam_gate_loss + cfam_gate_loss_pre
            graph_cfam_layers += 1
            if edge_gate_pre is not None:
                layerwise_gate_mean = layerwise_gate_mean + edge_gate_pre.mean()
                if self.lambda_layerwise_gate > 0.0:
                    layerwise_gate_loss = layerwise_gate_loss + (
                        edge_gate_pre.mean() - self.layerwise_gate_target
                    ).pow(2)
                layerwise_gate_layers += 1
        for layer_idx, layer in enumerate(self.backbone_layers):
            h = F.dropout(h, self.dropout, training=training)
            h = self.act_fn(layer(h, edge_index))

            # Layer-wise Local-IGM message routing: after each ordinary GNN
            # propagation, re-score the original one-hop edges using current
            # node states, add useful messages, and optionally subtract a
            # softly gated low-relevance context branch. Because h at layer l
            # already contains l-hop information, this implicitly filters
            # multi-hop information without materializing A^2/A^3 ego graphs.
            if self.use_graph_cfam:
                should_route = not (
                    self.layerwise_local_igm_skip_last
                    and layer_idx == num_backbone_layers - 1
                )
                if should_route:
                    h, edge_summary_l, domain_summary_l, cfam_gate_l, edge_gate_l, cfam_gate_loss_l = self.graph_cfam_adapt(
                        h,
                        edge_index,
                        training=training,
                        labels=labels,
                        label_mask=label_mask,
                    )
                    graph_cfam_gate_mean = graph_cfam_gate_mean + cfam_gate_l.mean()
                    graph_cfam_gate_loss = graph_cfam_gate_loss + cfam_gate_loss_l
                    graph_cfam_layers += 1
                    if edge_gate_l is not None:
                        layerwise_gate_mean = layerwise_gate_mean + edge_gate_l.mean()
                        if self.lambda_layerwise_gate > 0.0:
                            layerwise_gate_loss = layerwise_gate_loss + (
                                edge_gate_l.mean() - self.layerwise_gate_target
                            ).pow(2)
                        layerwise_gate_layers += 1
            elif self.use_layerwise_local_igm:
                should_route = not (
                    self.layerwise_local_igm_skip_last
                    and layer_idx == num_backbone_layers - 1
                )
                if should_route:
                    edge_summary_l, noise_summary_l, edge_gate_l = self.compute_edge_summaries(
                        h,
                        edge_index,
                        training=training,
                        labels=labels,
                        label_mask=label_mask,
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
        if graph_cfam_layers > 0:
            graph_cfam_gate_mean = graph_cfam_gate_mean / float(graph_cfam_layers)
            graph_cfam_gate_loss = graph_cfam_gate_loss / float(graph_cfam_layers)
        self._last_graph_cfam_gate_mean = graph_cfam_gate_mean.detach()
        self._last_graph_cfam_gate_loss = graph_cfam_gate_loss
        self._last_graph_cfam_layers = int(graph_cfam_layers)
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

        h_pre_enhance = h
        shortcut_summary = None
        cns_gate = None
        energy_uncertainty = None
        energy_uncertainty_neigh = None
        energy_uncertainty_delta = None
        if self.use_energy_cfam or self.use_energy_node_gate:
            energy_probe_logits = self.classifier(h)
            (
                energy_uncertainty,
                energy_uncertainty_neigh,
                energy_uncertainty_delta,
            ) = self.compute_structure_energy_uncertainty(
                energy_probe_logits,
                edge_index,
            )
        else:
            zero_energy = h.new_zeros(())
            self._last_energy_raw_mean = zero_energy
            self._last_energy_prop_mean = zero_energy
            self._last_energy_delta_mean = zero_energy
            self._last_energy_node_gate_mean = zero_energy
        if self.use_energy_node_gate:
            edge_summary, noise_summary, _ = self.compute_edge_summaries(
                h,
                edge_index,
                training=training,
                labels=labels,
                label_mask=label_mask,
            )
            enhanced = self.fuse_node_edge_representation(
                h,
                edge_summary,
                noise_summary=noise_summary,
                training=training,
            )
            node_gate = self.build_energy_node_gate(
                energy_uncertainty,
                energy_uncertainty_neigh,
                energy_uncertainty_delta,
            )
            if node_gate is None:
                node_gate = torch.full((h.size(0), 1), 0.5, device=h.device, dtype=h.dtype)
                self._last_energy_node_gate_mean = node_gate.detach().mean()
            z = h + node_gate * (enhanced - h)
            shortcut_summary = noise_summary
            cns_gate = node_gate.expand_as(z)
        elif self.use_graph_cfam and self.use_final_graph_cfam:
            z, causal_local_final, edge_summary, cfam_gate_final, _, cfam_gate_loss_final = self.graph_cfam_adapt(
                h,
                edge_index,
                training=training,
                uncertainty=energy_uncertainty,
                uncertainty_neigh=energy_uncertainty_neigh,
                uncertainty_delta=energy_uncertainty_delta,
                labels=labels,
                label_mask=label_mask,
            )
            shortcut_summary = edge_summary
            cns_gate = cfam_gate_final
            prev_layers = max(0, int(self._last_graph_cfam_layers))
            denom = float(prev_layers + 1)
            self._last_graph_cfam_gate_mean = (
                self._last_graph_cfam_gate_mean * float(prev_layers) + cfam_gate_final.mean().detach()
            ) / denom
            self._last_graph_cfam_gate_loss = (
                self._last_graph_cfam_gate_loss * float(prev_layers) + cfam_gate_loss_final
            ) / denom
            self._last_graph_cfam_layers = prev_layers + 1
        elif self.use_graph_cfam:
            edge_summary, _, _ = self.compute_edge_summaries(
                h,
                edge_index,
                training=training,
                labels=labels,
                label_mask=label_mask,
            )
            z = h
            shortcut_summary = edge_summary
            cns_gate = torch.full_like(z, 0.5)
        else:
            edge_summary, noise_summary, _ = self.compute_edge_summaries(
                h,
                edge_index,
                training=training,
                labels=labels,
                label_mask=label_mask,
            )
            if self.use_layerwise_local_igm and not self.layerwise_final_edge_fuse:
                z = self.node_edge_norm(h)
            else:
                z = self.fuse_node_edge_representation(
                    h,
                    edge_summary,
                    noise_summary=noise_summary,
                    training=training,
                )
            shortcut_summary = noise_summary
            cns_gate = torch.full_like(z, 0.5)
        z_raw = z
        z_denoised, _ = self.apply_latent_diffusion(z_raw, training=training)
        dag_vars = z.new_zeros(z.size(0), self.non_label_var_dim)
        mediator_gate = z.new_ones(self.d)
        causal_score = z.new_ones(self.dag_latent_dim)
        pollution_score = z.new_zeros(self.dag_latent_dim)
        dag_total = self.A_feat.new_zeros(self.dag_var_dim, self.dag_var_dim)
        z = z_denoised
        z_mediator = z_denoised
        if self.direct_z_spurious_mode == 'zero':
            z_spurious = z.new_zeros(z.size())
        elif self.direct_z_spurious_mode == 'z_adapter':
            z_spurious = self.spurious_norm(z + 0.1 * self.spurious_adapter(z))
        else:
            if shortcut_summary is None:
                shortcut_summary = z_raw - z_denoised
            else:
                shortcut_summary = shortcut_summary + (z_raw - z_denoised)
            z_spurious = self.spurious_norm(shortcut_summary + 0.1 * self.spurious_adapter(shortcut_summary))
        zero = z.new_zeros(())
        self._last_ica_cov_loss = zero
        self._last_ica_ng_loss = zero
        self._last_ica_gate_loss = zero
        self._last_ica_entropy_loss = zero
        self._last_ica_gate_mean = zero

        z_mediator = F.dropout(z_mediator, self.dropout, training=training)
        z_spurious = F.dropout(z_spurious, self.dropout, training=training)
        layerwise_spurious = None
        if self.use_layerwise_spurious_contexts and layerwise_states:
            layerwise_spurious = None
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
            h_global_context,
            layerwise_spurious,
            h_pre_enhance,
            cns_gate,
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

    def get_nego_source_representation(self, z_all, z_mediator, z_spurious):
        if self.nego_source == 'mediator':
            return z_mediator
        if self.nego_source == 'z':
            return z_all
        return z_spurious

    def negative_prompt_answers(self, z_source):
        if (not self.use_nego_prompt and not self.use_nego_context) or z_source is None or z_source.numel() == 0:
            return None
        source = z_source.detach() if self.nego_detach_source else z_source
        prompts = self.nego_prompts.unsqueeze(0).expand(source.size(0), -1, -1)
        source_expand = source.unsqueeze(1).expand(-1, self.c, -1)
        decoder_input = torch.cat([source_expand, prompts, source_expand * prompts], dim=-1)
        answer = self.nego_prompt_decoder(decoder_input.reshape(-1, self.d * 3)).view(source.size(0), self.c, self.d)
        answer = self.nego_prompt_norm(answer + prompts)
        return F.normalize(answer, dim=-1)

    def sample_mixed_nego_contexts(self, answers, labels=None, training=False):
        if answers is None or answers.numel() == 0 or answers.size(1) <= 1:
            return None

        num_samples, num_classes, _ = answers.shape
        device = answers.device
        class_ids = torch.arange(num_classes, device=device)
        if labels is None:
            valid = torch.ones(num_samples, num_classes, device=device, dtype=torch.bool)
        else:
            labels = labels.to(device=device, dtype=torch.long).view(-1).clamp(min=-1, max=num_classes - 1)
            valid = class_ids.unsqueeze(0) != labels.unsqueeze(1)
        valid_count = valid.sum(dim=1, keepdim=True)
        if (valid_count <= 0).any():
            valid = torch.ones_like(valid)
            valid_count = valid.sum(dim=1, keepdim=True)

        valid_float = valid.float()
        num_mix = max(1, self.nego_mix_k)
        if training:
            concentration = torch.full(
                (num_samples, num_mix, num_classes),
                self.nego_mix_alpha,
                device=device,
                dtype=answers.dtype,
            )
            gamma = torch.distributions.Gamma(
                concentration,
                torch.ones_like(concentration),
            ).sample()
            weights = gamma * valid_float.unsqueeze(1)
        else:
            base = (class_ids.to(dtype=answers.dtype).view(1, 1, -1) + 1.0)
            mix_ids = (
                torch.arange(num_mix, device=device, dtype=answers.dtype).view(1, -1, 1)
                + 1.0
            )
            weights = torch.sin(base * 12.9898 + mix_ids * 78.233).abs() + 1e-3
            weights = weights.expand(num_samples, -1, -1) * valid_float.unsqueeze(1)

        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        answers_expand = answers.unsqueeze(1).expand(
            num_samples,
            num_mix,
            num_classes,
            self.d,
        ).reshape(num_samples * num_mix, num_classes, self.d)
        mixed = torch.bmm(
            weights.reshape(num_samples * num_mix, 1, num_classes).to(dtype=answers.dtype),
            answers_expand,
        ).view(num_samples, num_mix, self.d)
        return F.normalize(mixed, dim=-1)

    def get_nego_contexts(self, z_source=None, y=None, sample_idx=None, training=False):
        if not self.use_nego_context or self.nego_context_weight <= 0.0:
            return None

        contexts = None
        if z_source is not None and y is not None and sample_idx is not None and sample_idx.numel() > 0:
            y_flat = y.squeeze(-1).long()
            labels = y_flat[sample_idx]
            answers = self.negative_prompt_answers(z_source.index_select(0, sample_idx))
            if answers is not None:
                if self.nego_context_mode == 'sample_mix':
                    contexts = self.sample_mixed_nego_contexts(
                        answers,
                        labels=labels,
                        training=training,
                    )
                    if contexts is not None:
                        self._last_nego_context_mean = contexts.mean().detach()
                        return self.nego_context_weight * F.normalize(contexts, dim=-1)

                class_contexts = []
                for cls_idx in range(self.c):
                    extra_mask = labels != cls_idx
                    if extra_mask.any():
                        ctx = answers[extra_mask, cls_idx, :].mean(dim=0)
                    else:
                        ctx = answers[:, cls_idx, :].mean(dim=0)
                    class_contexts.append(F.normalize(ctx, dim=0))
                contexts = torch.stack(class_contexts, dim=0)
                self._last_nego_context_mean = contexts.mean().detach()

        if contexts is None:
            valid = self.nego_context_valid
            if valid.any():
                contexts = self.nego_context_bank[valid]
            else:
                # Cold-start fallback for evaluation before the first EMA update.
                contexts = F.normalize(self.nego_prompts, dim=1)

        if contexts is None or contexts.numel() == 0:
            return None
        return self.nego_context_weight * F.normalize(contexts, dim=1)

    def compute_nego_loss(self, z_source, y, train_idx):
        zero = self.nego_prompts.new_zeros(())
        if not self.use_nego_prompt or self.lambda_nego <= 0.0:
            return zero, zero, zero
        if z_source is None or z_source.numel() == 0 or train_idx is None or train_idx.numel() == 0:
            return zero, zero, zero

        y_flat = y.squeeze(-1).long()
        labels = y_flat[train_idx]
        source_tr = z_source.index_select(0, train_idx)
        answers = self.negative_prompt_answers(source_tr)
        if answers is None:
            return zero, zero, zero

        proto_values = source_tr.detach()
        prototypes = source_tr.new_zeros(self.c, self.d)
        valid = torch.zeros(self.c, device=source_tr.device, dtype=torch.bool)
        for cls_idx in range(self.c):
            cls_mask = labels == cls_idx
            if cls_mask.any():
                prototypes[cls_idx] = F.normalize(proto_values[cls_mask].mean(dim=0), dim=0)
                valid[cls_idx] = True

        if not valid.any():
            return zero, zero, zero

        proto = F.normalize(prototypes, dim=1)
        sim = (answers * proto.unsqueeze(0)).sum(dim=-1) / self.nego_temp
        class_ids = torch.arange(self.c, device=labels.device).unsqueeze(0)
        target = (labels.unsqueeze(1) != class_ids).to(sim.dtype)
        valid_mask = valid.unsqueeze(0).expand_as(sim)
        raw = F.binary_cross_entropy_with_logits(sim, target, reduction='none')
        loss = raw[valid_mask].mean() if valid_mask.any() else zero

        with torch.no_grad():
            prob = torch.sigmoid(sim.detach())
            pos_mask = (target > 0.5) & valid_mask
            neg_mask = (target < 0.5) & valid_mask
            pos_score = prob[pos_mask].mean() if pos_mask.any() else zero
            neg_score = prob[neg_mask].mean() if neg_mask.any() else zero
        return loss, pos_score.detach(), neg_score.detach()

    @torch.no_grad()
    def update_nego_context_bank(self, contexts):
        if contexts is None or contexts.numel() == 0:
            return
        contexts = contexts.detach()
        if contexts.dim() == 3:
            contexts = contexts.mean(dim=0)
        contexts = F.normalize(contexts, dim=-1)
        take = min(contexts.size(0), self.c)
        if take <= 0:
            return
        old = self.nego_context_bank[:take]
        valid = self.nego_context_valid[:take]
        blended = torch.where(
            valid.unsqueeze(-1),
            self.nego_momentum * old + (1.0 - self.nego_momentum) * contexts[:take],
            contexts[:take],
        )
        self.nego_context_bank[:take] = F.normalize(blended, dim=1)
        self.nego_context_valid[:take] = True

    def class_proto_objective_enabled(self):
        return (
            self.lambda_class_proto_var > 0.0
            or self.lambda_class_proto_pos > 0.0
            or self.lambda_class_proto_neg > 0.0
            or self.lambda_class_proto_balance > 0.0
        )

    def compute_class_proto_assign_probs(self, z):
        source = z.detach() if self.class_proto_detach_assign else z
        logits = self.class_proto_assign_head(source) / self.class_proto_temp
        return F.softmax(logits, dim=-1)

    def build_batch_class_env_prototypes(self, z, labels, assign_probs):
        proto = z.new_zeros(self.c, self.class_proto_k, self.d)
        usage = z.new_zeros(self.c, self.class_proto_k)
        valid = torch.zeros(self.c, self.class_proto_k, device=z.device, dtype=torch.bool)
        values = z.detach()

        for cls_idx in range(self.c):
            cls_mask = labels == cls_idx
            if not cls_mask.any():
                continue
            cls_values = values[cls_mask]
            cls_assign = assign_probs[cls_mask].clamp_min(0.0)
            for proto_idx in range(self.class_proto_k):
                weight = cls_assign[:, proto_idx]
                mass = weight.sum()
                usage[cls_idx, proto_idx] = mass
                if mass > self.class_proto_min_mass:
                    ctx = (weight.unsqueeze(-1) * cls_values).sum(dim=0) / mass.clamp_min(1e-6)
                    proto[cls_idx, proto_idx] = F.normalize(ctx, dim=0)
                    valid[cls_idx, proto_idx] = True

        return proto, valid, usage

    def merge_class_env_prototypes(self, batch_proto, batch_valid):
        bank = self.class_env_proto_bank.to(device=batch_proto.device, dtype=batch_proto.dtype).detach()
        bank_valid = self.class_env_proto_valid.to(device=batch_proto.device)
        proto = torch.where(batch_valid.unsqueeze(-1), batch_proto, bank)
        valid = batch_valid | bank_valid
        return F.normalize(proto, dim=-1), valid

    def compute_class_env_proto_loss(self, z, y, train_idx):
        zero = z.new_zeros(())
        if not self.class_proto_objective_enabled():
            empty_probs = z.new_zeros(z.size(0), self.class_proto_k)
            return zero, zero, zero, zero, zero, empty_probs, None
        if z is None or z.numel() == 0 or train_idx is None or train_idx.numel() == 0:
            empty_probs = z.new_zeros(0, self.class_proto_k)
            return zero, zero, zero, zero, zero, empty_probs, None

        y_flat = self._flat_class_labels(y)
        labels = y_flat[train_idx]
        z_tr = z.index_select(0, train_idx)
        valid_label_mask = (labels >= 0) & (labels < self.c)
        if not valid_label_mask.any():
            empty_probs = z_tr.new_zeros(z_tr.size(0), self.class_proto_k)
            return zero, zero, zero, zero, zero, empty_probs, None

        z_valid = z_tr[valid_label_mask]
        labels_valid = labels[valid_label_mask]
        assign_probs = self.compute_class_proto_assign_probs(z_valid)
        batch_proto, batch_valid, batch_usage = self.build_batch_class_env_prototypes(
            z_valid,
            labels_valid,
            assign_probs,
        )
        proto, proto_valid = self.merge_class_env_prototypes(batch_proto, batch_valid)
        balance_losses = []
        target_usage = 1.0 / float(self.class_proto_k)
        for cls_idx in labels_valid.unique().tolist():
            cls_mask = labels_valid == cls_idx
            if cls_mask.any():
                mean_assign = assign_probs[cls_mask].mean(dim=0)
                balance_losses.append((mean_assign - target_usage).pow(2).mean())
        loss_balance = torch.stack(balance_losses).mean() if balance_losses else zero

        if not proto_valid.any():
            loss = self.lambda_class_proto_balance * loss_balance
            return loss, zero, zero, zero, loss_balance, assign_probs, {
                'proto': batch_proto.detach(),
                'valid': batch_valid.detach(),
                'usage': batch_usage.detach(),
            }

        z_norm = F.normalize(z_valid, dim=1)
        pos_proto = proto.index_select(0, labels_valid)
        pos_valid = proto_valid.index_select(0, labels_valid)
        pos_sims = (z_norm.unsqueeze(1) * pos_proto).sum(dim=-1)
        pos_weight = pos_valid.to(pos_sims.dtype)
        pos_count = pos_weight.sum(dim=1)
        has_pos = pos_count > 0

        loss_var = zero
        loss_pos = zero
        if has_pos.any():
            pos_mean = (pos_sims * pos_weight).sum(dim=1) / pos_count.clamp_min(1.0)
            loss_pos = -pos_mean[has_pos].mean()
            has_var = pos_count > 1
            if has_var.any():
                centered = pos_sims - pos_mean.unsqueeze(1)
                pos_var = (centered.pow(2) * pos_weight).sum(dim=1) / pos_count.clamp_min(1.0)
                loss_var = pos_var[has_var].mean()

        flat_proto = proto.reshape(self.c * self.class_proto_k, self.d)
        flat_valid = proto_valid.reshape(self.c * self.class_proto_k)
        sim_all = torch.matmul(z_norm, flat_proto.t()) / self.class_proto_temp
        class_ids = torch.arange(self.c, device=z.device).repeat_interleave(self.class_proto_k)
        neg_mask = (class_ids.unsqueeze(0) != labels_valid.unsqueeze(1)) & flat_valid.unsqueeze(0)
        if neg_mask.any():
            sim_neg = sim_all.masked_fill(~neg_mask, -1e9)
            loss_neg = torch.logsumexp(sim_neg, dim=1)
            has_neg = neg_mask.any(dim=1)
            loss_neg = loss_neg[has_neg].mean() if has_neg.any() else zero
        else:
            loss_neg = zero

        loss = (
            self.lambda_class_proto_var * loss_var
            + self.lambda_class_proto_pos * loss_pos
            + self.lambda_class_proto_neg * loss_neg
            + self.lambda_class_proto_balance * loss_balance
        )
        update_payload = {
            'proto': batch_proto.detach(),
            'valid': batch_valid.detach(),
            'usage': batch_usage.detach(),
        }
        return loss, loss_var, loss_pos, loss_neg, loss_balance, assign_probs.detach(), update_payload

    def update_class_env_prototypes(self, proto_payload):
        if proto_payload is None or not self.class_proto_objective_enabled():
            return
        proto = proto_payload.get('proto')
        valid = proto_payload.get('valid')
        usage = proto_payload.get('usage')
        if proto is None or valid is None or proto.numel() == 0:
            return

        proto = F.normalize(proto.to(device=self.class_env_proto_bank.device, dtype=self.class_env_proto_bank.dtype), dim=-1)
        valid = valid.to(device=self.class_env_proto_valid.device)
        if usage is None:
            usage = torch.zeros_like(self.class_env_proto_usage)
        else:
            usage = usage.to(device=self.class_env_proto_usage.device, dtype=self.class_env_proto_usage.dtype)

        momentum = self.class_proto_momentum
        for cls_idx in range(min(self.c, proto.size(0))):
            for proto_idx in range(min(self.class_proto_k, proto.size(1))):
                if not bool(valid[cls_idx, proto_idx]):
                    continue
                old_valid = bool(self.class_env_proto_valid[cls_idx, proto_idx])
                if old_valid:
                    blended = (
                        momentum * self.class_env_proto_bank[cls_idx, proto_idx]
                        + (1.0 - momentum) * proto[cls_idx, proto_idx]
                    )
                else:
                    blended = proto[cls_idx, proto_idx]
                self.class_env_proto_bank[cls_idx, proto_idx] = F.normalize(blended, dim=0)
                self.class_env_proto_valid[cls_idx, proto_idx] = True
                self.class_env_proto_usage[cls_idx, proto_idx] = (
                    momentum * self.class_env_proto_usage[cls_idx, proto_idx]
                    + (1.0 - momentum) * usage[cls_idx, proto_idx]
                )

    def merge_frontdoor_contexts(self, *context_sets):
        contexts = [ctx for ctx in context_sets if ctx is not None and ctx.numel() > 0]
        if not contexts:
            return None
        sample_contexts = [ctx for ctx in contexts if ctx.dim() == 3]
        global_contexts = [ctx for ctx in contexts if ctx.dim() == 2]
        if not sample_contexts:
            return torch.cat(global_contexts, dim=0)

        num_nodes = sample_contexts[0].size(0)
        merged = []
        for ctx in sample_contexts:
            if ctx.size(0) == num_nodes:
                merged.append(ctx)
        for ctx in global_contexts:
            merged.append(ctx.unsqueeze(0).expand(num_nodes, -1, -1))
        if not merged:
            return None
        return torch.cat(merged, dim=1)

    def build_frontdoor_contexts(
        self,
        gmm_contexts=None,
        global_contexts=None,
        layerwise_contexts=None,
        nego_contexts=None,
        training=False,
    ):
        if self.fd_context_source == 'nego_only':
            return self.sample_frontdoor_contexts(nego_contexts, training=training)
        return self.merge_frontdoor_contexts(
            gmm_contexts,
            global_contexts,
            layerwise_contexts,
            nego_contexts,
        )

    def count_frontdoor_contexts(self, contexts):
        if contexts is None or contexts.numel() == 0:
            return 0
        if contexts.dim() == 3:
            return int(contexts.size(1))
        return int(contexts.size(0))

    def sample_frontdoor_contexts(self, contexts, training=False):
        """
        Approximate the front-door intervention with K diverse contexts.

        During training we randomly sample K environments to mimic the paper's
        stochastic diversity augmentation. During evaluation we keep the subset
        deterministic so validation / test metrics stay stable across calls.
        """
        if contexts is None or contexts.size(0) == 0:
            return contexts

        context_dim = 1 if contexts.dim() == 3 else 0
        num_contexts = contexts.size(context_dim)
        if self.fd_sample_k <= 0 or num_contexts <= self.fd_sample_k:
            return contexts

        if training:
            indices = torch.randperm(num_contexts, device=contexts.device)[:self.fd_sample_k]
        else:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(self.context_sample_seed + num_contexts)
            indices = torch.randperm(num_contexts, generator=generator)[:self.fd_sample_k]
            indices = indices.to(contexts.device)
        return contexts.index_select(context_dim, indices)

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
        if contexts is None or contexts.numel() == 0:
            return base_logits, None

        if contexts.dim() == 3:
            if contexts.size(0) != z_mediator.size(0):
                return base_logits, None
            num_contexts = contexts.size(1)
            context_expand = contexts
        else:
            num_contexts = contexts.size(0)
            context_expand = contexts.unsqueeze(0).expand(z_mediator.size(0), -1, -1)
        if num_contexts <= 0:
            return base_logits, None
        mediator_expand = z_mediator.unsqueeze(1).expand(-1, num_contexts, -1)

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

    def compute_logit_consistency(self, clean_logits, aug_logits):
        if clean_logits is None or aug_logits is None or clean_logits.numel() == 0:
            return self.A_feat.new_zeros(())
        if clean_logits.size(-1) == 1:
            clean_prob = torch.sigmoid(clean_logits.detach()).clamp(1e-6, 1.0 - 1e-6)
            aug_prob = torch.sigmoid(aug_logits).clamp(1e-6, 1.0 - 1e-6)
            return F.binary_cross_entropy(aug_prob, clean_prob)
        clean_prob = F.softmax(clean_logits.detach(), dim=-1)
        aug_log = F.log_softmax(aug_logits, dim=-1)
        return F.kl_div(aug_log, clean_prob, reduction='batchmean')

    def compute_frontdoor_variance_consistency(self, clean_stack, aug_stack):
        if (
            clean_stack is None
            or aug_stack is None
            or clean_stack.numel() == 0
            or aug_stack.numel() == 0
            or clean_stack.size(1) <= 1
            or clean_stack.shape != aug_stack.shape
        ):
            return self.A_feat.new_zeros(())
        if clean_stack.size(-1) == 1:
            clean_pred = torch.sigmoid(clean_stack.detach())
            aug_pred = torch.sigmoid(aug_stack)
        else:
            clean_pred = F.softmax(clean_stack.detach(), dim=-1)
            aug_pred = F.softmax(aug_stack, dim=-1)
        clean_var = clean_pred.var(dim=1, unbiased=False)
        aug_var = aug_pred.var(dim=1, unbiased=False)
        return F.mse_loss(aug_var, clean_var)

    def sample_counterfactual_branch(self, branch):
        if self.cf_mode == 'zero':
            return torch.zeros_like(branch)
        if self.cf_mode == 'noise':
            if self.cf_noise_std <= 0.0:
                return branch
            feature_std = branch.detach().std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            return branch + torch.randn_like(branch) * feature_std * self.cf_noise_std

        if branch.size(0) <= 1:
            return branch
        perm = torch.randperm(branch.size(0), device=branch.device)
        if torch.equal(perm, torch.arange(branch.size(0), device=branch.device)):
            perm = torch.roll(perm, shifts=1)
        return branch.index_select(0, perm)

    def compute_counterfactual_loss(self, z_mediator, z_spurious, clean_fd_logits, contexts):
        """
        CauVQ-inspired counterfactual regularization.

        By default we intervene on the mediator branch, which is closest to
        CauVQ's perturbation of high-attention causal candidates. The spurious
        and both-branch targets are kept as ablations for invariance testing.
        In `cauvq` mode, counterfactual predictions follow CauVQ's global
        variance plus entropy regularizer. KL/MSE modes compare against the
        clean front-door prediction.
        """
        zero = z_mediator.new_zeros(())
        if (
            self.lambda_cf <= 0.0
            or z_mediator is None
            or z_spurious is None
            or z_mediator.numel() == 0
            or z_spurious.numel() == 0
        ):
            return zero, zero

        counter_logits = []
        for _ in range(self.cf_samples):
            z_mediator_cf = z_mediator
            z_spurious_cf = z_spurious
            if self.cf_target in ('mediator', 'both'):
                z_mediator_cf = self.sample_counterfactual_branch(z_mediator)
            if self.cf_target in ('spurious', 'both'):
                z_spurious_cf = self.sample_counterfactual_branch(z_spurious)
            fd_logits_cf, _ = self.frontdoor_logits_from_contexts(
                z_mediator_cf,
                z_spurious_cf,
                contexts,
            )
            counter_logits.append(fd_logits_cf)
        counter_logits = torch.stack(counter_logits, dim=0)

        if counter_logits.size(-1) == 1:
            counter_probs = torch.sigmoid(counter_logits)
        else:
            counter_probs = torch.softmax(counter_logits, dim=-1)
        flat_counter_probs = counter_probs.reshape(-1, counter_probs.size(-1))
        mean_counter_probs = flat_counter_probs.mean(dim=0, keepdim=True)
        cf_shift = (flat_counter_probs - mean_counter_probs).pow(2).mean()

        if self.cf_consistency == 'cauvq':
            eps = 1e-9
            if mean_counter_probs.size(-1) == 1:
                p_pos = mean_counter_probs.squeeze(0).clamp(eps, 1.0 - eps)
                entropy = -(
                    p_pos * p_pos.log()
                    + (1.0 - p_pos) * (1.0 - p_pos).clamp_min(eps).log()
                ).mean()
            else:
                entropy = -(
                    mean_counter_probs.clamp_min(eps)
                    * mean_counter_probs.clamp_min(eps).log()
                ).sum()
            return cf_shift + self.cf_beta * entropy, cf_shift.detach()

        clean_logits = clean_fd_logits.detach()
        if self.cf_consistency == 'mse':
            if clean_logits.size(-1) == 1:
                clean_probs = torch.sigmoid(clean_logits)
            else:
                clean_probs = torch.softmax(clean_logits, dim=-1)
            clean_probs = clean_probs.unsqueeze(0).expand_as(counter_probs)
            return F.mse_loss(counter_probs, clean_probs), cf_shift.detach()

        temp = self.cf_temp
        clean_target = torch.softmax(clean_logits / temp, dim=-1).unsqueeze(0)
        clean_target = clean_target.expand_as(counter_logits).detach()
        cf_log_probs = F.log_softmax(counter_logits / temp, dim=-1)
        loss_kl = F.kl_div(
            cf_log_probs.reshape(-1, cf_log_probs.size(-1)),
            clean_target.reshape(-1, clean_target.size(-1)),
            reduction='batchmean',
        ) * (temp ** 2)
        return loss_kl, cf_shift.detach()

    def forward(self, x, edge_index, training=False, labels=None, label_mask=None):
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
            h_global_context,
            layerwise_spurious,
            h_pre_enhance,
            cns_gate,
        ) = self.encode_representation(
            x,
            edge_index,
            training=training,
            labels=labels,
            label_mask=label_mask,
        )

        gmm_contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(training=training),
            training=training,
        )
        layerwise_contexts = self.get_layerwise_spurious_contexts(
            layerwise_spurious,
            self.compute_pseudo_env_probs(z_spurious) if self.num_envs > 1 else None,
        )
        if self.nego_context_mode == 'sample_mix':
            nego_source_all = self.get_nego_source_representation(z, z_mediator, z_spurious)
            if mediator_logits.size(-1) > 1:
                pseudo_labels = mediator_logits.detach().argmax(dim=-1)
            else:
                pseudo_labels = torch.zeros(z.size(0), device=z.device, dtype=torch.long)
            all_idx = torch.arange(z.size(0), device=z.device, dtype=torch.long)
            nego_contexts = self.get_nego_contexts(
                nego_source_all,
                pseudo_labels,
                all_idx,
                training=training,
            )
        else:
            nego_contexts = self.get_nego_contexts(training=False)
        contexts = self.build_frontdoor_contexts(
            gmm_contexts,
            self.get_global_contexts(h_global_context),
            layerwise_contexts,
            nego_contexts,
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
                h_global_context,
                layerwise_spurious,
                h_pre_enhance,
                cns_gate,
            )
        if self.eval_pred_mode == 'mediator':
            return mediator_logits
        if self.eval_pred_mode == 'frontdoor':
            return fd_logits
        return logits

    def compute_supervised_loss(self, logits, y, criterion, args):
        # if args.dataset in ('twitch', 'elliptic'):
        #     if y.shape[1] == 1 and logits.shape[1] > 1:
        #         true_label = F.one_hot(y.squeeze().long(), logits.shape[1]).float()
        #     else:
        #         true_label = y.float()
        #     sup_loss = criterion(logits, true_label)
        #     if sup_loss.dim() > 1:
        #         sup_loss = sup_loss.mean(dim=1)
        #     return sup_loss
        # return criterion(logits, y.squeeze().long())
        if args.dataset in ('twitch', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(logits, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(logits, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss

    def compute_energy_regularization_loss(self, logits):
        zero = self.A_feat.new_zeros(())
        if self.lambda_energy_reg <= 0.0 or logits is None or logits.numel() == 0:
            return zero

        flat_logits = logits.reshape(-1, logits.size(-1))
        logit_norm = flat_logits.norm(p=2, dim=-1)
        if flat_logits.size(-1) == 1:
            logit_mass = F.softplus(flat_logits.squeeze(-1))
        else:
            logit_mass = torch.logsumexp(flat_logits, dim=-1)

        loss = zero
        if self.energy_reg_norm_weight > 0.0:
            loss = loss + self.energy_reg_norm_weight * logit_norm.var(unbiased=False)
        if self.energy_reg_mass_weight > 0.0:
            loss = loss + self.energy_reg_mass_weight * logit_mass.var(unbiased=False)
        if self.energy_reg_mean_weight > 0.0:
            loss = loss + self.energy_reg_mean_weight * (
                logit_norm.mean().pow(2) + logit_mass.mean().pow(2)
            )
        return loss

    def _per_node_loss(self, loss):
        if loss.dim() <= 1:
            return loss
        return loss.reshape(loss.size(0), -1).mean(dim=1)

    def _degree_buckets(self, edge_index, num_nodes, node_idx):
        if self.dro_num_groups <= 1 or edge_index is None or edge_index.numel() == 0:
            return torch.zeros(node_idx.size(0), device=node_idx.device, dtype=torch.long)
        src, dst = edge_index
        deg = degree(src, num_nodes).to(device=node_idx.device) + degree(dst, num_nodes).to(device=node_idx.device)
        score = self._normalize_score(torch.log1p(deg.index_select(0, node_idx)), default_value=0.5)
        bucket = torch.floor(score * float(self.dro_num_groups)).long()
        return bucket.clamp_(0, self.dro_num_groups - 1)

    def _dro_group_ids(self, edge_index, num_nodes, train_idx, y):
        if self.dro_group_by == 'none':
            return torch.zeros(train_idx.size(0), device=train_idx.device, dtype=torch.long)

        degree_bucket = self._degree_buckets(edge_index, num_nodes, train_idx)
        if self.dro_group_by == 'degree':
            return degree_bucket

        labels = self._flat_class_labels(y).to(device=train_idx.device).index_select(0, train_idx)
        labels = labels.clamp_min(0)
        if self.dro_group_by == 'label':
            return labels.long()
        return labels.long() * self.dro_num_groups + degree_bucket

    def compute_entropy_dro_loss(self, raw_loss, edge_index, num_nodes, train_idx, y):
        per_node = self._per_node_loss(raw_loss)
        base_loss = per_node.mean()
        zero = base_loss.new_zeros(())
        if (
            self.lambda_entropy_dro <= 0.0
            or per_node.numel() <= 1
            or train_idx is None
            or train_idx.numel() != per_node.numel()
        ):
            self._last_dro_entropy = zero
            self._last_dro_max_weight = zero
            return base_loss, base_loss.detach(), zero, zero

        group_ids = self._dro_group_ids(edge_index, num_nodes, train_idx, y)
        group_losses = []
        for group_id in group_ids.unique(sorted=True):
            mask = group_ids == group_id
            if mask.any():
                group_losses.append(per_node[mask].mean())
        if len(group_losses) <= 1:
            self._last_dro_entropy = zero
            self._last_dro_max_weight = zero
            return base_loss, base_loss.detach(), zero, zero

        group_losses = torch.stack(group_losses)
        weights = F.softmax(group_losses.detach() / self.dro_entropy_beta, dim=0)
        dro_loss = (weights * group_losses).sum()
        mix = min(max(float(self.lambda_entropy_dro), 0.0), 1.0)
        robust_loss = (1.0 - mix) * base_loss + mix * dro_loss
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum()
        max_weight = weights.max()
        self._last_dro_entropy = entropy.detach()
        self._last_dro_max_weight = max_weight.detach()
        return robust_loss, base_loss.detach(), entropy.detach(), max_weight.detach()

       
    
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

    def linear_with_detached_head(self, x, head):
        weight = head.weight.detach()
        bias = head.bias.detach() if head.bias is not None else None
        return F.linear(x, weight, bias)

    def compute_role_supervision_loss(
        self,
        z_mediator,
        z_spurious,
        mediator_logits,
        env_logits_spu,
        y,
        criterion,
        args,
    ):
        """
        Label-free role supervision for the DAG partition.

        No ground-truth environment labels are used. The spurious branch induces
        pseudo-environments through env_logits_spu, while frozen-current heads
        are used for the uniform terms so the heads themselves do not learn to
        become uninformative.
        """
        zero = z_mediator.new_zeros(())
        if self.lambda_role <= 0.0 or z_mediator.numel() == 0:
            return zero, zero, zero, zero, zero

        loss_med_y = zero
        if self.role_med_y_weight > 0.0:
            loss_med_y = self.compute_supervised_loss(
                mediator_logits,
                y,
                criterion,
                args,
            ).mean()

        loss_spu_y = zero
        if self.role_spu_y_weight > 0.0:
            spu_label_logits = self.linear_with_detached_head(z_spurious, self.classifier)
            loss_spu_y = self.compute_uniform_loss(spu_label_logits)

        loss_spu_env = zero
        loss_med_env = zero
        if self.num_envs > 1 and env_logits_spu is not None and env_logits_spu.numel() > 0:
            if self.role_spu_env_weight > 0.0:
                pseudo_env_target = F.softmax(env_logits_spu.detach(), dim=-1)
                loss_spu_env = -(
                    pseudo_env_target
                    * F.log_softmax(env_logits_spu, dim=-1)
                ).sum(dim=-1).mean()

            if self.role_med_env_weight > 0.0:
                med_env_logits = self.linear_with_detached_head(z_mediator, self.env_classifier)
                loss_med_env = self.compute_env_uniform_loss(med_env_logits)

        loss_role = (
            self.role_med_y_weight * loss_med_y
            + self.role_spu_y_weight * loss_spu_y
            + self.role_spu_env_weight * loss_spu_env
            + self.role_med_env_weight * loss_med_env
        )
        return loss_role, loss_med_y, loss_spu_y, loss_spu_env, loss_med_env

    def compute_independence_loss(self, z_mediator, z_spurious):
        if z_mediator.numel() == 0:
            return self.A_feat.new_zeros(())
        z_med = F.normalize(z_mediator, dim=1)
        z_spu = F.normalize(z_spurious, dim=1)
        corr = (z_med * z_spu).sum(dim=1)
        return 0.5 * (corr ** 2).mean()

    def compute_frontdoor_variance_loss(self, logits_stack):
        if (
            self.lambda_var <= 0.0
            or logits_stack is None
            or logits_stack.numel() == 0
            or logits_stack.size(1) <= 1
        ):
            return self.A_feat.new_zeros(())
        if logits_stack.size(-1) == 1:
            pred_stack = torch.sigmoid(logits_stack)
        else:
            pred_stack = torch.softmax(logits_stack, dim=-1)
        return pred_stack.var(dim=1, unbiased=False).mean()

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

    def compute_enhance_semantic_loss(self, z, h_anchor, train_idx=None):
        """
        Keep the edge-enhanced node representation tied to the pre-enhancement
        graph semantics. The anchor is detached so this constrains the enhancer
        without pulling the backbone into a moving-target reconstruction loop.
        """
        if (
            self.lambda_enhance_sem <= 0.0
            or z is None
            or h_anchor is None
            or z.numel() == 0
            or h_anchor.numel() == 0
        ):
            return self.A_feat.new_zeros(())

        if train_idx is not None and train_idx.numel() > 0:
            z_view = z[train_idx]
            anchor = h_anchor[train_idx].detach()
        else:
            z_view = z
            anchor = h_anchor.detach()

        if self.enhance_sem_mode == 'mse':
            return F.mse_loss(F.normalize(z_view, dim=1), F.normalize(anchor, dim=1))
        return (1.0 - (F.normalize(z_view, dim=1) * F.normalize(anchor, dim=1)).sum(dim=1)).mean()

    def dag_regularization_loss(self, mediator_gate, dag_total):
        A = self.get_masked_A()
        if self.dag_ablate_acyclic_loss:
            loss_dag = A.new_zeros(())
        else:
            A_sq = A * A
            h_A = torch.trace(torch.matrix_exp(A_sq)) - self.dag_var_dim
            h_A_clipped = torch.clamp(h_A, min=-10.0, max=10.0)
            loss_dag = 0.5 * (h_A_clipped ** 2)
        loss_dag = loss_dag + self.lambda_l1 * torch.norm(A, p=1)

        # Optional soft sparsity on mediator gate to avoid trivial all-ones masks.
        if self.lambda_gate > 0.0:
            loss_dag = loss_dag + self.lambda_gate * mediator_gate.mean()

        if not self.dag_ablate_flow_consistency:
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
        self.update_nego_context_bank(
            state_payload.get('nego_contexts'),
        )
        self.update_class_env_prototypes(
            state_payload.get('class_env_proto'),
        )

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        y = y.to(x.device)
        train_idx = data.train_idx.to(device=x.device, dtype=torch.long)
        train_label_mask = torch.zeros(x.size(0), device=x.device, dtype=torch.bool)
        train_label_mask[train_idx] = True

        (
            _,
            z_all,
            edge_summary_all,
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
            h_pre_enhance_all,
            cns_gate_all,
        ) = self.forward(
            x,
            edge_index,
            training=True,
            labels=y,
            label_mask=train_label_mask,
        )

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
        nego_source_all = self.get_nego_source_representation(z_all, z_mediator_all, z_spurious_all)
        nego_contexts = self.get_nego_contexts(
            nego_source_all,
            y,
            train_idx,
            training=True,
        )
        contexts = self.build_frontdoor_contexts(
            gmm_contexts,
            global_contexts,
            layerwise_contexts,
            nego_contexts,
            training=True,
        )
        num_gmm_contexts = self.count_frontdoor_contexts(gmm_contexts)
        num_global_contexts = self.count_frontdoor_contexts(global_contexts)
        num_layerwise_contexts = self.count_frontdoor_contexts(layerwise_contexts)
        num_nego_contexts = self.count_frontdoor_contexts(nego_contexts)
        num_mixed_contexts = 0
        fd_logits_tr, fd_stack_tr = self.frontdoor_logits_from_contexts(med_tr, spu_tr, contexts)
        final_logits_tr = self.blend_logits(mediator_logits_tr, fd_logits_tr)
        loss_cf, cf_pred_shift = self.compute_counterfactual_loss(
            med_tr,
            spu_tr,
            fd_logits_tr,
            contexts,
        )

        zero = self.A_feat.new_zeros(())
        raw_loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args)
        raw_loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args)
        loss_energy_reg = self.compute_energy_regularization_loss(final_logits_tr)
        loss_class_neighbor_uncert = (
            zero
            if self._last_class_neighbor_uncert_loss is None
            else self._last_class_neighbor_uncert_loss
        )
        loss_cls, loss_cls_mean, dro_entropy, dro_max_weight = self.compute_entropy_dro_loss(
            raw_loss_cls,
            edge_index,
            x.size(0),
            train_idx,
            y,
        )
        loss_fd, loss_fd_mean, _, _ = self.compute_entropy_dro_loss(
            raw_loss_fd,
            edge_index,
            x.size(0),
            train_idx,
            y,
        )
        loss_var = self.compute_frontdoor_variance_loss(fd_stack_tr)
        loss_graph_delf = self.compute_graph_delf_loss(
            z_mediator_all,
            edge_summary_all,
            final_logits_tr,
            y,
            train_idx,
            criterion,
            args,
        )
        if self._last_graph_cfam_gate_loss is None:
            loss_graph_cfam_gate = self.A_feat.new_zeros(())
            graph_cfam_gate_mean = self.A_feat.new_zeros(())
        else:
            loss_graph_cfam_gate = self._last_graph_cfam_gate_loss
            graph_cfam_gate_mean = self._last_graph_cfam_gate_mean
        (
            loss_role,
            loss_role_med_y,
            loss_role_spu_y,
            loss_role_spu_env,
            loss_role_med_env,
        ) = self.compute_role_supervision_loss(
            med_tr,
            spu_tr,
            mediator_logits_tr,
            env_logits_spu,
            y_tr,
            criterion,
            args,
        )
        loss_dag = zero
        loss_dag_label = zero
        loss_ica_cov = zero if self._last_ica_cov_loss is None else self._last_ica_cov_loss
        loss_ica_ng = zero if self._last_ica_ng_loss is None else self._last_ica_ng_loss
        loss_ica_gate = zero if self._last_ica_gate_loss is None else self._last_ica_gate_loss
        loss_ica_entropy = zero if self._last_ica_entropy_loss is None else self._last_ica_entropy_loss
        loss_global_env = self.compute_global_env_consistency_loss(
            h_global_all,
            z_spurious_all,
            self.compute_pseudo_env_probs(z_spurious_all) if self.num_envs > 1 else None,
        )
        current_counterexample_penalty = self.counterexample_penalty.new_zeros(self.counterexample_penalty.size())
        loss_enhance_sem = self.compute_enhance_semantic_loss(
            z_all,
            h_pre_enhance_all,
            train_idx,
        )
        loss_diffusion = zero if self._last_diffusion_loss is None else self._last_diffusion_loss
        if self._last_layerwise_gate_loss is None:
            loss_layerwise_gate = self.A_feat.new_zeros(())
            layerwise_gate_mean = self.A_feat.new_zeros(())
        else:
            loss_layerwise_gate = self._last_layerwise_gate_loss
            layerwise_gate_mean = self._last_layerwise_gate_mean

        # DAG-Core keeps only the essential pseudo-environment constraints:
        # spurious features should form label-free contexts, and mediator
        # features should be invariant to those pseudo-contexts.
        env_logits_med = None
        env_logits_spu = None
        loss_env_med = zero
        loss_spu = zero

        loss_nego, nego_extra_score, nego_self_score = self.compute_nego_loss(
            nego_source_all,
            y,
            train_idx,
        )
        (
            loss_class_proto,
            loss_class_proto_var,
            loss_class_proto_pos,
            loss_class_proto_neg,
            loss_class_proto_balance,
            class_proto_assign_probs,
            class_proto_payload,
        ) = self.compute_class_env_proto_loss(
            z_all,
            y,
            train_idx,
        )

        loss_med = loss_role_med_y
        loss_fd_aug = zero
        loss_ind = zero
        loss_sem = zero
        loss_degree = zero
        loss_spu_y = zero
        loss_inv = zero
        loss_cns = zero
        loss_cns_cons = zero
        cns_complement_mean = zero
        cns_gate_mean = zero

        total_loss = (
            loss_cls
            + self.lambda_fd * loss_fd
            + self.lambda_cf * loss_cf
            + self.lambda_dag * loss_dag
            + self.lambda_dag_label * loss_dag_label
            + self.lambda_ica_cov * loss_ica_cov
            + self.lambda_ica_ng * loss_ica_ng
            + self.lambda_ica_gate * loss_ica_gate
            + self.lambda_ica_entropy * loss_ica_entropy
            + self.lambda_spu * loss_spu
            + self.lambda_role * loss_role
            + self.lambda_env * loss_env_med
            + self.lambda_global_env * loss_global_env
            + self.lambda_var * loss_var
            + self.lambda_latent_diffusion * loss_diffusion
            + self.lambda_layerwise_gate * loss_layerwise_gate
            + self.lambda_graph_cfam_gate * loss_graph_cfam_gate
            + self.lambda_graph_delf * loss_graph_delf
            + self.lambda_enhance_sem * loss_enhance_sem
            + self.lambda_nego * loss_nego
            + self.lambda_energy_reg * loss_energy_reg
            + self.lambda_class_neighbor_uncert * loss_class_neighbor_uncert
            + loss_class_proto
        )

        state_payload = None
        if update_state:
            state_payload = {
                'spu_tr': spu_tr.detach(),
                'env_probs_tr': env_probs_spu.detach() if env_probs_spu is not None else None,
                'edge_latent_tr': edge_latent_tr.detach(),
                'counterexample_penalty': current_counterexample_penalty.detach(),
                'nego_contexts': nego_contexts.detach() if nego_contexts is not None else None,
                'class_env_proto': class_proto_payload,
            }

        num_contexts = self.count_frontdoor_contexts(contexts)
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_cls_mean': loss_cls_mean,
            'loss_med': loss_med,
            'loss_fd': loss_fd,
            'loss_fd_mean': loss_fd_mean,
            'loss_cf': loss_cf,
            'loss_fd_aug': loss_fd_aug,
            'loss_var': loss_var,
            'loss_ind': loss_ind,
            'loss_dag': loss_dag,
            'loss_dag_label': loss_dag_label,
            'loss_ica_cov': loss_ica_cov,
            'loss_ica_ng': loss_ica_ng,
            'loss_ica_gate': loss_ica_gate,
            'loss_ica_entropy': loss_ica_entropy,
            'loss_role': loss_role,
            'loss_role_med_y': loss_role_med_y,
            'loss_role_spu_y': loss_role_spu_y,
            'loss_role_spu_env': loss_role_spu_env,
            'loss_role_med_env': loss_role_med_env,
            'loss_sem': loss_sem,
            'loss_degree': loss_degree,
            'loss_spu_y': loss_spu_y,
            'loss_spu': loss_spu,
            'loss_env_med': loss_env_med,
            'loss_inv': loss_inv,
            'loss_global_env': loss_global_env,
            'loss_diffusion': loss_diffusion,
            'loss_cns': loss_cns,
            'loss_cns_cons': loss_cns_cons,
            'loss_layerwise_gate': loss_layerwise_gate,
            'loss_graph_cfam_gate': loss_graph_cfam_gate,
            'loss_graph_delf': loss_graph_delf,
            'loss_enhance_sem': loss_enhance_sem,
            'loss_nego': loss_nego,
            'loss_energy_reg': loss_energy_reg,
            'loss_class_neighbor_uncert': loss_class_neighbor_uncert,
            'loss_class_proto': loss_class_proto,
            'loss_class_proto_var': loss_class_proto_var,
            'loss_class_proto_pos': loss_class_proto_pos,
            'loss_class_proto_neg': loss_class_proto_neg,
            'loss_class_proto_balance': loss_class_proto_balance,
            'nego_extra_score': nego_extra_score,
            'nego_self_score': nego_self_score,
            'class_proto_assign_entropy': (
                zero
                if class_proto_assign_probs is None or class_proto_assign_probs.numel() == 0
                else -(
                    class_proto_assign_probs
                    * class_proto_assign_probs.clamp_min(1e-12).log()
                ).sum(dim=1).mean().detach()
            ),
            'num_class_env_protos': torch.tensor(
                float(self.class_env_proto_valid.sum().item()),
                device=x.device,
            ),
            'dro_weight_entropy': dro_entropy.detach(),
            'dro_max_weight': dro_max_weight.detach(),
            'energy_raw_mean': (
                zero if self._last_energy_raw_mean is None else self._last_energy_raw_mean.detach()
            ),
            'energy_prop_mean': (
                zero if self._last_energy_prop_mean is None else self._last_energy_prop_mean.detach()
            ),
            'energy_delta_mean': (
                zero if self._last_energy_delta_mean is None else self._last_energy_delta_mean.detach()
            ),
            'energy_node_gate_mean': (
                zero if self._last_energy_node_gate_mean is None else self._last_energy_node_gate_mean.detach()
            ),
            'same_family_edge_ratio': (
                zero if self._last_same_family_edge_ratio is None else self._last_same_family_edge_ratio.detach()
            ),
            'explicit_family_edge_ratio': (
                zero
                if self._last_explicit_family_edge_ratio is None
                else self._last_explicit_family_edge_ratio.detach()
            ),
            'family_energy_conf_mean': (
                zero
                if self._last_family_energy_conf_mean is None
                else self._last_family_energy_conf_mean.detach()
            ),
            'family_diff_energy_mean': (
                zero
                if self._last_family_diff_energy_mean is None
                else self._last_family_diff_energy_mean.detach()
            ),
            'family_mask_mean': (
                zero if self._last_family_mask_mean is None else self._last_family_mask_mean.detach()
            ),
            'same_family_uncertainty_mean': (
                zero
                if self._last_same_family_uncertainty_mean is None
                else self._last_same_family_uncertainty_mean.detach()
            ),
            'diff_family_uncertainty_mean': (
                zero
                if self._last_diff_family_uncertainty_mean is None
                else self._last_diff_family_uncertainty_mean.detach()
            ),
            'class_neighbor_causal_gate_mean': (
                zero
                if self._last_class_neighbor_causal_gate_mean is None
                else self._last_class_neighbor_causal_gate_mean.detach()
            ),
            'cns_complement_mean': cns_complement_mean.detach(),
            'cns_gate_mean': cns_gate_mean.detach(),
            'cns_layer_complement_mean': zero.detach(),
            'cns_layer_gate_mean': zero.detach(),
            'cns_layer_layers': torch.tensor(0.0, device=x.device),
            'layerwise_gate_mean': layerwise_gate_mean.detach(),
            'layerwise_gate_layers': torch.tensor(float(self._last_layerwise_gate_layers), device=x.device),
            'graph_cfam_gate_mean': graph_cfam_gate_mean.detach(),
            'graph_cfam_layers': torch.tensor(float(self._last_graph_cfam_layers), device=x.device),
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'ica_gate_mean': (
                zero if self._last_ica_gate_mean is None else self._last_ica_gate_mean.detach()
            ),
            'causal_score_mean': causal_score.mean().detach(),
            'pollution_score_mean': pollution_score.mean().detach(),
            'counterexample_penalty_mean': self.counterexample_penalty.mean().detach(),
            'counterexample_penalty_batch_mean': current_counterexample_penalty.mean().detach(),
            'cf_pred_shift': cf_pred_shift.detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_mixed_contexts': torch.tensor(float(num_mixed_contexts), device=x.device),
            'num_gmm_contexts': torch.tensor(float(num_gmm_contexts), device=x.device),
            'num_global_contexts': torch.tensor(float(num_global_contexts), device=x.device),
            'num_layerwise_contexts': torch.tensor(float(num_layerwise_contexts), device=x.device),
            'num_nego_contexts': torch.tensor(float(num_nego_contexts), device=x.device),
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
