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
        self.main_layer_query = nn.Linear(self.d, self.d)
        self.main_state_proj = nn.Sequential(
            nn.Linear(self.d * 4, self.d),
            nn.ReLU(),
            nn.Dropout(p=getattr(args, 'dropout', 0.0)),
            nn.Linear(self.d, self.d),
        )
        self.main_state_gate = nn.Sequential(
            nn.Linear(self.d * 4, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.main_state_norm = nn.LayerNorm(self.d)
        self.main_path_weight = max(0.0, float(getattr(args, 'main_path_weight', 0.5)))
        self.main_gate_temp = max(1e-3, float(getattr(args, 'main_gate_temp', 1.0)))
        self._last_main_gate_mean = None

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
        self.graph_cfam_residual_blend = max(0.0, float(getattr(args, 'graph_cfam_residual_blend', 0.1)))
        self.graph_cfam_gate_temp = max(1e-3, float(getattr(args, 'graph_cfam_gate_temp', 1.0)))
        self.graph_cfam_gate_target = min(max(float(getattr(args, 'graph_cfam_gate_target', 0.5)), 0.0), 1.0)
        self.lambda_graph_cfam_gate = max(0.0, float(getattr(args, 'lambda_graph_cfam_gate', 0.0)))
        self.lambda_graph_delf = max(0.0, float(getattr(args, 'lambda_graph_delf', 0.0)))
        self.graph_delf_top_frac = min(max(float(getattr(args, 'graph_delf_top_frac', 0.2)), 0.0), 1.0)
        self.graph_delf_margin = float(getattr(args, 'graph_delf_margin', 0.2))
        self.graph_delf_shortcut_weight = max(0.0, float(getattr(args, 'graph_delf_shortcut_weight', 0.5)))
        self.graph_cfam_gate = nn.Sequential(
            nn.Linear(self.d * 5, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.graph_cfam_norm = nn.LayerNorm(self.d)
        self._last_graph_cfam_gate_loss = None
        self._last_graph_cfam_gate_mean = None
        self._last_graph_cfam_layers = 0

        self.noise_summary_norm = nn.LayerNorm(self.d)
        self.node_noise_fuser = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.node_noise_gate = nn.Linear(self.d * 3, 1)

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

        # Optional end-to-end ICA-like splitter over the enhanced node
        # representation z.  When enabled it replaces the DAG-derived
        # mediator/spurious split: independent components are softly assigned
        # to causal vs. spurious groups by a learnable component gate.
        self.use_ica_split = bool(getattr(args, 'use_ica_split', False))
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

        # Same-label multi-prototype guidance.
        self.proto_num = max(1, int(getattr(args, 'proto_num', 2)))
        self.proto_temp = max(1e-3, float(getattr(args, 'proto_temp', 0.2)))
        self.proto_gate_temp = max(1e-3, float(getattr(args, 'proto_gate_temp', 1.0)))
        self.proto_momentum = min(max(float(getattr(args, 'proto_momentum', 0.9)), 0.0), 1.0)
        self.proto_confidence = min(max(float(getattr(args, 'proto_confidence', 0.7)), 0.0), 1.0)
        self.proto_max_nodes_per_class = max(1, int(getattr(args, 'proto_max_nodes_per_class', 256)))
        self.proto_kmeans_iters = max(1, int(getattr(args, 'proto_kmeans_iters', 3)))
        self.proto_inject_weight = max(0.0, float(getattr(args, 'proto_inject_weight', 0.2)))
        self.proto_gate_target = min(max(float(getattr(args, 'proto_gate_target', 0.5)), 0.0), 1.0)
        self.lambda_proto_gate = max(0.0, float(getattr(args, 'lambda_proto_gate', 0.0)))
        self.lambda_proto_align = max(0.0, float(getattr(args, 'lambda_proto_align', 0.0)))
        self.use_proto_contexts = bool(getattr(args, 'use_proto_contexts', False))
        self.proto_context_weight = max(0.0, float(getattr(args, 'proto_context_weight', 1.0)))
        self.proto_fuser = nn.Sequential(
            nn.Linear(self.d * 4, self.d),
            nn.ReLU(),
            nn.Dropout(p=adapter_dropout),
            nn.Linear(self.d, self.d),
        )
        self.proto_gate = nn.Sequential(
            nn.Linear(self.d * 4, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.proto_norm = nn.LayerNorm(self.d)
        self.register_buffer('class_proto_keys', torch.zeros(self.c, self.proto_num, self.d))
        self.register_buffer('class_proto_values', torch.zeros(self.c, self.proto_num, self.d))
        self.register_buffer('class_proto_valid', torch.zeros(self.c, self.proto_num, dtype=torch.bool))
        self._last_proto_gate_loss = None
        self._last_proto_gate_mean = None
        self._last_proto_valid_ratio = None

        self.lambda_l1 = getattr(args, 'lambda_l1', 1e-5)
        self.lambda_dag = getattr(args, 'lambda_dag', 0.1)
        self.lambda_med = getattr(args, 'lambda_med', 0.5)
        self.lambda_spu = getattr(args, 'lambda_spu', 0.1)
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
        self.lambda_var = getattr(args, 'lambda_var', 0.05)
        self.lambda_ind = getattr(args, 'lambda_ind', 0.1)
        self.lambda_env = getattr(args, 'lambda_env', 0.1)
        self.lambda_inv = getattr(args, 'lambda_inv', 0.1)
        self.lambda_global_env = getattr(args, 'lambda_global_env', 0.0)
        # Local node-aware bi-smoothing consistency. This keeps the overall
        # front-door framework unchanged: the randomized local subgraph is
        # only used to regularize the mediator M, not to replace the
        # front-door prediction path.
        self.use_local_bismooth = bool(getattr(args, 'use_local_bismooth', False))
        self.lambda_bismooth = float(getattr(args, 'lambda_bismooth', 0.0))
        self.lambda_bismooth_cls = float(getattr(args, 'lambda_bismooth_cls', 0.0))
        self.lambda_enhance_sem = float(getattr(args, 'lambda_enhance_sem', 0.0))
        self.enhance_sem_mode = getattr(args, 'enhance_sem_mode', 'cosine')
        if self.enhance_sem_mode not in ('cosine', 'mse'):
            self.enhance_sem_mode = 'cosine'
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
        self.main_layer_query.reset_parameters()
        self._reset_module_parameters(self.main_state_proj)
        nn.init.zeros_(self.main_state_proj[-1].weight)
        nn.init.zeros_(self.main_state_proj[-1].bias)
        self._reset_module_parameters(self.main_state_gate)
        nn.init.zeros_(self.main_state_gate[-1].weight)
        nn.init.zeros_(self.main_state_gate[-1].bias)
        self.main_state_norm.reset_parameters()
        self._reset_module_parameters(self.edge_pair_encoder)
        self.edge_score_head.reset_parameters()
        self.edge_summary_norm.reset_parameters()
        self._reset_module_parameters(self.node_edge_fuser)
        nn.init.zeros_(self.node_edge_fuser[-1].weight)
        nn.init.zeros_(self.node_edge_fuser[-1].bias)
        self.node_edge_norm.reset_parameters()
        self._reset_module_parameters(self.graph_cfam_gate)
        nn.init.zeros_(self.graph_cfam_gate[-1].weight)
        nn.init.zeros_(self.graph_cfam_gate[-1].bias)
        self.graph_cfam_norm.reset_parameters()
        self._last_graph_cfam_gate_loss = None
        self._last_graph_cfam_gate_mean = None
        self._last_graph_cfam_layers = 0
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
        nn.init.normal_(self.nego_prompts, mean=0.0, std=0.02)
        self._reset_module_parameters(self.nego_prompt_decoder)
        self.nego_prompt_norm.reset_parameters()
        self._reset_module_parameters(self.proto_fuser)
        nn.init.zeros_(self.proto_fuser[-1].weight)
        nn.init.zeros_(self.proto_fuser[-1].bias)
        self._reset_module_parameters(self.proto_gate)
        nn.init.zeros_(self.proto_gate[-1].weight)
        nn.init.zeros_(self.proto_gate[-1].bias)
        self.proto_norm.reset_parameters()
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
        self.class_proto_keys.zero_()
        self.class_proto_values.zero_()
        self.class_proto_valid.zero_()
        self._last_nego_context_mean = None
        self._last_proto_gate_loss = None
        self._last_proto_gate_mean = None
        self._last_proto_valid_ratio = None
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
        self._last_main_gate_mean = None

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

    def _flat_labels_or_predictions(self, logits, labels=None, train_idx=None):
        probs = torch.softmax(logits.detach(), dim=-1)
        conf, pred = probs.max(dim=-1)
        keep = conf >= self.proto_confidence
        if labels is not None and train_idx is not None and train_idx.numel() > 0:
            label_ids = self._flat_class_labels(labels).to(device=pred.device)
            pred = pred.clone()
            keep = keep.clone()
            pred[train_idx] = label_ids[train_idx]
            keep[train_idx] = True
        return pred, conf, keep

    def _select_proto_seeds(self, samples, confidence, k):
        num_samples = samples.size(0)
        k = min(max(1, int(k)), num_samples)
        if k == num_samples:
            return samples[:k]

        chosen = [int(confidence.argmax().item())]
        while len(chosen) < k:
            sim = torch.matmul(samples, samples[chosen].t())
            best_sim = sim.max(dim=1).values
            best_sim[chosen] = 1.0
            next_idx = int(best_sim.argmin().item())
            if next_idx in chosen:
                break
            chosen.append(next_idx)
        return samples[chosen]

    def estimate_batch_prototypes(self, source, value, logits, labels=None, train_idx=None):
        proto_keys = source.new_zeros(self.c, self.proto_num, self.d)
        proto_values = value.new_zeros(self.c, self.proto_num, self.d)
        proto_valid = torch.zeros(self.c, self.proto_num, device=source.device, dtype=torch.bool)

        pred, conf, keep = self._flat_labels_or_predictions(logits, labels=labels, train_idx=train_idx)
        source_detached = F.normalize(source.detach(), dim=1)
        value_detached = value.detach()

        for cls_idx in range(self.c):
            cls_mask = keep & (pred == cls_idx)
            cls_idx_all = cls_mask.nonzero(as_tuple=False).squeeze(-1)
            if cls_idx_all.numel() == 0:
                continue

            if cls_idx_all.numel() > self.proto_max_nodes_per_class:
                cls_conf = conf.index_select(0, cls_idx_all)
                top_pos = cls_conf.topk(self.proto_max_nodes_per_class).indices
                cls_idx_all = cls_idx_all.index_select(0, top_pos)

            cls_source = source_detached.index_select(0, cls_idx_all)
            cls_value = value_detached.index_select(0, cls_idx_all)
            cls_conf = conf.index_select(0, cls_idx_all)
            k = min(self.proto_num, int(cls_source.size(0)))
            centers = self._select_proto_seeds(cls_source, cls_conf, k)

            for _ in range(self.proto_kmeans_iters):
                assign = torch.matmul(cls_source, centers.t()).argmax(dim=1)
                updated = []
                for slot in range(k):
                    slot_mask = assign == slot
                    if slot_mask.any():
                        updated.append(F.normalize(cls_source[slot_mask].mean(dim=0), dim=0))
                    else:
                        updated.append(centers[slot])
                centers = torch.stack(updated, dim=0)

            assign = torch.matmul(cls_source, centers.t()).argmax(dim=1)
            for slot in range(k):
                slot_mask = assign == slot
                if not slot_mask.any():
                    continue
                key = cls_source[slot_mask].mean(dim=0)
                val = cls_value[slot_mask].mean(dim=0)
                proto_keys[cls_idx, slot] = F.normalize(key, dim=0)
                proto_values[cls_idx, slot] = F.normalize(val, dim=0)
                proto_valid[cls_idx, slot] = True

        return proto_keys, proto_values, proto_valid

    def fuse_batch_and_bank_prototypes(self, batch_keys, batch_values, batch_valid):
        proto_keys = self.class_proto_keys.detach().clone()
        proto_values = self.class_proto_values.detach().clone()
        proto_valid = self.class_proto_valid.detach().clone()

        use_batch = batch_valid.any(dim=1)
        if use_batch.any():
            proto_keys[use_batch] = batch_keys[use_batch]
            proto_values[use_batch] = batch_values[use_batch]
            proto_valid[use_batch] = batch_valid[use_batch]
        return proto_keys, proto_values, proto_valid

    def build_proto_reference(
        self,
        z,
        shortcut_source,
        logits_hint,
        labels=None,
        train_idx=None,
        training=False,
    ):
        batch_keys = z.new_zeros(self.c, self.proto_num, self.d)
        batch_values = z.new_zeros(self.c, self.proto_num, self.d)
        batch_valid = torch.zeros(self.c, self.proto_num, device=z.device, dtype=torch.bool)
        if training:
            batch_keys, batch_values, batch_valid = self.estimate_batch_prototypes(
                shortcut_source,
                z,
                logits_hint,
                labels=labels,
                train_idx=train_idx,
            )

        proto_keys, proto_values, proto_valid = self.fuse_batch_and_bank_prototypes(
            batch_keys,
            batch_values,
            batch_valid,
        )

        pred, _, _ = self._flat_labels_or_predictions(logits_hint, labels=labels, train_idx=train_idx)
        source_norm = F.normalize(shortcut_source, dim=1)
        reference = z.new_zeros(z.size(0), self.d)
        valid_mask = torch.zeros(z.size(0), device=z.device, dtype=torch.bool)

        for cls_idx in range(self.c):
            node_mask = pred == cls_idx
            if not node_mask.any():
                continue
            valid_slots = proto_valid[cls_idx].nonzero(as_tuple=False).squeeze(-1)
            if valid_slots.numel() == 0:
                continue
            node_idx = node_mask.nonzero(as_tuple=False).squeeze(-1)
            cls_keys = proto_keys[cls_idx].index_select(0, valid_slots)
            cls_values = proto_values[cls_idx].index_select(0, valid_slots)
            sim = torch.matmul(source_norm.index_select(0, node_idx), cls_keys.t())
            if valid_slots.numel() > 1:
                own_slot = sim.argmax(dim=1, keepdim=True)
                masked = sim.clone()
                masked.scatter_(1, own_slot, -1e9)
                weights = torch.softmax(masked / self.proto_temp, dim=1)
            else:
                weights = sim.new_ones(sim.size(0), 1)
            reference.index_copy_(0, node_idx, torch.matmul(weights, cls_values))
            valid_mask.index_fill_(0, node_idx, True)

        diff = torch.abs(z - reference)
        gate_input = torch.cat([z, reference, diff, z * reference], dim=-1)
        proto_delta = self.proto_fuser(gate_input)
        z_proto = self.proto_norm(z + self.proto_inject_weight * proto_delta)
        gate = torch.sigmoid(self.proto_gate(gate_input) / self.proto_gate_temp)
        gate = torch.where(valid_mask.unsqueeze(-1), gate, gate.new_full(gate.shape, self.proto_gate_target))
        return z_proto, reference, gate, valid_mask, batch_keys, batch_values, batch_valid

    def build_strong_main_representation(
        self,
        z,
        base_mediator,
        backbone_states,
        proto_reference=None,
        proto_valid_mask=None,
        h_global_context=None,
        training=False,
    ):
        if backbone_states:
            layer_stack = torch.stack(backbone_states, dim=1)
            query = F.normalize(self.main_layer_query(z), dim=-1)
            keys = F.normalize(layer_stack, dim=-1)
            score = (keys * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(float(self.d))
            weight = torch.softmax(score, dim=1)
            layer_context = (weight.unsqueeze(-1) * layer_stack).sum(dim=1)
        else:
            layer_context = z

        if proto_reference is None:
            proto_reference = z
        elif proto_valid_mask is not None:
            proto_reference = torch.where(proto_valid_mask.unsqueeze(-1), proto_reference, z)

        global_context = h_global_context if h_global_context is not None else z
        anchor = 0.5 * (z + base_mediator)
        fusion_input = torch.cat([anchor, layer_context, proto_reference, global_context], dim=-1)
        delta = self.main_state_proj(fusion_input)
        delta = F.dropout(delta, self.dropout, training=training)
        gate = torch.sigmoid(self.main_state_gate(fusion_input) / self.main_gate_temp)
        main_repr = self.main_state_norm(anchor + self.main_path_weight * gate * delta)
        return main_repr, gate

    def get_proto_contexts(self):
        if not self.use_proto_contexts or self.proto_context_weight <= 0.0:
            return None
        valid = self.class_proto_valid.view(-1)
        if not valid.any():
            return None
        contexts = self.class_proto_values.view(-1, self.d)[valid]
        if contexts.numel() == 0:
            return None
        return self.proto_context_weight * F.normalize(contexts, dim=1)

    @torch.no_grad()
    def update_class_prototypes(self, proto_keys, proto_values, proto_valid):
        if proto_keys is None or proto_values is None or proto_valid is None:
            return
        for cls_idx in range(self.c):
            valid_slots = proto_valid[cls_idx].nonzero(as_tuple=False).squeeze(-1)
            if valid_slots.numel() == 0:
                continue
            for slot in valid_slots.tolist():
                new_key = F.normalize(proto_keys[cls_idx, slot].detach(), dim=0)
                new_value = F.normalize(proto_values[cls_idx, slot].detach(), dim=0)
                if self.class_proto_valid[cls_idx, slot]:
                    old_key = self.class_proto_keys[cls_idx, slot]
                    old_value = self.class_proto_values[cls_idx, slot]
                    new_key = F.normalize(
                        self.proto_momentum * old_key + (1.0 - self.proto_momentum) * new_key,
                        dim=0,
                    )
                    new_value = F.normalize(
                        self.proto_momentum * old_value + (1.0 - self.proto_momentum) * new_value,
                        dim=0,
                    )
                self.class_proto_keys[cls_idx, slot] = new_key
                self.class_proto_values[cls_idx, slot] = new_value
                self.class_proto_valid[cls_idx, slot] = True

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

    def compute_edge_summaries(self, h, edge_index, training=False):
        """
        Build both useful and low-relevance neighbor summaries.

        useful_summary_v = sum_u norm_uv * g_uv       * h_u
        noise_summary_v  = sum_u norm_uv * (1 - g_uv) * h_u

        The existing edge feature modes (mul/diff/degree/...) define g_uv.
        When edge_gate_mode='scalar', g_uv is one score per edge. When
        edge_gate_mode='vector', g_uv is a per-dimension edge gate so each
        hidden channel can keep or reject a neighbor independently.
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

    def _graph_cfam_energy(self, value):
        # Row-wise energy spectralization.  Squared activations highlight
        # dimensions that dominate local smooth/residual components, and the
        # normalization removes pure scale effects.
        energy = value.pow(2)
        denom = energy.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return energy / denom

    def graph_cfam_adapt(self, h, edge_index, training=False):
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
        adapted = h + self.edge_blend * causal_local + self.graph_cfam_residual_blend * residual
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

    def encode_representation(self, x, edge_index, training=False, labels=None, train_idx=None):
        x = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(self.input_proj(x))
        layerwise_states = []
        backbone_states = []
        layerwise_gate_loss = h.new_zeros(())
        layerwise_gate_mean = h.new_zeros(())
        layerwise_gate_layers = 0
        graph_cfam_gate_loss = h.new_zeros(())
        graph_cfam_gate_mean = h.new_zeros(())
        graph_cfam_layers = 0
        num_backbone_layers = len(self.backbone_layers)

        for layer_idx, layer in enumerate(self.backbone_layers):
            h = F.dropout(h, self.dropout, training=training)
            h = self.act_fn(layer(h, edge_index))
            backbone_states.append(h)

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
        if self.use_graph_cfam:
            z, causal_local_final, edge_summary, cfam_gate_final, _, cfam_gate_loss_final = self.graph_cfam_adapt(
                h,
                edge_index,
                training=training,
            )
            prev_layers = max(0, int(self._last_graph_cfam_layers))
            denom = float(prev_layers + 1)
            self._last_graph_cfam_gate_mean = (
                self._last_graph_cfam_gate_mean * float(prev_layers) + cfam_gate_final.mean().detach()
            ) / denom
            self._last_graph_cfam_gate_loss = (
                self._last_graph_cfam_gate_loss * float(prev_layers) + cfam_gate_loss_final
            ) / denom
            self._last_graph_cfam_layers = prev_layers + 1
        else:
            edge_summary, noise_summary, _ = self.compute_edge_summaries(h, edge_index, training=training)
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
        logits_hint = self.classifier(z)
        (
            z,
            proto_reference,
            mediator_gate,
            proto_valid_mask,
            batch_proto_keys,
            batch_proto_values,
            batch_proto_valid,
        ) = self.build_proto_reference(
            z,
            edge_summary,
            logits_hint,
            labels=labels,
            train_idx=train_idx,
            training=training,
        )
        base_mediator = z * mediator_gate
        z_spurious = z * (1.0 - mediator_gate)
        z_mediator, main_gate = self.build_strong_main_representation(
            z,
            base_mediator,
            backbone_states,
            proto_reference=proto_reference,
            proto_valid_mask=proto_valid_mask,
            h_global_context=h_global_context,
            training=training,
        )
        causal_score = mediator_gate.mean(dim=0)
        pollution_score = (1.0 - mediator_gate).mean(dim=0)
        dag_total = self.A_feat.new_zeros(self.dag_var_dim, self.dag_var_dim)
        zero = z.new_zeros(())
        self._last_ica_cov_loss = zero
        self._last_ica_ng_loss = zero
        self._last_ica_gate_loss = zero
        self._last_ica_entropy_loss = zero
        self._last_ica_gate_mean = mediator_gate.mean().detach()
        proto_gate_loss = (mediator_gate.mean() - self.proto_gate_target).pow(2)
        self._last_proto_gate_loss = proto_gate_loss
        self._last_proto_gate_mean = mediator_gate.mean().detach()
        self._last_proto_valid_ratio = proto_valid_mask.float().mean().detach()
        self._last_main_gate_mean = main_gate.mean().detach()

        z_mediator = F.dropout(z_mediator, self.dropout, training=training)
        z_spurious = F.dropout(z_spurious, self.dropout, training=training)
        layerwise_spurious = None
        if self.use_layerwise_spurious_contexts and layerwise_states:
            spurious_gate = (1.0 - mediator_gate).unsqueeze(0)
            layerwise_spurious = torch.stack(layerwise_states, dim=0) * spurious_gate
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
            proto_reference,
            proto_valid_mask,
            batch_proto_keys,
            batch_proto_values,
            batch_proto_valid,
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

    def get_nego_contexts(self, z_source=None, y=None, sample_idx=None, training=False):
        if not self.use_nego_context or self.nego_context_weight <= 0.0:
            return None

        contexts = None
        if training and z_source is not None and y is not None and sample_idx is not None and sample_idx.numel() > 0:
            y_flat = y.squeeze(-1).long()
            labels = y_flat[sample_idx]
            answers = self.negative_prompt_answers(z_source.index_select(0, sample_idx))
            if answers is not None:
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
        contexts = F.normalize(contexts.detach(), dim=1)
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

    def merge_frontdoor_contexts(self, *context_sets):
        contexts = [ctx for ctx in context_sets if ctx is not None and ctx.numel() > 0]
        if not contexts:
            return None
        return torch.cat(contexts, dim=0)

    def build_frontdoor_contexts(
        self,
        gmm_contexts=None,
        global_contexts=None,
        layerwise_contexts=None,
        nego_contexts=None,
        proto_contexts=None,
        training=False,
    ):
        if self.fd_context_source == 'nego_only':
            return self.sample_frontdoor_contexts(nego_contexts, training=training)
        return self.merge_frontdoor_contexts(
            gmm_contexts,
            global_contexts,
            layerwise_contexts,
            nego_contexts,
            proto_contexts,
        )

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

    def forward(self, x, edge_index, training=False, y=None, train_idx=None):
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
            proto_reference,
            proto_valid_mask,
            batch_proto_keys,
            batch_proto_values,
            batch_proto_valid,
        ) = self.encode_representation(
            x,
            edge_index,
            training=training,
            labels=y,
            train_idx=train_idx,
        )

        gmm_contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(training=training),
            training=training,
        )
        layerwise_contexts = self.get_layerwise_spurious_contexts(
            layerwise_spurious,
            self.compute_pseudo_env_probs(z_spurious) if self.num_envs > 1 else None,
        )
        nego_contexts = self.get_nego_contexts(training=False)
        proto_contexts = self.get_proto_contexts()
        contexts = self.build_frontdoor_contexts(
            gmm_contexts,
            self.get_global_contexts(h_global_context),
            layerwise_contexts,
            nego_contexts,
            proto_contexts,
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
                proto_reference,
                proto_valid_mask,
                batch_proto_keys,
                batch_proto_values,
                batch_proto_valid,
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

    def compute_proto_alignment_loss(self, z_mediator, proto_reference, proto_valid_mask, train_idx=None):
        if (
            self.lambda_proto_align <= 0.0
            or z_mediator is None
            or proto_reference is None
            or proto_valid_mask is None
            or z_mediator.numel() == 0
        ):
            return z_mediator.new_zeros(())

        if train_idx is not None and train_idx.numel() > 0:
            med = z_mediator[train_idx]
            ref = proto_reference[train_idx].detach()
            valid = proto_valid_mask[train_idx]
        else:
            med = z_mediator
            ref = proto_reference.detach()
            valid = proto_valid_mask

        if not valid.any():
            return med.new_zeros(())

        med = med[valid]
        ref = ref[valid]
        return (1.0 - (F.normalize(med, dim=1) * F.normalize(ref, dim=1)).sum(dim=1)).mean()

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
        self.update_class_prototypes(
            state_payload.get('proto_keys'),
            state_payload.get('proto_values'),
            state_payload.get('proto_valid'),
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
                z_mediator_smooth,
                _,
                mediator_logits_smooth,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = self.encode_representation(
                x,
                smooth_edge_index,
                training=True,
                labels=y,
                train_idx=train_idx,
            )

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
        y = y.to(x.device)
        train_idx = data.train_idx.to(device=x.device, dtype=torch.long)

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
            proto_reference_all,
            proto_valid_mask_all,
            batch_proto_keys,
            batch_proto_values,
            batch_proto_valid,
        ) = self.forward(
            x,
            edge_index,
            training=True,
            y=y,
            train_idx=train_idx,
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
        proto_contexts = self.get_proto_contexts()
        contexts = self.build_frontdoor_contexts(
            gmm_contexts,
            global_contexts,
            layerwise_contexts,
            nego_contexts,
            proto_contexts,
            training=True,
        )
        num_gmm_contexts = 0 if gmm_contexts is None else int(gmm_contexts.size(0))
        num_global_contexts = 0 if global_contexts is None else int(global_contexts.size(0))
        num_layerwise_contexts = 0 if layerwise_contexts is None else int(layerwise_contexts.size(0))
        num_nego_contexts = 0 if nego_contexts is None else int(nego_contexts.size(0))
        num_mixed_contexts = 0
        fd_logits_tr, fd_stack_tr = self.frontdoor_logits_from_contexts(med_tr, spu_tr, contexts)
        final_logits_tr = self.blend_logits(mediator_logits_tr, fd_logits_tr)
        loss_cf = self.A_feat.new_zeros(())
        cf_pred_shift = self.A_feat.new_zeros(())

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        zero = self.A_feat.new_zeros(())
        loss_main = self.compute_supervised_loss(mediator_logits_tr, y_tr, criterion, args).mean()
        loss_graph_delf = zero
        loss_graph_cfam_gate = zero
        graph_cfam_gate_mean = zero
        loss_role = zero
        loss_role_med_y = zero
        loss_role_spu_y = zero
        loss_role_spu_env = zero
        loss_role_med_env = zero
        loss_dag = zero
        loss_dag_label = zero
        loss_ica_cov = zero
        loss_ica_ng = zero
        loss_ica_gate = zero
        loss_ica_entropy = zero
        loss_global_env = zero
        current_counterexample_penalty = self.counterexample_penalty.new_zeros(self.counterexample_penalty.size())
        loss_enhance_sem = zero
        loss_proto_gate = zero if self._last_proto_gate_loss is None else self._last_proto_gate_loss
        proto_gate_mean = zero if self._last_proto_gate_mean is None else self._last_proto_gate_mean
        proto_valid_ratio = zero if self._last_proto_valid_ratio is None else self._last_proto_valid_ratio
        loss_proto_align = zero
        loss_bismooth = zero
        loss_bismooth_cls = zero
        bismooth_valid_ratio = zero
        loss_layerwise_gate = zero
        layerwise_gate_mean = zero

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

        loss_nego, nego_extra_score, nego_self_score = self.compute_nego_loss(
            nego_source_all,
            y,
            train_idx,
        )

        loss_med = loss_main
        loss_fd_aug = zero
        loss_var = zero
        loss_ind = zero
        loss_sem = zero
        loss_degree = zero
        loss_spu_y = zero
        loss_inv = zero

        total_loss = (
            loss_cls
            + 0.5 * loss_main
            + self.lambda_fd * loss_fd
            + self.lambda_spu * loss_spu
            + 0.5 * self.lambda_env * loss_env_med
            + self.lambda_nego * loss_nego
            + self.lambda_proto_gate * loss_proto_gate
        )

        state_payload = None
        if update_state:
            state_payload = {
                'spu_tr': spu_tr.detach(),
                'env_probs_tr': env_probs_spu.detach() if env_probs_spu is not None else None,
                'edge_latent_tr': edge_latent_tr.detach(),
                'counterexample_penalty': current_counterexample_penalty.detach(),
                'nego_contexts': nego_contexts.detach() if nego_contexts is not None else None,
                'proto_keys': batch_proto_keys.detach(),
                'proto_values': batch_proto_values.detach(),
                'proto_valid': batch_proto_valid.detach(),
            }

        num_contexts = 0 if contexts is None else int(contexts.size(0))
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_med': loss_med,
            'loss_fd': loss_fd,
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
            'loss_bismooth': loss_bismooth,
            'loss_bismooth_cls': loss_bismooth_cls,
            'loss_layerwise_gate': loss_layerwise_gate,
            'loss_graph_cfam_gate': loss_graph_cfam_gate,
            'loss_graph_delf': loss_graph_delf,
            'loss_enhance_sem': loss_enhance_sem,
            'loss_nego': loss_nego,
            'loss_proto_gate': loss_proto_gate,
            'loss_proto_align': loss_proto_align,
            'nego_extra_score': nego_extra_score,
            'nego_self_score': nego_self_score,
            'bismooth_valid_ratio': bismooth_valid_ratio.detach(),
            'layerwise_gate_mean': layerwise_gate_mean.detach(),
            'layerwise_gate_layers': torch.tensor(float(self._last_layerwise_gate_layers), device=x.device),
            'graph_cfam_gate_mean': graph_cfam_gate_mean.detach(),
            'graph_cfam_layers': torch.tensor(float(self._last_graph_cfam_layers), device=x.device),
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'proto_gate_mean': proto_gate_mean.detach(),
            'proto_valid_ratio': proto_valid_ratio.detach(),
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
            'num_proto_contexts': torch.tensor(float(0 if proto_contexts is None else int(proto_contexts.size(0))), device=x.device),
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
