import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_sparse import SparseTensor, matmul


def gcn_backbone_conv(x, edge_index):
    """
    CaNet-style GCN propagation used as the front-door encoder backbone.
    """
    num_nodes = x.size(0)
    row, col = edge_index
    deg = degree(col, num_nodes).float().clamp_min(1.0)
    deg_in = deg[col].pow(-0.5)
    deg_out = deg[row].pow(-0.5)
    value = torch.ones_like(row, dtype=x.dtype) * deg_in * deg_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
    return matmul(adj, x)


class FrontDoorBackboneLayer(nn.Module):
    """
    Single front-door encoder layer with a CaNet-style backbone choice.

    - gcn: normalized graph propagation + self-feature concatenation
    - gat: single-head attention aggregation with the same flavor as CaNet
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
                f"Front-door TriSplit backbone_type='{backbone_type}' is not implemented. "
                "Use 'gcn' or 'gat'."
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


class GlobalLinearAttention(nn.Module):
    """
    MLEI-style global linear attention used as the non-local diffusion channel.
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


class GraphFrontDoorTriSplit(nn.Module):
    """
    Front-door graph model with a direct three-way split after the GNN encoder.

    The model removes the latent DAG and instead learns a channel-wise softmax
    split of the post-GNN representation z into:
      1) causal: stable label-relevant features used for prediction;
      2) env: environment/spurious features used only to form contexts;
      3) residual: unrelated features that should be uninformative.

    The front-door branch uses causal features plus environment contexts. It
    never receives the current node's env representation, which avoids the
    spurious -> context -> label leak of the previous DAG-mixer path.
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
        self.global_info_mode = getattr(args, 'global_info_mode', 'advective')
        self.global_alpha = float(getattr(args, 'global_alpha', 0.2))
        self.global_beta = float(getattr(args, 'global_beta', 0.5))
        self.global_steps = max(1, int(getattr(args, 'global_steps', 1)))
        self.global_local_source = getattr(args, 'global_local_source', 'gcn')
        if self.use_global_info:
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
                    f"Unsupported global_info_mode='{self.global_info_mode}'. Use 'linear' or 'advective'."
                )
            self.global_fuse_norm = nn.LayerNorm(self.d)
        else:
            self.global_encoder = None
            self.global_fuse_norm = None

        self.classifier = nn.Linear(self.d, c)
        self.fd_classifier = nn.Linear(self.d, c)
        self.env_classifier = nn.Linear(self.d, self.num_envs)

        self.edge_score_temp = max(1e-3, float(getattr(args, 'edge_score_temp', 2.0)))
        self.edge_blend = max(0.0, float(getattr(args, 'edge_blend', 0.2)))
        self.edge_pair_encoder = nn.Sequential(
            nn.Linear(self.d * 4, self.d),
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

        # Channel-wise 3-way split: causal / env / residual.
        self.split_temp = max(1e-3, float(getattr(args, 'split_temp', 1.0)))
        prior = torch.tensor(
            [
                float(getattr(args, 'split_prior_causal', 0.5)),
                float(getattr(args, 'split_prior_env', 0.3)),
                float(getattr(args, 'split_prior_residual', 0.2)),
            ],
            dtype=torch.float32,
        ).clamp_min(1e-6)
        prior = prior / prior.sum()
        self.register_buffer('split_prior', prior)
        self.split_logits = Parameter(torch.zeros(self.d, 3))

        # Front-door fusion: causal + context -> intervened representation.
        self.fd_fuser = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.fd_norm = nn.LayerNorm(self.d)

        self.dropout = float(getattr(args, 'dropout', 0.0))
        self.gamma = float(getattr(args, 'gamma', 0.99))
        self.fd_blend = float(getattr(args, 'fd_blend', 0.5))
        self.current_fd_blend = self.fd_blend
        self.current_intervention_scale = 1.0
        self.fd_sample_k = max(0, int(getattr(args, 'K', 0)))
        self.context_sample_seed = int(getattr(args, 'seed', 0))
        self.context_mode = getattr(args, 'context_mode', 'both')
        if self.context_mode not in ('gmm', 'proto', 'both', 'none'):
            raise ValueError("context_mode must be one of: 'gmm', 'proto', 'both', 'none'")
        self.use_spu_gmm = bool(getattr(args, 'use_spu_gmm', True))
        requested_gmm_sample_k = int(getattr(args, 'gmm_sample_k', 0))
        if requested_gmm_sample_k <= 0:
            requested_gmm_sample_k = self.fd_sample_k
        self.gmm_sample_k = max(0, requested_gmm_sample_k)
        self.gmm_min_var = max(1e-6, float(getattr(args, 'gmm_min_var', 1e-4)))
        self.gmm_max_std = max(0.0, float(getattr(args, 'gmm_max_std', 0.2)))
        self.eval_gmm_noise = bool(getattr(args, 'eval_gmm_noise', False))

        # Loss weights.
        self.lambda_causal = float(getattr(args, 'lambda_causal', 1.0))
        self.lambda_spu = float(getattr(args, 'lambda_spu', 0.05))
        self.lambda_fd = float(getattr(args, 'lambda_fd', 0.5))
        self.lambda_env = float(getattr(args, 'lambda_env', 0.05))
        self.lambda_residual = float(getattr(args, 'lambda_residual', 0.01))
        self.lambda_orth = float(getattr(args, 'lambda_orth', 0.01))
        self.lambda_gate_balance = float(getattr(args, 'lambda_gate_balance', 0.01))
        self.lambda_gate_entropy = float(getattr(args, 'lambda_gate_entropy', 0.001))
        self.lambda_var = float(getattr(args, 'lambda_var', 0.0))
        self.lambda_inv = float(getattr(args, 'lambda_inv', 0.0))

        # Backward-compatible zeroed attributes used by older logging code.
        self.lambda_med = self.lambda_causal
        self.lambda_ind = self.lambda_orth
        self.lambda_fd_aug = float(getattr(args, 'lambda_fd_aug', 0.0))
        self.lambda_dag = float(getattr(args, 'lambda_dag', 0.0))
        self.lambda_dag_label = float(getattr(args, 'lambda_dag_label', 0.0))
        self.lambda_sem = float(getattr(args, 'lambda_sem', 0.0))
        self.lambda_spu_y = float(getattr(args, 'lambda_spu_y', 0.0))

        self.pseudo_env_balance = float(getattr(args, 'pseudo_env_balance', 1.0))

        self.register_buffer('gmm_spu_mean', torch.zeros(self.num_envs, self.d))
        self.register_buffer('gmm_spu_var', torch.ones(self.num_envs, self.d))
        self.register_buffer('gmm_spu_valid', torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer('proto_env_mean', torch.zeros(self.num_envs, self.d))
        self.register_buffer('proto_env_valid', torch.zeros(self.num_envs, dtype=torch.bool))

        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for layer in self.backbone_layers:
            layer.reset_parameters()
        if self.global_encoder is not None:
            self.global_encoder.reset_parameters()
            self.global_fuse_norm.reset_parameters()
        self.classifier.reset_parameters()
        self.fd_classifier.reset_parameters()
        self.env_classifier.reset_parameters()
        self._reset_module_parameters(self.edge_pair_encoder)
        self.edge_score_head.reset_parameters()
        self.edge_summary_norm.reset_parameters()
        self._reset_module_parameters(self.node_edge_fuser)
        nn.init.zeros_(self.node_edge_fuser[-1].weight)
        nn.init.zeros_(self.node_edge_fuser[-1].bias)
        self.node_edge_norm.reset_parameters()
        for module in self.fd_fuser:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.fd_norm.reset_parameters()

        prior = self.split_prior.clamp_min(1e-6)
        with torch.no_grad():
            self.split_logits.copy_(prior.log().unsqueeze(0).repeat(self.d, 1))

        self.gmm_spu_mean.zero_()
        self.gmm_spu_var.fill_(1.0)
        self.gmm_spu_valid.zero_()
        self.proto_env_mean.zero_()
        self.proto_env_valid.zero_()
        self.current_fd_blend = self.fd_blend
        self.current_intervention_scale = 1.0

    def _reset_module_parameters(self, module):
        for sub_module in module.modules():
            if sub_module is module:
                continue
            if hasattr(sub_module, 'reset_parameters'):
                sub_module.reset_parameters()

    def split_gates(self):
        return F.softmax(self.split_logits / self.split_temp, dim=-1)

    def split_representation(self, z):
        gates = self.split_gates()
        g_causal = gates[:, 0].unsqueeze(0)
        g_env = gates[:, 1].unsqueeze(0)
        g_res = gates[:, 2].unsqueeze(0)
        z_causal = z * g_causal
        z_env = z * g_env
        z_residual = z * g_res
        return z_causal, z_env, z_residual, gates

    def compute_edge_semantic_summary(self, h, edge_index, training=False):
        if edge_index.numel() == 0:
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
        edge_gate = torch.sigmoid(edge_logits)

        deg = degree(dst, h.size(0)).to(device=h.device, dtype=h.dtype).clamp_min(1.0)
        norm = deg[src].pow(-0.5) * deg[dst].pow(-0.5)
        edge_weight = torch.nan_to_num(norm * edge_gate, nan=0.0, posinf=0.0, neginf=0.0)

        edge_summary = h.new_zeros(h.size())
        edge_summary.index_add_(0, dst, edge_weight.unsqueeze(-1) * h[src])
        edge_summary = self.edge_summary_norm(edge_summary)
        return edge_summary

    def fuse_node_edge_representation(self, h, edge_summary, training=False):
        fuse_input = torch.cat([h, edge_summary, h * edge_summary], dim=-1)
        edge_delta = self.node_edge_fuser(fuse_input)
        edge_delta = F.dropout(edge_delta, self.dropout, training=training)
        return self.node_edge_norm(h + self.edge_blend * edge_delta)

    def encode_base_representation(self, x, edge_index, training=False):
        x = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(self.input_proj(x))
        for layer in self.backbone_layers:
            h = F.dropout(h, self.dropout, training=training)
            h = self.act_fn(layer(h, edge_index))

        if self.global_encoder is not None:
            if self.global_info_mode == 'linear':
                h_global = self.global_encoder(h, training=training)
            else:
                local_fn = None
                if self.global_local_source == 'edge':
                    local_fn = self.compute_edge_semantic_summary
                elif self.global_local_source != 'gcn':
                    raise ValueError(
                        f"Unsupported global_local_source='{self.global_local_source}'. Use 'edge' or 'gcn'."
                    )
                h_global = self.global_encoder(h, edge_index, training=training, local_fn=local_fn)
            h = h + self.global_alpha * self.global_fuse_norm(h_global)

        edge_summary = self.compute_edge_semantic_summary(h, edge_index, training=training)
        z = self.fuse_node_edge_representation(h, edge_summary, training=training)
        return z, edge_summary

    def encode_representation(self, x, edge_index, training=False):
        z, edge_summary = self.encode_base_representation(x, edge_index, training=training)
        z_causal, z_env, z_residual, gates = self.split_representation(z)
        z_causal = F.dropout(z_causal, self.dropout, training=training)
        z_env = F.dropout(z_env, self.dropout, training=training)
        z_residual = F.dropout(z_residual, self.dropout, training=training)
        causal_logits = self.classifier(z_causal)
        return z, edge_summary, z_causal, z_env, z_residual, gates, causal_logits

    def compute_pseudo_env_probs(self, z_env):
        if self.num_envs <= 1 or z_env.numel() == 0:
            return z_env.new_ones(z_env.size(0), 1)
        return F.softmax(self.env_classifier(z_env), dim=-1)

    def _fit_env_stats(self, z_env, env_probs):
        means = z_env.new_zeros(self.num_envs, self.d)
        vars_ = z_env.new_ones(self.num_envs, self.d)
        valid = torch.zeros(self.num_envs, device=z_env.device, dtype=torch.bool)
        if z_env is None or z_env.numel() == 0:
            return means, vars_, valid
        if env_probs is None or env_probs.numel() == 0 or env_probs.size(-1) != self.num_envs:
            env_probs = self.compute_pseudo_env_probs(z_env)

        env_probs = env_probs.detach().clamp_min(0.0)
        z_detached = z_env.detach()
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

    def _select_context_subset(self, contexts, training=False):
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
            indices = torch.randperm(num_contexts, generator=generator)[:self.fd_sample_k].to(contexts.device)
        return contexts.index_select(0, indices)

    def get_proto_contexts(self, z_env=None, env_probs=None, training=False):
        if self.context_mode not in ('proto', 'both'):
            return None
        if training and z_env is not None:
            means, _, valid = self._fit_env_stats(z_env, env_probs)
        else:
            means, valid = self.proto_env_mean, self.proto_env_valid
        valid_envs = valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_envs.numel() == 0:
            return None
        contexts = means.index_select(0, valid_envs)
        return F.normalize(contexts, dim=1)

    def sample_gmm_contexts(self, z_env=None, env_probs=None, training=False):
        if self.context_mode not in ('gmm', 'both'):
            return None
        if not self.use_spu_gmm or self.gmm_sample_k <= 0:
            return None

        if training and z_env is not None:
            means, vars_, valid = self._fit_env_stats(z_env, env_probs)
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

        use_noise = training or self.eval_gmm_noise
        if use_noise:
            if training:
                noise = torch.randn_like(mean)
                noise_scale = float(getattr(self, 'current_intervention_scale', 1.0))
            else:
                generator = torch.Generator(device=mean.device)
                generator.manual_seed(self.context_sample_seed + sample_k + int(valid_envs.numel()))
                noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
                noise_scale = 1.0
            contexts = mean + noise_scale * noise * std
        else:
            contexts = mean
        return F.normalize(contexts, dim=1)

    def build_frontdoor_contexts(self, z_env=None, env_probs=None, training=False):
        if self.context_mode == 'none':
            return None, {'proto': 0, 'gmm': 0, 'total_before_sample': 0}
        pieces = []
        proto_contexts = self.get_proto_contexts(z_env, env_probs, training=training)
        if proto_contexts is not None and proto_contexts.size(0) > 0:
            pieces.append(proto_contexts)
        gmm_contexts = self.sample_gmm_contexts(z_env, env_probs, training=training)
        if gmm_contexts is not None and gmm_contexts.size(0) > 0:
            pieces.append(gmm_contexts)
        if not pieces:
            return None, {'proto': 0, 'gmm': 0, 'total_before_sample': 0}
        contexts = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        info = {
            'proto': 0 if proto_contexts is None else int(proto_contexts.size(0)),
            'gmm': 0 if gmm_contexts is None else int(gmm_contexts.size(0)),
            'total_before_sample': int(contexts.size(0)),
        }
        contexts = self._select_context_subset(contexts, training=training)
        return contexts, info

    # Backward-compatible alias. If z_spurious is supplied, build fresh
    # contexts from it; otherwise use EMA buffers for evaluation.
    def get_frontdoor_contexts(self, z_spurious=None, env_probs=None):
        contexts, _ = self.build_frontdoor_contexts(
            z_spurious,
            env_probs,
            training=z_spurious is not None,
        )
        return contexts

    def frontdoor_logits_from_contexts(self, z_causal, contexts):
        base_logits = self.fd_classifier(z_causal)
        if contexts is None or contexts.size(0) == 0:
            return base_logits, None

        num_contexts = contexts.size(0)
        causal_expand = z_causal.unsqueeze(1).expand(-1, num_contexts, -1)
        context_expand = contexts.unsqueeze(0).expand(z_causal.size(0), -1, -1)
        fused_input = torch.cat([causal_expand, context_expand], dim=-1)
        fused = self.fd_fuser(fused_input.reshape(-1, self.d * 2)).view(z_causal.size(0), num_contexts, self.d)
        fused = self.fd_norm(fused + causal_expand)

        logits_stack = self.fd_classifier(fused.reshape(-1, self.d)).view(z_causal.size(0), num_contexts, self.c)
        fd_logits = logits_stack.mean(dim=1)
        return fd_logits, logits_stack

    def blend_logits(self, causal_logits, fd_logits, blend=None):
        if fd_logits is None:
            return causal_logits
        if blend is None:
            blend = float(getattr(self, 'current_fd_blend', self.fd_blend))
        blend = min(max(float(blend), 0.0), 1.0)
        return (1.0 - blend) * causal_logits + blend * fd_logits

    def forward(self, x, edge_index, training=False):
        (
            z,
            edge_summary,
            z_causal,
            z_env,
            z_residual,
            gates,
            causal_logits,
        ) = self.encode_representation(x, edge_index, training=training)

        contexts, _ = self.build_frontdoor_contexts(training=training)
        fd_logits, fd_stack = self.frontdoor_logits_from_contexts(z_causal, contexts)
        logits = self.blend_logits(causal_logits, fd_logits)

        if training:
            return (
                logits,
                z,
                edge_summary,
                z_causal,
                z_env,
                z_residual,
                gates,
                causal_logits,
                fd_logits,
                fd_stack,
            )
        return logits

    def compute_supervised_loss(self, logits, y, criterion, args):
        binary_loss_mode = getattr(args, 'binary_loss_mode', 'original_bce')
        if args.dataset in ('twitch', 'elliptic') and binary_loss_mode == 'ce' and logits.size(1) > 1:
            return criterion(logits, y.squeeze().long())
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

    def frozen_linear_logits(self, linear_module, z):
        bias = linear_module.bias.detach() if linear_module.bias is not None else None
        return F.linear(z, linear_module.weight.detach(), bias)

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
            return logits.new_zeros(())
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

    def compute_frontdoor_variance_loss(self, logits_stack):
        if logits_stack is None or logits_stack.size(1) <= 1:
            return self.split_logits.new_zeros(())
        probs = torch.softmax(logits_stack, dim=-1)
        return probs.var(dim=1, unbiased=False).mean()

    def pairwise_cosine_loss(self, a, b):
        if a.numel() == 0 or b.numel() == 0:
            return self.split_logits.new_zeros(())
        a_norm = F.normalize(a, dim=1)
        b_norm = F.normalize(b, dim=1)
        return ((a_norm * b_norm).sum(dim=1) ** 2).mean()

    def compute_trisplit_orth_loss(self, z_causal, z_env, z_residual):
        return (
            self.pairwise_cosine_loss(z_causal, z_env)
            + self.pairwise_cosine_loss(z_causal, z_residual)
            + self.pairwise_cosine_loss(z_env, z_residual)
        ) / 3.0

    def gate_balance_loss(self, gates):
        mean_gate = gates.mean(dim=0).clamp_min(1e-8)
        target = self.split_prior.to(device=gates.device, dtype=gates.dtype).clamp_min(1e-8)
        target = target / target.sum()
        return F.kl_div(mean_gate.log(), target, reduction='sum')

    def gate_entropy_loss(self, gates):
        return -(gates * gates.clamp_min(1e-8).log()).sum(dim=-1).mean()

    def update_frontdoor_state(self, z_env, env_probs=None):
        if z_env is None or z_env.numel() == 0:
            return
        means, vars_, valid = self._fit_env_stats(z_env, env_probs)
        momentum = min(max(float(self.gamma), 0.0), 1.0)
        for env_idx in valid.nonzero(as_tuple=False).squeeze(-1).tolist():
            mean = means[env_idx]
            var = vars_[env_idx]

            if self.proto_env_valid[env_idx]:
                proto_mean = momentum * self.proto_env_mean[env_idx] + (1.0 - momentum) * mean
            else:
                proto_mean = mean
            self.proto_env_mean[env_idx] = proto_mean
            self.proto_env_valid[env_idx] = True

            if self.use_spu_gmm:
                if self.gmm_spu_valid[env_idx]:
                    gmm_mean = momentum * self.gmm_spu_mean[env_idx] + (1.0 - momentum) * mean
                    gmm_var = momentum * self.gmm_spu_var[env_idx] + (1.0 - momentum) * var
                else:
                    gmm_mean = mean
                    gmm_var = var
                self.gmm_spu_mean[env_idx] = gmm_mean
                self.gmm_spu_var[env_idx] = gmm_var.clamp_min(self.gmm_min_var)
                self.gmm_spu_valid[env_idx] = True

    @torch.no_grad()
    def apply_state_update(self, state_payload):
        if state_payload is None:
            return
        self.update_frontdoor_state(
            state_payload.get('env_tr'),
            state_payload.get('env_probs_tr'),
        )

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx

        (
            _,
            _,
            z_causal_all,
            z_env_all,
            z_residual_all,
            gates,
            causal_logits_all,
        ) = self.encode_representation(x, edge_index, training=True)

        y_tr = y[train_idx]
        z_causal_tr = z_causal_all[train_idx]
        z_env_tr = z_env_all[train_idx]
        z_residual_tr = z_residual_all[train_idx]
        causal_logits_tr = causal_logits_all[train_idx]

        if self.num_envs > 1:
            env_logits_env = self.env_classifier(z_env_tr)
            env_probs = F.softmax(env_logits_env, dim=-1)
            loss_spu = self.compute_pseudo_env_loss(env_logits_env)
            env_logits_causal_frozen = self.frozen_linear_logits(self.env_classifier, z_causal_tr)
            env_logits_res_frozen = self.frozen_linear_logits(self.env_classifier, z_residual_tr)
            loss_env_causal = self.compute_env_uniform_loss(env_logits_causal_frozen)
            loss_res_env = self.compute_env_uniform_loss(env_logits_res_frozen)
        else:
            env_logits_env = None
            env_probs = None
            loss_spu = self.compute_uniform_loss(self.frozen_linear_logits(self.classifier, z_env_tr))
            loss_env_causal = self.split_logits.new_zeros(())
            loss_res_env = self.split_logits.new_zeros(())

        contexts, context_info = self.build_frontdoor_contexts(z_env_tr, env_probs, training=True)
        fd_logits_tr, fd_stack_tr = self.frontdoor_logits_from_contexts(z_causal_tr, contexts)
        final_logits_tr = self.blend_logits(causal_logits_tr, fd_logits_tr)

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_causal = self.compute_supervised_loss(causal_logits_tr, y_tr, criterion, args).mean()
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()

        residual_label_logits = self.frozen_linear_logits(self.classifier, z_residual_tr)
        loss_res_label = self.compute_uniform_loss(residual_label_logits)
        loss_residual = loss_res_label + loss_res_env

        loss_orth = self.compute_trisplit_orth_loss(z_causal_tr, z_env_tr, z_residual_tr)
        loss_gate_balance = self.gate_balance_loss(gates)
        loss_gate_entropy = self.gate_entropy_loss(gates)
        loss_var = self.compute_frontdoor_variance_loss(fd_stack_tr)
        loss_inv = self.compute_pseudo_env_invariance_loss(causal_logits_tr, env_probs)

        zero = self.split_logits.new_zeros(())
        loss_fd_aug = zero
        loss_dag = zero
        loss_dag_label = zero
        loss_sem = zero
        loss_degree = zero
        loss_spu_y = zero

        total_loss = (
            loss_cls
            + self.lambda_causal * loss_causal
            + self.lambda_fd * loss_fd
            + self.lambda_spu * loss_spu
            + self.lambda_env * loss_env_causal
            + self.lambda_residual * loss_residual
            + self.lambda_orth * loss_orth
            + self.lambda_gate_balance * loss_gate_balance
            + self.lambda_gate_entropy * loss_gate_entropy
            + self.lambda_var * loss_var
            + self.lambda_inv * loss_inv
        )

        state_payload = None
        if update_state:
            state_payload = {
                'env_tr': z_env_tr.detach(),
                'env_probs_tr': env_probs.detach() if env_probs is not None else None,
            }

        num_contexts = 0 if contexts is None else int(contexts.size(0))
        num_proto_contexts = int(context_info.get('proto', 0))
        num_gmm_contexts = int(context_info.get('gmm', 0))

        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_causal': loss_causal,
            'loss_med': loss_causal,
            'loss_fd': loss_fd,
            'loss_fd_aug': loss_fd_aug,
            'loss_var': loss_var,
            'loss_orth': loss_orth,
            'loss_ind': loss_orth,
            'loss_spu': loss_spu,
            'loss_env_causal': loss_env_causal,
            'loss_env_med': loss_env_causal,
            'loss_residual': loss_residual,
            'loss_gate_balance': loss_gate_balance,
            'loss_gate_entropy': loss_gate_entropy,
            'loss_inv': loss_inv,
            'loss_dag': loss_dag,
            'loss_dag_label': loss_dag_label,
            'loss_sem': loss_sem,
            'loss_degree': loss_degree,
            'loss_spu_y': loss_spu_y,
            'gate_causal_mean': gates[:, 0].mean().detach(),
            'gate_env_mean': gates[:, 1].mean().detach(),
            'gate_residual_mean': gates[:, 2].mean().detach(),
            'gate_entropy_value': loss_gate_entropy.detach(),
            'mediator_gate_mean': gates[:, 0].mean().detach(),
            'causal_score_mean': gates[:, 0].mean().detach(),
            'pollution_score_mean': gates[:, 1].mean().detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_proto_contexts': torch.tensor(float(num_proto_contexts), device=x.device),
            'num_gmm_contexts': torch.tensor(float(num_gmm_contexts), device=x.device),
            'num_mixed_contexts': torch.tensor(0.0, device=x.device),
            'state_payload': state_payload,
        }

    def loss_compute(self, data, criterion, args):
        losses = self.compute_losses(data, criterion, args, update_state=True)
        return (
            losses['total_loss'],
            losses['loss_cls'].item(),
            (self.lambda_orth * losses['loss_orth']).item(),
            0.0,
            (self.lambda_fd * losses['loss_fd']).item(),
        )
