import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, softmax
from torch_sparse import SparseTensor, matmul


def gcn_backbone_conv(x, edge_index):
    num_nodes = x.size(0)
    row, col = edge_index
    deg = degree(col, num_nodes).to(device=x.device, dtype=x.dtype).clamp_min(1.0)
    deg_in = deg[col].pow(-0.5)
    deg_out = deg[row].pow(-0.5)
    value = torch.nan_to_num(deg_in * deg_out, nan=0.0, posinf=0.0, neginf=0.0)
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


class FrontDoorLatentMixer(nn.Module):
    """
    Token mixer for M + C front-door prediction.

    Token order is [mediator, spurious, context, label_query].  The label token
    sees mediator and context, while spurious information can only pass through
    the context token.
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
        self.attn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(self.num_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)])

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
        attn_mask = tokens.new_zeros(blocked.shape).masked_fill(blocked, -1e9)

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


class GraphCFAMCIPTNeGo(nn.Module):
    """
    Clean main line:
    backbone -> Graph-CFAM node enhancement -> CIPT dual adapters ->
    NeGo/context construction -> front-door aggregation.
    """

    def __init__(self, d_in, c, args, device):
        super().__init__()
        self.device = device
        self.d = int(args.hidden_channels)
        self.c = int(c)
        self.num_layers = max(1, int(getattr(args, 'num_layers', 2)))
        self.num_envs = max(1, int(getattr(args, 'train_env_num', 1)))
        self.backbone_type = getattr(args, 'backbone_type', 'gcn')
        self.variant = bool(getattr(args, 'variant', False))
        self.dropout = float(getattr(args, 'dropout', 0.0))
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

        self.edge_feat_mode = getattr(args, 'edge_feat_mode', 'mul')
        self.edge_gate_mode = getattr(args, 'edge_gate_mode', 'vector')
        if self.edge_gate_mode not in ('scalar', 'vector'):
            self.edge_gate_mode = 'vector'
        self.edge_score_temp = max(1e-3, float(getattr(args, 'edge_score_temp', 2.0)))
        self.edge_blend = max(0.0, float(getattr(args, 'edge_blend', 0.2)))
        edge_feat_dim = self._get_edge_feat_dim(self.edge_feat_mode)
        edge_gate_out_dim = 1 if self.edge_gate_mode == 'scalar' else self.d
        self.edge_pair_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.edge_score_head = nn.Linear(self.d, edge_gate_out_dim)
        self.edge_summary_norm = nn.LayerNorm(self.d)
        self.noise_summary_norm = nn.LayerNorm(self.d)

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
        self.env_classifier = nn.Linear(self.d, self.num_envs)
        self.fd_fuser = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.fd_norm = nn.LayerNorm(self.d)
        self.use_dag_mixer = bool(getattr(args, 'use_dag_mixer', True))
        self.frontdoor_mixer = FrontDoorLatentMixer(
            self.d,
            num_heads=getattr(args, 'dag_mixer_heads', 1),
            num_layers=getattr(args, 'dag_mixer_layers', 2),
            dropout=self.dropout,
        )

        self.fd_blend = float(getattr(args, 'fd_blend', 0.5))
        self.eval_pred_mode = getattr(args, 'eval_pred_mode', 'blend')
        if self.eval_pred_mode not in ('blend', 'mediator', 'frontdoor'):
            self.eval_pred_mode = 'blend'
        self.fd_sample_k = max(0, int(getattr(args, 'K', 0)))
        self.context_sample_seed = int(getattr(args, 'seed', 0))
        self.use_env_contexts = bool(getattr(args, 'use_env_contexts', True))
        self.env_context_weight = max(0.0, float(getattr(args, 'env_context_weight', 1.0)))
        self.env_context_momentum = min(max(float(getattr(args, 'env_context_momentum', 0.9)), 0.0), 1.0)
        self.env_context_detach = bool(getattr(args, 'env_context_detach', False))

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
        if self.fd_context_source not in ('mixed', 'env_only', 'nego_only'):
            self.fd_context_source = 'mixed'
        self.nego_prompts = Parameter(torch.zeros(self.c, self.d))
        self.nego_prompt_decoder = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Dropout(p=adapter_dropout),
            nn.Linear(self.d, self.d),
        )
        self.nego_prompt_norm = nn.LayerNorm(self.d)

        self.lambda_fd = float(getattr(args, 'lambda_fd', 0.5))
        self.register_buffer('env_context_bank', torch.zeros(self.num_envs, self.d))
        self.register_buffer('env_context_valid', torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer('nego_context_bank', torch.zeros(self.c, self.d))
        self.register_buffer('nego_context_valid', torch.zeros(self.c, dtype=torch.bool))
        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for layer in self.backbone_layers:
            layer.reset_parameters()
        self._reset_module_parameters(self.edge_pair_encoder)
        self.edge_score_head.reset_parameters()
        self.edge_summary_norm.reset_parameters()
        self.noise_summary_norm.reset_parameters()
        self._reset_module_parameters(self.graph_cfam_gate)
        nn.init.zeros_(self.graph_cfam_gate[-1].weight)
        nn.init.zeros_(self.graph_cfam_gate[-1].bias)
        self.graph_cfam_norm.reset_parameters()
        self._reset_module_parameters(self.causal_adapter)
        self._reset_module_parameters(self.spurious_adapter)
        self.causal_norm.reset_parameters()
        self.spurious_norm.reset_parameters()
        self.classifier.reset_parameters()
        self.fd_classifier.reset_parameters()
        self.env_classifier.reset_parameters()
        self._reset_module_parameters(self.fd_fuser)
        self.fd_norm.reset_parameters()
        self.frontdoor_mixer.reset_parameters()
        nn.init.normal_(self.nego_prompts, mean=0.0, std=0.02)
        self._reset_module_parameters(self.nego_prompt_decoder)
        self.nego_prompt_norm.reset_parameters()
        self.env_context_bank.zero_()
        self.env_context_valid.zero_()
        self.nego_context_bank.zero_()
        self.nego_context_valid.zero_()
        self._last_graph_cfam_gate_loss = None
        self._last_graph_cfam_gate_mean = None

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

    def _degree_pair_feature(self, deg_src, deg_dst, deg_max):
        log_deg_src = torch.log1p(deg_src)
        log_deg_dst = torch.log1p(deg_dst)
        return (torch.maximum(log_deg_src, log_deg_dst) / deg_max.clamp_min(1.0)).unsqueeze(-1)

    def build_edge_feat(self, h_src, h_dst, deg_src, deg_dst, deg_max):
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
        edge_logits = self.edge_score_head(edge_hidden) / self.edge_score_temp
        edge_gate = torch.sigmoid(edge_logits)
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

    def _graph_cfam_energy(self, value):
        energy = value.pow(2)
        denom = energy.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return energy / denom

    def graph_cfam_adapt(self, h, edge_index, training=False):
        smooth, noise_summary, edge_gate = self.compute_edge_summaries(h, edge_index, training=training)
        residual = h - smooth
        smooth_energy = self._graph_cfam_energy(smooth)
        residual_energy = self._graph_cfam_energy(residual)
        gate_input = torch.cat([h, smooth, residual, smooth_energy, residual_energy], dim=-1)
        gate = torch.sigmoid(self.graph_cfam_gate(gate_input) / self.graph_cfam_gate_temp)

        causal_local = gate * smooth
        domain_local = (1.0 - gate) * smooth + noise_summary
        adapted = h + self.edge_blend * causal_local + self.graph_cfam_residual_blend * residual
        adapted = F.dropout(adapted, self.dropout, training=training)
        adapted = self.graph_cfam_norm(adapted)
        gate_loss = (gate.mean() - self.graph_cfam_gate_target).pow(2)
        self._last_graph_cfam_gate_loss = gate_loss
        self._last_graph_cfam_gate_mean = gate.mean().detach()
        return adapted, causal_local, domain_local, gate, edge_gate, gate_loss

    def encode_representation(self, x, edge_index, training=False):
        h = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(self.input_proj(h))
        for layer in self.backbone_layers:
            h = F.dropout(h, self.dropout, training=training)
            h = self.act_fn(layer(h, edge_index))

        z, causal_local, domain_local, cfam_gate, edge_gate, cfam_gate_loss = self.graph_cfam_adapt(
            h,
            edge_index,
            training=training,
        )
        causal_seed = z + causal_local
        spurious_seed = z + domain_local
        z_mediator = self.causal_norm(causal_seed + self.causal_adapter(causal_seed))
        z_spurious = self.spurious_norm(spurious_seed + self.spurious_adapter(spurious_seed))
        z_mediator = F.dropout(z_mediator, self.dropout, training=training)
        z_spurious = F.dropout(z_spurious, self.dropout, training=training)
        mediator_logits = self.classifier(z_mediator)
        env_logits = self.env_classifier(z_spurious)
        return {
            'z': z,
            'z_mediator': z_mediator,
            'z_spurious': z_spurious,
            'mediator_logits': mediator_logits,
            'env_logits': env_logits,
            'cfam_gate': cfam_gate,
            'edge_gate': edge_gate,
            'loss_graph_cfam_gate': cfam_gate_loss,
        }

    def compute_pseudo_env_probs(self, z_spurious):
        if self.num_envs <= 1 or z_spurious.numel() == 0:
            return z_spurious.new_ones(z_spurious.size(0), 1)
        return F.softmax(self.env_classifier(z_spurious), dim=-1)

    def get_env_contexts(self, z_spurious=None, env_probs=None, training=False):
        if not self.use_env_contexts or self.env_context_weight <= 0.0:
            return None

        if z_spurious is None or z_spurious.numel() == 0:
            valid = self.env_context_valid
            if valid.any():
                return self.env_context_weight * F.normalize(self.env_context_bank[valid], dim=1)
            return None

        values = z_spurious.detach() if self.env_context_detach else z_spurious
        if env_probs is None:
            env_probs = self.compute_pseudo_env_probs(z_spurious)
        if self.env_context_detach:
            env_probs = env_probs.detach()
        env_probs = env_probs.clamp_min(0.0)

        contexts = []
        for env_idx in range(env_probs.size(1)):
            weight = env_probs[:, env_idx]
            mass = weight.sum()
            if mass > 1e-6:
                context_vec = (weight.unsqueeze(-1) * values).sum(dim=0) / mass.clamp_min(1e-6)
                contexts.append(F.normalize(context_vec, dim=0))

        if not contexts:
            return None
        contexts = torch.stack(contexts, dim=0)
        return self.env_context_weight * contexts

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

    def _flat_class_labels(self, y):
        if y.dim() > 1 and y.size(1) > 1:
            return y.argmax(dim=1).long()
        return y.squeeze().long()

    def get_nego_contexts(self, z_source=None, y=None, sample_idx=None, training=False):
        if not self.use_nego_context or self.nego_context_weight <= 0.0:
            return None

        contexts = None
        if training and z_source is not None and y is not None and sample_idx is not None and sample_idx.numel() > 0:
            labels = self._flat_class_labels(y)[sample_idx]
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

        if contexts is None:
            valid = self.nego_context_valid
            if valid.any():
                contexts = self.nego_context_bank[valid]
            else:
                contexts = F.normalize(self.nego_prompts, dim=1)

        if contexts is None or contexts.numel() == 0:
            return None
        return self.nego_context_weight * F.normalize(contexts, dim=1)

    def _normalize_score(self, values, default_value=0.5):
        if values.numel() == 0:
            return values
        v_min = values.min()
        v_max = values.max()
        if (v_max - v_min).abs() < 1e-4:
            return (values - values.detach()) + default_value
        return (values - v_min) / (v_max - v_min + 1e-8)

    def compute_nego_loss(self, z_source, y, train_idx):
        zero = self.nego_prompts.new_zeros(())
        if not self.use_nego_prompt or self.lambda_nego <= 0.0:
            return zero, zero, zero
        if z_source is None or z_source.numel() == 0 or train_idx is None or train_idx.numel() == 0:
            return zero, zero, zero

        labels = self._flat_class_labels(y)[train_idx]
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

    def compute_graph_delf_loss(
        self,
        z_mediator,
        z_shortcut,
        final_logits_train,
        y,
        train_idx,
        criterion,
        args,
    ):
        """
        Graph-DELF auxiliary decoupling.

        Hard/shortcut-heavy train nodes are pulled toward stable same-class
        mediator prototypes and pushed away from same-class shortcut prototypes.
        """
        zero = z_mediator.new_zeros(())
        if (
            self.lambda_graph_delf <= 0.0
            or train_idx is None
            or train_idx.numel() <= 1
            or z_mediator.numel() == 0
            or z_shortcut is None
            or z_shortcut.numel() == 0
            or self.graph_delf_top_frac <= 0.0
        ):
            return zero

        device = z_mediator.device
        train_idx = train_idx.to(device=device, dtype=torch.long)
        y = y.to(device)

        with torch.no_grad():
            raw_loss = self.compute_supervised_loss(
                final_logits_train,
                y[train_idx],
                criterion,
                args,
            )
            if raw_loss.dim() > 1:
                raw_loss = raw_loss.mean(dim=1)
            shortcut_energy = z_shortcut[train_idx].detach().norm(dim=1)
            hard_score = self._normalize_score(raw_loss.detach(), default_value=0.5)
            shortcut_score = self._normalize_score(shortcut_energy, default_value=0.5)
            ambiguous_score = hard_score + shortcut_score
            top_k = max(1, int(round(float(train_idx.numel()) * self.graph_delf_top_frac)))
            top_k = min(top_k, int(train_idx.numel()))
            ambiguous_pos = ambiguous_score.topk(top_k).indices
            ambiguous_mask = torch.zeros(train_idx.numel(), device=train_idx.device, dtype=torch.bool)
            ambiguous_mask[ambiguous_pos] = True

        labels_train = self._flat_class_labels(y)[train_idx]
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

    def merge_frontdoor_contexts(self, env_contexts=None, nego_contexts=None):
        if self.fd_context_source == 'env_only':
            contexts = [env_contexts]
        elif self.fd_context_source == 'nego_only':
            contexts = [nego_contexts]
        else:
            contexts = [env_contexts, nego_contexts]
        contexts = [ctx for ctx in contexts if ctx is not None and ctx.numel() > 0]
        if not contexts:
            return None
        return torch.cat(contexts, dim=0)

    def sample_frontdoor_contexts(self, contexts, training=False):
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

    def frontdoor_logits_from_contexts(self, z_mediator, z_spurious, contexts):
        base_logits = self.fd_classifier(z_mediator)
        if contexts is None or contexts.size(0) == 0:
            return base_logits, None

        num_contexts = contexts.size(0)
        mediator_expand = z_mediator.unsqueeze(1).expand(-1, num_contexts, -1)
        spurious_expand = z_spurious.unsqueeze(1).expand(-1, num_contexts, -1)
        context_expand = contexts.unsqueeze(0).expand(z_mediator.size(0), -1, -1)

        if self.use_dag_mixer:
            fused = self.frontdoor_mixer(
                mediator_expand.reshape(-1, self.d),
                spurious_expand.reshape(-1, self.d),
                context_expand.reshape(-1, self.d),
            ).view(z_mediator.size(0), num_contexts, self.d)
        else:
            fused_input = torch.cat([mediator_expand, context_expand], dim=-1)
            fused = self.fd_fuser(fused_input.reshape(-1, self.d * 2)).view(z_mediator.size(0), num_contexts, self.d)
        fused = self.fd_norm(fused + mediator_expand)
        logits_stack = self.fd_classifier(fused.reshape(-1, self.d)).view(z_mediator.size(0), num_contexts, self.c)
        return logits_stack.mean(dim=1), logits_stack

    def blend_logits(self, mediator_logits, fd_logits):
        if fd_logits is None:
            return mediator_logits
        return (1.0 - self.fd_blend) * mediator_logits + self.fd_blend * fd_logits

    def forward(self, x, edge_index, training=False):
        enc = self.encode_representation(x, edge_index, training=training)
        env_probs = self.compute_pseudo_env_probs(enc['z_spurious'])
        env_contexts = self.get_env_contexts(enc['z_spurious'], env_probs, training=training)
        nego_contexts = self.get_nego_contexts(training=False)
        contexts = self.sample_frontdoor_contexts(
            self.merge_frontdoor_contexts(env_contexts, nego_contexts),
            training=training,
        )
        fd_logits, fd_stack = self.frontdoor_logits_from_contexts(
            enc['z_mediator'],
            enc['z_spurious'],
            contexts,
        )
        logits = self.blend_logits(enc['mediator_logits'], fd_logits)

        if training:
            enc.update({
                'logits': logits,
                'fd_logits': fd_logits,
                'fd_stack': fd_stack,
                'env_contexts': env_contexts,
                'nego_contexts': nego_contexts,
                'num_contexts': 0 if contexts is None else int(contexts.size(0)),
            })
            return enc
        if self.eval_pred_mode == 'mediator':
            return enc['mediator_logits']
        if self.eval_pred_mode == 'frontdoor':
            return fd_logits
        return logits

    def compute_supervised_loss(self, logits, y, criterion, args):
        if getattr(args, 'dataset', None) in ('twitch', 'elliptic'):
            if y.dim() == 1 or y.size(-1) == 1:
                labels = y.view(-1).long()
                if logits.size(1) > 1:
                    target = F.one_hot(labels, logits.size(1)).to(dtype=logits.dtype)
                else:
                    target = labels.to(dtype=logits.dtype).view(-1, 1)
            else:
                target = y.to(dtype=logits.dtype)
            loss = criterion(logits, target)
            if loss.dim() > 1:
                loss = loss.mean(dim=1)
            return loss
        return criterion(logits, y.squeeze(-1).long())

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y.to(data.x.device)
        train_idx = data.train_idx.to(device=x.device, dtype=torch.long)
        enc = self.encode_representation(x, edge_index, training=True)

        med_tr = enc['z_mediator'][train_idx]
        spu_tr = enc['z_spurious'][train_idx]
        y_tr = y[train_idx]
        mediator_logits_tr = enc['mediator_logits'][train_idx]

        env_probs_tr = self.compute_pseudo_env_probs(spu_tr)
        env_contexts = self.get_env_contexts(spu_tr, env_probs_tr, training=True)
        nego_source_all = self.get_nego_source_representation(
            enc['z'],
            enc['z_mediator'],
            enc['z_spurious'],
        )
        nego_contexts = self.get_nego_contexts(
            nego_source_all,
            y,
            train_idx,
            training=True,
        )
        contexts = self.sample_frontdoor_contexts(
            self.merge_frontdoor_contexts(env_contexts, nego_contexts),
            training=True,
        )
        fd_logits_tr, fd_stack_tr = self.frontdoor_logits_from_contexts(med_tr, spu_tr, contexts)
        final_logits_tr = self.blend_logits(mediator_logits_tr, fd_logits_tr)

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        loss_graph_cfam_gate = enc['loss_graph_cfam_gate']
        graph_cfam_gate_mean = enc['cfam_gate'].mean().detach()
        loss_graph_delf = self.compute_graph_delf_loss(
            enc['z_mediator'],
            enc['z_spurious'],
            final_logits_tr,
            y,
            train_idx,
            criterion,
            args,
        )
        loss_nego, nego_extra_score, nego_self_score = self.compute_nego_loss(
            nego_source_all,
            y,
            train_idx,
        )
        total_loss = (
            loss_cls
            + self.lambda_fd * loss_fd
            + self.lambda_graph_cfam_gate * loss_graph_cfam_gate
            + self.lambda_graph_delf * loss_graph_delf
            + self.lambda_nego * loss_nego
        )

        state_payload = None
        if update_state:
            state_payload = {
                'env_contexts': env_contexts.detach() if env_contexts is not None else None,
                'nego_contexts': nego_contexts.detach() if nego_contexts is not None else None,
            }

        zero = total_loss.new_zeros(())
        num_contexts = 0 if contexts is None else int(contexts.size(0))
        num_env_contexts = 0 if env_contexts is None else int(env_contexts.size(0))
        num_nego_contexts = 0 if nego_contexts is None else int(nego_contexts.size(0))
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_fd': loss_fd,
            'loss_graph_cfam_gate': loss_graph_cfam_gate,
            'loss_graph_delf': loss_graph_delf,
            'loss_nego': loss_nego,
            'loss_med': zero,
            'loss_cf': zero,
            'loss_ind': zero,
            'loss_dag': zero,
            'loss_dag_label': zero,
            'loss_spu': zero,
            'loss_env_med': zero,
            'nego_extra_score': nego_extra_score,
            'nego_self_score': nego_self_score,
            'graph_cfam_gate_mean': graph_cfam_gate_mean,
            'edge_gate_mean': zero if enc['edge_gate'] is None else enc['edge_gate'].mean().detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_env_contexts': torch.tensor(float(num_env_contexts), device=x.device),
            'num_nego_contexts': torch.tensor(float(num_nego_contexts), device=x.device),
            'state_payload': state_payload,
            'fd_stack': fd_stack_tr,
        }

    @torch.no_grad()
    def update_env_context_bank(self, contexts):
        if contexts is None or contexts.numel() == 0:
            return
        contexts = F.normalize(contexts.detach(), dim=1)
        take = min(contexts.size(0), self.num_envs)
        if take <= 0:
            return
        old = self.env_context_bank[:take]
        valid = self.env_context_valid[:take]
        blended = torch.where(
            valid.unsqueeze(-1),
            self.env_context_momentum * old + (1.0 - self.env_context_momentum) * contexts[:take],
            contexts[:take],
        )
        self.env_context_bank[:take] = F.normalize(blended, dim=1)
        self.env_context_valid[:take] = True

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

    @torch.no_grad()
    def apply_state_update(self, state_payload):
        if state_payload is None:
            return
        self.update_env_context_bank(state_payload.get('env_contexts'))
        self.update_nego_context_bank(state_payload.get('nego_contexts'))

    def loss_compute(self, data, criterion, args):
        losses = self.compute_losses(data, criterion, args, update_state=True)
        return (
            losses['total_loss'],
            losses['loss_cls'].item(),
            (self.lambda_fd * losses['loss_fd']).item(),
            (self.lambda_graph_cfam_gate * losses['loss_graph_cfam_gate']).item(),
            (self.lambda_nego * losses['loss_nego']).item(),
        )


GraphFrontDoorDAG = GraphCFAMCIPTNeGo
