import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_sparse import SparseTensor, matmul


def gcn_backbone_conv(x, edge_index):
    """
    CaNet-style GCN propagation used as the shared graph encoder backbone.
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
    Single shared graph encoder layer.

    This mirrors the CaNet backbone choices so the paper-style front-door
    model can reuse the same graph encoder family under OOD evaluation.
    """

    def __init__(self, in_features, out_features, backbone_type='gcn', residual=True, variant=False):
        super().__init__()
        self.backbone_type = backbone_type
        self.out_features = out_features
        self.residual = residual
        self.variant = variant

        if backbone_type == 'gcn':
            self.weight = nn.Parameter(torch.FloatTensor(in_features * 2, out_features))
        elif backbone_type == 'gat':
            self.leakyrelu = nn.LeakyReLU()
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            self.att = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        else:
            raise NotImplementedError(
                f"Front-door backbone_type='{backbone_type}' is not implemented. "
                "Use 'gcn' or 'gat' to match the CaNet-style backbone."
            )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.backbone_type == 'gat':
            nn.init.xavier_uniform_(self.att.data, gain=1.414)

    def specialspmm(self, edge_index, values, size, h):
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=values, sparse_sizes=size)
        return matmul(adj, h)

    def forward(self, x, edge_index):
        if self.backbone_type == 'gcn':
            if self.variant: #不做标准的gcn归一化，目的是让度数较大的节点的影响力更强
                adj = torch.sparse_coo_tensor(
                    edge_index,
                    torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype),
                    size=(x.size(0), x.size(0)),
                )
                h_neigh = torch.sparse.mm(adj, x)
            else:
                h_neigh = gcn_backbone_conv(x, edge_index) #标准的GCN归一化
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


class GraphFrontDoor(nn.Module):
    """
    Paper-style non-DAG front-door model for graph OOD.

    Mapping the CIPT framework to graphs:
    1) a shared graph encoder extracts node representations;
    2) two lightweight adapters decouple them into causal and spurious views;
    3) the causal branch predicts labels while the spurious branch is pushed
       toward a uniform label distribution;
    4) diverse environment-specific spurious contexts are sampled and combined
       with the causal feature through an adaptive augmentation module to
       approximate the front-door intervention.
    """

    def __init__(self, d_in, c, args, device):
        super().__init__()
        self.device = device
        self.d = args.hidden_channels
        self.c = c
        self.num_envs = max(1, int(getattr(args, 'K', getattr(args, 'train_env_num', 1))))
        self.num_layers = max(1, int(getattr(args, 'num_layers', 2)))
        self.backbone_type = getattr(args, 'backbone_type', 'gcn')
        self.variant = getattr(args, 'variant', False)
        self.dropout = getattr(args, 'dropout', 0.0)
        self.gamma = getattr(args, 'gamma', 0.99)
        self.fd_blend = getattr(args, 'fd_blend', 0.5)
        self.fd_sample_k = max(0, int(getattr(args, 'K', 0)))
        self.context_sample_seed = int(getattr(args, 'seed', 0))
        self.context_gate_temp = float(getattr(args, 'context_gate_temp', 1.0))
        self.proto_aug_k = max(0, int(getattr(args, 'proto_aug_k', 0)))
        self.proto_mix_alpha = max(1e-3, float(getattr(args, 'proto_mix_alpha', 1.0)))
        self.use_spu_gmm = bool(getattr(args, 'use_spu_gmm', False))
        self.gmm_alpha = float(getattr(args, 'gmm_alpha', 0.0))
        self.gmm_sample_k = max(0, int(getattr(args, 'gmm_sample_k', 0)))
        self.gmm_min_var = max(1e-6, float(getattr(args, 'gmm_min_var', 1e-4)))
        self.gmm_max_std = max(0.0, float(getattr(args, 'gmm_max_std', 1.0)))

        # Dirichlet-Barycentric GMM virtual environment sampler.
        # The original sampler drew from one observed pseudo-environment at a time.
        # This variant first samples a Dirichlet mixture over valid environments,
        # then builds a barycentric Gaussian whose mean/variance stay inside the
        # observed spurious distribution hull. It is intended to create realistic
        # virtual environments rather than unconstrained off-manifold contexts.
        self.gmm_cap_by_fd_k = bool(getattr(args, 'gmm_cap_by_fd_k', False))
        self.virtual_dir_alpha = max(1e-3, float(getattr(args, 'virtual_dir_alpha', 0.5)))
        self.virtual_between_scale = max(0.0, float(getattr(args, 'virtual_between_scale', 0.15)))
        self.virtual_sample_temp = max(0.0, float(getattr(args, 'virtual_sample_temp', 0.35)))
        self.virtual_maha_max = float(getattr(args, 'virtual_maha_max', 4.0))
        self.virtual_eval_noise = bool(getattr(args, 'virtual_eval_noise', False))

        self.lambda_med = getattr(args, 'lambda_med', 0.5)
        self.lambda_spu = getattr(args, 'lambda_spu', 0.1)
        self.lambda_fd = getattr(args, 'lambda_fd', 0.5)
        self.lambda_fd_aug = getattr(args, 'lambda_fd_aug', 0.5)
        self.lambda_var = getattr(args, 'lambda_var', 0.05)
        self.lambda_ind = getattr(args, 'lambda_ind', 0.1)
        self.lambda_spu_env = getattr(args, 'lambda_spu_env', 0.05)
        self.lambda_env_causal = getattr(args, 'lambda_env_causal', 0.0)
        self.lambda_split_gate = getattr(args, 'lambda_split_gate', 0.01)
        self.gate_binary_weight = getattr(args, 'gate_binary_weight', 0.1)
        self.lambda_context_recon = getattr(args, 'lambda_context_recon', 0.1)
        self.ind_loss_type = getattr(args, 'ind_loss_type', 'mi')
        self.hsic_sigma = float(getattr(args, 'hsic_sigma', 0.0))
        self.hsic_max_samples = max(2, int(getattr(args, 'hsic_max_samples', 256)))
        self.lambda_bootstrap = getattr(args, 'lambda_bootstrap', 0.0)
        self.ttt_feat_drop = float(getattr(args, 'ttt_feat_drop', 0.1))
        self.ttt_edge_drop = float(getattr(args, 'ttt_edge_drop', 0.1))
        self.ttt_ema = float(getattr(args, 'ttt_ema', 0.99))
        self.ttt_reward_conf = float(getattr(args, 'ttt_reward_conf', 1.0))
        self.ttt_reward_consistency = float(getattr(args, 'ttt_reward_consistency', 0.5))
        self.ttt_policy_entropy = float(getattr(args, 'ttt_policy_entropy', 0.01))

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
        self.bootstrap_projector = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.bootstrap_predictor = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.rl_context_policy = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, 1),
        )
        # Paper-style causal decomposition: two lightweight adapters.
        self.causal_adapter = nn.Linear(self.d, self.d)
        self.spurious_adapter = nn.Linear(self.d, self.d)
        self.split_gate = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.causal_norm = nn.LayerNorm(self.d)
        self.spurious_norm = nn.LayerNorm(self.d)

        self.classifier = nn.Linear(self.d, c)
        self.spurious_classifier = nn.Linear(self.d, c)
        self.fd_classifier = nn.Linear(self.d, c)
        self.env_classifier = nn.Linear(self.d, self.num_envs)

        # Learn the observed causal-context composition, then reuse it for
        # front-door intervention with prototype/Dirichlet-Barycentric GMM contexts.
        self.context_film = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d * 2),
        )
        self.context_interaction = nn.Sequential(
            nn.Linear(self.d * 3, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.composer_norm = nn.LayerNorm(self.d)

        self.target_input_proj = copy.deepcopy(self.input_proj)
        self.target_backbone_layers = copy.deepcopy(self.backbone_layers)
        self.target_bootstrap_projector = copy.deepcopy(self.bootstrap_projector)
        self._set_target_requires_grad(False)

        self.register_buffer('proto_spu_env', torch.zeros(self.num_envs, self.d))
        self.register_buffer('proto_spu_env_valid', torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer('gmm_spu_mean', torch.zeros(self.num_envs, self.d))
        self.register_buffer('gmm_spu_var', torch.ones(self.num_envs, self.d))
        self.register_buffer('gmm_spu_valid', torch.zeros(self.num_envs, dtype=torch.bool))

        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for layer in self.backbone_layers:
            layer.reset_parameters()
        self.causal_adapter.reset_parameters()
        self.spurious_adapter.reset_parameters()
        self._reset_module_parameters(self.split_gate)
        nn.init.zeros_(self.split_gate[-1].weight)
        nn.init.zeros_(self.split_gate[-1].bias)
        self.causal_norm.reset_parameters()
        self.spurious_norm.reset_parameters()
        self.classifier.reset_parameters()
        self.spurious_classifier.reset_parameters()
        self.fd_classifier.reset_parameters()
        self.env_classifier.reset_parameters()
        self._reset_module_parameters(self.context_film)
        self._reset_module_parameters(self.context_interaction)
        nn.init.zeros_(self.context_film[-1].weight)
        nn.init.zeros_(self.context_film[-1].bias)
        nn.init.zeros_(self.context_interaction[-1].weight)
        nn.init.zeros_(self.context_interaction[-1].bias)
        self.composer_norm.reset_parameters()
        self.proto_spu_env.zero_()
        self.proto_spu_env_valid.zero_()
        self.gmm_spu_mean.zero_()
        self.gmm_spu_var.fill_(1.0)
        self.gmm_spu_valid.zero_()
        self._reset_module_parameters(self.bootstrap_projector)
        self._reset_module_parameters(self.bootstrap_predictor)
        self._reset_module_parameters(self.rl_context_policy)
        nn.init.zeros_(self.rl_context_policy[-1].weight)
        nn.init.zeros_(self.rl_context_policy[-1].bias)
        self.update_bootstrap_target(momentum=0.0)

    def _reset_module_parameters(self, module):
        for sub_module in module.modules():
            if hasattr(sub_module, 'reset_parameters'):
                sub_module.reset_parameters()

    def _set_target_requires_grad(self, requires_grad):
        for module in (
            self.target_input_proj,
            self.target_backbone_layers,
            self.target_bootstrap_projector,
        ):
            for param in module.parameters():
                param.requires_grad = requires_grad

    @torch.no_grad()
    def update_bootstrap_target(self, momentum=None):
        if momentum is None:
            momentum = self.ttt_ema
        online_modules = (
            self.input_proj,
            self.backbone_layers,
            self.bootstrap_projector,
        )
        target_modules = (
            self.target_input_proj,
            self.target_backbone_layers,
            self.target_bootstrap_projector,
        )
        for online_module, target_module in zip(online_modules, target_modules):
            for online_param, target_param in zip(online_module.parameters(), target_module.parameters()):
                target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)

    def encode_backbone(self, x, edge_index, training=False):
        return self.encode_backbone_with(
            self.input_proj,
            self.backbone_layers,
            x,
            edge_index,
            training=training,
        )

    def encode_target_backbone(self, x, edge_index):
        return self.encode_backbone_with(
            self.target_input_proj,
            self.target_backbone_layers,
            x,
            edge_index,
            training=False,
        )

    def encode_backbone_with(self, input_proj, backbone_layers, x, edge_index, training=False):
        x = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(input_proj(x))
        for layer in backbone_layers:
            h = F.dropout(h, self.dropout, training=training)
            h = self.act_fn(layer(h, edge_index))
        return h

    def decompose_representation(self, h, training=False):
        split_gate = torch.sigmoid(self.split_gate(h))
        causal_base = h + self.causal_adapter(h)
        spurious_base = h + self.spurious_adapter(h)
        z_causal = self.causal_norm(split_gate * causal_base)
        z_spurious = self.spurious_norm((1.0 - split_gate) * spurious_base)
        z_causal = F.dropout(z_causal, self.dropout, training=training)
        z_spurious = F.dropout(z_spurious, self.dropout, training=training)
        causal_logits = self.classifier(z_causal)
        spurious_logits = self.spurious_classifier(z_spurious)
        return z_causal, z_spurious, causal_logits, spurious_logits, split_gate

    def encode_representation(self, x, edge_index, training=False):
        h = self.encode_backbone(x, edge_index, training=training)
        z_causal, z_spurious, causal_logits, spurious_logits, split_gate = self.decompose_representation(
            h,
            training=training,
        )
        return h, z_causal, z_spurious, causal_logits, spurious_logits, split_gate

    def infer_pseudo_envs(self, z_spurious):
        if self.num_envs <= 1 or z_spurious.numel() == 0:
            env = torch.zeros(z_spurious.size(0), device=z_spurious.device, dtype=torch.long)
            logits = z_spurious.new_zeros(z_spurious.size(0), 1)
            probs = torch.ones_like(logits)
            return env, logits, probs

        logits = self.env_classifier(z_spurious)
        probs = F.softmax(logits, dim=-1)
        env = probs.argmax(dim=-1).detach()
        return env, logits, probs

    def compute_pseudo_env_loss(self, env_probs):
        if env_probs is None or env_probs.size(-1) <= 1:
            return self.classifier.weight.new_zeros(())
        env_probs = env_probs.clamp_min(1e-8)
        node_entropy = -(env_probs * env_probs.log()).sum(dim=-1).mean()
        mean_probs = env_probs.mean(dim=0).clamp_min(1e-8)
        balance = (mean_probs * (mean_probs * env_probs.size(-1)).log()).sum()
        return node_entropy + balance

    def update_spurious_env_prototypes(self, z_spurious, pseudo_envs=None, env_probs=None):
        if z_spurious is None or z_spurious.numel() == 0:
            return

        if env_probs is not None and env_probs.numel() > 0 and env_probs.size(-1) == self.num_envs:
            env_probs = env_probs.detach().clamp_min(0.0)
            for env_idx in range(self.num_envs):
                weights = env_probs[:, env_idx]
                mass = weights.sum()
                if mass <= 1e-8:
                    continue
                vec = (z_spurious * weights.unsqueeze(-1)).sum(dim=0).detach() / mass.clamp_min(1e-8)
                if self.proto_spu_env_valid[env_idx]:
                    vec = self.gamma * self.proto_spu_env[env_idx] + (1.0 - self.gamma) * vec
                self.proto_spu_env[env_idx] = F.normalize(vec, dim=0)
                self.proto_spu_env_valid[env_idx] = True
            return

        if pseudo_envs is None or pseudo_envs.numel() == 0:
            return
        env_values = pseudo_envs.squeeze().long()
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
    def update_spurious_gmm(self, z_spurious, pseudo_envs=None, env_probs=None):
        if not self.use_spu_gmm or z_spurious is None or z_spurious.numel() == 0:
            return

        if env_probs is not None and env_probs.numel() > 0 and env_probs.size(-1) == self.num_envs:
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
                var = var.clamp_min(self.gmm_min_var)
                if self.gmm_spu_valid[env_idx]:
                    mean = self.gamma * self.gmm_spu_mean[env_idx] + (1.0 - self.gamma) * mean
                    var = self.gamma * self.gmm_spu_var[env_idx] + (1.0 - self.gamma) * var
                self.gmm_spu_mean[env_idx] = mean
                self.gmm_spu_var[env_idx] = var.clamp_min(self.gmm_min_var)
                self.gmm_spu_valid[env_idx] = True
            return

        if pseudo_envs is None or pseudo_envs.numel() == 0:
            return
        env_values = pseudo_envs.squeeze().long()
        for env_idx in range(self.num_envs):
            mask_e = env_values == env_idx
            if not mask_e.any():
                continue
            z_env = z_spurious[mask_e].detach()
            mean = z_env.mean(dim=0)
            if z_env.size(0) > 1:
                var = z_env.var(dim=0, unbiased=False)
            else:
                var = self.gmm_spu_var[env_idx]
            var = var.clamp_min(self.gmm_min_var)
            if self.gmm_spu_valid[env_idx]:
                mean = self.gamma * self.gmm_spu_mean[env_idx] + (1.0 - self.gamma) * mean
                var = self.gamma * self.gmm_spu_var[env_idx] + (1.0 - self.gamma) * var
            self.gmm_spu_mean[env_idx] = mean
            self.gmm_spu_var[env_idx] = var.clamp_min(self.gmm_min_var)
            self.gmm_spu_valid[env_idx] = True

    @torch.no_grad()
    def apply_state_update(self, state_payload):
        if state_payload is None:
            return
        self.update_spurious_env_prototypes(
            state_payload['spu_tr'],
            state_payload.get('pseudo_env_tr'),
            state_payload.get('env_probs_tr'),
        )
        self.update_spurious_gmm(
            state_payload['spu_tr'],
            state_payload.get('pseudo_env_tr'),
            state_payload.get('env_probs_tr'),
        )

    def get_frontdoor_contexts(self, z_spurious=None, envs=None, env_probs=None):
        context_map = {}
        if self.proto_spu_env_valid.any():
            valid_envs = self.proto_spu_env_valid.nonzero(as_tuple=False).squeeze(-1).tolist()
            for env_idx in valid_envs:
                context_map[int(env_idx)] = self.proto_spu_env[env_idx].detach()

        used_soft_env = False
        if z_spurious is not None and env_probs is not None and env_probs.numel() > 0:
            if env_probs.size(-1) == self.num_envs:
                env_probs = env_probs.detach().clamp_min(0.0)
                for env_idx in range(self.num_envs):
                    weights = env_probs[:, env_idx]
                    mass = weights.sum()
                    if mass > 1e-8:
                        context_map[int(env_idx)] = (
                            z_spurious * weights.unsqueeze(-1)
                        ).sum(dim=0).detach() / mass.clamp_min(1e-8)
                        used_soft_env = True

        if not used_soft_env and z_spurious is not None and envs is not None and envs.numel() > 0:
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

    def sample_gmm_contexts(self, training=False):
        """Sample realistic virtual spurious contexts with a Dirichlet-Barycentric GMM.

        Each observed pseudo environment keeps an EMA diagonal Gaussian
        N(mu_e, sigma_e^2). Instead of drawing contexts only from one observed
        environment, we sample or construct mixture weights over the valid
        environments and create a virtual barycentric Gaussian:

            mu_v  = sum_e w_e * mu_e
            var_v = geometric_mean_e(var_e; w_e)
                    + virtual_between_scale * Var_w(mu_e)

        This produces extra contexts that lie in the distributional hull of the
        observed environments, which is usually safer than an unconstrained
        generative model when only a few environments are available.
        """
        if not self.use_spu_gmm or self.gmm_alpha <= 0.0 or self.gmm_sample_k <= 0:
            return None

        valid_envs = self.gmm_spu_valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_envs.numel() == 0:
            return None

        sample_k = self.gmm_sample_k
        # Old behavior capped GMM contexts by K/fd_sample_k. For virtual
        # environment expansion this cap is often too restrictive, so it is
        # disabled by default and can be re-enabled with --gmm_cap_by_fd_k.
        if self.gmm_cap_by_fd_k and self.fd_sample_k > 0:
            sample_k = min(sample_k, self.fd_sample_k)
        if sample_k <= 0:
            return None

        means = self.gmm_spu_mean.index_select(0, valid_envs)
        variances = self.gmm_spu_var.index_select(0, valid_envs).clamp_min(self.gmm_min_var)

        num_envs = means.size(0)
        device = means.device
        dtype = means.dtype

        if training:
            # Sparse Dirichlet weights (alpha < 1) create realistic new
            # environments near observed ones; larger alpha moves toward
            # smoother barycentric mixtures.
            alpha = torch.full(
                (num_envs,),
                self.virtual_dir_alpha,
                device=device,
                dtype=dtype,
            )
            weights = torch.distributions.Dirichlet(alpha).sample((sample_k,))
        else:
            # Deterministic evaluation: original environments, pairwise
            # barycenters, then the global barycenter. This avoids random test
            # noise while still evaluating expanded distributional contexts.
            candidates = [torch.eye(num_envs, device=device, dtype=dtype)]

            pair_weights = []
            for i in range(num_envs):
                for j in range(i + 1, num_envs):
                    weight = torch.zeros(num_envs, device=device, dtype=dtype)
                    weight[i] = 0.5
                    weight[j] = 0.5
                    pair_weights.append(weight)
            if pair_weights:
                candidates.append(torch.stack(pair_weights, dim=0))

            candidates.append(
                torch.full(
                    (1, num_envs),
                    1.0 / float(num_envs),
                    device=device,
                    dtype=dtype,
                )
            )

            weights = torch.cat(candidates, dim=0)
            repeat = (sample_k + weights.size(0) - 1) // weights.size(0)
            weights = weights.repeat(repeat, 1)[:sample_k]

        virtual_mean = weights @ means

        # Weighted geometric mean is more stable than an arithmetic average
        # when per-dimension variances differ greatly across environments.
        log_variances = variances.clamp_min(self.gmm_min_var).log()
        virtual_within_var = (weights @ log_variances).exp()

        # Add a controllable between-environment term so mixed environments can
        # cover plausible distribution shifts between observed environment
        # centers without exploding into off-manifold samples.
        centered_means = means.unsqueeze(0) - virtual_mean.unsqueeze(1)
        virtual_between_var = (
            weights.unsqueeze(-1) * centered_means.pow(2)
        ).sum(dim=1)

        virtual_var = (
            virtual_within_var
            + self.virtual_between_scale * virtual_between_var
        ).clamp_min(self.gmm_min_var)

        std = virtual_var.sqrt()
        if self.gmm_max_std > 0.0:
            std = std.clamp_max(self.gmm_max_std)

        if training or self.virtual_eval_noise:
            contexts = virtual_mean + self.virtual_sample_temp * torch.randn_like(virtual_mean) * std
        else:
            contexts = virtual_mean

        # Realism gate: if a sampled point is too far from every observed
        # environment Gaussian, fall back to its barycentric mean. The distance
        # is averaged over feature dimensions to keep the threshold independent
        # of hidden dimensionality. Set --virtual_maha_max <= 0 to disable.
        if self.virtual_maha_max > 0.0:
            maha = (
                (contexts.unsqueeze(1) - means.unsqueeze(0)).pow(2)
                / variances.unsqueeze(0).clamp_min(self.gmm_min_var)
            ).mean(dim=-1)
            nearest_maha = maha.min(dim=1).values
            valid_context = nearest_maha <= self.virtual_maha_max
            contexts = torch.where(valid_context.unsqueeze(-1), contexts, virtual_mean)

        return F.normalize(contexts, dim=1)

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

    def score_frontdoor_contexts(self, z_causal, contexts):
        num_contexts = contexts.size(0)
        causal_expand = z_causal.unsqueeze(1).expand(-1, num_contexts, -1)
        context_expand = contexts.unsqueeze(0).expand(z_causal.size(0), -1, -1)
        policy_input = torch.cat(
            [causal_expand, context_expand, causal_expand * context_expand],
            dim=-1,
        )
        return self.rl_context_policy(policy_input).squeeze(-1)

    def compose_causal_context(self, z_causal, z_context):
        film = self.context_film(z_context)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        interaction = self.context_interaction(
            torch.cat([z_causal, z_context, z_causal * z_context], dim=-1)
        )
        composed = z_causal + gamma * z_causal + beta + interaction
        return self.composer_norm(composed)

    def interventional_logits_from_contexts(
        self,
        z_causal,
        contexts,
        training=False,
        use_context_policy=False,
        return_context_weights=False,
    ):
        base_logits = self.fd_classifier(z_causal)
        if contexts is None or contexts.size(0) == 0:
            if return_context_weights:
                return base_logits, None, None
            return base_logits, None

        num_contexts = contexts.size(0)
        causal_expand = z_causal.unsqueeze(1).expand(-1, num_contexts, -1)
        context_expand = contexts.unsqueeze(0).expand(z_causal.size(0), -1, -1)

        aug = self.compose_causal_context(causal_expand, context_expand)
        aug = F.dropout(aug, self.dropout, training=training)

        logits_stack = self.fd_classifier(aug.reshape(-1, self.d)).view(z_causal.size(0), num_contexts, self.c)
        context_weights = None
        if use_context_policy:
            policy_scores = self.score_frontdoor_contexts(z_causal, contexts)
            context_weights = F.softmax(policy_scores, dim=-1)
            fd_logits = (context_weights.unsqueeze(-1) * logits_stack).sum(dim=1)
        else:
            fd_logits = logits_stack.mean(dim=1)
        if return_context_weights:
            return fd_logits, logits_stack, context_weights
        return fd_logits, logits_stack

    def blend_logits(self, causal_logits, fd_logits):
        if fd_logits is None:
            return causal_logits
        return (1.0 - self.fd_blend) * causal_logits + self.fd_blend * fd_logits

    def blend_gmm_logits(self, base_logits, gmm_logits):
        if gmm_logits is None or self.gmm_alpha <= 0.0:
            return base_logits
        alpha = min(1.0, max(0.0, self.gmm_alpha))
        return (1.0 - alpha) * base_logits + alpha * gmm_logits

    def forward(self, x, edge_index, training=False, use_context_policy=False):
        h, z_causal, z_spurious, causal_logits, spurious_logits, split_gate = self.encode_representation(
            x,
            edge_index,
            training=training,
        )

        contexts = self.sample_frontdoor_contexts(
            self.get_frontdoor_contexts(),
            training=training,
        )
        fd_logits, fd_stack = self.interventional_logits_from_contexts(
            z_causal,
            contexts,
            training=training,
            use_context_policy=use_context_policy,
        )
        logits = self.blend_logits(causal_logits, fd_logits)
        gmm_contexts = self.sample_gmm_contexts(training=training)
        gmm_fd_logits = None
        if gmm_contexts is not None:
            gmm_fd_logits, _ = self.interventional_logits_from_contexts(
                z_causal,
                gmm_contexts,
                training=training,
                use_context_policy=use_context_policy,
            )
            gmm_logits = self.blend_logits(causal_logits, gmm_fd_logits)
            logits = self.blend_gmm_logits(logits, gmm_logits)

        if training:
            return (
                logits,
                h,
                z_causal,
                z_spurious,
                causal_logits,
                spurious_logits,
                fd_logits,
                fd_stack,
                split_gate,
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

    def compute_uniform_spurious_loss(self, logits):
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

    def _subsample_for_dependence(self, z_causal, z_spurious):
        if z_causal.size(0) <= self.hsic_max_samples:
            return z_causal, z_spurious
        indices = torch.linspace(
            0,
            z_causal.size(0) - 1,
            steps=self.hsic_max_samples,
            device=z_causal.device,
        ).long()
        return z_causal.index_select(0, indices), z_spurious.index_select(0, indices)

    def _rbf_kernel(self, z):
        sq_dist = torch.cdist(z, z, p=2).pow(2)
        if self.hsic_sigma > 0.0:
            sigma2 = z.new_tensor(self.hsic_sigma ** 2).clamp_min(1e-8)
        else:
            positive = sq_dist.detach()[sq_dist.detach() > 0]
            if positive.numel() == 0:
                sigma2 = z.new_tensor(1.0)
            else:
                sigma2 = positive.median().clamp_min(1e-8)
        return torch.exp(-sq_dist / (2.0 * sigma2))

    def compute_hsic_loss(self, z_causal, z_spurious):
        z_causal, z_spurious = self._subsample_for_dependence(z_causal, z_spurious)
        n = z_causal.size(0)
        if n <= 1:
            return z_causal.new_zeros(())
        z_causal = F.normalize(z_causal, dim=1)
        z_spurious = F.normalize(z_spurious, dim=1)
        k_causal = self._rbf_kernel(z_causal)
        k_spurious = self._rbf_kernel(z_spurious)
        eye = torch.eye(n, device=z_causal.device, dtype=z_causal.dtype)
        ones = torch.full((n, n), 1.0 / n, device=z_causal.device, dtype=z_causal.dtype)
        center = eye - ones
        k_causal = center @ k_causal @ center
        k_spurious = center @ k_spurious @ center
        return (k_causal * k_spurious).sum() / ((n - 1) ** 2)

    def compute_corr_loss(self, z_causal, z_spurious):
        z_causal = (z_causal - z_causal.mean(dim=0, keepdim=True)) / z_causal.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
        z_spurious = (z_spurious - z_spurious.mean(dim=0, keepdim=True)) / z_spurious.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
        corr = torch.matmul(z_causal.t(), z_spurious) / max(1, z_causal.size(0) - 1)
        return corr.pow(2).mean()

    def compute_mi_loss(self, z_causal, z_spurious):
        z_causal, z_spurious = self._subsample_for_dependence(z_causal, z_spurious)
        n = z_causal.size(0)
        if n <= 2:
            return z_causal.new_zeros(())

        z_causal = z_causal - z_causal.mean(dim=0, keepdim=True)
        z_spurious = z_spurious - z_spurious.mean(dim=0, keepdim=True)
        z_causal = z_causal / z_causal.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
        z_spurious = z_spurious / z_spurious.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)

        cross_corr = torch.matmul(z_causal.t(), z_spurious) / max(1, n - 1)
        singular_values = torch.linalg.svdvals(cross_corr)
        rho2 = singular_values.pow(2).clamp(max=1.0 - 1e-4)
        mi = -0.5 * torch.log1p(-rho2).mean()
        return torch.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_independence_loss(self, z_causal, z_spurious):
        if z_causal.numel() == 0:
            return z_causal.new_zeros(())
        if self.ind_loss_type in ('mi', 'mutual_info', 'mutual_information'):
            return self.compute_mi_loss(z_causal, z_spurious)
        if self.ind_loss_type == 'hsic':
            return self.compute_hsic_loss(z_causal, z_spurious)
        if self.ind_loss_type == 'corr':
            return self.compute_corr_loss(z_causal, z_spurious)
        z_causal = F.normalize(z_causal, dim=1)
        z_spurious = F.normalize(z_spurious, dim=1)
        corr = (z_causal * z_spurious).sum(dim=1)
        return 0.5 * (corr ** 2).mean()

    def compute_frontdoor_variance_loss(self, logits_stack):
        if logits_stack is None or logits_stack.size(1) <= 1:
            return logits_stack.new_zeros(()) if logits_stack is not None else self.classifier.weight.new_zeros(())
        probs = torch.softmax(logits_stack, dim=-1)
        return probs.var(dim=1, unbiased=False).mean()

    def compute_split_gate_loss(self, split_gate):
        if split_gate is None or split_gate.numel() == 0:
            return self.classifier.weight.new_zeros(())
        channel_balance = (split_gate.mean(dim=0) - 0.5).pow(2).mean()
        binary_pressure = (split_gate * (1.0 - split_gate)).mean()
        return channel_balance + self.gate_binary_weight * binary_pressure

    def compute_context_reconstruction_loss(self, h, z_causal, z_spurious):
        if h.numel() == 0:
            return self.classifier.weight.new_zeros(())
        h_hat = self.compose_causal_context(z_causal, z_spurious)
        target = F.layer_norm(h.detach(), (h.size(-1),))
        return F.mse_loss(h_hat, target)

    def compute_context_supervised_loss(self, logits_stack, y, criterion, args):
        if logits_stack is None or logits_stack.size(1) == 0:
            return self.classifier.weight.new_zeros(())
        num_nodes, num_contexts, _ = logits_stack.shape
        logits_flat = logits_stack.reshape(num_nodes * num_contexts, self.c)
        y_flat = y.repeat_interleave(num_contexts, dim=0)
        return self.compute_supervised_loss(logits_flat, y_flat, criterion, args).mean()

    def augment_graph_view(self, x, edge_index, feat_drop=None, edge_drop=None):
        if feat_drop is None:
            feat_drop = self.ttt_feat_drop
        if edge_drop is None:
            edge_drop = self.ttt_edge_drop

        x_aug = x
        if feat_drop > 0.0:
            keep_prob = max(1e-4, 1.0 - feat_drop)
            mask = torch.empty_like(x).bernoulli_(keep_prob)
            x_aug = x * mask / keep_prob

        edge_aug = edge_index
        if edge_drop > 0.0 and edge_index.size(1) > 1:
            keep_mask = torch.rand(edge_index.size(1), device=edge_index.device) > edge_drop
            if keep_mask.any():
                edge_aug = edge_index[:, keep_mask]
        return x_aug, edge_aug

    def compute_bootstrap_loss(self, x, edge_index, idx=None):
        x_online, edge_online = self.augment_graph_view(x, edge_index)
        x_target, edge_target = self.augment_graph_view(x, edge_index)
        h_online = self.encode_backbone(x_online, edge_online, training=True)
        pred_online = self.bootstrap_predictor(self.bootstrap_projector(h_online))
        with torch.no_grad():
            h_target = self.encode_target_backbone(x_target, edge_target)
            target = self.target_bootstrap_projector(h_target).detach()
        if idx is not None:
            pred_online = pred_online[idx]
            target = target[idx]
        pred_online = F.normalize(pred_online, dim=-1)
        target = F.normalize(target, dim=-1)
        return 2.0 - 2.0 * (pred_online * target).sum(dim=-1).mean()

    def prediction_confidence(self, logits):
        if logits.size(-1) == 1:
            probs = torch.sigmoid(logits)
            return torch.maximum(probs, 1.0 - probs).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        return probs.max(dim=-1).values

    def rl_context_adaptation_loss(self, x, edge_index, adapt_idx):
        contexts = self.sample_frontdoor_contexts(
            self.get_frontdoor_contexts(),
            training=False,
        )
        if contexts is None or contexts.size(0) <= 1 or adapt_idx.numel() == 0:
            return self.classifier.weight.new_zeros(())

        _, z_causal, _, causal_logits, _, _ = self.encode_representation(
            x,
            edge_index,
            training=False,
        )
        z_causal = z_causal[adapt_idx].detach()
        causal_logits = causal_logits[adapt_idx].detach()

        fd_logits, logits_stack, context_weights = self.interventional_logits_from_contexts(
            z_causal,
            contexts,
            training=False,
            use_context_policy=True,
            return_context_weights=True,
        )
        final_logits = self.blend_logits(causal_logits, fd_logits)

        with torch.no_grad():
            conf_reward = self.prediction_confidence(logits_stack.reshape(-1, self.c)).view(
                z_causal.size(0),
                contexts.size(0),
            )

            x_aug, edge_aug = self.augment_graph_view(x, edge_index)
            _, z_aug, _, causal_aug, _, _ = self.encode_representation(
                x_aug,
                edge_aug,
                training=False,
            )
            fd_aug, _ = self.interventional_logits_from_contexts(
                z_aug[adapt_idx],
                contexts,
                training=False,
                use_context_policy=False,
            )
            aug_logits = self.blend_logits(causal_aug[adapt_idx], fd_aug)
            target_probs = F.softmax(aug_logits, dim=-1)
            context_probs = F.softmax(logits_stack, dim=-1)
            consistency_reward = -(context_probs * (
                context_probs.clamp_min(1e-8).log()
                - target_probs.unsqueeze(1).clamp_min(1e-8).log()
            )).sum(dim=-1)
            reward = (
                self.ttt_reward_conf * conf_reward
                + self.ttt_reward_consistency * consistency_reward
            )
            reward = reward - reward.mean(dim=1, keepdim=True)

        expected_reward = (context_weights * reward).sum(dim=-1).mean()
        entropy = -(context_weights * context_weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        confident_entropy = F.softmax(final_logits, dim=-1)
        confident_entropy = -(confident_entropy * confident_entropy.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return -expected_reward - self.ttt_policy_entropy * entropy + 0.01 * confident_entropy

    def adapt_test_time_policy(self, x, edge_index, adapt_idx, steps=1, lr=1e-3):
        if steps <= 0 or adapt_idx is None or adapt_idx.numel() == 0:
            return
        contexts = self.sample_frontdoor_contexts(
            self.get_frontdoor_contexts(),
            training=False,
        )
        if contexts is None or contexts.size(0) <= 1:
            return
        was_training = self.training
        self.train()
        params = list(self.rl_context_policy.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = self.rl_context_adaptation_loss(x, edge_index, adapt_idx)
            if not loss.requires_grad:
                break
            loss.backward()
            optimizer.step()
        if was_training:
            self.train()
        else:
            self.eval()

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx

        (
            _,
            h_all,
            z_causal_all,
            z_spurious_all,
            causal_logits_all,
            spurious_logits_all,
            _,
            _,
            split_gate_all,
        ) = self.forward(x, edge_index, training=True)

        y_tr = y[train_idx]
        h_tr = h_all[train_idx]
        causal_tr = z_causal_all[train_idx]
        spurious_tr = z_spurious_all[train_idx]
        split_gate_tr = split_gate_all[train_idx]
        causal_logits_tr = causal_logits_all[train_idx]
        spurious_logits_tr = spurious_logits_all[train_idx]
        pseudo_env_tr, env_logits_spurious, env_probs_spurious = self.infer_pseudo_envs(spurious_tr)

        contexts = self.sample_frontdoor_contexts(
            self.get_frontdoor_contexts(spurious_tr, pseudo_env_tr, env_probs_spurious),
            training=True,
        )
        num_base_contexts = 0 if contexts is None else int(contexts.size(0))
        contexts, num_mixed_contexts = self.augment_frontdoor_contexts(
            contexts,
            training=True,
        )
        fd_logits_tr, fd_stack_tr = self.interventional_logits_from_contexts(
            causal_tr,
            contexts,
            training=True,
        )
        final_logits_tr = self.blend_logits(causal_logits_tr, fd_logits_tr)
        gmm_contexts = self.sample_gmm_contexts(training=True)
        gmm_fd_logits_tr = None
        if gmm_contexts is not None:
            gmm_fd_logits_tr, _ = self.interventional_logits_from_contexts(
                causal_tr,
                gmm_contexts,
                training=True,
            )
            gmm_logits_tr = self.blend_logits(causal_logits_tr, gmm_fd_logits_tr)
            final_logits_tr = self.blend_gmm_logits(final_logits_tr, gmm_logits_tr)

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_med = self.compute_supervised_loss(causal_logits_tr, y_tr, criterion, args).mean()
        loss_spu = self.compute_uniform_spurious_loss(spurious_logits_tr)
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        fd_aug_stack_tr = None
        if fd_stack_tr is not None and num_mixed_contexts > 0:
            fd_aug_stack_tr = fd_stack_tr[:, num_base_contexts:, :]
        loss_fd_aug = self.compute_context_supervised_loss(fd_aug_stack_tr, y_tr, criterion, args)
        loss_var = self.compute_frontdoor_variance_loss(fd_stack_tr)
        loss_ind = self.compute_independence_loss(causal_tr, spurious_tr)
        loss_split_gate = self.compute_split_gate_loss(split_gate_tr)
        loss_context_recon = self.compute_context_reconstruction_loss(h_tr, causal_tr, spurious_tr)
        if self.lambda_bootstrap > 0.0:
            loss_bootstrap = self.compute_bootstrap_loss(x, edge_index, train_idx)
        else:
            loss_bootstrap = self.classifier.weight.new_zeros(())
        if pseudo_env_tr.numel() > 0 and self.num_envs > 1:
            env_logits_causal = F.linear(
                causal_tr,
                self.env_classifier.weight.detach(),
                self.env_classifier.bias.detach() if self.env_classifier.bias is not None else None,
            )
            loss_env_causal = self.compute_env_uniform_loss(env_logits_causal)
            loss_spu_env = self.compute_pseudo_env_loss(env_probs_spurious)
        else:
            loss_env_causal = self.classifier.weight.new_zeros(())
            loss_spu_env = self.classifier.weight.new_zeros(())

        total_loss = (
            loss_cls
            + self.lambda_med * loss_med
            + self.lambda_spu * loss_spu
            + self.lambda_fd * loss_fd
            + self.lambda_fd_aug * loss_fd_aug
            + self.lambda_var * loss_var
            + self.lambda_ind * loss_ind
            + self.lambda_env_causal * loss_env_causal
            + self.lambda_spu_env * loss_spu_env
            + self.lambda_split_gate * loss_split_gate
            + self.lambda_context_recon * loss_context_recon
            + self.lambda_bootstrap * loss_bootstrap
        )

        state_payload = None
        if update_state:
            state_payload = {
                'spu_tr': spurious_tr.detach(),
                'pseudo_env_tr': pseudo_env_tr.detach(),
                'env_probs_tr': env_probs_spurious.detach(),
            }

        num_contexts = 0 if contexts is None else int(contexts.size(0))
        num_gmm_contexts = 0 if gmm_contexts is None else int(gmm_contexts.size(0))
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_med': loss_med,
            'loss_spu': loss_spu,
            'loss_fd': loss_fd,
            'loss_fd_aug': loss_fd_aug,
            'loss_var': loss_var,
            'loss_ind': loss_ind,
            'loss_split_gate': loss_split_gate,
            'loss_context_recon': loss_context_recon,
            'loss_env_causal': loss_env_causal,
            'loss_spu_env': loss_spu_env,
            'loss_bootstrap': loss_bootstrap,
            'causal_norm_mean': causal_tr.norm(dim=1).mean().detach(),
            'spurious_norm_mean': spurious_tr.norm(dim=1).mean().detach(),
            'split_gate_mean': split_gate_tr.mean().detach(),
            'split_gate_std': split_gate_tr.std(unbiased=False).detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_mixed_contexts': torch.tensor(float(num_mixed_contexts), device=x.device),
            'num_gmm_contexts': torch.tensor(float(num_gmm_contexts), device=x.device),
            'state_payload': state_payload,
        }

    def loss_compute(self, data, criterion, args):
        losses = self.compute_losses(data, criterion, args, update_state=True)
        return (
            losses['total_loss'],
            losses['loss_cls'].item(),
            (self.lambda_ind * losses['loss_ind']).item(),
            (self.lambda_fd * losses['loss_fd']).item(),
        )
