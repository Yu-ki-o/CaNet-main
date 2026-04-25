import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
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


class GraphFrontDoorDAG(nn.Module):
    """
    Front-door graph model with a feature-only DAG.

    Main idea:
    1) Learn a DAG A over hidden feature dimensions only (not prototypes + labels).
    2) Use DAG-derived structural scores to construct a soft mediator mask M.
    3) Split node representations into mediator/spurious parts.
    4) Keep the front-door aggregation path by averaging predictions over
       environment-specific spurious contexts.

    Compared with the previous prototype-reconstruction DAG, this version:
    - removes prototype/label reconstruction from DAG learning,
    - uses the DAG directly as a structured prior for mediator discovery,
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

        # Feature-only DAG.
        self.A_feat = Parameter(torch.zeros(self.d, self.d))
        # Learnable base score per hidden dimension; DAG structure refines it.
        self.gate_base = Parameter(torch.zeros(self.d))

        # Front-door fusion: mediator + context -> intervened representation.
        self.fd_fuser = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
        )
        self.fd_norm = nn.LayerNorm(self.d)

        self.dropout = getattr(args, 'dropout', 0.0)
        self.gamma = getattr(args, 'gamma', 0.99)
        self.fd_blend = getattr(args, 'fd_blend', 0.5)
        self.fd_sample_k = max(0, int(getattr(args, 'K', 0)))
        self.context_sample_seed = int(getattr(args, 'seed', 0))

        self.lambda_l1 = getattr(args, 'lambda_l1', 1e-5)
        self.lambda_dag = getattr(args, 'lambda_dag', 0.1)
        self.lambda_med = getattr(args, 'lambda_med', 0.5)
        self.lambda_spu = getattr(args, 'lambda_spu', 0.1)
        self.lambda_fd = getattr(args, 'lambda_fd', 0.5)
        self.lambda_var = getattr(args, 'lambda_var', 0.05)
        self.lambda_ind = getattr(args, 'lambda_ind', 0.1)
        self.lambda_env = getattr(args, 'lambda_env', 0.1)
        self.lambda_inv = getattr(args, 'lambda_inv', 0.1)
        self.lambda_gate = getattr(args, 'lambda_gate', 0.0)

        self.mediator_temp = getattr(args, 'mediator_temp', 8.0)
        self.mediator_threshold = getattr(args, 'mediator_threshold', 0.5)
        self.low_temp = getattr(args, 'low_temp', 8.0)
        self.low_threshold = getattr(args, 'low_threshold', 0.35)
        self.pollution_coeff = getattr(args, 'pollution_coeff', 1.0)

        # Only keep env-level spurious prototypes for front-door contexts.
        self.register_buffer('proto_spu_env', torch.zeros(self.num_envs, self.d))
        self.register_buffer('proto_spu_env_valid', torch.zeros(self.num_envs, dtype=torch.bool))

        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for layer in self.backbone_layers:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        self.fd_classifier.reset_parameters()
        self.env_classifier.reset_parameters()
        for module in self.fd_fuser:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.fd_norm.reset_parameters()
        nn.init.uniform_(self.A_feat, -0.01, 0.01)
        nn.init.zeros_(self.gate_base)
        self.proto_spu_env.zero_()
        self.proto_spu_env_valid.zero_()

    def get_masked_A(self):
        return self.A_feat - torch.diag(torch.diag(self.A_feat))

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
        Use only the feature DAG to produce mediator scores.

        - reach_score: how strongly each hidden dimension participates in the
          learned DAG's total structural flow.
        - pollution_score: penalize dimensions strongly coupled with low-score
          dimensions.
        - gate_base: lets the task losses directly refine mediator selection,
          while the DAG acts as a structured prior.
        """
        A = self.get_masked_A()
        A_sq = A * A
        C_tot = torch.matrix_exp(A_sq)

        feature_flow = C_tot - torch.diag(torch.diag(C_tot))
        reach_score = self._normalize_score(feature_flow.mean(dim=1), default_value=0.5)

        symmetric_flow = 0.5 * (feature_flow + feature_flow.t())
        low_weight = torch.sigmoid(self.low_temp * (self.low_threshold - reach_score))
        low_weight = low_weight / low_weight.sum().clamp_min(1e-8)
        pollution_score = torch.matmul(symmetric_flow, low_weight)
        pollution_score = self._normalize_score(pollution_score, default_value=0.0)

        base_score = torch.sigmoid(self.gate_base)
        causal_score = self._normalize_score(base_score + reach_score, default_value=0.5)
        mediator_logit = causal_score - self.pollution_coeff * pollution_score - self.mediator_threshold
        mediator_gate = torch.sigmoid(self.mediator_temp * mediator_logit)
        return causal_score, pollution_score, mediator_gate, C_tot

    def encode_representation(self, x, edge_index, training=False):
        x = F.dropout(x, self.dropout, training=training)
        z = self.act_fn(self.input_proj(x))
        for layer in self.backbone_layers:
            z = F.dropout(z, self.dropout, training=training)
            z = self.act_fn(layer(z, edge_index))

        causal_score, pollution_score, mediator_gate, dag_total = self.get_causal_effect_and_mask()
        z_mediator = F.dropout(z * mediator_gate.unsqueeze(0), self.dropout, training=training)
        z_spurious = F.dropout(z * (1.0 - mediator_gate).unsqueeze(0), self.dropout, training=training)
        mediator_logits = self.classifier(z_mediator)
        return (
            z,
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

    def frontdoor_logits_from_contexts(self, z_mediator, contexts):
        base_logits = self.fd_classifier(z_mediator)
        if contexts is None or contexts.size(0) == 0:
            return base_logits, None

        num_contexts = contexts.size(0)
        mediator_expand = z_mediator.unsqueeze(1).expand(-1, num_contexts, -1)
        context_expand = contexts.unsqueeze(0).expand(z_mediator.size(0), -1, -1)
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
        fd_logits, fd_stack = self.frontdoor_logits_from_contexts(z_mediator, contexts)
        logits = self.blend_logits(mediator_logits, fd_logits)

        if training:
            return (
                logits,
                z,
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
        h_A = torch.trace(torch.matrix_exp(A_sq)) - self.d
        h_A_clipped = torch.clamp(h_A, min=-10.0, max=10.0)
        loss_dag = 0.5 * (h_A_clipped ** 2) + self.lambda_l1 * torch.norm(A, p=1)

        # Optional soft sparsity on mediator gate to avoid trivial all-ones masks.
        if self.lambda_gate > 0.0:
            loss_dag = loss_dag + self.lambda_gate * mediator_gate.mean()

        # Mild consistency term: high-flow nodes should align with selected mediators.
        flow_score = self._normalize_score((dag_total - torch.diag(torch.diag(dag_total))).mean(dim=1), default_value=0.5)
        loss_dag = loss_dag + 0.1 * F.mse_loss(mediator_gate, flow_score.detach())
        return loss_dag

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
        fd_logits_tr, fd_stack_tr = self.frontdoor_logits_from_contexts(med_tr, contexts)
        final_logits_tr = self.blend_logits(mediator_logits_tr, fd_logits_tr)

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_med = self.compute_supervised_loss(mediator_logits_tr, y_tr, criterion, args).mean()
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        loss_var = self.compute_frontdoor_variance_loss(fd_stack_tr)
        loss_ind = self.compute_independence_loss(med_tr, spu_tr)
        loss_dag = self.dag_regularization_loss(mediator_gate, dag_total)

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
            + self.lambda_var * loss_var
            + self.lambda_ind * loss_ind
            + self.lambda_dag * loss_dag
            + self.lambda_spu * loss_spu
            + self.lambda_env * loss_env_med
            + self.lambda_inv * loss_inv
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
            'loss_var': loss_var,
            'loss_ind': loss_ind,
            'loss_dag': loss_dag,
            'loss_spu': loss_spu,
            'loss_env_med': loss_env_med,
            'loss_inv': loss_inv,
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'causal_score_mean': causal_score.mean().detach(),
            'pollution_score_mean': pollution_score.mean().detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
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
