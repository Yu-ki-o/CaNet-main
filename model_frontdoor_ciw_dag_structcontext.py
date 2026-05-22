"""
Graph Front-Door CIW-DAG + StableStruct Context.

This variant keeps the original DAG + front-door framework as the main model.
It only borrows the StableGNN idea of high-level neighborhood variables: each
node softly pools its neighbors into K structural tokens.  These tokens are
injected into the existing DAG structure slice, and their DAG pollution score is
used to build a structural spurious context for front-door context sampling.

No StableGNN HSIC/CVD/sample reweighting is used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_frontdoor_ciw_dag import GraphFrontDoorCIWDAG


class StableStructEncoder(nn.Module):
    """Node-centric high-level neighborhood structural tokens.

    For every directed edge u -> v, the assignment network softly assigns the
    neighbor u to one of K structural roles/tokens of the center node v.  The K
    token embeddings are weighted summaries of neighbors.  Unlike the previous
    compressed implementation, this module preserves all K tokens and returns a
    flattened [N, K * token_dim] structure variable so that the DAG can see the
    token groups separately.
    """

    def __init__(self, hidden_dim, struct_dim, num_tokens=4, dropout=0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.struct_dim = int(struct_dim)
        self.num_tokens = max(1, int(num_tokens))
        if self.struct_dim % self.num_tokens != 0:
            raise ValueError(
                f'struct_dim={self.struct_dim} must be divisible by num_struct_tokens={self.num_tokens}. '
                'Choose a token count that divides the DAG edge/structure slice width.'
            )
        self.token_dim = self.struct_dim // self.num_tokens

        pair_dim = 4 * self.hidden_dim + 2
        self.assign_mlp = nn.Sequential(
            nn.Linear(pair_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_tokens),
        )
        self.msg_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.token_dim),
            nn.ReLU(),
            nn.LayerNorm(self.token_dim),
        )
        self.flat_norm = nn.LayerNorm(self.struct_dim)

    def reset_parameters(self):
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, z, edge_index):
        if z.dim() != 2:
            raise ValueError(f'StableStructEncoder expects z with shape [N, H], got {tuple(z.shape)}')
        if z.size(1) != self.hidden_dim:
            raise ValueError(
                f'StableStructEncoder hidden_dim={self.hidden_dim}, but z has dim={z.size(1)}. '
                f'Pass --struct_hidden_dim {z.size(1)} or set it equal to the encoder output size.'
            )

        src, dst = edge_index
        num_nodes = z.size(0)
        device = z.device
        dtype = z.dtype

        deg = z.new_zeros(num_nodes, 1)
        deg.index_add_(0, dst, torch.ones(src.size(0), 1, device=device, dtype=dtype))
        log_deg = torch.log1p(deg)

        z_src = z[src]
        z_dst = z[dst]
        pair_feat = torch.cat(
            [
                z_src,
                z_dst,
                torch.abs(z_src - z_dst),
                z_src * z_dst,
                log_deg[src],
                log_deg[dst],
            ],
            dim=-1,
        )

        assign = torch.softmax(self.assign_mlp(pair_feat), dim=-1)  # [E, K]
        msg = self.msg_proj(z_src)                                  # [E, token_dim]

        K = self.num_tokens
        D = self.token_dim
        tokens = z.new_zeros(num_nodes, K, D)
        normalizer = z.new_zeros(num_nodes, K, 1)

        # K is small; this works on older PyTorch versions.
        for k in range(K):
            weight = assign[:, k:k + 1]
            tokens[:, k, :].index_add_(0, dst, weight * msg)
            normalizer[:, k, :].index_add_(0, dst, weight)

        tokens = tokens / normalizer.clamp_min(1.0)
        struct_flat = self.flat_norm(tokens.reshape(num_nodes, K * D))
        return struct_flat, tokens, assign, log_deg


class GraphFrontDoorCIWStructContextDAG(GraphFrontDoorCIWDAG):
    """DAG/front-door model with structural spurious context sampling."""

    def __init__(self, d_in, c, args, device):
        super().__init__(d_in, c, args, device)

        self.use_stable_struct = bool(getattr(args, 'use_stable_struct', True))
        self.use_struct_context = bool(getattr(args, 'use_struct_context', True))
        self.num_struct_tokens = max(1, int(getattr(args, 'num_struct_tokens', 4)))
        self.struct_hidden_dim = int(getattr(args, 'struct_hidden_dim', 0)) or int(getattr(args, 'hidden_channels', 64))
        self.struct_context_hidden_dim = int(getattr(args, 'struct_context_hidden_dim', 0)) or int(getattr(args, 'hidden_channels', 64))

        start, stop, step = self.edge_var_slice.indices(self.non_label_var_dim)
        if step != 1:
            raise ValueError('StableStruct injection requires a contiguous edge_var_slice.')
        self.struct_slice = slice(start, stop)
        self.struct_dim = int(stop - start)
        if self.struct_dim <= 0:
            raise ValueError('edge_var_slice/struct_slice must have positive width.')
        if self.struct_dim % self.num_struct_tokens != 0:
            raise ValueError(
                f'edge/structure slice width {self.struct_dim} must be divisible by '
                f'num_struct_tokens={self.num_struct_tokens}. Try --num_struct_tokens 1, 2, 4, or 8 depending on the slice width.'
            )
        self.struct_token_dim = self.struct_dim // self.num_struct_tokens

        self.struct_inject_alpha = float(getattr(args, 'struct_inject_alpha', 0.5))
        self.struct_factor_scale = float(getattr(args, 'struct_factor_scale', 0.0))
        if self.struct_factor_scale <= 0.0:
            self.struct_factor_scale = math.sqrt(max(1.0, float(self.non_label_var_dim) / float(self.struct_dim)))

        # Keep auxiliary terms off by default so DAG + front-door remains the main frame.
        self.lambda_struct_aux = float(getattr(args, 'lambda_struct_aux', 0.0))
        self.lambda_struct_token_div = float(getattr(args, 'lambda_struct_token_div', 0.0))
        self.lambda_struct_env = float(getattr(args, 'lambda_struct_env', 0.0))
        self.struct_spu_threshold = float(getattr(args, 'struct_spu_threshold', getattr(args, 'mediator_threshold', 0.5)))
        self.struct_spu_temp = float(getattr(args, 'struct_spu_temp', getattr(args, 'mediator_temp', 8.0)))

        if self.use_stable_struct:
            self.struct_encoder = StableStructEncoder(
                hidden_dim=self.struct_hidden_dim,
                struct_dim=self.struct_dim,
                num_tokens=self.num_struct_tokens,
                dropout=float(getattr(args, 'dropout', 0.0)),
            )
            self.struct_mix_gate = nn.Sequential(
                nn.Linear(2 * self.struct_dim, self.struct_dim),
                nn.ReLU(),
                nn.Linear(self.struct_dim, self.struct_dim),
            )
            self.struct_aux_classifier = nn.Linear(self.struct_dim, self.c)
            self.struct_context_fuser = nn.Sequential(
                nn.Linear(self.struct_context_hidden_dim + self.struct_dim, self.struct_context_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(self.struct_context_hidden_dim),
            )
        else:
            self.struct_encoder = None
            self.struct_mix_gate = None
            self.struct_aux_classifier = None
            self.struct_context_fuser = None

        self._reset_struct_context_modules()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'struct_encoder'):
            self._reset_struct_context_modules()

    def _reset_struct_context_modules(self):
        if not getattr(self, 'use_stable_struct', False):
            return
        modules = (self.struct_encoder, self.struct_mix_gate, self.struct_aux_classifier, self.struct_context_fuser)
        for module in modules:
            if module is None:
                continue
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            else:
                for sub in module.modules():
                    if sub is module:
                        continue
                    if hasattr(sub, 'reset_parameters'):
                        sub.reset_parameters()

    def _forward_rich(self, x, edge_index, training):
        out = super().forward(x, edge_index, training=training)
        if torch.is_tensor(out):
            raise RuntimeError(
                'Expected rich tuple from GraphFrontDoorDAG.forward(..., training=True), but got a tensor. '
                'Check the parent model forward API.'
            )
        if not isinstance(out, (tuple, list)) or len(out) <= 10:
            raise RuntimeError('Parent forward did not return the expected rich output tuple/list.')
        return out

    def forward(self, x, edge_index, training=False):
        if training:
            return super().forward(x, edge_index, training=True)
        return self.predict_logits(x, edge_index)

    def encode_structure(self, z_all, edge_index):
        if not self.use_stable_struct:
            zeros = z_all.new_zeros(z_all.size(0), self.struct_dim)
            return zeros, None, None, None
        struct_flat, struct_tokens, struct_assign, log_deg = self.struct_encoder(z_all, edge_index)
        struct_flat = struct_flat * self.struct_factor_scale
        return struct_flat, struct_tokens, struct_assign, log_deg

    def inject_struct_into_dag(self, dag_vars, struct_flat):
        if not self.use_stable_struct:
            return dag_vars, dag_vars.new_ones(dag_vars.size(0), self.struct_dim)
        old_struct = dag_vars[:, self.struct_slice]
        mix = torch.sigmoid(self.struct_mix_gate(torch.cat([old_struct, struct_flat], dim=-1)))
        alpha = min(max(float(self.struct_inject_alpha), 0.0), 1.0)
        mixed_struct = mix * old_struct + (1.0 - mix) * struct_flat
        new_struct = (1.0 - alpha) * old_struct + alpha * mixed_struct
        dag_vars_new = dag_vars.clone()
        dag_vars_new[:, self.struct_slice] = new_struct
        return dag_vars_new, mix

    def _struct_token_diversity_loss(self, tokens):
        if tokens is None or self.lambda_struct_token_div <= 0.0:
            return self.A_feat.new_zeros(())
        tok = F.normalize(tokens, dim=-1)
        sim = torch.matmul(tok, tok.transpose(1, 2))
        K = sim.size(1)
        eye = torch.eye(K, device=sim.device, dtype=torch.bool).unsqueeze(0)
        off_diag = sim.masked_select(~eye)
        return off_diag.pow(2).mean() if off_diag.numel() > 0 else self.A_feat.new_zeros(())

    def _assign_entropy(self, assign):
        if assign is None:
            return self.A_feat.new_zeros(())
        return -(assign * assign.clamp_min(1e-8).log()).sum(dim=-1).mean()

    def _node_struct_label_strength(self):
        A = self.get_masked_A()
        label_A = A[:self.non_label_var_dim, self.label_var_slice].abs().mean(dim=1)
        struct_strength = label_A[self.struct_slice].mean()
        mask = torch.ones(self.non_label_var_dim, dtype=torch.bool, device=A.device)
        mask[self.struct_slice] = False
        node_strength = label_A[mask].mean() if mask.any() else label_A.new_zeros(())
        return node_strength.detach(), struct_strength.detach()

    def _struct_score_from_pollution_or_dag(self, pollution_score):
        """Return a length-struct_dim score for structural context masking.

        In some versions of the parent DAG model, ``pollution_score`` is defined
        on encoder/feature dimensions rather than on DAG-variable dimensions.  In
        that case ``pollution_score[self.struct_slice]`` can be empty because the
        structural DAG slice starts beyond the feature-score length.  The previous
        implementation assumed the two spaces had identical indexing, which caused
        ``shape '[K, d]' is invalid for input of size 0``.

        Fallback: derive a DAG-variable-level structural non-causal score from the
        learned adjacency.  We use the inverse normalized parent strength from
        structural variables to label sinks: structural variables weakly supported
        as label parents are treated as environment/spurious context candidates.
        This keeps the overall framework as DAG + front-door and avoids depending
        on StableGNN-style extra objectives.
        """
        device = self.A_feat.device
        dtype = self.A_feat.dtype

        if torch.is_tensor(pollution_score):
            flat_score = pollution_score.reshape(-1)
            if flat_score.numel() >= self.struct_slice.stop:
                raw_score = flat_score[self.struct_slice]
                if raw_score.numel() == self.struct_dim:
                    return raw_score

        # Fallback to DAG adjacency in DAG-variable space.
        try:
            A = self.get_masked_A()
            parent_strength = self._non_label_label_parent_strength(A)
            raw_score = 1.0 - parent_strength[self.struct_slice]
            if raw_score.numel() == self.struct_dim:
                return raw_score
        except Exception:
            pass

        # Last-resort neutral score.  This produces a nonzero but not saturated
        # structural context mask and keeps training running.
        return torch.full((self.struct_dim,), float(self.struct_spu_threshold), device=device, dtype=dtype)

    def _struct_pollution_strength(self, pollution_score):
        if (not self.use_stable_struct) or (not self.use_struct_context):
            return self.A_feat.new_zeros(())
        return self._struct_score_from_pollution_or_dag(pollution_score).detach().mean()

    def _structure_spurious_from_dag(self, dag_vars, pollution_score):
        """Use DAG-derived structural score to extract structural spurious context."""
        struct_vars = dag_vars[:, self.struct_slice]
        if (not self.use_stable_struct) or (not self.use_struct_context):
            mask = struct_vars.new_zeros(self.struct_dim)
            return struct_vars.new_zeros(struct_vars.shape), mask

        raw_score = self._struct_score_from_pollution_or_dag(pollution_score)
        dim_mask = torch.sigmoid(self.struct_spu_temp * (raw_score - self.struct_spu_threshold))

        # Token-level smoothing: all dimensions within a structural token share an
        # averaged pollution gate, keeping the high-level-variable granularity.
        if self.num_struct_tokens > 1 and self.struct_dim % self.num_struct_tokens == 0:
            token_mask = dim_mask.reshape(self.num_struct_tokens, self.struct_token_dim).mean(dim=1, keepdim=True)
            dim_mask = token_mask.repeat(1, self.struct_token_dim).reshape(-1)

        struct_spu = struct_vars * dim_mask.unsqueeze(0)
        return struct_spu, dim_mask

    def _fuse_env_context(self, z_spurious, struct_spu):
        if (not self.use_stable_struct) or (not self.use_struct_context):
            return z_spurious
        if z_spurious.size(1) != self.struct_context_hidden_dim:
            raise ValueError(
                f'struct_context_fuser expects z_spurious dim={self.struct_context_hidden_dim}, '
                f'but got {z_spurious.size(1)}. Pass --struct_context_hidden_dim {z_spurious.size(1)}.'
            )
        return self.struct_context_fuser(torch.cat([z_spurious, struct_spu], dim=-1))

    def _compute_struct_aux_loss(self, dag_vars_tr, y_tr, criterion, args):
        if (not self.use_stable_struct) or self.lambda_struct_aux <= 0.0:
            return self.A_feat.new_zeros(())
        logits = self.struct_aux_classifier(dag_vars_tr[:, self.struct_slice])
        return self.compute_supervised_loss(logits, y_tr, criterion, args).mean()

    def _compute_struct_env_loss(self, env_logits_context):
        if (not self.use_stable_struct) or self.lambda_struct_env <= 0.0 or self.num_envs <= 1:
            return self.A_feat.new_zeros(())
        return self.compute_pseudo_env_loss(env_logits_context)

    def predict_logits(self, x, edge_index):
        # Rich pass gives z_all, dag_vars, mediator, and spurious representations.
        out = self._forward_rich(x, edge_index, training=True)
        z_all = out[1]
        dag_vars_raw = out[3]
        z_mediator_all = out[4]
        z_spurious_all = out[5]
        pollution_score = out[8]
        mediator_logits_all = out[10]

        struct_flat, _, _, _ = self.encode_structure(z_all, edge_index)
        dag_vars_struct, _ = self.inject_struct_into_dag(dag_vars_raw, struct_flat)
        struct_spu_all, _ = self._structure_spurious_from_dag(dag_vars_struct, pollution_score)
        env_context_all = self._fuse_env_context(z_spurious_all, struct_spu_all)

        contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(env_context_all, None, training=False),
            training=False,
        )
        fd_logits_all, _ = self.frontdoor_logits_from_contexts(z_mediator_all, env_context_all, contexts)
        return self.blend_logits(mediator_logits_all, fd_logits_all)

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx

        forward_out = self._forward_rich(x, edge_index, training=True)
        z_all = forward_out[1]
        dag_vars_raw_all = forward_out[3]
        z_mediator_all = forward_out[4]
        z_spurious_all = forward_out[5]
        mediator_gate = forward_out[6]
        causal_score = forward_out[7]
        pollution_score = forward_out[8]
        dag_total = forward_out[9]
        mediator_logits_all = forward_out[10]

        struct_flat_all, struct_tokens_all, struct_assign, _ = self.encode_structure(z_all, edge_index)
        dag_vars_all, struct_mix = self.inject_struct_into_dag(dag_vars_raw_all, struct_flat_all)
        struct_spu_all, struct_spu_mask = self._structure_spurious_from_dag(dag_vars_all, pollution_score)
        env_context_all = self._fuse_env_context(z_spurious_all, struct_spu_all)

        y_tr = y[train_idx]
        med_tr = z_mediator_all[train_idx]
        spu_tr = z_spurious_all[train_idx]
        env_context_tr = env_context_all[train_idx]
        dag_vars_tr = dag_vars_all[train_idx]
        edge_latent_tr = dag_vars_tr[:, self.edge_var_slice]
        mediator_logits_tr = mediator_logits_all[train_idx]

        # Environment inference and context sampling now use fused node+structure
        # spurious context instead of node-only z_spurious.
        env_logits_context = self.env_classifier(env_context_tr)
        env_probs_context = F.softmax(env_logits_context, dim=-1) if self.num_envs > 1 else None

        contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(env_context_tr, env_probs_context, training=True),
            training=True,
        )
        num_gmm_contexts = 0 if contexts is None else int(contexts.size(0))
        fd_logits_tr, fd_stack_tr = self.frontdoor_logits_from_contexts(med_tr, env_context_tr, contexts)
        final_logits_tr = self.blend_logits(mediator_logits_tr, fd_logits_tr)

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        loss_dag = self.dag_regularization_loss(mediator_gate, dag_total)
        loss_dag_rec, loss_dag_feat, loss_dag_label_rec = self.dag_reconstruction_loss(
            dag_vars_all,
            y,
            train_idx,
            criterion,
            args,
            env=getattr(data, 'env', None),
        )
        loss_dag_label = loss_dag_rec

        if self.num_envs > 1:
            env_logits_med = self.env_classifier(med_tr)
            loss_env_med = self.compute_env_uniform_loss(env_logits_med)
            loss_spu = self.compute_pseudo_env_loss(env_logits_context)
        else:
            loss_env_med = self.A_feat.new_zeros(())
            loss_spu = self.compute_uniform_loss(self.classifier(env_context_tr))

        zero = self.A_feat.new_zeros(())
        loss_med = zero
        loss_fd_aug = zero
        loss_var = zero
        loss_ind = zero
        loss_sem = zero
        loss_degree = zero
        loss_spu_y = zero
        loss_inv = zero
        loss_causal_mask_cls = zero
        if self.lambda_causal_mask_cls > 0.0:
            causal_expand = self.dag_gate_expander(causal_score.unsqueeze(0)).squeeze(0)
            causal_mask = torch.sigmoid(self.mediator_temp * (causal_expand - self.mediator_threshold))
            z_causal_mask = z_all * causal_mask.unsqueeze(0)
            loss_causal_mask_cls = self.compute_supervised_loss(
                self.classifier(z_causal_mask[train_idx]), y_tr, criterion, args
            ).mean()

        loss_struct_aux = self._compute_struct_aux_loss(dag_vars_tr, y_tr, criterion, args)
        loss_struct_token_div = self._struct_token_diversity_loss(struct_tokens_all)
        loss_struct_env = self._compute_struct_env_loss(env_logits_context)
        assign_entropy = self._assign_entropy(struct_assign)
        node_label_strength, struct_label_strength = self._node_struct_label_strength()
        struct_pollution_strength = self._struct_pollution_strength(pollution_score)

        total_loss = (
            loss_cls
            + self.lambda_fd * loss_fd
            + self.lambda_dag * loss_dag
            + self.lambda_dag_rec * loss_dag_rec
            + self.lambda_spu * loss_spu
            + self.lambda_env * loss_env_med
            + self.lambda_causal_mask_cls * loss_causal_mask_cls
            + self.lambda_struct_aux * loss_struct_aux
            + self.lambda_struct_token_div * loss_struct_token_div
            + self.lambda_struct_env * loss_struct_env
        )

        state_payload = None
        if update_state:
            # The EMA/GMM state now tracks the fused node+struct spurious context.
            state_payload = {
                'spu_tr': env_context_tr.detach(),
                'env_probs_tr': env_probs_context.detach() if env_probs_context is not None else None,
                'edge_latent_tr': edge_latent_tr.detach(),
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
            'loss_dag_rec': loss_dag_rec,
            'loss_dag_feat': loss_dag_feat,
            'loss_dag_label_rec': loss_dag_label_rec,
            'loss_sem': loss_sem,
            'loss_degree': loss_degree,
            'loss_spu_y': loss_spu_y,
            'loss_spu': loss_spu,
            'loss_env_med': loss_env_med,
            'loss_inv': loss_inv,
            'loss_causal_mask_cls': loss_causal_mask_cls,
            'loss_struct_aux': loss_struct_aux,
            'loss_struct_token_div': loss_struct_token_div,
            'loss_struct_env': loss_struct_env,
            'struct_assign_entropy': assign_entropy.detach(),
            'struct_mix_mean': struct_mix.mean().detach() if torch.is_tensor(struct_mix) else zero.detach(),
            'struct_factor_norm': struct_flat_all.norm(dim=1).mean().detach(),
            'struct_spu_mask_mean': struct_spu_mask.mean().detach() if torch.is_tensor(struct_spu_mask) else zero.detach(),
            'struct_spu_context_norm': struct_spu_all.norm(dim=1).mean().detach(),
            'env_context_norm': env_context_tr.norm(dim=1).mean().detach(),
            'dag_node_label_strength': node_label_strength,
            'dag_struct_label_strength': struct_label_strength,
            'dag_struct_pollution_strength': struct_pollution_strength,
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'causal_score_mean': causal_score.mean().detach(),
            'pollution_score_mean': pollution_score.mean().detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_mixed_contexts': torch.tensor(0.0, device=x.device),
            'num_gmm_contexts': torch.tensor(float(num_gmm_contexts), device=x.device),
            'state_payload': state_payload,
        }
