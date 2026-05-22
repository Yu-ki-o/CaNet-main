"""
Graph Front-Door CIW-DAG + StableStruct.

This variant keeps the original CIW-DAG/front-door model, but adds a
StableGNN-inspired node-centric structural variable module.  Instead of treating
single edges as independent features, it softly pools each target node's
neighbors into K high-level structural tokens, projects them into the DAG
non-label variable space, and injects the result into the original edge/structure
slice of dag_vars.  The injected structure factors then participate in the
existing DAG label-parent mask, CIW reconstruction objective, DAG regularizers,
and front-door training path.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_frontdoor_ciw_dag import GraphFrontDoorCIWDAG


class StableStructEncoder(nn.Module):
    """Node-centric high-level neighborhood structural variables.

    For every directed edge u -> v, the assignment network softly assigns the
    neighbor u to one of K structural roles/tokens of the center node v.  The
    K token embeddings are weighted neighbor summaries and are finally projected
    into one structure factor per node.
    """

    def __init__(self, hidden_dim, struct_dim, num_tokens=4, dropout=0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.struct_dim = int(struct_dim)
        self.num_tokens = int(num_tokens)

        pair_dim = 4 * self.hidden_dim + 2
        self.assign_mlp = nn.Sequential(
            nn.Linear(pair_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_tokens),
        )
        self.msg_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.struct_dim),
            nn.ReLU(),
            nn.LayerNorm(self.struct_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(self.num_tokens * self.struct_dim, self.struct_dim),
            nn.ReLU(),
            nn.LayerNorm(self.struct_dim),
        )

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
        msg = self.msg_proj(z_src)                                  # [E, D]

        K = self.num_tokens
        D = self.struct_dim
        tokens = z.new_zeros(num_nodes, K, D)
        normalizer = z.new_zeros(num_nodes, K, 1)

        # K is small; the loop is simpler and works on old PyTorch versions.
        for k in range(K):
            weight = assign[:, k:k + 1]
            tokens[:, k, :].index_add_(0, dst, weight * msg)
            normalizer[:, k, :].index_add_(0, dst, weight)

        tokens = tokens / normalizer.clamp_min(1.0)
        struct_factor = self.out_proj(tokens.reshape(num_nodes, K * D))
        return struct_factor, tokens, assign, log_deg


class GraphFrontDoorCIWStableStructDAG(GraphFrontDoorCIWDAG):
    """CIW-DAG/front-door model with StableGNN-style neighborhood variables."""

    def __init__(self, d_in, c, args, device):
        super().__init__(d_in, c, args, device)

        self.use_stable_struct = bool(getattr(args, 'use_stable_struct', True))
        self.num_struct_tokens = max(1, int(getattr(args, 'num_struct_tokens', 4)))
        self.struct_hidden_dim = int(getattr(args, 'struct_hidden_dim', 0)) or int(getattr(args, 'hidden_channels', 64))

        start, stop, step = self.edge_var_slice.indices(self.non_label_var_dim)
        if step != 1:
            raise ValueError('StableStruct injection requires a contiguous edge_var_slice.')
        self.struct_slice = slice(start, stop)
        self.struct_dim = int(stop - start)
        if self.struct_dim <= 0:
            raise ValueError('edge_var_slice/struct_slice must have positive width.')

        self.struct_inject_alpha = float(getattr(args, 'struct_inject_alpha', 0.5))
        self.struct_factor_scale = float(getattr(args, 'struct_factor_scale', 0.0))
        if self.struct_factor_scale <= 0.0:
            # Equalize the total scale of a low-dimensional structure slice against
            # the full non-label factor vector.
            self.struct_factor_scale = math.sqrt(max(1.0, float(self.non_label_var_dim) / float(self.struct_dim)))

        self.lambda_struct_aux = float(getattr(args, 'lambda_struct_aux', 0.02))
        self.lambda_struct_token_div = float(getattr(args, 'lambda_struct_token_div', 0.0))
        self.struct_readout_blend = float(getattr(args, 'struct_readout_blend', 0.15))
        self.struct_readout_blend = min(max(self.struct_readout_blend, 0.0), 1.0)

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
        else:
            self.struct_encoder = None
            self.struct_mix_gate = None
            self.struct_aux_classifier = None

        self._reset_stable_struct_modules()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'struct_encoder'):
            self._reset_stable_struct_modules()

    def _reset_stable_struct_modules(self):
        if not getattr(self, 'use_stable_struct', False):
            return
        if self.struct_encoder is not None:
            self.struct_encoder.reset_parameters()
        for module in (self.struct_mix_gate, self.struct_aux_classifier):
            if module is None:
                continue
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            elif hasattr(module, 'modules'):
                for sub in module.modules():
                    if sub is module:
                        continue
                    if hasattr(sub, 'reset_parameters'):
                        sub.reset_parameters()

    def _forward_rich(self, x, edge_index, training):
        out = super().forward(x, edge_index, training=training)
        if torch.is_tensor(out):
            raise RuntimeError(
                'Expected rich tuple from GraphFrontDoorDAG.forward(..., training=True), '
                'but got a tensor. Check the parent model forward API.'
            )
        if not isinstance(out, (tuple, list)) or len(out) <= 10:
            raise RuntimeError('Parent forward did not return the expected rich output tuple/list.')
        return out

    def forward(self, x, edge_index, training=False):
        # Training code calls compute_losses, which explicitly requests the rich
        # parent forward.  Evaluation utilities call model(x, edge_index) and expect
        # a logits tensor, so return structure-aware prediction logits here.
        if training:
            return super().forward(x, edge_index, training=True)
        return self.predict_logits(x, edge_index)

    def encode_structure(self, z_all, edge_index):
        if not self.use_stable_struct:
            zeros = z_all.new_zeros(z_all.size(0), self.struct_dim)
            return zeros, None, None, None
        struct_factor, struct_tokens, struct_assign, log_deg = self.struct_encoder(z_all, edge_index)
        struct_factor = struct_factor * self.struct_factor_scale
        return struct_factor, struct_tokens, struct_assign, log_deg

    def inject_struct_into_dag(self, dag_vars, struct_factor):
        if not self.use_stable_struct:
            return dag_vars, dag_vars.new_ones(dag_vars.size(0), self.struct_dim)
        old_struct = dag_vars[:, self.struct_slice]
        mix = torch.sigmoid(self.struct_mix_gate(torch.cat([old_struct, struct_factor], dim=-1)))
        alpha = min(max(float(self.struct_inject_alpha), 0.0), 1.0)
        # mix controls old-vs-new inside the injectable portion; alpha controls the
        # maximum amount of structure replacement.
        mixed_struct = mix * old_struct + (1.0 - mix) * struct_factor
        new_struct = (1.0 - alpha) * old_struct + alpha * mixed_struct
        dag_vars_new = dag_vars.clone()
        dag_vars_new[:, self.struct_slice] = new_struct
        return dag_vars_new, mix

    def _struct_token_diversity_loss(self, tokens):
        if tokens is None or self.lambda_struct_token_div <= 0.0:
            return self.A_feat.new_zeros(())
        # Encourage different structural tokens to avoid exact collapse, but keep it
        # optional and weak because overly strong decorrelation can hurt semantics.
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

    def _structure_aware_logits(self, base_logits, dag_vars_struct):
        if (not self.use_stable_struct) or self.struct_readout_blend <= 0.0:
            return base_logits, base_logits.new_zeros(())
        struct_logits = self._dag_label_logits_from_factors(dag_vars_struct)
        final_logits = (1.0 - self.struct_readout_blend) * base_logits + self.struct_readout_blend * struct_logits
        return final_logits, struct_logits

    @torch.no_grad()
    def _maybe_detach_eval_context(self, x):
        return x

    def predict_logits(self, x, edge_index):
        # Use a rich pass even in eval so that we can access z_all and dag_vars.
        out = self._forward_rich(x, edge_index, training=True)
        z_all = out[1]
        dag_vars_all = out[3]
        z_mediator_all = out[4]
        z_spurious_all = out[5]
        mediator_logits_all = out[10]

        contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(z_spurious_all, None, training=False),
            training=False,
        )
        fd_logits_all, _ = self.frontdoor_logits_from_contexts(z_mediator_all, z_spurious_all, contexts)
        base_logits = self.blend_logits(mediator_logits_all, fd_logits_all)

        struct_factor_all, _, _, _ = self.encode_structure(z_all, edge_index)
        dag_vars_struct, _ = self.inject_struct_into_dag(dag_vars_all, struct_factor_all)
        final_logits, _ = self._structure_aware_logits(base_logits, dag_vars_struct)
        return final_logits

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

        struct_factor_all, struct_tokens_all, struct_assign, _ = self.encode_structure(z_all, edge_index)
        dag_vars_all, struct_mix = self.inject_struct_into_dag(dag_vars_raw_all, struct_factor_all)

        y_tr = y[train_idx]
        med_tr = z_mediator_all[train_idx]
        spu_tr = z_spurious_all[train_idx]
        dag_vars_tr = dag_vars_all[train_idx]
        edge_latent_tr = dag_vars_tr[:, self.edge_var_slice]
        mediator_logits_tr = mediator_logits_all[train_idx]
        env_logits_spu = self.env_classifier(spu_tr)
        env_probs_spu = F.softmax(env_logits_spu, dim=-1) if self.num_envs > 1 else None

        contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(spu_tr, env_probs_spu, training=True),
            training=True,
        )
        num_gmm_contexts = 0 if contexts is None else int(contexts.size(0))
        fd_logits_tr, fd_stack_tr = self.frontdoor_logits_from_contexts(med_tr, spu_tr, contexts)
        base_logits_tr = self.blend_logits(mediator_logits_tr, fd_logits_tr)
        final_logits_tr, struct_logits_tr = self._structure_aware_logits(base_logits_tr, dag_vars_tr)

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
            loss_spu = self.compute_pseudo_env_loss(env_logits_spu)
        else:
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
        loss_causal_mask_cls = zero
        if self.lambda_causal_mask_cls > 0.0:
            causal_expand = self.dag_gate_expander(causal_score.unsqueeze(0)).squeeze(0)
            causal_mask = torch.sigmoid(self.mediator_temp * (causal_expand - self.mediator_threshold))
            z_causal_mask = z_all * causal_mask.unsqueeze(0)
            loss_causal_mask_cls = self.compute_supervised_loss(
                self.classifier(z_causal_mask[train_idx]), y_tr, criterion, args
            ).mean()

        loss_struct_aux = zero
        if self.use_stable_struct and self.lambda_struct_aux > 0.0:
            # Auxiliary prediction through the DAG-masked structure-aware factors.
            if torch.is_tensor(struct_logits_tr):
                loss_struct_aux = self.compute_supervised_loss(struct_logits_tr, y_tr, criterion, args).mean()

        loss_struct_token_div = self._struct_token_diversity_loss(struct_tokens_all)
        assign_entropy = self._assign_entropy(struct_assign)
        node_label_strength, struct_label_strength = self._node_struct_label_strength()

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
        )

        state_payload = None
        if update_state:
            state_payload = {
                'spu_tr': spu_tr.detach(),
                'env_probs_tr': env_probs_spu.detach() if env_probs_spu is not None else None,
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
            'struct_assign_entropy': assign_entropy.detach(),
            'struct_mix_mean': struct_mix.mean().detach() if torch.is_tensor(struct_mix) else zero.detach(),
            'struct_factor_norm': struct_factor_all.norm(dim=1).mean().detach(),
            'dag_node_label_strength': node_label_strength,
            'dag_struct_label_strength': struct_label_strength,
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'causal_score_mean': causal_score.mean().detach(),
            'pollution_score_mean': pollution_score.mean().detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_mixed_contexts': torch.tensor(0.0, device=x.device),
            'num_gmm_contexts': torch.tensor(float(num_gmm_contexts), device=x.device),
            'state_payload': state_payload,
        }
