"""
Graph Front-Door CIW-DAG model.

This file keeps the original CIPT/front-door graph framework, but replaces the
DAG-Core shortcut objective with a CIW-style DAG construction objective matching
Eq. (1)-(2) of "Causal-Guided Strength Differential Independence Sample
Weighting for OOD Generalization":

  feature factors: reconstruct every non-label factor from its DAG parents;
  label factor: mask parent factors by A and map them to logits with M;
  L_rec = feature-factor L2 reconstruction + label-factor CE.

The model still uses the learned DAG total effect exp(A*A) to build the mediator
mask and keeps the original front-door context aggregation path.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_frontdoor_dag_core import GraphFrontDoorDAG


class GraphFrontDoorCIWDAGEdge(GraphFrontDoorDAG):
    """CIPT/front-door model with CIW DAG construction and DAG-internal edge factor injection."""

    def __init__(self, d_in, c, args, device):
        super().__init__(d_in, c, args, device)

        # Eq. (1), label branch: M(A_y^T \odot F).  We keep A as the mask/strength
        # and use a separate mapper M instead of letting A itself become the whole
        # classifier.  This prevents the DAG adjacency from degenerating into a
        # pure linear label head.
        self.dag_label_mapper = nn.Linear(self.non_label_var_dim, self.c)

        # Optional light nonlinear reconstructor for the parent signal.  The default
        # is identity because Eq. (1) uses A_i,:F directly for feature factors.
        self.use_dag_feature_mlp = bool(getattr(args, 'use_dag_feature_mlp', False))
        if self.use_dag_feature_mlp:
            self.dag_feature_mapper = nn.Sequential(
                nn.Linear(self.non_label_var_dim, self.non_label_var_dim),
                nn.ReLU(),
                nn.Linear(self.non_label_var_dim, self.non_label_var_dim),
            )
        else:
            self.dag_feature_mapper = nn.Identity()

        self.lambda_dag_rec = float(getattr(args, 'lambda_dag_rec', getattr(args, 'lambda_dag_label', 0.05)))
        self.lambda_dag_feat = float(getattr(args, 'lambda_dag_feat', 1.0))
        self.lambda_dag_label_rec = float(getattr(args, 'lambda_dag_label_rec', 1.0))
        self.lambda_dag_proto = float(getattr(args, 'lambda_dag_proto', 0.0))
        self.lambda_causal_mask_cls = float(getattr(args, 'lambda_causal_mask_cls', 0.0))
        self.dag_rec_detach_input = bool(getattr(args, 'dag_rec_detach_input', False))
        self.dag_rec_use_abs_mask = bool(getattr(args, 'dag_rec_use_abs_mask', True))
        self.dag_proto_min_count = max(1, int(getattr(args, 'dag_proto_min_count', 1)))
        self.zero_current_spurious_in_fd = bool(getattr(args, 'zero_current_spurious_in_fd', True))
        self.eval_gmm_use_mean = bool(getattr(args, 'eval_gmm_use_mean', True))

        # Edge-DAG injection.  The learned edge context is projected into the
        # existing edge_var_slice and then mixed with the original edge latent.
        # Therefore edge evidence still passes through A, DAG reconstruction,
        # label-parent masking, and the front-door/DAG readout instead of bypassing
        # the DAG module with an independent edge classifier.
        self.use_edge_dag = bool(getattr(args, 'use_edge_dag', True))
        self.edge_hidden_dim = int(getattr(args, 'edge_hidden_dim', getattr(args, 'hidden_channels', 64)))
        self.edge_var_dim = self._infer_index_dim(self.edge_var_slice, self.non_label_var_dim)
        edge_factor_scale_arg = getattr(args, 'edge_factor_scale', None)
        if edge_factor_scale_arg is None:
            edge_factor_scale_arg = math.sqrt(max(1, self.non_label_var_dim) / max(1, self.edge_var_dim))
        self.edge_factor_scale = float(edge_factor_scale_arg)
        self.edge_inject_dropout = float(getattr(args, 'edge_inject_dropout', getattr(args, 'dropout', 0.0)))
        self.lambda_edge_dag = float(getattr(args, 'lambda_edge_dag', 0.05))
        self.lambda_edge_gate_balance = float(getattr(args, 'lambda_edge_gate_balance', 0.0))
        self.edge_readout_blend = float(getattr(args, 'edge_readout_blend', 0.2))

        self.edge_msg_mlp = nn.Sequential(
            nn.Linear(4 * self.edge_hidden_dim, self.edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.edge_inject_dropout),
            nn.Linear(self.edge_hidden_dim, self.edge_hidden_dim),
        )
        self.edge_gate_mlp = nn.Sequential(
            nn.Linear(4 * self.edge_hidden_dim, max(1, self.edge_hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(1, self.edge_hidden_dim // 2), 1),
        )
        self.edge_factor_proj = nn.Linear(self.edge_hidden_dim, self.edge_var_dim)
        self.edge_factor_norm = nn.LayerNorm(self.edge_var_dim)
        self.edge_mix_gate = nn.Sequential(
            nn.Linear(2 * self.edge_var_dim, max(1, self.edge_var_dim)),
            nn.ReLU(),
            nn.Linear(max(1, self.edge_var_dim), self.edge_var_dim),
        )
        self.node_edge_readout_gate = nn.Sequential(
            nn.Linear(2 * self.non_label_var_dim, max(1, self.non_label_var_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(1, self.non_label_var_dim // 2), 2),
        )

        for module_name in (
            'edge_msg_mlp',
            'edge_gate_mlp',
            'edge_factor_proj',
            'edge_factor_norm',
            'edge_mix_gate',
            'node_edge_readout_gate',
        ):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for submodule in module.modules() if hasattr(module, 'modules') else [module]:
                if submodule is module and hasattr(module, 'modules'):
                    # Sequential/LayerNorm/Linear reset is handled by child modules,
                    # except single modules like LayerNorm/Linear with no children.
                    pass
                if hasattr(submodule, 'reset_parameters'):
                    submodule.reset_parameters()
        self.current_fd_blend = self.fd_blend
        self._reset_ciw_modules()


    @staticmethod
    def _infer_index_dim(indexer, fallback_dim):
        if isinstance(indexer, slice):
            start = 0 if indexer.start is None else int(indexer.start)
            stop = int(fallback_dim if indexer.stop is None else indexer.stop)
            step = 1 if indexer.step is None else int(indexer.step)
            return max(0, (stop - start + step - 1) // step)
        if torch.is_tensor(indexer):
            return int(indexer.numel())
        try:
            return len(indexer)
        except TypeError:
            return 1

    def _edge_slice_set(self, dag_vars, value):
        out = dag_vars.clone()
        out[:, self.edge_var_slice] = value
        return out

    def compute_edge_factor(self, z, edge_index):
        if (not self.use_edge_dag) or edge_index is None or edge_index.numel() == 0:
            edge_factor = z.new_zeros((z.size(0), self.edge_var_dim))
            edge_gate = z.new_zeros((0, 1))
            edge_mix = z.new_zeros((z.size(0), self.edge_var_dim))
            return edge_factor, edge_gate, edge_mix

        src, dst = edge_index
        z_src = z.index_select(0, src)
        z_dst = z.index_select(0, dst)
        edge_feat = torch.cat([z_src, z_dst, torch.abs(z_src - z_dst), z_src * z_dst], dim=-1)

        edge_msg = self.edge_msg_mlp(edge_feat)
        edge_gate = torch.sigmoid(self.edge_gate_mlp(edge_feat))
        weighted_msg = edge_gate * edge_msg

        edge_context = z.new_zeros(z.shape)
        edge_context.index_add_(0, dst, weighted_msg)
        deg = z.new_zeros((z.size(0), 1))
        deg.index_add_(0, dst, torch.ones_like(edge_gate))
        edge_context = edge_context / deg.clamp_min(1.0)

        edge_factor = self.edge_factor_norm(self.edge_factor_proj(edge_context)) * self.edge_factor_scale
        # edge_mix is returned later after seeing the old edge slice.
        edge_mix = edge_factor.new_zeros(edge_factor.shape)
        return edge_factor, edge_gate, edge_mix

    def inject_edge_factor_into_dag(self, dag_vars, z, edge_index):
        if not self.use_edge_dag:
            zero_gate = dag_vars.new_zeros((0, 1))
            zero_mix = dag_vars.new_zeros((dag_vars.size(0), self.edge_var_dim))
            return dag_vars, zero_gate, zero_mix
        edge_factor, edge_gate, _ = self.compute_edge_factor(z, edge_index)
        old_edge = dag_vars[:, self.edge_var_slice]
        mix = torch.sigmoid(self.edge_mix_gate(torch.cat([old_edge, edge_factor], dim=-1)))
        mixed_edge = mix * old_edge + (1.0 - mix) * edge_factor
        return self._edge_slice_set(dag_vars, mixed_edge), edge_gate, mix

    def edge_aware_dag_logits(self, base_logits, dag_vars_edge, dag_vars_original=None):
        edge_logits = self._dag_label_logits_from_factors(dag_vars_edge)
        if dag_vars_original is None or self.edge_readout_blend <= 0.0:
            return edge_logits, edge_logits, None
        gate_input = torch.cat([dag_vars_original, dag_vars_edge], dim=-1)
        readout_w = torch.softmax(self.node_edge_readout_gate(gate_input), dim=-1)
        # Blend original front-door/mediator logits with an edge-injected DAG readout.
        # edge_readout_blend caps the edge branch so it cannot dominate at init.
        w_edge = self.edge_readout_blend * readout_w[:, 1:2]
        final_logits = (1.0 - w_edge) * base_logits + w_edge * edge_logits
        return final_logits, edge_logits, readout_w

    def reset_parameters(self):
        # Base __init__ calls reset_parameters before CIW modules exist.  Guarding
        # keeps that path safe and also resets CIW modules on later run resets.
        super().reset_parameters()
        if hasattr(self, 'dag_label_mapper'):
            self._reset_ciw_modules()

    def _reset_ciw_modules(self):
        self.dag_label_mapper.reset_parameters()
        if hasattr(self, 'dag_feature_mapper') and hasattr(self.dag_feature_mapper, 'modules'):
            for module in self.dag_feature_mapper.modules():
                if module is self.dag_feature_mapper:
                    continue
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
        self.current_fd_blend = self.fd_blend

    def _label_to_onehot(self, labels, num_rows=None):
        if labels.dim() > 1 and labels.size(1) == self.c:
            out = labels.float()
        elif labels.dim() > 1 and labels.size(1) > 1:
            out = labels.float()
        else:
            out = F.one_hot(labels.view(-1).long().clamp_min(0), num_classes=self.c).float()
        if num_rows is not None:
            out = out[:num_rows]
        return out

    def _non_label_label_parent_strength(self, A):
        """Compress all label sink columns into one parent mask over non-label factors."""
        label_A = A[:self.non_label_var_dim, self.label_var_slice]
        if self.dag_rec_use_abs_mask:
            strength = label_A.abs().mean(dim=1)
        else:
            strength = label_A.mean(dim=1)
        # Keep scale stable across runs.  Detach min/max? No: causal strength should
        # still train through A; clamp only for numerical stability.
        strength = strength / strength.max().clamp_min(1e-8)
        return strength

    def _dag_label_logits_from_factors(self, dag_vars, A=None):
        if A is None:
            A = self.get_masked_A()
        parent_strength = self._non_label_label_parent_strength(A).unsqueeze(0)
        masked_factors = dag_vars * parent_strength
        return self.dag_label_mapper(masked_factors)

    def _make_dag_prototypes(self, dag_vars, labels, env, train_idx):
        """Build lightweight env/class and class prototypes for CIW-style DAG learning."""
        if train_idx is None or train_idx.numel() == 0:
            return None, None, None
        z = dag_vars[train_idx]
        y_raw = labels[train_idx]
        if y_raw.dim() > 1 and y_raw.size(1) > 1:
            y_cls = y_raw.argmax(dim=1)
        else:
            y_cls = y_raw.view(-1).long()
        if env is None:
            e_cls = torch.zeros_like(y_cls)
            env_count = 1
        else:
            e = env[train_idx]
            e_cls = e.view(-1).long() if e.dim() > 1 else e.long()
            env_count = max(self.num_envs, int(e_cls.max().item()) + 1 if e_cls.numel() > 0 else 1)

        proto_list = []
        proto_labels = []
        proto_envs = []
        for env_idx in range(env_count):
            for cls_idx in range(self.c):
                mask = (e_cls == env_idx) & (y_cls == cls_idx)
                if int(mask.sum().item()) >= self.dag_proto_min_count:
                    proto_list.append(z[mask].mean(dim=0))
                    proto_labels.append(cls_idx)
                    proto_envs.append(env_idx)

        # Cross-domain invariant class prototypes: one per class, averaged over all
        # training envs.  This is a compact analogue of the paper's B_cr^d.
        for cls_idx in range(self.c):
            mask = (y_cls == cls_idx)
            if int(mask.sum().item()) >= self.dag_proto_min_count:
                proto_list.append(z[mask].mean(dim=0))
                proto_labels.append(cls_idx)
                proto_envs.append(-1)

        if not proto_list:
            return None, None, None
        proto_z = torch.stack(proto_list, dim=0)
        proto_y = torch.tensor(proto_labels, device=dag_vars.device, dtype=torch.long).view(-1, 1)
        proto_e = torch.tensor(proto_envs, device=dag_vars.device, dtype=torch.long)
        return proto_z, proto_y, proto_e

    def dag_reconstruction_loss(self, dag_vars, labels, train_idx, criterion, args, env=None):
        """
        CIW Eq. (1)-(2) DAG construction loss.

        Orientation in this code: A[source, target].  Therefore target-factor
        reconstruction is F @ A[:, target].  Label reconstruction uses a separate
        M over A-masked parent factors.
        """
        if dag_vars.numel() == 0 or train_idx.numel() == 0:
            return self.A_feat.new_zeros(()), self.A_feat.new_zeros(()), self.A_feat.new_zeros(())

        A = self.get_masked_A()
        z_train = dag_vars[train_idx]
        y_train = labels[train_idx]
        if self.dag_rec_detach_input:
            z_for_rec = z_train.detach()
        else:
            z_for_rec = z_train

        # Eq. (1), feature factors: G_i = A_i,: F.  With source->target
        # orientation this is F @ A[:, non_label].  We only reconstruct non-label
        # factors; label sinks are excluded from feature parents by the allowed mask.
        A_feat_to_feat = A[:self.non_label_var_dim, :self.non_label_var_dim]
        feature_parent_signal = torch.matmul(z_for_rec, A_feat_to_feat)
        recon_features = self.dag_feature_mapper(feature_parent_signal)
        loss_feat = F.mse_loss(recon_features, z_train.detach())

        # Eq. (1), label factor: M(A_y^T \odot F), then Eq. (2) CE.
        label_logits = self._dag_label_logits_from_factors(z_for_rec, A=A)
        loss_label = self.compute_supervised_loss(label_logits, y_train, criterion, args).mean()

        loss_proto = self.A_feat.new_zeros(())
        if self.lambda_dag_proto > 0.0:
            proto_z, proto_y, _ = self._make_dag_prototypes(dag_vars, labels, env, train_idx)
            if proto_z is not None and proto_z.size(0) > 0:
                proto_input = proto_z.detach() if self.dag_rec_detach_input else proto_z
                proto_parent_signal = torch.matmul(proto_input, A_feat_to_feat)
                proto_recon = self.dag_feature_mapper(proto_parent_signal)
                proto_feat_loss = F.mse_loss(proto_recon, proto_z.detach())
                proto_logits = self._dag_label_logits_from_factors(proto_input, A=A)
                proto_label_loss = self.compute_supervised_loss(proto_logits, proto_y, criterion, args).mean()
                loss_proto = proto_feat_loss + proto_label_loss

        loss_rec = (
            self.lambda_dag_feat * loss_feat
            + self.lambda_dag_label_rec * loss_label
            + self.lambda_dag_proto * loss_proto
        )
        return loss_rec, loss_feat.detach(), loss_label.detach()

    def dag_label_loss(self, dag_vars, labels, train_idx, criterion, args):
        """Backward-compatible label branch, now using M(A_y^T \odot F)."""
        if dag_vars.numel() == 0 or train_idx.numel() == 0:
            return self.A_feat.new_zeros(())
        logits = self._dag_label_logits_from_factors(dag_vars[train_idx])
        return self.compute_supervised_loss(logits, labels[train_idx], criterion, args).mean()

    def blend_logits(self, med_logits, fd_logits):
        if fd_logits is None:
            return med_logits
        blend = float(getattr(self, 'current_fd_blend', self.fd_blend))
        return (1.0 - blend) * med_logits + blend * fd_logits

    def sample_gmm_contexts(self, z_spurious=None, env_probs=None, training=False):
        # CIW/CIPT evaluation should be deterministic.  By default use the learned
        # GMM means at eval time instead of reintroducing random context noise.
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
        if (not training) and self.eval_gmm_use_mean:
            return F.normalize(mean, dim=1)

        var = vars_.index_select(0, env_indices).clamp_min(self.gmm_min_var)
        std = var.sqrt()
        if self.gmm_max_std > 0.0:
            std = std.clamp_max(self.gmm_max_std)
        noise = torch.randn_like(mean)
        contexts = mean + noise * std
        return F.normalize(contexts, dim=1)

    def frontdoor_logits_from_contexts(self, z_mediator, z_spurious, contexts):
        # Keep the front-door adjustment, but default to preventing the current
        # node's spurious representation from leaking through the DAG mixer path.
        if self.zero_current_spurious_in_fd and z_spurious is not None:
            z_spurious = torch.zeros_like(z_spurious)
        return super().frontdoor_logits_from_contexts(z_mediator, z_spurious, contexts)


    def _parent_rich_forward(self, x, edge_index):
        """Return the parent model's rich tuple used by compute_losses.

        In this codebase the base model usually returns a diagnostic tuple when
        called with training=True, while evaluation helpers call model(x, edge_index)
        and expect a logits tensor.  Keeping this helper separate prevents us from
        accidentally treating an eval logits tensor as a tuple of hidden states.
        """
        out = super().forward(x, edge_index, training=True)
        if not isinstance(out, (tuple, list)) or len(out) <= 10:
            raise RuntimeError(
                "GraphFrontDoorCIWDAGEdge expected the parent forward(..., training=True) "
                "to return the rich tuple with hidden states and DAG variables."
            )
        return out

    def predict_logits(self, x, edge_index):
        """Evaluation-time logits using the edge-injected DAG branch.

        evaluate_full() in the original project calls model(x, edge_index) and
        expects a tensor.  This method reproduces the same node/front-door readout
        used in training, injects learned edge factors into dag_vars[:, edge_var_slice],
        and returns only final logits.
        """
        if not self.use_edge_dag:
            out = super().forward(x, edge_index, training=False)
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

        forward_out = self._parent_rich_forward(x, edge_index)
        z_all = forward_out[1]
        dag_vars_raw_all = forward_out[3]
        z_mediator_all = forward_out[4]
        z_spurious_all = forward_out[5]
        mediator_logits_all = forward_out[10]

        if (not torch.is_tensor(z_all)) or z_all.dim() != 2:
            out = super().forward(x, edge_index, training=False)
            return out[0] if isinstance(out, (tuple, list)) else out

        env_probs_spu = None
        if self.num_envs > 1:
            env_probs_spu = F.softmax(self.env_classifier(z_spurious_all), dim=-1)
        contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(z_spurious_all, env_probs_spu, training=False),
            training=False,
        )
        fd_logits_all, _ = self.frontdoor_logits_from_contexts(
            z_mediator_all, z_spurious_all, contexts
        )
        base_logits_all = self.blend_logits(mediator_logits_all, fd_logits_all)

        dag_vars_edge_all, _, _ = self.inject_edge_factor_into_dag(
            dag_vars_raw_all, z_all, edge_index
        )
        final_logits_all, _, _ = self.edge_aware_dag_logits(
            base_logits_all, dag_vars_edge_all, dag_vars_raw_all
        )
        return final_logits_all

    def forward(self, x, edge_index, training=False):
        # Evaluation helpers expect a logits tensor, not the rich tuple returned by
        # the parent when training=True.  Returning predict_logits here fixes the
        # earlier shape error where eval logits were mistakenly parsed as z_all.
        if not training:
            return self.predict_logits(x, edge_index)

        out = super().forward(x, edge_index, training=True)
        if not self.use_edge_dag or not isinstance(out, (tuple, list)) or len(out) <= 10:
            return out

        out_list = list(out)
        z_all = out_list[1]
        dag_vars_raw = out_list[3]
        if (not torch.is_tensor(z_all)) or z_all.dim() != 2:
            return out

        z_mediator = out_list[4]
        z_spurious = out_list[5]
        mediator_logits = out_list[10]
        env_probs_spu = None
        if self.num_envs > 1:
            env_probs_spu = F.softmax(self.env_classifier(z_spurious), dim=-1)
        contexts = self.sample_frontdoor_contexts(
            self.sample_gmm_contexts(z_spurious, env_probs_spu, training=True),
            training=True,
        )
        fd_logits, _ = self.frontdoor_logits_from_contexts(z_mediator, z_spurious, contexts)
        base_logits = self.blend_logits(mediator_logits, fd_logits)
        dag_vars_edge, _, _ = self.inject_edge_factor_into_dag(dag_vars_raw, z_all, edge_index)
        final_logits, _, _ = self.edge_aware_dag_logits(base_logits, dag_vars_edge, dag_vars_raw)

        out_list[0] = final_logits
        out_list[3] = dag_vars_edge
        out_list[10] = final_logits
        return tuple(out_list)

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx

        forward_out = super().forward(x, edge_index, training=True)
        z_all = forward_out[1]
        dag_vars_raw_all = forward_out[3]
        z_mediator_all = forward_out[4]
        z_spurious_all = forward_out[5]
        mediator_gate = forward_out[6]
        causal_score = forward_out[7]
        pollution_score = forward_out[8]
        dag_total = forward_out[9]
        mediator_logits_all = forward_out[10]

        dag_vars_all, edge_gate_all, edge_mix_all = self.inject_edge_factor_into_dag(
            dag_vars_raw_all, z_all, edge_index
        )

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
        final_logits_tr, edge_dag_logits_tr, node_edge_readout_w = self.edge_aware_dag_logits(
            base_logits_tr, dag_vars_all[train_idx], dag_vars_raw_all[train_idx]
        )

        loss_cls = self.compute_supervised_loss(final_logits_tr, y_tr, criterion, args).mean()
        loss_fd = self.compute_supervised_loss(fd_logits_tr, y_tr, criterion, args).mean()
        loss_edge_dag = self.compute_supervised_loss(edge_dag_logits_tr, y_tr, criterion, args).mean()
        if edge_mix_all.numel() > 0 and self.lambda_edge_gate_balance > 0.0:
            loss_edge_gate_balance = ((edge_mix_all.mean() - 0.5) ** 2)
        else:
            loss_edge_gate_balance = self.A_feat.new_zeros(())
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
            # Paper Eq. (5) style auxiliary: z_invariant = z \odot Ca.  The model's
            # main classifier already uses mediator_gate; this term directly tests
            # whether the raw total-effect mask is label-predictive.
            causal_expand = self.dag_gate_expander(causal_score.unsqueeze(0)).squeeze(0)
            causal_mask = torch.sigmoid(self.mediator_temp * (causal_expand - self.mediator_threshold))
            z_causal_mask = z_all * causal_mask.unsqueeze(0)
            loss_causal_mask_cls = self.compute_supervised_loss(
                self.classifier(z_causal_mask[train_idx]), y_tr, criterion, args
            ).mean()

        total_loss = (
            loss_cls
            + self.lambda_fd * loss_fd
            + self.lambda_dag * loss_dag
            + self.lambda_dag_rec * loss_dag_rec
            + self.lambda_spu * loss_spu
            + self.lambda_env * loss_env_med
            + self.lambda_causal_mask_cls * loss_causal_mask_cls
            + self.lambda_edge_dag * loss_edge_dag
            + self.lambda_edge_gate_balance * loss_edge_gate_balance
        )

        state_payload = None
        if update_state:
            state_payload = {
                'spu_tr': spu_tr.detach(),
                'env_probs_tr': env_probs_spu.detach() if env_probs_spu is not None else None,
                'edge_latent_tr': dag_vars_all[train_idx][:, self.edge_var_slice].detach(),
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
            'loss_edge_dag': loss_edge_dag,
            'loss_edge_gate_balance': loss_edge_gate_balance,
            'edge_gate_mean': edge_gate_all.mean().detach() if edge_gate_all.numel() > 0 else self.A_feat.new_zeros(()),
            'edge_mix_mean': edge_mix_all.mean().detach() if edge_mix_all.numel() > 0 else self.A_feat.new_zeros(()),
            'node_edge_readout_edge_mean': (node_edge_readout_w[:, 1].mean().detach() if node_edge_readout_w is not None else self.A_feat.new_zeros(())),
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'causal_score_mean': causal_score.mean().detach(),
            'pollution_score_mean': pollution_score.mean().detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_mixed_contexts': torch.tensor(0.0, device=x.device),
            'num_gmm_contexts': torch.tensor(float(num_gmm_contexts), device=x.device),
            'state_payload': state_payload,
        }
