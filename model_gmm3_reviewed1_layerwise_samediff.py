import torch
import torch.nn as nn
import torch.nn.functional as F

from model_gmm3_reviewed1_graph_cfam_nego import GraphFrontDoorDAG as BaseGraphFrontDoorDAG


class GraphFrontDoorDAG(BaseGraphFrontDoorDAG):
    """
    Layer-wise same/different ego-relation front-door model.

    This variant turns each GNN layer into two relation subgraphs:
    - same path: runs the selected GNN backbone on class-consistent edges;
    - different path: runs the same GNN backbone on boundary edges.

    The subgraphs are computed layer by layer from predicted class relations,
    not from ground-truth labels.  The different path produces a dimension
    gate that guides the same-path GNN summary, but it is not written back
    into the next-layer node state.  The final guided same-path state is used
    as z_causal, and the gated different-path state is used as
    z_spurious/context.
    Graph-CFAM, multi-ratio contexts, global contexts, and NeGo contexts are
    intentionally disabled in this lightweight research variant.
    """

    def __init__(self, d_in, c, args, device):
        super().__init__(d_in, c, args, device)

        self.use_graph_cfam = False
        self.use_pre_gnn_graph_cfam = False
        self.use_final_graph_cfam = False
        self.use_layerwise_local_igm = True
        self.use_multi_ratio_spurious_fd = False
        self.multi_ratio_spurious_fd_as_main = False
        self.lambda_multi_ratio_fd = 0.0
        self.lambda_multi_ratio_fd_worst = 0.0
        self.lambda_multi_ratio_fd_cons = 0.0
        self.use_layerwise_spurious_contexts = False
        self.use_nego_context = False
        self.use_nego_prompt = False
        self.use_global_contexts = False

        self.same_diff_temp = max(1e-3, float(getattr(args, 'same_diff_temp', 1.0)))
        self.same_diff_detach_pred = bool(getattr(args, 'same_diff_detach_pred', True))
        self.same_diff_hard = bool(getattr(args, 'same_diff_hard', True))
        self.same_diff_edge_threshold = min(
            max(float(getattr(args, 'same_diff_edge_threshold', 0.5)), 0.0),
            1.0,
        )
        self.same_path_blend = max(0.0, float(getattr(args, 'same_path_blend', 1.0)))
        self.diff_path_blend = max(0.0, float(getattr(args, 'diff_path_blend', 1.0)))
        self.layer_fuse_blend = max(0.0, float(getattr(args, 'same_diff_layer_fuse_blend', 1.0)))
        self.same_diff_gate_blend = min(
            max(float(getattr(args, 'same_diff_gate_blend', 1.0)), 0.0),
            1.0,
        )
        self.same_diff_context_source = getattr(args, 'same_diff_context_source', 'diff')
        if self.same_diff_context_source not in ('diff', 'same_gate', 'mixed'):
            self.same_diff_context_source = 'diff'
        requested_context_k = int(getattr(args, 'same_diff_context_k', 0))
        self.same_diff_context_k = max(0, requested_context_k)
        self.use_same_diff_final_fuse = bool(
            getattr(args, 'use_same_diff_final_fuse', True)
        )
        self.use_same_diff_layerwise_fuse = bool(
            getattr(args, 'use_same_diff_layerwise_fuse', False)
        )
        self.lambda_same_diff_sep = max(
            0.0,
            float(getattr(args, 'lambda_same_diff_sep', 0.05)),
        )

        self.same_path_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d * 2, self.d),
                nn.ReLU(),
                nn.Linear(self.d, self.d),
            )
            for _ in range(self.num_layers)
        ])
        self.same_gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d * 2, self.d),
                nn.ReLU(),
                nn.Linear(self.d, self.d),
            )
            for _ in range(self.num_layers)
        ])
        self.diff_gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d * 2, self.d),
                nn.ReLU(),
                nn.Linear(self.d, self.d),
            )
            for _ in range(self.num_layers)
        ])
        self.diff_path_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d * 2, self.d),
                nn.ReLU(),
                nn.Linear(self.d, self.d),
            )
            for _ in range(self.num_layers)
        ])
        self.layer_fuse_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d * 2, self.d),
                nn.ReLU(),
                nn.Linear(self.d, self.d),
            )
            for _ in range(self.num_layers)
        ])
        self.same_path_norms = nn.ModuleList([nn.LayerNorm(self.d) for _ in range(self.num_layers)])
        self.diff_path_norms = nn.ModuleList([nn.LayerNorm(self.d) for _ in range(self.num_layers)])
        self.layer_fuse_norms = nn.ModuleList([nn.LayerNorm(self.d) for _ in range(self.num_layers)])
        self.same_summary_norms = nn.ModuleList([nn.LayerNorm(self.d) for _ in range(self.num_layers)])
        self.diff_summary_norms = nn.ModuleList([nn.LayerNorm(self.d) for _ in range(self.num_layers)])
        self.final_causal_norm = nn.LayerNorm(self.d)
        self.final_spurious_norm = nn.LayerNorm(self.d)
        self.final_same_context_norm = nn.LayerNorm(self.d)

        self._last_same_edge_weight_mean = None
        self._last_diff_edge_weight_mean = None
        self._last_diff_dim_gate_mean = None
        self._last_same_dim_gate_mean = None
        self._last_same_context_mean = None
        self._last_same_diff_sep_loss = None
        self._last_same_diff_final_fuse_gate_mean = None
        self._last_same_diff_layerwise_fuse_gate_mean = None
        self.reset_same_diff_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'same_path_mlps'):
            self.reset_same_diff_parameters()

    def reset_same_diff_parameters(self):
        for modules in (
            self.same_path_mlps,
            self.same_gate_mlps,
            self.diff_gate_mlps,
            self.diff_path_mlps,
            self.layer_fuse_mlps,
        ):
            for module in modules:
                self._reset_module_parameters(module)
        for norms in (
            self.same_path_norms,
            self.diff_path_norms,
            self.layer_fuse_norms,
            self.same_summary_norms,
            self.diff_summary_norms,
        ):
            for norm in norms:
                norm.reset_parameters()
        self.final_causal_norm.reset_parameters()
        self.final_spurious_norm.reset_parameters()
        self.final_same_context_norm.reset_parameters()
        self._last_same_edge_weight_mean = None
        self._last_diff_edge_weight_mean = None
        self._last_diff_dim_gate_mean = None
        self._last_same_dim_gate_mean = None
        self._last_same_context_mean = None
        self._last_same_diff_sep_loss = None
        self._last_same_diff_final_fuse_gate_mean = None
        self._last_same_diff_layerwise_fuse_gate_mean = None

    def _class_relation_weights(self, h, edge_index):
        src, dst = edge_index
        logits = self.classifier(h)
        if self.same_diff_detach_pred:
            logits = logits.detach()

        if logits.size(-1) <= 1:
            prob_pos = torch.sigmoid(logits / self.same_diff_temp)
            probs = torch.cat([1.0 - prob_pos, prob_pos], dim=-1)
        else:
            probs = F.softmax(logits / self.same_diff_temp, dim=-1)

        if self.same_diff_hard:
            pseudo = probs.argmax(dim=-1)
            same_weight = (pseudo[src] == pseudo[dst]).to(dtype=h.dtype)
        else:
            same_weight = (probs[src] * probs[dst]).sum(dim=-1).clamp(0.0, 1.0)
        diff_weight = (1.0 - same_weight).clamp(0.0, 1.0)
        return same_weight, diff_weight

    def _split_relation_edges(self, edge_index, same_weight):
        if edge_index.numel() == 0:
            return edge_index, edge_index

        threshold = 0.5 if self.same_diff_hard else self.same_diff_edge_threshold
        same_mask = same_weight >= threshold
        diff_mask = ~same_mask
        return edge_index[:, same_mask], edge_index[:, diff_mask]

    def _same_diff_layer(self, h, edge_index, layer_idx, training=False):
        if edge_index.numel() == 0:
            zero = h.new_zeros(h.size())
            one = h.new_ones(h.size())
            return h, h, zero, h, zero, one, h.new_zeros(()), h.new_zeros(()), h.new_zeros(())

        same_weight, diff_weight = self._class_relation_weights(h, edge_index)
        same_edge_index, diff_edge_index = self._split_relation_edges(edge_index, same_weight)

        same_gnn = self.act_fn(self.backbone_layers[layer_idx](h, same_edge_index))
        diff_gnn = self.act_fn(self.backbone_layers[layer_idx](h, diff_edge_index))

        diff_summary = diff_gnn - h
        diff_summary = self.diff_summary_norms[layer_idx](diff_summary)
        diff_dim_gate = torch.sigmoid(
            self.diff_gate_mlps[layer_idx](torch.cat([h, diff_summary], dim=-1))
        )
        gated_diff_summary = diff_summary * diff_dim_gate
        diff_delta = self.diff_path_mlps[layer_idx](torch.cat([h, gated_diff_summary], dim=-1))
        diff_delta = F.dropout(diff_delta, self.dropout, training=training)
        z_diff = self.diff_path_norms[layer_idx](diff_gnn + self.diff_path_blend * diff_delta)

        same_summary = same_gnn - h
        same_summary = self.same_summary_norms[layer_idx](same_summary)
        same_dim_gate = torch.sigmoid(
            self.same_gate_mlps[layer_idx](torch.cat([h, same_summary], dim=-1))
        )
        guided_same_gate = same_dim_gate * (
            (1.0 - self.same_diff_gate_blend)
            + self.same_diff_gate_blend * diff_dim_gate
        )
        guided_same_summary = same_summary * guided_same_gate
        same_context = same_summary * (1.0 - guided_same_gate)
        same_delta = self.same_path_mlps[layer_idx](torch.cat([h, guided_same_summary], dim=-1))
        same_delta = F.dropout(same_delta, self.dropout, training=training)
        z_same = self.same_path_norms[layer_idx](same_gnn + self.same_path_blend * same_delta)

        same_norm = F.normalize(z_same, dim=1)
        diff_norm = F.normalize(z_diff, dim=1)
        same_diff_sep_loss = (same_norm * diff_norm).sum(dim=1).pow(2).mean()

        fused_delta = self.layer_fuse_mlps[layer_idx](torch.cat([h, z_same], dim=-1))
        fused_delta = F.dropout(fused_delta, self.dropout, training=training)
        h_next = self.layer_fuse_norms[layer_idx](h + self.layer_fuse_blend * fused_delta)
        if self.use_same_diff_layerwise_fuse:
            layer_edge_summary, _, layer_edge_gate = self.compute_edge_summaries(
                h_next,
                edge_index,
                training=training,
            )
            h_next = self.fuse_node_edge_representation(
                h_next,
                layer_edge_summary,
                noise_summary=None,
                training=training,
            )
            layerwise_fuse_gate_mean = (
                layer_edge_gate.mean()
                if layer_edge_gate is not None
                else h_next.new_zeros(())
            )
        else:
            layerwise_fuse_gate_mean = h_next.new_zeros(())

        self._last_same_edge_weight_mean = same_weight.mean().detach()
        self._last_diff_edge_weight_mean = diff_weight.mean().detach()
        self._last_diff_dim_gate_mean = diff_dim_gate.mean().detach()
        self._last_same_dim_gate_mean = guided_same_gate.mean().detach()
        self._last_same_context_mean = same_context.mean().detach()
        return (
            h_next,
            z_same,
            z_diff,
            guided_same_summary,
            same_context,
            diff_dim_gate,
            same_weight.mean(),
            same_diff_sep_loss,
            layerwise_fuse_gate_mean,
        )

    def encode_representation(self, x, edge_index, training=False):
        x = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(self.input_proj(x))
        h_pre_enhance = h

        layerwise_gate_loss = h.new_zeros(())
        layerwise_gate_mean = h.new_zeros(())
        same_diff_sep_loss = h.new_zeros(())
        layerwise_fuse_gate_mean = h.new_zeros(())
        layerwise_gate_layers = 0
        z_causal = h
        z_spurious = h.new_zeros(h.size())
        edge_summary = h
        same_context = h.new_zeros(h.size())
        diff_dim_gate = h.new_ones(h.size())

        for layer_idx in range(self.num_layers):
            h = F.dropout(h, self.dropout, training=training)
            (
                h,
                z_causal,
                z_spurious,
                edge_summary,
                same_context,
                diff_dim_gate,
                same_mean,
                sep_loss_l,
                fuse_gate_mean_l,
            ) = self._same_diff_layer(
                h,
                edge_index,
                layer_idx,
                training=training,
            )
            layerwise_gate_mean = layerwise_gate_mean + same_mean
            same_diff_sep_loss = same_diff_sep_loss + sep_loss_l
            layerwise_fuse_gate_mean = layerwise_fuse_gate_mean + fuse_gate_mean_l
            if self.lambda_layerwise_gate > 0.0:
                layerwise_gate_loss = layerwise_gate_loss + (
                    same_mean - self.layerwise_gate_target
                ).pow(2)
            layerwise_gate_layers += 1

        if layerwise_gate_layers > 0:
            layerwise_gate_mean = layerwise_gate_mean / float(layerwise_gate_layers)
            layerwise_gate_loss = layerwise_gate_loss / float(layerwise_gate_layers)
            same_diff_sep_loss = same_diff_sep_loss / float(layerwise_gate_layers)
            layerwise_fuse_gate_mean = layerwise_fuse_gate_mean / float(layerwise_gate_layers)

        self._last_layerwise_gate_mean = layerwise_gate_mean.detach()
        self._last_layerwise_gate_loss = layerwise_gate_loss
        self._last_layerwise_gate_layers = int(layerwise_gate_layers)
        self._last_same_diff_sep_loss = same_diff_sep_loss
        self._last_same_diff_layerwise_fuse_gate_mean = layerwise_fuse_gate_mean.detach()
        self._last_graph_cfam_gate_mean = h.new_zeros(())
        self._last_graph_cfam_gate_loss = h.new_zeros(())
        self._last_graph_cfam_layers = 0

        z_causal = self.final_causal_norm(z_causal)
        z_spurious = self.final_spurious_norm(z_spurious)
        edge_summary = self.final_causal_norm(edge_summary)
        same_context = self.final_same_context_norm(same_context)
        if self.use_same_diff_final_fuse:
            final_edge_summary, _, final_edge_gate = self.compute_edge_summaries(
                z_causal,
                edge_index,
                training=training,
            )
            z_causal = self.fuse_node_edge_representation(
                z_causal,
                final_edge_summary,
                noise_summary=None,
                training=training,
            )
            edge_summary = final_edge_summary
            if final_edge_gate is not None:
                self._last_same_diff_final_fuse_gate_mean = final_edge_gate.mean().detach()
            else:
                self._last_same_diff_final_fuse_gate_mean = z_causal.new_zeros(())
        else:
            self._last_same_diff_final_fuse_gate_mean = z_causal.new_zeros(())
        z_causal = F.dropout(z_causal, self.dropout, training=training)
        z_spurious = F.dropout(z_spurious, self.dropout, training=training)
        edge_summary = F.dropout(edge_summary, self.dropout, training=training)
        same_context = F.dropout(same_context, self.dropout, training=training)
        self._last_same_context = same_context

        z = z_causal
        dag_vars = z.new_zeros(z.size(0), self.non_label_var_dim)
        mediator_gate = z.new_ones(self.d)
        causal_score = z.new_ones(self.dag_latent_dim)
        pollution_score = z.new_zeros(self.dag_latent_dim)
        dag_total = self.A_feat.new_zeros(self.dag_var_dim, self.dag_var_dim)
        cns_gate = torch.full_like(z, 0.5)

        zero = z.new_zeros(())
        self._last_ica_cov_loss = zero
        self._last_ica_ng_loss = zero
        self._last_ica_gate_loss = zero
        self._last_ica_entropy_loss = zero
        self._last_ica_gate_mean = zero
        self._last_lirs_proto_records = []

        mediator_logits = self.classifier(z_causal)
        return (
            z,
            edge_summary,
            dag_vars,
            z_causal,
            z_spurious,
            mediator_logits,
            mediator_gate,
            causal_score,
            pollution_score,
            dag_total,
            None,
            None,
            h_pre_enhance,
            cns_gate,
        )

    def sample_gmm_contexts(self, z_spurious=None, env_probs=None, training=False):
        if z_spurious is None or z_spurious.numel() == 0:
            return None
        return F.normalize(z_spurious, dim=1).unsqueeze(1)

    def sample_other_node_contexts(self, source, training=False):
        if source is None or source.numel() == 0:
            return None

        num_nodes = source.size(0)
        if num_nodes <= 1:
            return F.normalize(source, dim=1).unsqueeze(1)

        sample_k = self.same_diff_context_k
        if sample_k <= 0:
            sample_k = max(2, int(getattr(self, 'fd_sample_k', 0)))
        sample_k = max(1, int(sample_k))

        if training:
            context_idx = torch.randint(
                num_nodes - 1,
                (num_nodes, sample_k),
                device=source.device,
            )
        else:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(self.context_sample_seed + num_nodes + sample_k + 17)
            context_idx = torch.randint(
                num_nodes - 1,
                (num_nodes, sample_k),
                generator=generator,
            ).to(source.device)

        self_idx = torch.arange(num_nodes, device=source.device).view(-1, 1)
        context_idx = context_idx + (context_idx >= self_idx).to(dtype=context_idx.dtype)
        contexts = source.index_select(0, context_idx.reshape(-1)).view(
            num_nodes,
            sample_k,
            source.size(1),
        )
        return F.normalize(contexts, dim=-1)

    def build_same_diff_contexts(self, edge_summary, z_spurious, training=False):
        if self.same_diff_context_source == 'same_gate':
            same_context = getattr(self, '_last_same_context', edge_summary)
            return self.sample_other_node_contexts(same_context, training=training)
        if self.same_diff_context_source == 'mixed':
            same_context = getattr(self, '_last_same_context', edge_summary)
            same_contexts = self.sample_other_node_contexts(same_context, training=training)
            diff_contexts = F.normalize(z_spurious, dim=1).unsqueeze(1)
            if same_contexts is None:
                return diff_contexts
            return torch.cat(
                [
                    same_contexts,
                    diff_contexts,
                ],
                dim=1,
            )
        return F.normalize(z_spurious, dim=1).unsqueeze(1)

    def forward(self, x, edge_index, training=False):
        (
            z,
            edge_summary,
            dag_vars,
            z_causal,
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
        ) = self.encode_representation(x, edge_index, training=training)

        contexts = self.build_same_diff_contexts(edge_summary, z_spurious, training=training)
        fd_logits, fd_stack = self.frontdoor_logits_from_contexts(z_causal, z_spurious, contexts)
        logits = self.blend_logits(mediator_logits, fd_logits)

        if training:
            return (
                logits,
                z,
                edge_summary,
                dag_vars,
                z_causal,
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
        return logits

    def compute_losses(self, data, criterion, args, update_state=False):
        losses = super().compute_losses(data, criterion, args, update_state=update_state)
        zero = losses['total_loss'].new_zeros(())
        loss_same_diff_sep = (
            zero
            if self._last_same_diff_sep_loss is None
            else self._last_same_diff_sep_loss
        )
        losses['loss_same_diff_sep'] = loss_same_diff_sep
        losses['total_loss'] = (
            losses['total_loss']
            + self.lambda_same_diff_sep * loss_same_diff_sep
        )
        return losses
