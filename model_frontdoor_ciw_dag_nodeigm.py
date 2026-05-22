"""
NodeIGM-style causal-edge front-door CIW-DAG model.

This variant keeps the original DAG-CIW + front-door objective as the main
framework, but changes the graph that the DAG/front-door encoder sees:

  1) run a probe GNN pass on the original graph;
  2) score every edge using the endpoint node representations;
  3) split edges into causal and environment edges, with optional high-degree
     protection inspired by NodeIGM;
  4) run the original GraphFrontDoorCIWDAG on the learned causal subgraph;
  5) optionally mix environment edges back into the causal subgraph and add a
     representation-stability loss.

The edge splitter is intentionally graph-structure level, not a final-logit
fusion branch. The final prediction is still produced by the inherited
DAG-CIW/front-door path.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_frontdoor_ciw_dag import GraphFrontDoorCIWDAG


class EdgeImportanceMLP(nn.Module):
    def __init__(self, hidden_dim, mlp_hidden=None, dropout=0.0):
        super().__init__()
        mlp_hidden = int(mlp_hidden or hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(4 * hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mlp_hidden, max(mlp_hidden // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(mlp_hidden // 2, 8), 1),
        )

    def forward(self, z, edge_index):
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]
        edge_feat = torch.cat(
            [z_src, z_dst, torch.abs(z_src - z_dst), z_src * z_dst],
            dim=-1,
        )
        logits = self.net(edge_feat).squeeze(-1)
        scores = torch.sigmoid(logits)
        return logits, scores


class GraphFrontDoorNodeIGMDAG(GraphFrontDoorCIWDAG):
    """GraphFrontDoorCIWDAG with NodeIGM-style causal edge filtering."""

    def __init__(self, d_in, c, args, device):
        super().__init__(d_in, c, args, device)

        self.use_nodeigm_edges = bool(getattr(args, 'use_nodeigm_edges', True))
        self.use_causal_edges_for_eval = bool(getattr(args, 'use_causal_edges_for_eval', True))
        self.use_env_edge_mixup = bool(getattr(args, 'use_env_edge_mixup', True))

        self.nodeigm_hidden_dim = int(getattr(args, 'nodeigm_hidden_dim', getattr(args, 'hidden_channels', 64)))
        self.edge_score_threshold = float(getattr(args, 'edge_score_threshold', 0.5))
        self.degree_threshold = float(getattr(args, 'degree_threshold', 50.0))
        self.min_causal_edge_ratio = float(getattr(args, 'min_causal_edge_ratio', 0.15))
        self.max_causal_edge_ratio = float(getattr(args, 'max_causal_edge_ratio', 1.0))
        self.target_causal_edge_ratio = float(getattr(args, 'target_causal_edge_ratio', 0.65))

        self.lambda_edge_ratio = float(getattr(args, 'lambda_edge_ratio', 0.01))
        self.lambda_edge_label = float(getattr(args, 'lambda_edge_label', 0.02))
        self.lambda_envmix = float(getattr(args, 'lambda_envmix', 0.1))
        self.lambda_envmix_vrex = float(getattr(args, 'lambda_envmix_vrex', 0.1))
        self.envmix_ratios = self._parse_ratios(getattr(args, 'envmix_ratios', '0.1,1.0'))
        self.edge_label_min_edges = int(getattr(args, 'edge_label_min_edges', 16))
        self.current_epoch = 0
        self.total_epochs = int(getattr(args, 'epochs', 1))

        self.edge_importance = EdgeImportanceMLP(
            hidden_dim=self.nodeigm_hidden_dim,
            mlp_hidden=int(getattr(args, 'nodeigm_edge_mlp_hidden', self.nodeigm_hidden_dim)),
            dropout=float(getattr(args, 'dropout', 0.0)),
        )

        self._reset_nodeigm_modules()

    @staticmethod
    def _parse_ratios(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        out = []
        for item in str(value).split(','):
            item = item.strip()
            if item:
                out.append(float(item))
        return out

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'edge_importance'):
            self._reset_nodeigm_modules()

    def _reset_nodeigm_modules(self):
        for module in self.edge_importance.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def _rich_forward(self, x, edge_index):
        out = super().forward(x, edge_index, training=True)
        if torch.is_tensor(out):
            raise RuntimeError(
                'Expected rich tuple output from GraphFrontDoorDAG.forward(..., training=True), '
                'but got a tensor. Please check the parent forward signature.'
            )
        return out

    def _unpack_rich(self, out):
        return {
            'base_logits': out[0],
            'z_all': out[1],
            'dag_vars_all': out[3],
            'z_mediator_all': out[4],
            'z_spurious_all': out[5],
            'mediator_gate': out[6],
            'causal_score': out[7],
            'pollution_score': out[8],
            'dag_total': out[9],
            'mediator_logits_all': out[10],
        }

    def _node_degrees(self, edge_index, num_nodes):
        src, dst = edge_index
        deg = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float)
        deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        return deg

    def _topk_min_mask(self, scores, mask, min_ratio, max_ratio):
        num_edges = scores.numel()
        if num_edges == 0:
            return mask
        min_keep = int(math.ceil(max(0.0, min_ratio) * num_edges))
        max_keep = int(math.ceil(max(0.0, min(max_ratio, 1.0)) * num_edges))
        max_keep = max(max_keep, min_keep, 1)
        current = int(mask.sum().item())
        if current < min_keep:
            k = min(min_keep, num_edges)
            top_idx = torch.topk(scores, k=k, largest=True).indices
            add_mask = torch.zeros_like(mask)
            add_mask[top_idx] = True
            mask = mask | add_mask
        if max_ratio < 1.0 and int(mask.sum().item()) > max_keep:
            # Keep protected edges first, then fill remaining slots by score.
            selected_idx = mask.nonzero(as_tuple=False).view(-1)
            selected_scores = scores[selected_idx]
            keep_local = torch.topk(selected_scores, k=max_keep, largest=True).indices
            new_mask = torch.zeros_like(mask)
            new_mask[selected_idx[keep_local]] = True
            mask = new_mask
        return mask

    def extract_causal_env_edges(self, x, edge_index, z_probe=None, detach_probe=False):
        """Return causal/env edge split and diagnostics.

        z_probe should usually be the node representation from a probe full-graph
        pass. The split is hard, but the score MLP is trained with edge-level
        ratio and train-label homophily losses.
        """
        if z_probe is None:
            probe_out = self._rich_forward(x, edge_index)
            z_probe = self._unpack_rich(probe_out)['z_all']
        if detach_probe:
            z_for_score = z_probe.detach()
        else:
            z_for_score = z_probe

        if z_for_score.dim() != 2 or z_for_score.size(-1) != self.nodeigm_hidden_dim:
            raise RuntimeError(
                f'NodeIGM edge scorer expected z dimension {self.nodeigm_hidden_dim}, '
                f'but got {tuple(z_for_score.shape)}. Pass --nodeigm_hidden_dim {z_for_score.size(-1)}.'
            )

        edge_logits, edge_scores = self.edge_importance(z_for_score, edge_index)
        src, dst = edge_index
        deg = self._node_degrees(edge_index, z_for_score.size(0))
        if self.degree_threshold > 0:
            high_degree_mask = (deg[src] > self.degree_threshold) | (deg[dst] > self.degree_threshold)
        else:
            high_degree_mask = torch.zeros_like(edge_scores, dtype=torch.bool)

        learned_mask = edge_scores > self.edge_score_threshold
        causal_mask = learned_mask | high_degree_mask
        causal_mask = self._topk_min_mask(edge_scores, causal_mask, self.min_causal_edge_ratio, self.max_causal_edge_ratio)
        env_mask = ~causal_mask

        if int(causal_mask.sum().item()) == 0:
            causal_mask = torch.ones_like(causal_mask, dtype=torch.bool)
            env_mask = ~causal_mask

        causal_edge_index = edge_index[:, causal_mask]
        env_edge_index = edge_index[:, env_mask]

        return {
            'causal_edge_index': causal_edge_index,
            'env_edge_index': env_edge_index,
            'edge_logits': edge_logits,
            'edge_scores': edge_scores,
            'causal_mask': causal_mask,
            'env_mask': env_mask,
            'high_degree_mask': high_degree_mask,
            'degree': deg,
        }

    def _edge_label_loss(self, edge_logits, edge_index, y, train_idx):
        if self.lambda_edge_label <= 0.0 or edge_logits.numel() == 0:
            return edge_logits.new_zeros(())
        src, dst = edge_index
        train_mask = torch.zeros(y.size(0), device=y.device, dtype=torch.bool)
        train_mask[train_idx] = True
        edge_train = train_mask[src] & train_mask[dst]
        if int(edge_train.sum().item()) < self.edge_label_min_edges:
            return edge_logits.new_zeros(())

        y_flat = y
        if y_flat.dim() > 1 and y_flat.size(1) > 1:
            y_cls = y_flat.argmax(dim=1)
        else:
            y_cls = y_flat.view(-1).long()
        target = (y_cls[src[edge_train]] == y_cls[dst[edge_train]]).float()
        logits = edge_logits[edge_train]
        return F.binary_cross_entropy_with_logits(logits, target)

    def _edge_ratio_loss(self, edge_scores):
        if self.lambda_edge_ratio <= 0.0 or edge_scores.numel() == 0:
            return edge_scores.new_zeros(())
        mean_score = edge_scores.mean()
        target = edge_scores.new_tensor(self.target_causal_edge_ratio)
        return (mean_score - target).pow(2)

    def _sample_env_edges(self, env_edge_index, ratio):
        num_env = env_edge_index.size(1)
        if num_env == 0 or ratio <= 0:
            return env_edge_index[:, :0]
        k = int(math.ceil(float(ratio) * num_env))
        k = max(1, min(k, num_env))
        perm = torch.randperm(num_env, device=env_edge_index.device)[:k]
        return env_edge_index[:, perm]

    def _mixed_edge_index(self, causal_edge_index, env_edge_index, ratio):
        sampled = self._sample_env_edges(env_edge_index, ratio)
        if sampled.size(1) == 0:
            return causal_edge_index
        return torch.cat([causal_edge_index, sampled], dim=1)

    def _env_edge_mixup_loss(self, x, causal_edge_index, env_edge_index, z_ref, train_idx):
        if (not self.use_env_edge_mixup) or self.lambda_envmix <= 0.0 or len(self.envmix_ratios) == 0:
            return z_ref.new_zeros(()), z_ref.new_zeros(()), 0
        if env_edge_index.size(1) == 0:
            return z_ref.new_zeros(()), z_ref.new_zeros(()), 0

        risks = []
        for ratio in self.envmix_ratios:
            mixed_edge_index = self._mixed_edge_index(causal_edge_index, env_edge_index, ratio)
            mixed_out = self._rich_forward(x, mixed_edge_index)
            mixed = self._unpack_rich(mixed_out)
            # Compare mediator representations, not raw logits, so the DAG/front-door
            # representation is stable when environment edges are mixed back in.
            z_mixed = mixed['z_mediator_all']
            risk = F.mse_loss(z_mixed[train_idx], z_ref[train_idx].detach())
            risks.append(risk)
        if not risks:
            return z_ref.new_zeros(()), z_ref.new_zeros(()), 0
        risk_tensor = torch.stack(risks)
        mean_risk = risk_tensor.mean()
        vrex = risk_tensor.var(unbiased=False) if risk_tensor.numel() > 1 else z_ref.new_zeros(())
        return mean_risk, vrex, len(risks)

    def forward(self, x, edge_index, training=False):
        # For standard training rich output, compute_losses calls _rich_forward directly.
        # For evaluation, return logits from the causal subgraph so eval.py remains unchanged.
        if training or (not self.use_nodeigm_edges) or (not self.use_causal_edges_for_eval):
            return super().forward(x, edge_index, training=training)

        probe_out = self._rich_forward(x, edge_index)
        z_probe = self._unpack_rich(probe_out)['z_all']
        split = self.extract_causal_env_edges(x, edge_index, z_probe=z_probe, detach_probe=True)
        return super().forward(x, split['causal_edge_index'], training=False)

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx

        # 1) Probe full graph and learn/split causal vs environment edges.
        probe_out = self._rich_forward(x, edge_index)
        probe = self._unpack_rich(probe_out)
        if self.use_nodeigm_edges:
            split = self.extract_causal_env_edges(x, edge_index, z_probe=probe['z_all'], detach_probe=False)
            causal_edge_index = split['causal_edge_index']
            env_edge_index = split['env_edge_index']
        else:
            split = None
            causal_edge_index = edge_index
            env_edge_index = edge_index[:, :0]

        # 2) Main DAG-CIW/front-door path runs on the causal subgraph.
        causal_out = self._rich_forward(x, causal_edge_index)
        unpacked = self._unpack_rich(causal_out)
        z_all = unpacked['z_all']
        dag_vars_all = unpacked['dag_vars_all']
        z_mediator_all = unpacked['z_mediator_all']
        z_spurious_all = unpacked['z_spurious_all']
        mediator_gate = unpacked['mediator_gate']
        causal_score = unpacked['causal_score']
        pollution_score = unpacked['pollution_score']
        dag_total = unpacked['dag_total']
        mediator_logits_all = unpacked['mediator_logits_all']

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

        if split is not None:
            loss_edge_ratio = self._edge_ratio_loss(split['edge_scores'])
            loss_edge_label = self._edge_label_loss(split['edge_logits'], edge_index, y, train_idx)
            loss_envmix, loss_envmix_vrex, num_edge_mix_contexts = self._env_edge_mixup_loss(
                x, causal_edge_index, env_edge_index, z_mediator_all, train_idx
            )
            causal_edge_ratio = split['causal_mask'].float().mean().detach()
            high_degree_ratio = split['high_degree_mask'].float().mean().detach()
            edge_score_mean = split['edge_scores'].mean().detach()
        else:
            loss_edge_ratio = zero
            loss_edge_label = zero
            loss_envmix = zero
            loss_envmix_vrex = zero
            num_edge_mix_contexts = 0
            causal_edge_ratio = torch.tensor(1.0, device=x.device)
            high_degree_ratio = torch.tensor(0.0, device=x.device)
            edge_score_mean = torch.tensor(1.0, device=x.device)

        total_loss = (
            loss_cls
            + self.lambda_fd * loss_fd
            + self.lambda_dag * loss_dag
            + self.lambda_dag_rec * loss_dag_rec
            + self.lambda_spu * loss_spu
            + self.lambda_env * loss_env_med
            + self.lambda_causal_mask_cls * loss_causal_mask_cls
            + self.lambda_edge_ratio * loss_edge_ratio
            + self.lambda_edge_label * loss_edge_label
            + self.lambda_envmix * (loss_envmix + self.lambda_envmix_vrex * loss_envmix_vrex)
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
            'loss_edge_ratio': loss_edge_ratio,
            'loss_edge_label': loss_edge_label,
            'loss_envmix': loss_envmix,
            'loss_envmix_vrex': loss_envmix_vrex,
            'mediator_gate_mean': mediator_gate.mean().detach(),
            'causal_score_mean': causal_score.mean().detach(),
            'pollution_score_mean': pollution_score.mean().detach(),
            'num_contexts': torch.tensor(float(num_contexts), device=x.device),
            'num_mixed_contexts': torch.tensor(float(num_edge_mix_contexts), device=x.device),
            'num_gmm_contexts': torch.tensor(float(num_gmm_contexts), device=x.device),
            'edge_score_mean': edge_score_mean,
            'causal_edge_ratio': causal_edge_ratio,
            'high_degree_edge_ratio': high_degree_ratio,
            'num_causal_edges': torch.tensor(float(causal_edge_index.size(1)), device=x.device),
            'num_env_edges': torch.tensor(float(env_edge_index.size(1)), device=x.device),
            'state_payload': state_payload,
        }
