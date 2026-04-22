import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_sparse import SparseTensor, matmul


def gcn_conv(x, edge_index):
    num_nodes = x.shape[0]
    row, col = edge_index
    deg = degree(col, num_nodes).float()
    deg_in = (1.0 / deg[col]).sqrt()
    deg_out = (1.0 / deg[row]).sqrt()
    value = torch.ones_like(row, dtype=x.dtype) * deg_in * deg_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
    return matmul(adj, x)


class GraphConvolutionBase(nn.Module):
    def __init__(self, in_features, out_features, residual=False):
        super().__init__()
        self.residual = residual
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if self.residual:
            self.weight_r = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.residual:
            self.weight_r.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, x0=None):
        hi = gcn_conv(x, adj)
        output = torch.mm(hi, self.weight)
        if self.residual:
            output = output + torch.mm(x, self.weight_r)
        return output


class CaNetConv(nn.Module):
    """
    Environment-conditioned propagation layer reused from the existing codebase.
    For the GCN backbone, it matches Eq. (11)/(12):
        sum_k e_k [W_D^(k) * A_norm x + W_S^(k) * x]
    """

    def __init__(self, in_features, out_features, num_envs, residual=True, backbone_type="gcn", variant=False, device=None):
        super().__init__()
        self.backbone_type = backbone_type
        self.out_features = out_features
        self.residual = residual
        self.num_envs = num_envs
        self.variant = variant
        self.device = device

        if backbone_type == "gcn":
            self.weights = Parameter(torch.FloatTensor(num_envs, in_features * 2, out_features))
        elif backbone_type == "gat":
            self.leakyrelu = nn.LeakyReLU()
            self.weights = nn.Parameter(torch.zeros(num_envs, in_features, out_features))
            self.a = nn.Parameter(torch.zeros(num_envs, 2 * out_features, 1))
        else:
            raise NotImplementedError(f"Unsupported backbone_type: {backbone_type}")

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weights.data.uniform_(-stdv, stdv)
        if self.backbone_type == "gat":
            nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def specialspmm(self, adj, spm, size, h):
        adj = SparseTensor(row=adj[0], col=adj[1], value=spm, sparse_sizes=size)
        return matmul(adj, h)

    def forward(self, x, adj, env_weights, weights=None):
        if weights is None:
            weights = self.weights

        if self.backbone_type == "gcn":
            if not self.variant:
                hi = gcn_conv(x, adj)
            else:
                adj_sp = torch.sparse_coo_tensor(
                    adj,
                    torch.ones(adj.shape[1], device=x.device, dtype=x.dtype),
                    size=(x.shape[0], x.shape[0]),
                ).coalesce()
                hi = torch.sparse.mm(adj_sp, x)
            hi = torch.cat([hi, x], dim=1)
            hi = hi.unsqueeze(0).repeat(self.num_envs, 1, 1)
            outputs = torch.matmul(hi, weights).transpose(1, 0)
        else:
            xi = x.unsqueeze(0).repeat(self.num_envs, 1, 1)
            h = torch.matmul(xi, weights)
            num_nodes = x.size(0)
            adj_loops, _ = remove_self_loops(adj)
            adj_loops, _ = add_self_loops(adj_loops, num_nodes=num_nodes)
            edge_h = torch.cat((h[:, adj_loops[0, :], :], h[:, adj_loops[1, :], :]), dim=2)
            logits = self.leakyrelu(torch.matmul(edge_h, self.a)).squeeze(2)
            logits_max, _ = torch.max(logits, dim=1, keepdim=True)
            edge_e = torch.exp(logits - logits_max)

            outputs = []
            eps = 1e-8
            ones = torch.ones(num_nodes, 1, device=x.device, dtype=x.dtype)
            for env_id in range(self.num_envs):
                edge_e_k = edge_e[env_id]
                e_expsum_k = self.specialspmm(adj_loops, edge_e_k, torch.Size([num_nodes, num_nodes]), ones) + eps
                hi_k = self.specialspmm(adj_loops, edge_e_k, torch.Size([num_nodes, num_nodes]), h[env_id])
                outputs.append(hi_k / e_expsum_k)
            outputs = torch.stack(outputs, dim=1)

        envs = env_weights.unsqueeze(2).repeat(1, 1, self.out_features)
        output = torch.sum(envs * outputs, dim=1)
        if self.residual:
            output = output + x
        return output


class LinearGraphTransformer(nn.Module):
    """Implements Eq. (3)-(6) for the global environment inference branch."""

    def __init__(self, hidden_channels, num_envs):
        super().__init__()
        self.query_proj = nn.Linear(hidden_channels, hidden_channels)
        self.key_proj = nn.Linear(hidden_channels, hidden_channels)
        self.value_proj = nn.Linear(hidden_channels, hidden_channels)
        self.env_proj = nn.Linear(hidden_channels, num_envs)
        self.reset_parameters()

    def reset_parameters(self):
        for module in (self.query_proj, self.key_proj, self.value_proj, self.env_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        num_nodes = x.size(0)
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        q = q / q.norm(p="fro").clamp_min(1e-12)
        k = k / k.norm(p="fro").clamp_min(1e-12)

        ones = torch.ones(num_nodes, 1, device=x.device, dtype=x.dtype)
        attn_norm = 1.0 + (q @ (k.transpose(0, 1) @ ones)) / max(num_nodes, 1)
        global_repr = v + (q @ (k.transpose(0, 1) @ v)) / max(num_nodes, 1)
        global_repr = global_repr / attn_norm.clamp_min(1e-12)  #N*H
        global_logits = self.env_proj(global_repr) # N*K
        return global_repr, global_logits


class MLEI(nn.Module):
    """
    Multi-Level Environment Inference (AAAI 2025).
    This implementation follows Eq. (3)-(16) in the paper while keeping the
    same external interface as `model.py`.
    """

    def __init__(self, d, c, args, device):
        super().__init__()
        self.hidden_channels = args.hidden_channels
        self.num_layers = args.num_layers
        self.num_envs = args.K
        self.tau = args.tau
        self.dropout = getattr(args, "dropout", 0.0)
        self.backbone_type = args.backbone_type
        self.pred_mode = getattr(args, "mlei_pred_mode", "fused")
        self.device = device

        self.convs = nn.ModuleList()
        self.env_enc = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d, self.hidden_channels))
        self.fcs.append(nn.Linear(self.hidden_channels, c))
        self.global_model = LinearGraphTransformer(self.hidden_channels, self.num_envs)
        # Build K environment-specific global representations before weighting
        # them by the sampled global environment assignments.
        self.global_env_transforms = nn.ModuleList(
            [nn.Linear(self.hidden_channels, self.hidden_channels, bias=False) for _ in range(self.num_envs)]
        )
        self.global_transform = nn.Linear(self.hidden_channels, self.hidden_channels)
        for _ in range(self.num_layers):
            self.convs.append(
                CaNetConv(
                    self.hidden_channels,
                    self.hidden_channels,
                    self.num_envs,
                    backbone_type=self.backbone_type,
                    residual=True,
                    device=device,
                    variant=getattr(args, "variant", False),
                )
            )
            if getattr(args, "env_type", "node") == "node":
                self.env_enc.append(nn.Linear(self.hidden_channels, self.num_envs))
            elif getattr(args, "env_type", "node") == "graph":
                self.env_enc.append(GraphConvolutionBase(self.hidden_channels, self.num_envs, residual=True))
            else:
                raise NotImplementedError(f"Unsupported env_type: {getattr(args, 'env_type', 'node')}")
        self.act_fn = nn.ReLU()
        self.env_type = getattr(args, "env_type", "node")
        self.reset_parameters()

    def reset_parameters(self):
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight)
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)

        self.global_model.reset_parameters()

        for transform in self.global_env_transforms:
            nn.init.xavier_uniform_(transform.weight)

        nn.init.xavier_uniform_(self.global_transform.weight)
        if self.global_transform.bias is not None:
            nn.init.zeros_(self.global_transform.bias)

        for enc in self.env_enc:
            if hasattr(enc, "reset_parameters"):
                enc.reset_parameters()
            else:
                nn.init.xavier_uniform_(enc.weight)
                if enc.bias is not None:
                    nn.init.zeros_(enc.bias)

        for conv in self.convs:
            conv.reset_parameters()

    def _sample_environment(self, logits, training):
        if training:
            return F.gumbel_softmax(logits, tau=self.tau, dim=-1)
        return F.softmax(logits, dim=-1)

    def _regularization_loss(self, env_samples, logits):
        log_pi = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # Strictly matching Eq. (8)/(15): the loss contains the negative of the
        # environment regularizer. Since sum_k e_{v,k} log K is a constant shift,
        # we keep only the trainable part -sum_k e_{v,k} log pi_{v,k}.
        return -torch.mean(torch.sum(env_samples * log_pi, dim=-1))

    def _predict_global(self, global_repr, global_env, training):
        # First produce K environment-specific global representations, then
        # aggregate them with the sampled global environment probabilities.
        global_repr_per_env = torch.stack(
            [transform(global_repr) for transform in self.global_env_transforms],
            dim=1,
        )  # [N, K, H]
        global_context = torch.sum(global_env.unsqueeze(-1) * global_repr_per_env, dim=1)
        global_hidden = self.act_fn(self.global_transform(global_context))
        global_hidden = F.dropout(global_hidden, p=self.dropout, training=training)
        return self.fcs[-1](global_hidden), global_repr_per_env

    def _predict_local(self, local_repr, training):
        local_repr = F.dropout(local_repr, p=self.dropout, training=training)
        return self.fcs[-1](local_repr)

    def _merge_logits(self, local_logits, global_logits, fused_logits, pred_mode=None):
        mode = pred_mode or self.pred_mode
        if mode == "local":
            return local_logits
        if mode == "global":
            return global_logits
        if mode == "fused":
            return fused_logits
        raise ValueError(f"Unsupported mlei_pred_mode: {mode}")

    def forward(self, x, adj, idx=None, training=False, return_details=False, pred_mode=None):
        use_training = training or self.training

        x = F.dropout(x, p=self.dropout, training=use_training)
        h0 = self.act_fn(self.fcs[0](x)) # N*H

        #这里的global_repr对应于原文(4)式中的ZG，global_env_logits对应于(5)式中的经过W相乘后的ZG
        global_repr, global_env_logits = self.global_model(h0)
        global_env = self._sample_environment(global_env_logits, use_training)  # N*K
        global_logits, global_repr_per_env = self._predict_global(global_repr, global_env, use_training)

        # Eq. (11)-(13): each layer first models local relations, then injects the
        # global environment, and finally fuses the two views before the next layer.
        layer_inputs = []
        local_layer_outputs = []
        global_guided_outputs = []
        fused_layer_outputs = []
        local_env_logits_list = []
        local_env_list = []
        fused_repr = h0
        last_local_repr = h0
        last_global_guided_repr = h0
        for layer_id, conv in enumerate(self.convs):
            layer_inputs.append(fused_repr)
            layer_input = F.dropout(fused_repr, p=self.dropout, training=use_training)
            if self.env_type == "node":
                env_logits = self.env_enc[layer_id](layer_input)
            else:
                env_logits = self.env_enc[layer_id](layer_input, adj, h0)
            env_sample = self._sample_environment(env_logits, use_training)

            local_repr = self.act_fn(conv(layer_input, adj, env_sample))
            global_guided_repr = self.act_fn(conv(layer_input, adj, global_env))
            fused_repr = 0.5 * (local_repr + global_guided_repr)

            local_layer_outputs.append(local_repr)
            global_guided_outputs.append(global_guided_repr)
            fused_layer_outputs.append(fused_repr)
            local_env_logits_list.append(env_logits)
            local_env_list.append(env_sample)
            last_global_guided_repr = global_guided_repr
            last_local_repr = local_repr

        local_logits = self._predict_local(last_local_repr, use_training)
        fused_logits = self._predict_local(fused_repr, use_training)

        merged_logits = self._merge_logits(local_logits, global_logits, fused_logits, pred_mode=pred_mode)

        if return_details:
            return {
                "logits": merged_logits,
                "local_logits": local_logits,
                "global_logits": global_logits,
                "fused_logits": fused_logits,
                "local_repr": last_local_repr,
                "stage1_inputs": layer_inputs,
                "stage1_local_outputs": local_layer_outputs,
                "global_guided_outputs": global_guided_outputs,
                "fused_layer_outputs": fused_layer_outputs,
                "last_local_repr": last_local_repr,
                "last_global_guided_repr": last_global_guided_repr,
                "fused_repr": fused_repr,
                "global_repr": global_repr,
                "global_repr_per_env": global_repr_per_env,
                "global_env": global_env,
                "global_env_logits": global_env_logits,
                "local_envs": local_env_list,
                "local_env_logits": local_env_logits_list,
            }
        return merged_logits

    def sup_loss_calc(self, y, pred, criterion, args):
        if args.dataset in ("twitch", "elliptic"):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss

    def loss_compute(self, d, criterion, args):
        details = self.forward(d.x, d.edge_index, idx=d.train_idx, training=True, return_details=True)
        train_idx = d.train_idx

        # Eq. (15): the local-view loss is defined on the fused representation,
        # so the supervision and layer-wise regularization should optimize the
        # same branch.
        local_pred_loss = self.sup_loss_calc(d.y[train_idx], details["fused_logits"][train_idx], criterion, args)
        global_pred_loss = self.sup_loss_calc(d.y[train_idx], details["global_logits"][train_idx], criterion, args)

        local_reg_terms = []
        for env_sample, env_logits in zip(details["local_envs"], details["local_env_logits"]):
            local_reg_terms.append(self._regularization_loss(env_sample[train_idx], env_logits[train_idx]))
        if local_reg_terms:
            local_reg = torch.stack(local_reg_terms).mean()
        else:
            local_reg = torch.tensor(0.0, device=d.x.device)

        global_reg = self._regularization_loss(
            details["global_env"][train_idx],
            details["global_env_logits"][train_idx],
        )

        loss_local = local_pred_loss + local_reg
        loss_global = global_pred_loss + global_reg
        total_loss = loss_local + args.lamda * loss_global
        self._last_loss_breakdown = {
            "total_loss": total_loss.detach(),
            "loss_local": loss_local.detach(),
            "loss_global": loss_global.detach(),
            "local_pred_loss": local_pred_loss.detach(),
            "global_pred_loss": global_pred_loss.detach(),
            "local_reg": local_reg.detach(),
            "global_reg": global_reg.detach(),
        }
        return total_loss


class CaNet(MLEI):
    """Compatibility alias for the existing training pipeline."""

    pass
