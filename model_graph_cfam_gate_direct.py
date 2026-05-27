import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, softmax
from torch_sparse import SparseTensor, matmul


def gcn_backbone_conv(x, edge_index):
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
            raise NotImplementedError("Use backbone_type='gcn' or 'gat'.")
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
            out = torch.matmul(torch.cat([h_neigh, x], dim=1), self.weight)
        else:
            h = torch.matmul(x, self.weight)
            num_nodes = x.size(0)
            att_edge_index, _ = remove_self_loops(edge_index)
            att_edge_index, _ = add_self_loops(att_edge_index, num_nodes=num_nodes)
            edge_h = torch.cat([h[att_edge_index[0]], h[att_edge_index[1]]], dim=1)
            logits = self.leakyrelu(torch.matmul(edge_h, self.att)).squeeze(1)
            alpha = softmax(logits, att_edge_index[1], num_nodes=num_nodes)
            out = self.specialspmm(att_edge_index, alpha, torch.Size([num_nodes, num_nodes]), h)
        return out + x if self.residual else out


class GraphFrontDoorDAG(nn.Module):
    """
    Direct Graph-CFAM gate model.

    Main path:
        x -> GNN -> relation/dimension gate -> enhanced z -> linear classifier

    Removed from the prediction path:
        NeGo, front-door contexts, z_spurious, latent diffusion, DAG/ICA split.

    Kept for the current training script:
        method names and zero-valued diagnostic losses expected by main.
    """

    def __init__(self, d_in, c, args, device):
        super().__init__()
        self.d_in = d_in
        self.c = c
        self.device = device
        self.d = int(getattr(args, 'hidden_channels', 64))
        self.dropout = float(getattr(args, 'dropout', 0.0))
        self.num_layers = int(getattr(args, 'num_layers', 2))
        self.backbone_type = getattr(args, 'backbone_type', getattr(args, 'backbone', 'gcn'))
        self.variant = bool(getattr(args, 'variant', False))
        self.edge_feat_mode = getattr(args, 'edge_feat_mode', 'mul')
        self.edge_relation_model = getattr(args, 'edge_relation_model', 'mlp')
        if self.edge_relation_model not in ('mlp', 'transformer'):
            self.edge_relation_model = 'mlp'
        self.edge_gate_mode = getattr(args, 'edge_gate_mode', 'vector')
        if self.edge_gate_mode not in ('scalar', 'vector'):
            self.edge_gate_mode = 'vector'

        self.edge_score_temp = max(1e-3, float(getattr(args, 'edge_score_temp', 2.0)))
        self.edge_blend = max(0.0, float(getattr(args, 'edge_blend', 0.2)))
        self.nonfeature_blend = max(0.0, float(getattr(args, 'nonfeature_blend', 0.2)))
        self.gate_target = min(max(float(getattr(args, 'layerwise_gate_target', 0.5)), 0.0), 1.0)
        self.lambda_layerwise_gate = max(0.0, float(getattr(args, 'lambda_layerwise_gate', 0.0)))
        self.lambda_layer_pred_var = max(0.0, float(getattr(args, 'lambda_layer_pred_var', 0.0)))
        self.lambda_layer_pred_cls = max(0.0, float(getattr(args, 'lambda_layer_pred_cls', 0.0)))

        self.input_proj = nn.Linear(d_in, self.d)
        self.input_norm = nn.LayerNorm(self.d)
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
        self.act_fn = nn.ReLU()

        edge_in_dim = self._edge_feature_dim()
        gate_out_dim = self.d if self.edge_gate_mode == 'vector' else 1
        self.edge_gate_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, self.d),
            nn.ReLU(),
            nn.Linear(self.d, gate_out_dim),
        )
        heads = max(1, int(getattr(args, 'edge_transformer_heads', 1)))
        self.edge_pair_attn = nn.MultiheadAttention(
            self.d,
            num_heads=heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.edge_pair_gate = nn.Linear(self.d * 4, gate_out_dim)

        self.enhance_norms = nn.ModuleList([nn.LayerNorm(self.d) for _ in range(self.num_layers)])
        self.layer_classifiers = nn.ModuleList([nn.Linear(self.d, c) for _ in range(self.num_layers)])
        self.classifier = nn.Linear(self.d, c)

        self._last_layer_gate_loss = None
        self._last_layer_gate_mean = None
        self._last_layer_pred_var = None
        self._last_layer_pred_cls = None
        self._last_graph_cfam_layers = 0

        self._init_compat_attrs(args)
        self.reset_parameters()

    def _init_compat_attrs(self, args):
        self.use_dag_mixer = False
        self.use_latent_diffusion = False
        self.use_enhanced_as_causal = True
        self.use_ica_split = False
        self.ica_dim = int(getattr(args, 'ica_components', min(16, self.d)))
        self.dag_latent_dim = int(getattr(args, 'dag_latent_dim', min(16, self.d)))
        self.A_feat = Parameter(torch.zeros(self.dag_latent_dim * 2 + self.c, self.dag_latent_dim * 2 + self.c))
        self.register_buffer('counterexample_penalty', torch.zeros(self.dag_latent_dim))
        for name in (
            'lambda_med', 'lambda_fd', 'lambda_fd_aug', 'lambda_ind',
            'lambda_cf',
            'lambda_dag', 'lambda_dag_label', 'lambda_ica_cov',
            'lambda_ica_ng', 'lambda_ica_gate', 'lambda_ica_entropy',
            'lambda_spu', 'lambda_role', 'lambda_spu_y', 'lambda_env',
            'lambda_inv', 'lambda_var', 'lambda_global_env',
            'lambda_entropy_dro',
            'lambda_latent_diffusion', 'lambda_graph_cfam_gate',
            'lambda_graph_delf', 'lambda_enhance_sem', 'lambda_nego',
            'lambda_class_proto_var', 'lambda_class_proto_pos',
            'lambda_class_proto_neg', 'lambda_class_proto_balance',
        ):
            setattr(self, name, 0.0)

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.input_norm.reset_parameters()
        for layer in self.backbone_layers:
            layer.reset_parameters()
        for module in self.edge_gate_mlp:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.edge_pair_attn._reset_parameters()
        self.edge_pair_gate.reset_parameters()
        for norm in self.enhance_norms:
            norm.reset_parameters()
        for clf in self.layer_classifiers:
            clf.reset_parameters()
        self.classifier.reset_parameters()
        nn.init.zeros_(self.A_feat)
        self.counterexample_penalty.zero_()

    def _edge_feature_dim(self):
        mode = self.edge_feat_mode
        dim = self.d
        if mode in ('mul', 'diff', 'signed_diff', 'degree'):
            return dim
        if mode in ('mul_diff', 'mul_signed_diff', 'mul_degree', 'diff_degree'):
            return dim * 2
        if mode in ('concat', 'concat_diff', 'mul_diff_degree', 'mul_signed_diff_degree'):
            return dim * 3
        return dim

    def _normalize_score(self, score, default_value=0.5):
        if score.numel() == 0:
            return score
        min_v = score.min()
        max_v = score.max()
        if torch.isclose(max_v, min_v):
            return torch.full_like(score, float(default_value))
        return (score - min_v) / (max_v - min_v).clamp_min(1e-6)

    def compute_edge_features(self, h, edge_index):
        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]
        mul = h_src * h_dst
        diff = torch.abs(h_src - h_dst)
        signed_diff = h_src - h_dst
        deg_score = degree(dst, h.size(0)).to(device=h.device, dtype=h.dtype)
        deg_feat = self._normalize_score(torch.log1p(deg_score.index_select(0, dst))).unsqueeze(-1)
        deg_feat = deg_feat.expand(-1, self.d)

        mode = self.edge_feat_mode
        if mode == 'diff':
            return diff
        if mode == 'signed_diff':
            return signed_diff
        if mode == 'degree':
            return deg_feat
        if mode == 'mul_diff':
            return torch.cat([mul, diff], dim=-1)
        if mode == 'mul_signed_diff':
            return torch.cat([mul, signed_diff], dim=-1)
        if mode == 'concat':
            return torch.cat([h_src, h_dst, mul], dim=-1)
        if mode == 'concat_diff':
            return torch.cat([h_src, h_dst, diff], dim=-1)
        if mode == 'mul_degree':
            return torch.cat([mul, deg_feat], dim=-1)
        if mode == 'diff_degree':
            return torch.cat([diff, deg_feat], dim=-1)
        if mode == 'mul_diff_degree':
            return torch.cat([mul, diff, deg_feat], dim=-1)
        if mode == 'mul_signed_diff_degree':
            return torch.cat([mul, signed_diff, deg_feat], dim=-1)
        return mul

    def compute_edge_gate(self, h, edge_index):
        if edge_index.numel() == 0:
            shape = (0, self.d) if self.edge_gate_mode == 'vector' else (0, 1)
            return h.new_zeros(shape)

        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]
        if self.edge_relation_model == 'transformer':
            tokens = torch.stack([h_dst, h_src], dim=1)
            pair, _ = self.edge_pair_attn(tokens, tokens, tokens, need_weights=False)
            pair_feat = torch.cat(
                [pair[:, 0], pair[:, 1], pair[:, 0] * pair[:, 1], torch.abs(pair[:, 0] - pair[:, 1])],
                dim=-1,
            )
            logits = self.edge_pair_gate(pair_feat)
        else:
            logits = self.edge_gate_mlp(self.compute_edge_features(h, edge_index))
        gate = torch.sigmoid(logits / self.edge_score_temp)
        if self.edge_gate_mode == 'scalar':
            gate = gate.expand(-1, self.d)
        return gate

    def graph_cfam_adapt(self, h, edge_index, layer_idx, training=False):
        if edge_index.numel() == 0:
            zero = h.new_zeros(h.size())
            gate = h.new_full(h.size(), 0.5)
            return h, zero, zero, gate

        src, dst = edge_index
        gate = self.compute_edge_gate(h, edge_index)
        useful_msg = gate * h[src]
        nonfeature_msg = (1.0 - gate) * h[src]

        useful = h.new_zeros(h.size()).index_add_(0, dst, useful_msg)
        nonfeature = h.new_zeros(h.size()).index_add_(0, dst, nonfeature_msg)
        denom = h.new_zeros(h.size(0), 1).index_add_(
            0,
            dst,
            torch.ones(dst.size(0), 1, device=h.device, dtype=h.dtype),
        ).clamp_min(1.0)
        useful = useful / denom
        nonfeature = nonfeature / denom

        adapted = h + self.edge_blend * useful + self.nonfeature_blend * nonfeature
        adapted = self.enhance_norms[layer_idx](adapted)
        adapted = F.dropout(adapted, self.dropout, training=training)
        return adapted, useful, nonfeature, gate

    def encode_representation(self, x, edge_index, y=None, train_idx=None, criterion=None, args=None, training=False):
        h = self.input_norm(self.input_proj(x))
        layer_gate_losses = []
        layer_gate_means = []
        layer_pred_vars = []
        layer_pred_cls = []

        for layer_idx, layer in enumerate(self.backbone_layers):
            h = F.dropout(h, self.dropout, training=training)
            h = self.act_fn(layer(h, edge_index))
            h, useful, nonfeature, gate = self.graph_cfam_adapt(h, edge_index, layer_idx, training=training)

            layer_gate_means.append(gate.mean())
            layer_gate_losses.append((gate.mean() - self.gate_target).pow(2))

            if training and train_idx is not None and train_idx.numel() > 0:
                full_repr = self.enhance_norms[layer_idx](h + self.edge_blend * nonfeature)
                base_repr = self.enhance_norms[layer_idx](h - self.edge_blend * nonfeature)
                logits_stack = torch.stack(
                    [
                        self.layer_classifiers[layer_idx](h[train_idx]),
                        self.layer_classifiers[layer_idx](full_repr[train_idx]),
                        self.layer_classifiers[layer_idx](base_repr[train_idx]),
                    ],
                    dim=1,
                )
                if logits_stack.size(-1) == 1:
                    pred_stack = torch.sigmoid(logits_stack)
                else:
                    pred_stack = F.softmax(logits_stack, dim=-1)
                layer_pred_vars.append(pred_stack.var(dim=1, unbiased=False).mean())
                if y is not None and criterion is not None and self.lambda_layer_pred_cls > 0.0:
                    layer_pred_cls.append(self.compute_supervised_loss(
                        logits_stack[:, 0, :],
                        y[train_idx],
                        criterion,
                        args,
                    ).mean())

        zero = h.new_zeros(())
        self._last_layer_gate_loss = torch.stack(layer_gate_losses).mean() if layer_gate_losses else zero
        self._last_layer_gate_mean = torch.stack(layer_gate_means).mean() if layer_gate_means else zero
        self._last_layer_pred_var = torch.stack(layer_pred_vars).mean() if layer_pred_vars else zero
        self._last_layer_pred_cls = torch.stack(layer_pred_cls).mean() if layer_pred_cls else zero
        self._last_graph_cfam_layers = len(layer_gate_means)
        return h

    def forward(self, x, edge_index, training=False):
        z = self.encode_representation(x, edge_index, training=training)
        return self.classifier(z)

    def compute_supervised_loss(self, out, target, criterion, args):
        if target.dim() > 1 and target.size(-1) == 1:
            target = target.squeeze(-1)
        if out.size(-1) == 1:
            return criterion(out, target.float().view_as(out))
        return criterion(out, target.long())

    def compute_losses(self, data, criterion, args, update_state=False):
        x, edge_index, y = data.x, data.edge_index, data.y.to(data.x.device)
        train_idx = data.train_idx.to(device=x.device, dtype=torch.long)
        z = self.encode_representation(
            x,
            edge_index,
            y=y,
            train_idx=train_idx,
            criterion=criterion,
            args=args,
            training=True,
        )
        logits = self.classifier(z)
        raw_loss = self.compute_supervised_loss(logits[train_idx], y[train_idx], criterion, args)
        loss_cls = raw_loss.mean()
        zero = loss_cls.new_zeros(())
        loss_layerwise_gate = self._last_layer_gate_loss if self._last_layer_gate_loss is not None else zero
        loss_layer_pred_var = self._last_layer_pred_var if self._last_layer_pred_var is not None else zero
        loss_layer_pred_cls = self._last_layer_pred_cls if self._last_layer_pred_cls is not None else zero
        total_loss = (
            loss_cls
            + self.lambda_layerwise_gate * loss_layerwise_gate
            + self.lambda_layer_pred_var * loss_layer_pred_var
            + self.lambda_layer_pred_cls * loss_layer_pred_cls
        )
        return self._loss_dict(total_loss, loss_cls, loss_layerwise_gate, loss_layer_pred_var, loss_layer_pred_cls, zero, x)

    def _loss_dict(self, total_loss, loss_cls, loss_layerwise_gate, loss_layer_pred_var, loss_layer_pred_cls, zero, x):
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_cls_mean': loss_cls.detach(),
            'loss_med': zero,
            'loss_fd': zero,
            'loss_fd_mean': zero,
            'loss_cf': zero,
            'loss_fd_aug': zero,
            'loss_var': loss_layer_pred_var,
            'loss_ind': zero,
            'loss_dag': zero,
            'loss_dag_label': zero,
            'loss_ica_cov': zero,
            'loss_ica_ng': zero,
            'loss_ica_gate': zero,
            'loss_ica_entropy': zero,
            'loss_role': zero,
            'loss_role_med_y': zero,
            'loss_role_spu_y': zero,
            'loss_role_spu_env': zero,
            'loss_role_med_env': zero,
            'loss_sem': zero,
            'loss_degree': zero,
            'loss_spu_y': zero,
            'loss_spu': zero,
            'loss_env_med': zero,
            'loss_inv': zero,
            'loss_global_env': zero,
            'loss_diffusion': zero,
            'loss_cns': zero,
            'loss_cns_cons': zero,
            'loss_layerwise_gate': loss_layerwise_gate,
            'loss_graph_cfam_gate': zero,
            'loss_graph_delf': zero,
            'loss_enhance_sem': zero,
            'loss_nego': zero,
            'loss_class_proto': zero,
            'loss_class_proto_var': zero,
            'loss_class_proto_pos': zero,
            'loss_class_proto_neg': zero,
            'loss_class_proto_balance': zero,
            'nego_extra_score': zero,
            'nego_self_score': zero,
            'class_proto_assign_entropy': zero,
            'num_class_env_protos': torch.tensor(0.0, device=x.device),
            'dro_weight_entropy': zero,
            'dro_max_weight': zero,
            'cns_complement_mean': zero,
            'cns_gate_mean': zero,
            'cns_layer_complement_mean': zero,
            'cns_layer_gate_mean': zero,
            'cns_layer_layers': torch.tensor(0.0, device=x.device),
            'layerwise_gate_mean': (
                zero if self._last_layer_gate_mean is None else self._last_layer_gate_mean.detach()
            ),
            'layerwise_gate_layers': torch.tensor(float(self._last_graph_cfam_layers), device=x.device),
            'graph_cfam_gate_mean': (
                zero if self._last_layer_gate_mean is None else self._last_layer_gate_mean.detach()
            ),
            'graph_cfam_layers': torch.tensor(float(self._last_graph_cfam_layers), device=x.device),
            'mediator_gate_mean': (
                zero if self._last_layer_gate_mean is None else self._last_layer_gate_mean.detach()
            ),
            'ica_gate_mean': zero,
            'causal_score_mean': zero,
            'pollution_score_mean': zero,
            'counterexample_penalty_mean': self.counterexample_penalty.mean().detach(),
            'counterexample_penalty_batch_mean': zero,
            'cf_pred_shift': zero,
            'num_contexts': torch.tensor(0.0, device=x.device),
            'num_mixed_contexts': torch.tensor(0.0, device=x.device),
            'num_gmm_contexts': torch.tensor(0.0, device=x.device),
            'num_global_contexts': torch.tensor(0.0, device=x.device),
            'num_layerwise_contexts': torch.tensor(0.0, device=x.device),
            'num_nego_contexts': torch.tensor(0.0, device=x.device),
            'state_payload': None,
        }

    def apply_state_update(self, state_payload):
        return None

    def loss_compute(self, data, criterion, args):
        losses = self.compute_losses(data, criterion, args, update_state=True)
        return losses['total_loss'], losses['loss_cls'].item(), 0.0, 0.0, 0.0
