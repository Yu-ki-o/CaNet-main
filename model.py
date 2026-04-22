import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops, add_self_loops, degree, add_remaining_self_loops
from data_utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
from torch_sparse import SparseTensor, matmul

def gcn_conv(x, edge_index):
    N = x.shape[0]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    value = torch.ones_like(row) * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    return matmul(adj, x) # [N, D]

class GraphConvolutionBase(nn.Module):
    def __init__(self, in_features, out_features, residual=False):
        super(GraphConvolutionBase, self).__init__()
        self.residual = residual
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if self.residual:
            self.weight_r = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
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
    # 复用原版算子：当 K=1 且传入 dummy e 时，退化为标准的 GCN/GAT 骨干网络
    def __init__(self, in_features, out_features, K, residual=True, backbone_type='gcn', variant=False, device=None):
        super(CaNetConv, self).__init__()
        self.backbone_type = backbone_type
        self.out_features = out_features
        self.residual = residual
        if backbone_type == 'gcn':
            self.weights = Parameter(torch.FloatTensor(K, in_features*2, out_features))
        elif backbone_type == 'gat':
            self.leakyrelu = nn.LeakyReLU()
            self.weights = nn.Parameter(torch.zeros(K, in_features, out_features))
            self.a = nn.Parameter(torch.zeros(K, 2 * out_features, 1))
        self.K = K
        self.device = device
        self.variant = variant
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weights.data.uniform_(-stdv, stdv)
        if self.backbone_type == 'gat':
            nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def specialspmm(self, adj, spm, size, h):
        adj = SparseTensor(row=adj[0], col=adj[1], value=spm, sparse_sizes=size)
        return matmul(adj, h)

    def forward(self, x, adj, e, weights=None):
        if weights == None:
            weights = self.weights
        if self.backbone_type == 'gcn':
            if not self.variant:
                hi = gcn_conv(x, adj)
            else:
                adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]).to(self.device), size=(x.shape[0],x.shape[0])).to(self.device)
                hi = torch.sparse.mm(adj_sp, x)
            hi = torch.cat([hi, x], 1)
            hi = hi.unsqueeze(0).repeat(self.K, 1, 1)  # [K, N, D*2]
            outputs = torch.matmul(hi, weights) # [K, N, D]
            outputs = outputs.transpose(1, 0)  # [N, K, D]
        elif self.backbone_type == 'gat':
            xi = x.unsqueeze(0).repeat(self.K, 1, 1)  # [K, N, D]
            h = torch.matmul(xi, weights) # [K, N, D]
            N = x.size()[0]
            adj_loops, _ = remove_self_loops(adj)
            adj_loops, _ = add_self_loops(adj_loops, num_nodes=N)
            edge_h = torch.cat((h[:, adj_loops[0, :], :], h[:, adj_loops[1, :], :]), dim=2)  # [K, E, 2*D]
            logits = self.leakyrelu(torch.matmul(edge_h, self.a)).squeeze(2)
            logits_max , _ = torch.max(logits, dim=1, keepdim=True)
            edge_e = torch.exp(logits-logits_max)  # [K, E]

            outputs = []
            eps = 1e-8
            for k in range(self.K):
                edge_e_k = edge_e[k, :] # [E]
                e_expsum_k = self.specialspmm(adj_loops, edge_e_k, torch.Size([N, N]), torch.ones(N, 1).to(self.device)) + eps
                
                hi_k = self.specialspmm(adj_loops, edge_e_k, torch.Size([N, N]), h[k])
                hi_k = torch.div(hi_k, e_expsum_k)  # [N, D]
                outputs.append(hi_k)
            outputs = torch.stack(outputs, dim=1) # [N, K, D]

        es = e.unsqueeze(2).repeat(1, 1, self.out_features)  # [N, K, D]
        output = torch.sum(torch.mul(es, outputs), dim=1)  # [N, D]

        if self.residual:
            output = output + x

        return output


class ExpertNet(nn.Module):
    """
    单环境专家网络 (Single-Environment Expert Network)
    作为集成架构的基石模块，专门提取特定环境分布下的节点特征。
    """
    def __init__(self, d, c, args, device):
        super(ExpertNet, self).__init__()
        self.convs = nn.ModuleList()
        # 强制 K=1，使其成为无环境干预的标准骨干网络
        for _ in range(args.num_layers):
            self.convs.append(CaNetConv(args.hidden_channels, args.hidden_channels, K=1, backbone_type=args.backbone_type, residual=True, device=device, variant=args.variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d, args.hidden_channels))
        self.fcs.append(nn.Linear(args.hidden_channels, c))
        self.act_fn = nn.ReLU()
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.device = device

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, adj, training=None):
        if training is None:
            training = self.training
        x = F.dropout(x, self.dropout, training=training)
        h = self.act_fn(self.fcs[0](x))

        # 构建一个全为 1 的 Dummy 权重，以适配 CaNetConv 的输入维度
        e_dummy = torch.ones(x.shape[0], 1).to(self.device)

        for con in self.convs:
            h = F.dropout(h, self.dropout, training=training)
            h = self.act_fn(con(h, adj, e_dummy))

        h = F.dropout(h, self.dropout, training=training)
        out = self.fcs[-1](h)
        return out


class CaNet(nn.Module):
    """
    因果多专家均值集成模型 (Causal Multi-Expert Mean Ensemble)
    直接替代原版 CaNet 的接口，包含 N 个并行专家。
    训练时：基于物理标签计算 [自身环境损失] + [跨环境惩罚损失]。
    推理时：无参数算术平均 (等价于因果后门调整)。
    """
    def __init__(self, d, c, args, device):
        super(CaNet, self).__init__()
        # 借助启动脚本中的 --K 参数来决定并行训练的专家数量
        self.num_experts = args.K
        self.expert_agg = getattr(args, 'expert_agg', 'mean')
        self.experts = nn.ModuleList([ExpertNet(d, c, args, device) for _ in range(self.num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(d, args.gate_hidden),
            nn.ReLU(),
            nn.Linear(args.gate_hidden, self.num_experts),
        )
        self.device = device

    def reset_parameters(self):
        for expert in self.experts:
            expert.reset_parameters()
        for module in self.gate:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def compute_expert_weights(self, x):
        gate_logits = self.gate(x)
        return torch.softmax(gate_logits, dim=-1)

    def forward(self, x, adj, idx=None, training=False):
        # 1. 专家独立推断 (Independent Inference)
        logits_list = [expert(x, adj, training=training) for expert in self.experts]

        logits_stack = torch.stack(logits_list, dim=0) # [num_experts, N, C]

        if self.expert_agg == 'gate':
            expert_weights = self.compute_expert_weights(x)  # [N, K]
            weighted_logits = torch.einsum('nk,knc->nc', expert_weights, logits_stack)
            return weighted_logits

        # 2. 后门均值组合 (Backdoor Adjustment via Arithmetic Mean)
        avg_logits = torch.mean(logits_stack, dim=0)   # [N, C]
        return avg_logits

    def sup_loss_calc(self, y, pred, criterion, args):
        # 与原版保持一致的监督损失计算逻辑
        if args.dataset in ('twitch', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            target = y.squeeze(1)
            # CrossEntropyLoss expects raw logits (it internally applies log_softmax).
            loss = criterion(pred, target)
        return loss

    def loss_compute(self, d, criterion, args):
        total_loss = 0.0
        
        # 获取当前训练集中存在的所有唯一物理环境标签
        unique_envs = torch.unique(d.env[d.train_idx])
        num_unique_envs = len(unique_envs)

        # 收集所有专家在训练模式下的 Logits
        logits_list = [expert(d.x, d.edge_index, training=True) for expert in self.experts]

        # 复用原脚本中的 --lamda 作为跨环境惩罚项的超参数权重 alpha
        alpha_other = args.lamda 

        for i, expert_logits in enumerate(logits_list):
            # 动态映射：将第 i 个专家绑定到某一个真实的物理环境上
            env_id = unique_envs[i % num_unique_envs]

            # ==========================================
            # 模块 A: 自身专属环境上的拟合损失 (L_self)
            # ==========================================
            env_self_mask = (d.env == env_id)[d.train_idx]
            train_idx_self = d.train_idx[env_self_mask]
            
            loss_self = 0.0
            if len(train_idx_self) > 0:
                loss_self = self.sup_loss_calc(d.y[train_idx_self], expert_logits[train_idx_self], criterion, args)

            # ==========================================
            # 模块 B: 在其余偏移环境上的惩罚损失 (L_other)
            # ==========================================
            env_other_mask = (d.env != env_id)[d.train_idx]
            train_idx_other = d.train_idx[env_other_mask]
            
            loss_other = 0.0
            if len(train_idx_other) > 0:
                if getattr(args, 'other_env_reduce', 'sample') == 'env':
                    other_env_losses = []
                    for other_env_id in unique_envs:
                        if other_env_id == env_id:
                            continue
                        env_k_mask = (d.env == other_env_id)[d.train_idx]
                        train_idx_k = d.train_idx[env_k_mask]
                        if len(train_idx_k) > 0:
                            loss_k = self.sup_loss_calc(d.y[train_idx_k], expert_logits[train_idx_k], criterion, args)
                            other_env_losses.append(loss_k)
                    if len(other_env_losses) > 0:
                        loss_other = torch.stack(other_env_losses).mean()
                else:
                    loss_other = self.sup_loss_calc(d.y[train_idx_other], expert_logits[train_idx_other], criterion, args)

            # 将专家的损失整合到全局总损失中
            total_loss += (loss_self + alpha_other * loss_other)

        # 返回所有专家的平均聚合损失以反向传播
        return total_loss / self.num_experts
