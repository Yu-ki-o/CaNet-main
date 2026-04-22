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
    """
    对应 CaNet 中 mixture-of-expert GNN predictor 的基础图传播算子。
    这里先做一阶邻居聚合，再交给环境相关的专家权重进行加权组合。
    """
    N = x.shape[0]
    row, col = edge_index #这里是无向图，每条边出现两次
    d = degree(col, N).float() #计算节点的度数
    #计算每条边的权重，边连接两端节点度数大的边的权重反而小
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    value = torch.ones_like(row) * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    #节点吸收邻居节点信息生成新节点表示，等价于消息传递
    return matmul(adj, x) # [N, D]

class GraphConvolutionBase(nn.Module):
    """
    对应论文中的 environment estimator q_φ(E|G) 里的图结构编码器版本。
    当 `env_type == graph` 时，用图卷积把当前节点表示映射成环境 logits。
    """

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
        self.weight_r.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, x0):
        hi = gcn_conv(x, adj)
        output = torch.mm(hi, self.weight)
        if self.residual:
            output = output + torch.mm(x, self.weight_r)
        return output

class CaNetConv(nn.Module):
    """
    对应论文中的 mixture-of-expert GNN predictor p_θ(Y | G, E)。

    其核心思想是：
    1. 为 K 个潜在环境分别维护一组专家参数；
    2. 先对每个环境独立计算一份邻居传播结果；
    3. 再用环境估计器输出的环境分布 e 对 K 个专家输出做加权求和。
    """

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
            # GCN 骨干：先做标准图卷积邻居聚合，再和节点自身表示拼接，
            # 对应论文里“feature propagation units conditioned on environments”。
            if not self.variant:
                hi = gcn_conv(x, adj)
            else:#和gcn_conv差个归一化，高度数节点的影响会更大
                adj = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]).to(self.device), size=(x.shape[0],x.shape[0])).to(self.device)
                hi = torch.sparse.mm(adj, x)
            hi = torch.cat([hi, x], 1)
            hi = hi.unsqueeze(0).repeat(self.K, 1, 1)  # [K, N, D*2]
            outputs = torch.matmul(hi, weights) # [K, N, D]
            outputs = outputs.transpose(1, 0)  # [N, K, D]
        elif self.backbone_type == 'gat':
            # GAT 骨干：每个潜在环境对应一套独立注意力参数，
            # 先分别计算环境特定的 attention aggregation，再由 e 做环境加权。
            xi = x.unsqueeze(0).repeat(self.K, 1, 1)  # [K, N, D]
            h = torch.matmul(xi, weights) # [K, N, D]
            N = x.size()[0]
            adj, _ = remove_self_loops(adj)
            adj, _ = add_self_loops(adj, num_nodes=N)
            edge_h = torch.cat((h[:, adj[0, :], :], h[:, adj[1, :], :]), dim=2)  # [K, E, 2*D]
            logits = self.leakyrelu(torch.matmul(edge_h, self.a)).squeeze(2)
            logits_max , _ = torch.max(logits, dim=1, keepdim=True)
            edge_e = torch.exp(logits-logits_max)  # [K, E]

            outputs = []
            eps = 1e-8
            for k in range(self.K):
                edge_e_k = edge_e[k, :] # [E]
                e_expsum_k = self.specialspmm(adj, edge_e_k, torch.Size([N, N]), torch.ones(N, 1).cuda()) + eps
                assert not torch.isnan(e_expsum_k).any()

                hi_k = self.specialspmm(adj, edge_e_k, torch.Size([N, N]), h[k])
                hi_k = torch.div(hi_k, e_expsum_k)  # [N, D]
                outputs.append(hi_k)
            outputs = torch.stack(outputs, dim=1) # [N, K, D]

        es = e.unsqueeze(2).repeat(1, 1, self.out_features)  # [N, K, D]
        # 对应论文里的 backdoor-adjustment 近似实现：
        # 用环境分布 e 对 K 个环境专家的输出进行加权组合。
        output = torch.sum(torch.mul(es, outputs), dim=1)  # [N, D]

        if self.residual:
            output = output + x

        return output

class CaNet(nn.Module):
    """
    CaNet 主体对应论文中的两个核心部分：
    1. environment estimator：逐层推断节点所属的潜在环境分布；
    2. mixture-of-expert GNN predictor：在环境条件下做图传播并预测标签。

    整体训练目标是学习 p_θ(Y | do(G)) 的可计算近似，
    通过环境估计 + 环境条件专家传播来削弱环境混杂偏差。
    """
    def __init__(self, d, c, args, device):
        super(CaNet, self).__init__()
        # 论文中的 feature propagation units：每层都是一个环境条件专家卷积。
        self.convs = nn.ModuleList()
        for _ in range(args.num_layers):
            self.convs.append(CaNetConv(args.hidden_channels, args.hidden_channels, args.K, backbone_type=args.backbone_type, residual=True, device=device, variant=args.variant))
        # 两层 MLP：输入映射 + 最终标签预测头。
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d, args.hidden_channels))
        self.fcs.append(nn.Linear(args.hidden_channels, c))
        # 论文中的 environment estimator q_φ(E|G)。
        # node: 仅基于节点表示推断环境；
        # graph: 结合图结构信息推断环境。
        self.env_enc = nn.ModuleList()
        for _ in range(args.num_layers):
            if args.env_type == 'node':
                self.env_enc.append(nn.Linear(args.hidden_channels, args.K))
            elif args.env_type == 'graph':
                self.env_enc.append(GraphConvolutionBase(args.hidden_channels, args.K, residual=True))
            else:
                raise NotImplementedError
        self.act_fn = nn.ReLU()
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.tau = args.tau
        self.env_type = args.env_type
        self.device = device

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        for enc in self.env_enc:
            enc.reset_parameters()

    def forward(self, x, adj, idx=None, training=False):
        self.training = training
        # 输入编码：把原始节点特征映射到隐空间，作为环境估计器和专家传播器的共同输入。
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x))
        h0 = h.clone()

        reg = 0
        for i,con in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            if self.training:
                # 逐层环境推断：得到环境 logits，再通过 Gumbel-Softmax 近似采样环境分布 e。
                if self.env_type == 'node':
                    logit = self.env_enc[i](h)
                else:
                    logit = self.env_enc[i](h, adj, h0)
                e = F.gumbel_softmax(logit, tau=self.tau, dim=-1)
                # 对应论文 ELBO / regularization 项中的环境分布约束。
                reg += self.reg_loss(e, logit)
            else:
                # 测试时使用 soft assignment，而不是采样。
                if self.env_type == 'node':
                    e = F.softmax(self.env_enc[i](h), dim=-1)
                else:
                    e = F.softmax(self.env_enc[i](h, adj, h0), dim=-1)
            # 环境条件专家传播：由 e 选择并组合 K 个环境专家的输出。
            h = self.act_fn(con(h, adj, e))

        h = F.dropout(h, self.dropout, training=self.training)
        out = self.fcs[-1](h)
        if self.training:
            return out, reg / self.num_layers
        else:
            return out

    def reg_loss(self, z, logit, logit_0 = None):
        # 对应论文中 environment estimator 的 regularization term，
        # 作用是避免环境分布塌缩到单一环境。
        log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, logit.size(1))
        return torch.mean(torch.sum(
            torch.mul(z, log_pi), dim=1))

    def sup_loss_calc(self, y, pred, criterion, args):
        # 对应论文中的 predictive term，即对目标标签 Y 的监督学习项。
        if args.dataset in ('twitch', 'elliptic'):
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
        # 论文训练目标的可执行版本：
        # 监督预测损失 + 环境分布正则项（由 lamda 控制强度）。
        logits, reg_loss = self.forward(d.x, d.edge_index, idx=d.train_idx, training=True)
        sup_loss = self.sup_loss_calc(d.y[d.train_idx], logits[d.train_idx], criterion, args)
        loss = sup_loss + args.lamda * reg_loss
        return loss
