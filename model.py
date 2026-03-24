import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops, add_self_loops, degree, add_remaining_self_loops
from data_utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
from torch_sparse import SparseTensor, matmul


class CrossAlignedEnvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(CrossAlignedEnvEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # --- 1. 局部视图提取 (Local GCN) ---
        self.weight_local = Parameter(torch.FloatTensor(in_channels, in_channels))
        
        # --- 2. 全局视图提取 (Self-Attention) ---
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(in_channels)
        
        # --- 3. 跨视图交叉注意力 (Cross-Attention Alignment) ---
        # 灵感来自论文 3.2 节：用一种视图去 Query 另一种视图
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(in_channels)
        
        # --- 4. 融合门控 (Gate) ---
        # 替代简单的 alpha 加权，用非线性映射融合净化后的特征
        self.gate = nn.Linear(in_channels * 2, in_channels)
        
        # 伪环境分类器
        self.fc = nn.Linear(in_channels, out_channels)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_channels)
        self.weight_local.data.uniform_(-stdv, stdv)
        self.self_attn._reset_parameters()
        self.cross_attn._reset_parameters()
        self.layer_norm1.reset_parameters()
        self.layer_norm2.reset_parameters()
        self.gate.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, adj, x0=None):
        # --- A. 获取局部视图 (含有拓扑噪声) ---
        hi = gcn_conv(x, adj)
        local_embed = torch.mm(hi, self.weight_local)  # [N, D]
        
        # --- B. 获取全局视图 (纯净的宏观特征) ---
        x_seq = x.unsqueeze(0)  # [1, N, D]
        attn_out, _ = self.self_attn(x_seq, x_seq, x_seq)
        global_embed = self.layer_norm1(x_seq + attn_out)  # [1, N, D]
        
        # --- C. 交叉注意力净化 (The Magic Happens Here) ---
        # Q = Global, K = Local, V = Local
        # 让纯净的全局特征去“挑选”有用的局部特征，过滤掉误导性的拓扑边
        local_seq = local_embed.unsqueeze(0)  # [1, N, D]
        cross_out, cross_weights = self.cross_attn(query=global_embed, key=local_seq, value=local_seq)
        
        # 得到被全局语义“净化”后的局部特征
        aligned_embed = self.layer_norm2(global_embed + cross_out).squeeze(0)  # [N, D]
        global_embed_sq = global_embed.squeeze(0)  # [N, D]
        
        # --- D. 最终融合 ---
        # 拼接原始全局特征和被净化后的跨视图特征，通过 MLP 融合
        concat_embed = torch.cat([global_embed_sq, aligned_embed], dim=-1)  # [N, 2D]
        final_embed = torch.relu(self.gate(concat_embed))  # [N, D]
        
        return self.fc(final_embed)


# --- 2. 局部 GCN + 全局 Transformer 融合模块 (适用于小图: Cora, Citeseer) ---
class LocalGlobalEnvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(LocalGlobalEnvEncoder, self).__init__()
        self.in_channels = in_channels
        self.weight_local = Parameter(torch.FloatTensor(in_channels, in_channels))
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(in_channels)
        self.alpha = nn.Parameter(torch.tensor(0.0)) # 自适应融合权重
        self.fc = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_channels)
        self.weight_local.data.uniform_(-stdv, stdv)
        self.self_attn._reset_parameters()
        self.layer_norm.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, adj, x0=None):
        hi = gcn_conv(x, adj) 
        local_embed = torch.mm(hi, self.weight_local)
        
        x_seq = x.unsqueeze(0) 
        attn_out, _ = self.self_attn(x_seq, x_seq, x_seq)
        global_embed = self.layer_norm(x_seq + attn_out).squeeze(0) 
        
        weight = torch.sigmoid(self.alpha)
        combined_embed = weight * local_embed + (1.0 - weight) * global_embed
        return self.fc(combined_embed)

    def get_fusion_weights(self):
        local_w = torch.sigmoid(self.alpha).item()
        return local_w, 1.0 - local_w

# --- 方案 A: 纯虚拟节点 (Pure VN) - 完全无视连边，只用节点自身特征融合全局大盘 ---
class PureVirtualNodeEnvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PureVirtualNodeEnvEncoder, self).__init__()
        self.in_channels = in_channels
        
        # 处理全局特征的 MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )
        # 处理节点自身特征的 MLP (替代了 GCN)
        self.node_mlp = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.global_mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.node_mlp.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, adj=None, x0=None):
        # 1. 节点自身表示 (绝对不使用 adj)
        node_embed = self.node_mlp(x)
        
        # 2. 全局宏观表示
        global_pool = x.mean(dim=0, keepdim=True) 
        global_embed = self.global_mlp(global_pool).expand_as(x)
        
        # 3. 直接相加融合 (纯粹的前馈网络)
        combined_embed = node_embed + global_embed
        return self.fc(combined_embed)

# --- 方案 B: 结合版虚拟节点 (Combined VN) - 局部 GCN 结合 全局大盘，带自适应权重 ---
class CombinedVirtualNodeEnvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CombinedVirtualNodeEnvEncoder, self).__init__()
        self.in_channels = in_channels
        
        # 局部 GCN 权重
        self.weight_local = Parameter(torch.FloatTensor(in_channels, in_channels))
        # 全局 MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )
        # 自适应权重
        self.alpha = nn.Parameter(torch.tensor(0.0)) 
        self.fc = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_channels)
        self.weight_local.data.uniform_(-stdv, stdv)
        for layer in self.global_mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, adj, x0=None):
        # 1. 局部 GCN 表示 (使用 adj)
        hi = gcn_conv(x, adj)
        local_embed = torch.mm(hi, self.weight_local)
        
        # 2. 全局宏观表示
        global_pool = x.mean(dim=0, keepdim=True) 
        global_embed = self.global_mlp(global_pool).expand_as(x)
        
        # 3. 自适应加权融合
        weight = torch.sigmoid(self.alpha)
        combined_embed = weight * local_embed + (1.0 - weight) * global_embed
        return self.fc(combined_embed)

    def get_fusion_weights(self):
        local_w = torch.sigmoid(self.alpha).item()
        return local_w, 1.0 - local_w


#---------------------------------------------
#新增transformer模块
import torch.nn as nn
import torch.nn.functional as F

class GlobalTransformerEnvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(GlobalTransformerEnvEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 使用 PyTorch 原生的多头自注意力机制
        # batch_first=True 表示输入形状为 [Batch, Seq_len, Features]
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(in_channels)
        
        # 映射到 K 个伪环境的分类器
        self.fc = nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.self_attn._reset_parameters()
        self.layer_norm.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, adj=None, x0=None):
        # 原始节点特征 x 形状: [N, D]
        # 在 Transformer 中，我们需要将其视为 Batch Size 为 1，序列长度为 N 的序列: [1, N, D]
        x_seq = x.unsqueeze(0)
        
        # 全局自注意力计算：每个节点与整张图上的所有节点计算 Attention
        # 这里不传入 adj 掩码，意味着感受野是整张全图
        attn_out, _ = self.self_attn(x_seq, x_seq, x_seq)
        
        # 残差连接 + LayerNorm
        out = self.layer_norm(x_seq + attn_out)
        out = out.squeeze(0)  # 还原形状回 [N, D]
        
        # 输出 K 个伪环境的 Logits
        logits = self.fc(out)  # [N, K]
        return logits

#---------------------------------------------------

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
        self.weight_r.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, x0):
        hi = gcn_conv(x, adj)
        output = torch.mm(hi, self.weight)
        if self.residual:
            output = output + torch.mm(x, self.weight_r)
        return output

class CaNetConv(nn.Module):

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
                adj = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]).to(self.device), size=(x.shape[0],x.shape[0])).to(self.device)
                hi = torch.sparse.mm(adj, x)
            hi = torch.cat([hi, x], 1)
            hi = hi.unsqueeze(0).repeat(self.K, 1, 1)  # [K, N, D*2]
            outputs = torch.matmul(hi, weights) # [K, N, D]
            outputs = outputs.transpose(1, 0)  # [N, K, D]
        elif self.backbone_type == 'gat':
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
        output = torch.sum(torch.mul(es, outputs), dim=1)  # [N, D]

        if self.residual:
            output = output + x

        return output

class CaNet(nn.Module):
    def __init__(self, d, c, args, device):
        super(CaNet, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(args.num_layers):
            self.convs.append(CaNetConv(args.hidden_channels, args.hidden_channels, args.K, backbone_type=args.backbone_type, residual=True, device=device, variant=args.variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d, args.hidden_channels))
        self.fcs.append(nn.Linear(args.hidden_channels, c))
        self.env_enc = nn.ModuleList()
        for _ in range(args.num_layers):
            if args.env_type == 'node':
                self.env_enc.append(nn.Linear(args.hidden_channels, args.K))
            elif args.env_type == 'graph':
                self.env_enc.append(GraphConvolutionBase(args.hidden_channels, args.K, residual=True))
            #新增transformer分支
            elif args.env_type == 'transformer':
                # 新增 Transformer 分支，默认使用 4 个注意力头
                self.env_enc.append(GlobalTransformerEnvEncoder(args.hidden_channels, args.K, num_heads=4))
            elif args.env_type == 'local_global':
                self.env_enc.append(LocalGlobalEnvEncoder(args.hidden_channels, args.K, num_heads=4))
            elif args.env_type == 'pure_vn':
                self.env_enc.append(PureVirtualNodeEnvEncoder(args.hidden_channels, args.K))
            elif args.env_type == 'combined_vn':
                self.env_enc.append(CombinedVirtualNodeEnvEncoder(args.hidden_channels, args.K))
            elif args.env_type == 'cross_align':
                self.env_enc.append(CrossAlignedEnvEncoder(args.hidden_channels, args.K, num_heads=4))
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
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x))
        h0 = h.clone()

        reg = 0
        for i,con in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            if self.training:
                if self.env_type == 'node':
                    logit = self.env_enc[i](h)
                else:
                    logit = self.env_enc[i](h, adj, h0)
                e = F.gumbel_softmax(logit, tau=self.tau, dim=-1)
                reg += self.reg_loss(e, logit)
            else:
                if self.env_type == 'node':
                    e = F.softmax(self.env_enc[i](h), dim=-1)
                else:
                    e = F.softmax(self.env_enc[i](h, adj, h0), dim=-1)
            h = self.act_fn(con(h, adj, e))

        h = F.dropout(h, self.dropout, training=self.training)
        out = self.fcs[-1](h)
        if self.training:
            return out, reg / self.num_layers
        else:
            return out

    def reg_loss(self, z, logit, logit_0 = None):
        log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, logit.size(1))
        return torch.mean(torch.sum(
            torch.mul(z, log_pi), dim=1))

    def sup_loss_calc(self, y, pred, criterion, args):
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
        logits, reg_loss = self.forward(d.x, d.edge_index, idx=d.train_idx, training=True)
        sup_loss = self.sup_loss_calc(d.y[d.train_idx], logits[d.train_idx], criterion, args)
        loss = sup_loss + args.lamda * reg_loss
        return loss