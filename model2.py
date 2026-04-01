import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops, add_self_loops, degree, add_remaining_self_loops
from data_utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
from torch_sparse import SparseTensor, matmul

#CIW论文模块
class CIWModule(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(CIWModule, self).__init__()
        
        # 1. 样本权重生成器 (利用原始输入特征生成权重，避免干扰深层语义)
        self.weight_net = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus() # 保证权重严格为正
        )
        
        # 2. 因果效应评估向量
        # 初始化为 0 (经过 sigmoid 后起步为 0.5，平等看待所有特征)
        self.causal_logits = nn.Parameter(torch.zeros(hidden_channels))

    def reset_parameters(self):
        """重置 CIW 模块的权重"""
        # 1. 重置样本权重生成网络
        for m in self.weight_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
        # 2. 重置因果效应评估向量 (归零，使其经过 sigmoid 后恢复到公平的 0.5)
        nn.init.zeros_(self.causal_logits)

    def forward(self, x_raw, h_init, idx=None):
        # 1. 计算因果效应 Ca
        Ca = torch.sigmoid(self.causal_logits)
        
        # 2. 生成强度引导掩码 S (【核心修复】：必须使用 detach 截断梯度，防止网络作弊！)
        Ca_detached = Ca.detach()
        S = 1.0 - torch.ger(Ca_detached, Ca_detached)
        
        # 3. 生成节点初始权重
        w = self.weight_net(x_raw) + 1e-5 # [N_all, 1]
        
        ciw_loss = torch.tensor(0.0, device=x_raw.device)
        w_norm_train = None
        
        if self.training and idx is not None:
            # 4. 【杜绝数据泄露】：只对当前传入的训练集节点进行权重归一化
            w_train = w[idx]
            w_norm_train = w_train / (w_train.mean() + 1e-8)
            
            # 提取训练集的 hidden 特征，并使用 GELU 非线性激活模拟 RFF 空间
            h_train = h_init[idx]
            h_nl = F.gelu(h_train)
            
            # 5. 计算加权协方差矩阵
            N_train = h_train.shape[0]
            # 计算加权均值
            mean_w = torch.sum(w_norm_train * h_nl, dim=0, keepdim=True) / N_train
            h_centered = h_nl - mean_w
            # 加权协方差: (X - \mu)^T * diag(W) * (X - \mu)
            cov_w = torch.matmul(h_centered.T, w_norm_train * h_centered) / (N_train - 1 + 1e-8)
            
            # 6. 施加差异化掩码
            masked_cov = S * cov_w
            
            # 7. 【尺度修复】：计算剔除对角线后的 Mean Square Error，防止 Loss 爆炸
            D = cov_w.shape[0]
            eye = torch.eye(D, device=x_raw.device)
            off_diagonal_mask = 1.0 - eye
            
            # 除以非对角线元素的总个数 D * (D - 1)
            ciw_loss = torch.sum((masked_cov * off_diagonal_mask) ** 2) / (D * (D - 1) + 1e-8)
            
        return w_norm_train, Ca, ciw_loss


# ================= 新增：端到端因果解耦模块 =================
class End2EndCausalDisentangler(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(End2EndCausalDisentangler, self).__init__()
        
        # 1. 非线性编码器：吸收原始离散/连续特征的非线性流形
        self.nonlinear_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )
        
        # 2. 纯线性解混层：等效于 ICA 解混矩阵 W (禁止使用 bias 和激活函数)
        self.unmixing_layer = nn.Linear(hidden_channels, out_channels, bias=False)

    def forward(self, x):
        h = self.nonlinear_encoder(x)
        z = self.unmixing_layer(h)
        return z

    def compute_ica_loss(self, z):
        """计算数值稳定且无泄露的可导 ICA 损失"""
        # 1. 零均值化
        z_centered = z - z.mean(dim=0)
        
        # 2. 协方差惩罚 (正交化与单位方差约束)
        # 使用 max 防止 batch_size 为 1 时的除零错误
        N = max(z.shape[0], 2)
        cov = torch.mm(z_centered.T, z_centered) / (N - 1)
        eye = torch.eye(cov.shape[0], device=z.device)
        cov_loss = F.mse_loss(cov, eye)
        
        # 3. 负熵最大化 (使用数值极度稳定的 Log-Cosh 展开式)
        z_abs = z_centered.abs()
        g_z = z_abs - math.log(2.0) + torch.log1p(torch.exp(-2.0 * z_abs))
        g_z_mean = g_z.mean(dim=0)
        
        # 标准高斯分布的 log-cosh 期望约等于 0.3745
        negentropy = (g_z_mean - 0.3745).pow(2).mean()
        
        # 4. 最终可导 ICA 损失 (强约束协方差，寻找独立方向)
        ica_loss = 10.0 * cov_loss - 1.0 * negentropy 
        return ica_loss
# ============================================================



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
    
#这里是对应CIW模块时的修改  
class CaNet(nn.Module):
    def __init__(self, d, c, args, device):
        super(CaNet, self).__init__()
        
        # 1. 基础特征映射
        self.proj = nn.Linear(d, args.hidden_channels)
        
        # 2. 挂载 CIW 差异化因果解耦模块 (确保你在上方定义了 CIWModule)
        self.ciw = CIWModule(in_channels=d, hidden_channels=args.hidden_channels)
        
        # 3. CaNet 混合专家卷积层
        self.convs = nn.ModuleList()
        
        # 4. 全连接预测层
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(args.hidden_channels, args.hidden_channels))
        self.fcs.append(nn.Linear(args.hidden_channels, c))
        
        # 5. 环境推断器容器
        self.env_enc = nn.ModuleList()
        
        # 联合初始化 convs 和 env_enc
        for _ in range(args.num_layers):
            self.convs.append(CaNetConv(args.hidden_channels, args.hidden_channels, args.K, backbone_type=args.backbone_type, residual=True, device=device, variant=args.variant))
            
            if args.env_type == 'node':
                self.env_enc.append(nn.Linear(args.hidden_channels, args.K))
            elif args.env_type == 'graph':
                self.env_enc.append(GraphConvolutionBase(args.hidden_channels, args.K, residual=True))
            # 新增 transformer 分支
            elif args.env_type == 'transformer':
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
        """重置整个 CaNet 及所有子模块的权重"""
        # 1. 重置基础特征投影层
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
            
        # 2. 重置我们新加的 CIW 模块
        self.ciw.reset_parameters()

        # 3. 重置所有的 MoE 图卷积层
        for conv in self.convs:
            conv.reset_parameters()

        # 4. 重置全连接预测层
        for fc in self.fcs:
            if isinstance(fc, nn.Linear):
                nn.init.xavier_uniform_(fc.weight)
                if fc.bias is not None:
                    nn.init.zeros_(fc.bias)

        # 5. 重置环境推断器 (兼容各种复杂的 EnvEncoder)
        for enc in self.env_enc:
            if hasattr(enc, 'reset_parameters'):
                enc.reset_parameters()
            else:
                # 兼容简单的 nn.Linear 推断器
                for m in enc.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

    def forward(self, x, adj, idx=None, training=False):
        self.training = training
        
        # 1. 基础映射
        h_init = self.act_fn(self.proj(x))
        
        # 2. 通过 CIW 模块获取：训练集样本权重、因果效应向量、差异化解耦损失
        w_norm_train, Ca, ciw_loss = self.ciw(x, h_init, idx if training else None)
        
        # 3. 特征过滤/提纯：保留因果强度高的特征，过滤环境噪音
        h_causal = h_init * Ca.unsqueeze(0)
        
        # 4. 将提纯后的纯净因果特征送入 CaNet 进行图路由
        x_dropout = F.dropout(h_causal, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x_dropout))
        h0 = h.clone()

        reg = 0
        for i, con in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            
            # 【核心修复】：根据 env_type 动态决定传入多少个参数
            if self.env_type == 'node':
                logit = self.env_enc[i](h)
            else:
                # graph, transformer 等高级模式，需要提供邻接矩阵 adj 和初始特征 h0
                logit = self.env_enc[i](h, adj, h0)
            
            # 环境推断与 Gumbel-Softmax 采样
            if self.training:
                e = F.gumbel_softmax(logit, tau=self.tau, dim=-1)
                reg += self.reg_loss(e, logit)
            else:
                e = F.softmax(logit, dim=-1)
                
            # 使用推断出的伪环境 e 加权专家网络
            h = self.act_fn(con(h, adj, e))

        h = F.dropout(h, self.dropout, training=self.training)
        out = self.fcs[-1](h)
        
        if self.training:
            # 返回包含了训练集归一化权重 w_norm_train 和因果效应 Ca 的结果
            return out, reg / self.num_layers, ciw_loss, w_norm_train, Ca
        else:
            return out

    def reg_loss(self, z, logit, logit_0 = None):
        """CaNet 的伪环境分布正则化，强迫接近均匀分布"""
        log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, logit.size(1))
        return torch.mean(torch.sum(torch.mul(z, log_pi), dim=1))

    def sup_loss_calc_weighted(self, y, pred, w_norm_train, criterion, args):
        """支持 CIW 样本权重的加权分类损失 (精准靶向消除混淆偏差)"""
        if args.dataset in ('twitch', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = F.binary_cross_entropy_with_logits(pred, true_label.squeeze(1).to(torch.float), reduction='none')
            if len(loss.shape) > 1:
                loss = loss.mean(dim=1)
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            # 使用 reduction='none' 提取出每个节点的 loss
            loss = F.nll_loss(out, target, reduction='none')
            
        # CIW 核心：用 batch 内均值为 1 的权重去乘每个样本的 loss，然后再求均值
        weighted_loss = (loss * w_norm_train.squeeze()).mean()
        return weighted_loss

    def loss_compute(self, dataset, criterion, args):
        """四大多任务联合优化引擎"""
        logits, reg_loss, ciw_loss, w_norm_train, Ca = self.forward(
            dataset.x, dataset.edge_index, idx=dataset.train_idx, training=True
        )
        
        # 1. 核心任务：加权预测损失 (精准传入前向传播计算好的 w_norm_train)
        sup_loss = self.sup_loss_calc_weighted(
            dataset.y[dataset.train_idx], 
            logits[dataset.train_idx], 
            w_norm_train, 
            criterion, args
        )
        
        # 2. L1 稀疏正则化 (防作弊紧箍咒：迫使无用的特征维度 Ca 降为 0)
        l1_loss = torch.norm(Ca, p=1)
        
        # 3. 动态配置权重 (建议在 main.py 的 args 中增加这几个超参数以方便调优)
        lamda_reg = args.lamda  # 默认的 CaNet 环境正则化系数
        lamda_ciw = getattr(args, 'lamda_ciw', 1.0) # 建议调优区间: 0.1 ~ 2.0
        lamda_l1 = getattr(args, 'lamda_l1', 1e-4)  # 建议保持在 1e-4 ~ 1e-3
        
        # 终极联合优化：加权分类 + 伪环境正则 + CIW差异化解耦 + L1稀疏
        loss = sup_loss + lamda_reg * reg_loss + lamda_ciw * ciw_loss + lamda_l1 * l1_loss
        return loss


# class CaNet(nn.Module):
#     def __init__(self, d, c, args, device):
#         super(CaNet, self).__init__()
#         # 下面是CIW模块时对应的修改
#         self.proj = nn.Linear(d, args.hidden_channels)
#         self.ciw = CIWModule(in_channels=d, hidden_channels=args.hidden_channels)

#         # 实例化端到端因果解耦器 (将原始维度 d 映射到 args.hidden_channels)
#         # self.causal_disentangler = End2EndCausalDisentangler(
#         #     in_channels=d, 
#         #     hidden_channels=256,
#         #     out_channels=args.hidden_channels 
#         # )
#         self.convs = nn.ModuleList()
#         for _ in range(args.num_layers):
#             self.convs.append(CaNetConv(args.hidden_channels, args.hidden_channels, args.K, backbone_type=args.backbone_type, residual=True, device=device, variant=args.variant))
#         self.fcs = nn.ModuleList()
#         #这里修改了，原本前面的维度是d，现在是args.hidden_channels
#         self.fcs.append(nn.Linear(d, args.hidden_channels))
#         self.fcs.append(nn.Linear(args.hidden_channels, c))
#         self.env_enc = nn.ModuleList()
#         for _ in range(args.num_layers):
#             if args.env_type == 'node':
#                 self.env_enc.append(nn.Linear(args.hidden_channels, args.K))
#             elif args.env_type == 'graph':
#                 self.env_enc.append(GraphConvolutionBase(args.hidden_channels, args.K, residual=True))
#             #新增transformer分支
#             elif args.env_type == 'transformer':
#                 # 新增 Transformer 分支，默认使用 4 个注意力头
#                 self.env_enc.append(GlobalTransformerEnvEncoder(args.hidden_channels, args.K, num_heads=4))
#             elif args.env_type == 'local_global':
#                 self.env_enc.append(LocalGlobalEnvEncoder(args.hidden_channels, args.K, num_heads=4))
#             elif args.env_type == 'pure_vn':
#                 self.env_enc.append(PureVirtualNodeEnvEncoder(args.hidden_channels, args.K))
#             elif args.env_type == 'combined_vn':
#                 self.env_enc.append(CombinedVirtualNodeEnvEncoder(args.hidden_channels, args.K))
#             elif args.env_type == 'cross_align':
#                 self.env_enc.append(CrossAlignedEnvEncoder(args.hidden_channels, args.K, num_heads=4))
#             else:
#                 raise NotImplementedError
#         self.act_fn = nn.ReLU()
#         self.dropout = args.dropout
#         self.num_layers = args.num_layers
#         self.tau = args.tau
#         self.env_type = args.env_type
#         self.device = device

#     def reset_parameters(self):
#         # 增加 disentangler 的重置
#         for m in self.causal_disentangler.modules():
#             if isinstance(m, nn.Linear):
#                 m.reset_parameters()
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.reset_parameters()
#         for conv in self.convs:
#             conv.reset_parameters()
#         for fc in self.fcs:
#             fc.reset_parameters()
#         for enc in self.env_enc:
#             enc.reset_parameters()
#     def forward(self, x, adj, idx=None, training=False):
#         self.training = training
        
#         # 1. 原始特征流经非线性解耦器，得到纯净的独立潜变量 Z
#         z = self.causal_disentangler(x)
        
#         ica_loss = torch.tensor(0.0, device=x.device)
#         if self.training and idx is not None:
#             # 【杜绝数据泄露】：严格只在训练节点 (idx) 上计算 ICA 独立性约束
#             z_train = z[idx]
#             ica_loss = self.causal_disentangler.compute_ica_loss(z_train)
            
#         # 2. 将纯净的 Z 送入后续的 GNN 处理流程
#         x_dropout = F.dropout(z, self.dropout, training=self.training)
#         h = self.act_fn(self.fcs[0](x_dropout))
#         h0 = h.clone()

#         reg = 0
#         for i, con in enumerate(self.convs):
#             h = F.dropout(h, self.dropout, training=self.training)
#             if self.training:
#                 if self.env_type == 'node':
#                     logit = self.env_enc[i](h)
#                 else:
#                     logit = self.env_enc[i](h, adj, h0)
#                 e = F.gumbel_softmax(logit, tau=self.tau, dim=-1)
#                 reg += self.reg_loss(e, logit)
#             else:
#                 if self.env_type == 'node':
#                     e = F.softmax(self.env_enc[i](h), dim=-1)
#                 else:
#                     e = F.softmax(self.env_enc[i](h, adj, h0), dim=-1)
#             h = self.act_fn(con(h, adj, e))

#         h = F.dropout(h, self.dropout, training=self.training)
#         out = self.fcs[-1](h)
        
#         if self.training:
#             return out, reg / self.num_layers, ica_loss
#         else:
#             return out

#     def reg_loss(self, z, logit, logit_0 = None):
#         log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, logit.size(1))
#         return torch.mean(torch.sum(torch.mul(z, log_pi), dim=1))

#     def sup_loss_calc(self, y, pred, criterion, args):
#         if args.dataset in ('twitch', 'elliptic'):
#             if y.shape[1] == 1:
#                 true_label = F.one_hot(y, y.max() + 1).squeeze(1)
#             else:
#                 true_label = y
#             loss = criterion(pred, true_label.squeeze(1).to(torch.float))
#         else:
#             out = F.log_softmax(pred, dim=1)
#             target = y.squeeze(1)
#             loss = criterion(out, target)
#         return loss

#     def loss_compute(self, d, criterion, args):
#         # 确保传入了 idx=d.train_idx，以便精准计算 ICA loss
#         logits, reg_loss, ica_loss = self.forward(d.x, d.edge_index, idx=d.train_idx, training=True)
#         sup_loss = self.sup_loss_calc(d.y[d.train_idx], logits[d.train_idx], criterion, args)
        
#         # 联合优化：监督损失 + 伪环境正则损失 + 端到端非线性 ICA 损失
#         # 你可以通过传入额外的 args.lambda_ica 来动态控制，此处默认设为 0.1
#         loss = sup_loss + args.lamda * reg_loss + 0.1 * ica_loss
#         return loss


        
    # def forward(self, x, adj, idx=None, training=False):
    #     self.training = training
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     h = self.act_fn(self.fcs[0](x))
    #     h0 = h.clone()

    #     reg = 0
    #     for i,con in enumerate(self.convs):
    #         h = F.dropout(h, self.dropout, training=self.training)
    #         if self.training:
    #             if self.env_type == 'node':
    #                 logit = self.env_enc[i](h)
    #             else:
    #                 logit = self.env_enc[i](h, adj, h0)
    #             e = F.gumbel_softmax(logit, tau=self.tau, dim=-1)
    #             reg += self.reg_loss(e, logit)
    #         else:
    #             if self.env_type == 'node':
    #                 e = F.softmax(self.env_enc[i](h), dim=-1)
    #             else:
    #                 e = F.softmax(self.env_enc[i](h, adj, h0), dim=-1)
    #         h = self.act_fn(con(h, adj, e))

    #     h = F.dropout(h, self.dropout, training=self.training)
    #     out = self.fcs[-1](h)
    #     if self.training:
    #         return out, reg / self.num_layers
    #     else:
    #         return out

    # def reg_loss(self, z, logit, logit_0 = None):
    #     log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, logit.size(1))
    #     return torch.mean(torch.sum(
    #         torch.mul(z, log_pi), dim=1))

    # def sup_loss_calc(self, y, pred, criterion, args):
    #     if args.dataset in ('twitch', 'elliptic'):
    #         if y.shape[1] == 1:
    #             true_label = F.one_hot(y, y.max() + 1).squeeze(1)
    #         else:
    #             true_label = y
    #         loss = criterion(pred, true_label.squeeze(1).to(torch.float))
    #     else:
    #         out = F.log_softmax(pred, dim=1)
    #         target = y.squeeze(1)
    #         loss = criterion(out, target)
    #     return loss

    # def loss_compute(self, d, criterion, args):
    #     logits, reg_loss = self.forward(d.x, d.edge_index, idx=d.train_idx, training=True)
    #     sup_loss = self.sup_loss_calc(d.y[d.train_idx], logits[d.train_idx], criterion, args)
    #     loss = sup_loss + args.lamda * reg_loss
    #     return loss