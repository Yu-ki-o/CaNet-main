import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class CaNetBasicConv(MessagePassing):
    """
    完全复刻 CaNet 源码中 backbone_type='gcn' 且 K=1 时的骨干网络。
    包含了对称归一化、[hi, x] 显式拼接、以及残差连接。
    """
    def __init__(self, in_channels, out_channels, residual=True):
        super(CaNetBasicConv, self).__init__(aggr='add') 
        self.residual = residual
        self.lin = nn.Linear(in_channels * 2, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.lin.out_features)
        self.lin.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        hi = self.propagate(edge_index, x=x, norm=norm)
        cat_x = torch.cat([hi, x], dim=1)
        out = self.lin(cat_x)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


# class CausalFeatureTransformer(nn.Module):
#     """
#     完全体：带 [CLS] Label Token 的因果 Transformer。
#     序列长度为 d_feat + 1，完美对齐全局因果矩阵 A。
#     """
#     def __init__(self, d_feat, d_emb=64, num_heads=4):
#         super(CausalFeatureTransformer, self).__init__()
#         self.d_feat = d_feat
#         self.d_emb = d_emb
#         self.num_heads = num_heads
#         self.d_k = d_emb // num_heads
        
#         # 1. 特征 Token 字典
#         self.feat_emb = nn.Parameter(torch.randn(d_feat, d_emb))
        
#         # 2. 🌟 核心创新：引入因果标签 Token (类似于 BERT 的 [CLS])
#         self.label_token = nn.Parameter(torch.randn(1, 1, d_emb))
        
#         # 3. 多头注意力组件
#         self.q_proj = nn.Linear(d_emb, d_emb)
#         self.k_proj = nn.Linear(d_emb, d_emb)
#         self.v_proj = nn.Linear(d_emb, d_emb)
#         self.o_proj = nn.Linear(d_emb, d_emb)
        
#         # 控制因果先验矩阵对注意力机制介入强度的温度参数
#         self.alpha = nn.Parameter(torch.tensor(1.0))
        
#         # 4. FFN
#         self.ffn = nn.Sequential(
#             nn.Linear(d_emb, d_emb * 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(d_emb * 2, d_emb)
#         )
#         self.norm1 = nn.LayerNorm(d_emb)
#         self.norm2 = nn.LayerNorm(d_emb)

#     def reset_parameters(self):
#         nn.init.normal_(self.feat_emb, std=0.02)
#         nn.init.normal_(self.label_token, std=0.02)
#         self.q_proj.reset_parameters()
#         self.k_proj.reset_parameters()
#         self.v_proj.reset_parameters()
#         self.o_proj.reset_parameters()
#         for m in self.ffn:
#             if isinstance(m, nn.Linear):
#                 m.reset_parameters()
#         self.norm1.reset_parameters()
#         self.norm2.reset_parameters()
#         nn.init.constant_(self.alpha, 1.0)

#     def forward(self, Z, A_full):
#         """
#         Z: [N, d_feat]
#         A_full: [d_feat+1, d_feat+1] - 完整的上帝视角因果图
#         """
#         N = Z.size(0)
        
#         # 防爆层：强制 LayerNorm 防止大标量引发 Softmax 饱和
#         Z_norm = F.layer_norm(Z, (self.d_feat,))
        
#         # 1. 特征 Token 序列: [N, d_feat, d_emb]
#         X_feats = Z_norm.unsqueeze(-1) * self.feat_emb.unsqueeze(0) 
        
#         # 2. 🌟 拼接 Label Token，形成 d+1 长度的完整序列
#         label_tokens = self.label_token.expand(N, -1, -1)
#         X = torch.cat([X_feats, label_tokens], dim=1) # [N, d_feat+1, d_emb]
        
#         X_norm = self.norm1(X)
#         seq_len = self.d_feat + 1
        
#         Q = self.q_proj(X_norm).view(N, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         K = self.k_proj(X_norm).view(N, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         V = self.v_proj(X_norm).view(N, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
#         # 3. 注入完整的 d+1 维因果掩码偏置
#         causal_bias = self.alpha * A_full.t().unsqueeze(0).unsqueeze(0)
#         scores = scores + causal_bias
        
#         attn = F.softmax(scores, dim=-1)
#         out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(N, seq_len, self.d_emb)
#         out = self.o_proj(out)
        
#         X = X + out
#         X = X + self.ffn(self.norm2(X))
        
#         # 4. 🌟 取出最后一列 (Label Token) 作为最终的因果特征浓缩表征
#         Z_label = X[:, -1, :] # 形状: [N, d_emb]
        
#         return Z_label


class CausalFeatureTransformer(nn.Module):
    """
    完全体：带 [CLS] Label Token 的因果 Transformer。
    【新增防爆机制】：利用特征的节点独立性，通过 Chunking (切片) 彻底解决十万级大图的 OOM 问题。
    """
    def __init__(self, d_feat, d_emb=64, num_heads=4):
        super(CausalFeatureTransformer, self).__init__()
        self.d_feat = d_feat
        self.d_emb = d_emb
        self.num_heads = num_heads
        self.d_k = d_emb // num_heads
        
        self.feat_emb = nn.Parameter(torch.randn(d_feat, d_emb))
        self.label_token = nn.Parameter(torch.randn(1, 1, d_emb))
        
        self.q_proj = nn.Linear(d_emb, d_emb)
        self.k_proj = nn.Linear(d_emb, d_emb)
        self.v_proj = nn.Linear(d_emb, d_emb)
        self.o_proj = nn.Linear(d_emb, d_emb)
        
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
        self.ffn = nn.Sequential(
            nn.Linear(d_emb, d_emb * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_emb * 2, d_emb)
        )
        self.norm1 = nn.LayerNorm(d_emb)
        self.norm2 = nn.LayerNorm(d_emb)

    def reset_parameters(self):
        nn.init.normal_(self.feat_emb, std=0.02)
        nn.init.normal_(self.label_token, std=0.02)
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.o_proj.reset_parameters()
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        nn.init.constant_(self.alpha, 1.0)

    def _forward_chunk(self, Z_chunk, A_full):
        """
        处理单个块的核心前向逻辑
        """
        N = Z_chunk.size(0)
        
        Z_norm = F.layer_norm(Z_chunk, (self.d_feat,))
        X_feats = Z_norm.unsqueeze(-1) * self.feat_emb.unsqueeze(0) 
        
        label_tokens = self.label_token.expand(N, -1, -1)
        X = torch.cat([X_feats, label_tokens], dim=1) 
        
        X_norm = self.norm1(X)
        seq_len = self.d_feat + 1
        
        Q = self.q_proj(X_norm).view(N, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(X_norm).view(N, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(X_norm).view(N, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        causal_bias = self.alpha * A_full.t().unsqueeze(0).unsqueeze(0)
        scores = scores + causal_bias
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(N, seq_len, self.d_emb)
        out = self.o_proj(out)
        
        X = X + out
        X = X + self.ffn(self.norm2(X))
        
        Z_label = X[:, -1, :] 
        return Z_label

    def forward(self, Z, A_full, chunk_size=2048):
        """
        支持超大规模图的防爆切片封装层
        """
        N = Z.size(0)
        
        # 智能动态降级 Chunk Size (针对 Pubmed 这种特征维度极高的猛兽)
        if self.d_feat > 200:
            chunk_size = 512
        
        # 如果节点很少，直接全量算
        if N <= chunk_size:
            return self._forward_chunk(Z, A_full)
            
        # 🌟 显存救星：时间换空间的张量切片 (Chunking)
        Z_causal_chunks = []
        for i in range(0, N, chunk_size):
            Z_c = Z[i : i + chunk_size]
            out_c = self._forward_chunk(Z_c, A_full)
            Z_causal_chunks.append(out_c)
            
        # 拼接回完整大图的形状，PyTorch的自动求导机制完美支持这种操作
        Z_causal_full = torch.cat(Z_causal_chunks, dim=0)
        return Z_causal_full

class GraphCIW(nn.Module):
    def __init__(self, d_in, c, args, device):
        super(GraphCIW, self).__init__()
        self.device = device
        self.d = args.hidden_channels  
        self.c = c                     
        self.num_envs = args.train_env_num 
        
        self.pre_fc = nn.Linear(d_in, self.d)
        
        self.conv1 = CaNetBasicConv(self.d, self.d, residual=True)
        self.conv2 = CaNetBasicConv(self.d, self.d, residual=True)
        
        self.act_fn = nn.GELU()

        # 引入因果特征 Transformer
        self.causal_transformer = CausalFeatureTransformer(self.d, d_emb=64, num_heads=4)

        # 此时分类器只接 Label Token 的 64 维特征！
        self.classifier = nn.Linear(64, c) 

        # 严格遵循 d+1 维度（d 个特征节点，1 个标签节点）
        self.A = Parameter(torch.zeros(self.d + 1, self.d + 1))
        nn.init.uniform_(self.A, a=-0.05, b=0.05)
        
        self.M_dag = nn.Linear(1, self.c, bias=False)

        # 🌟 【新增】分布感知的高斯重构方差参数 (对数形式)
        # 为 d 个隐式特征各分配一个可学习的方差
        self.log_vars = Parameter(torch.zeros(self.d))

        self.gamma = args.gamma if hasattr(args, 'gamma') else 0.99
        self.tau = args.tau if hasattr(args, 'tau') else 0.5 
        
        #  原型的维度也需要对齐到 Transformer 的输出维度 d_emb (64)
        self.register_buffer('proto_in', torch.zeros(self.num_envs, c, 64))
        self.register_buffer('proto_cr', torch.zeros(c, 64))

        self.dropout = getattr(args, 'dropout', 0.0)
        self.lambda_l1 = getattr(args, 'lambda_l1', 1e-5)
        self.lambda_cl = getattr(args, 'lambda_cl', 0.1) 
        self.lambda_dag = getattr(args, 'lambda_dag', 0.1) 

        self.dag_gate = Parameter(torch.tensor([0.0]))
        self.reset_parameters()

    def get_masked_A(self):
        mask = torch.ones_like(self.A)
        mask[self.d:, :] = 0.0 # 标签节点不能指向特征节点
        mask.fill_diagonal_(0.0)
        return self.A * mask
    
    def reset_parameters(self):
        self.pre_fc.reset_parameters()  
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.classifier.reset_parameters()
        self.M_dag.reset_parameters()
        self.causal_transformer.reset_parameters()
        nn.init.uniform_(self.A, -0.01, 0.01)

        # 🌟 【新增】方差参数初始化
        nn.init.constant_(self.log_vars, 0.0) # 0.0 代表初始方差为 1 (exp(0))

        self.proto_in.zero_()
        self.proto_cr.zero_()
            
        if hasattr(self, 'dag_gate'):
            nn.init.constant_(self.dag_gate, 0.0)

    def get_causal_effect(self):
        """提取用于阶段一的效应屏蔽值"""
        A_no_diag = self.get_masked_A()
        A_sq = A_no_diag * A_no_diag
        
        A_sq_2 = torch.matmul(A_sq, A_sq)
        C_tot = A_sq + (A_sq_2 / 2.0)

        Ca = C_tot[:self.d, self.d] 
        Ca_min, Ca_max = Ca.min(), Ca.max()
        
        if Ca_max - Ca_min < 1e-4:
            Ca_mask = (Ca - Ca.detach()) + 0.5 
        else:
            temperature = 0.05
            Ca_norm = (Ca - Ca_min) / (Ca_max - Ca_min + 1e-8)
            Ca_mask = torch.sigmoid((Ca_norm - 0.5) / temperature)
            
        return Ca_mask

    def update_prototypes(self, z_local, z_invariant, y, envs):
        unique_classes = torch.unique(y)
        for c in unique_classes:
            c_idx = int(c.item())
            mask_c = (y.squeeze() == c)
            
            z_inv_c_mean = z_invariant[mask_c].mean(dim=0)
            
            if self.proto_cr[c_idx].abs().sum() < 1e-6:
                self.proto_cr[c_idx] = F.normalize(z_inv_c_mean.detach(), dim=0)
            else:
                self.proto_cr[c_idx] = F.normalize(self.gamma * self.proto_cr[c_idx] + (1 - self.gamma) * z_inv_c_mean.detach(), dim=0)
            
            if envs is not None:
                for e in range(self.num_envs):
                    mask_ce = mask_c & (envs.squeeze() == e)
                    if mask_ce.any():
                        z_loc_ce_mean = z_local[mask_ce].mean(dim=0)
                        
                        if self.proto_in[e, c_idx].abs().sum() < 1e-6:
                            self.proto_in[e, c_idx] = F.normalize(z_loc_ce_mean.detach(), dim=0)
                        else:
                            self.proto_in[e, c_idx] = F.normalize(self.gamma * self.proto_in[e, c_idx] + (1 - self.gamma) * z_loc_ce_mean.detach(), dim=0)

    def compute_contrastive_loss(self, z_local, z_invariant, y, envs):
        y_idx = y.squeeze().long()
        z_inv_norm = F.normalize(z_invariant, dim=1)
        z_loc_norm = F.normalize(z_local, dim=1)

        logits_cr = torch.matmul(z_inv_norm, self.proto_cr.t()) / self.tau
        loss_cl_cr = F.cross_entropy(logits_cr, y_idx)

        loss_cl_in = 0.0
        if envs is not None:
            e_idx = envs.squeeze().long()
            protos_e = self.proto_in[e_idx] 
            logits_in = torch.bmm(protos_e, z_loc_norm.unsqueeze(2)).squeeze(2) / self.tau 
            loss_cl_in = F.cross_entropy(logits_in, y_idx)

        return loss_cl_in, loss_cl_cr

    def dag_reconstruction_loss(self, z_local, y_local, criterion, args):
        A_no_diag = self.get_masked_A()
        
        z_norm = F.layer_norm(z_local, (self.d,))
        y_factor_micro = y_local.float().view(-1, 1) / max(1.0, float(self.c - 1))
        
        F_micro = torch.cat([z_norm, y_factor_micro], dim=1)
        rec_F_micro = torch.matmul(F_micro, A_no_diag)
        
        # loss_rec_X_micro = F.mse_loss(rec_F_micro[:, :self.d], F_micro[:, :self.d])
        
        # ---------------------------------------------------------
        # 🌟 【改进】分布感知的类 DG-LRT 高斯似然重构
        diff = F_micro[:, :self.d] - rec_F_micro[:, :self.d]
        
        # 保险措施 1：数值截断 (防止方差彻底爆炸或坍缩到引发 NaN)
        # -5 对应极小方差，5 对应极大方差。你可以根据情况微调这个范围
        clamped_log_vars = torch.clamp(self.log_vars, min=-5.0, max=5.0)
        
        # 精度矩阵 (方差的倒数)
        precision = torch.exp(-clamped_log_vars)
        
        # 高斯负对数似然损失 (Gaussian NLL)
        nll_loss = 0.5 * precision * (diff**2) + 0.5 * clamped_log_vars
        
        # 保险措施 2：轻量级 L2 正则化
        # 给 log_vars 施加一个很弱的 L2 惩罚，防止模型为了逃避重构一味调大方差
        var_reg = 1e-4 * torch.mean(clamped_log_vars ** 2)
        
        # 微观特征重构损失 = NLL + 方差正则
        loss_rec_X_micro = nll_loss.mean() + var_reg
        # ---------------------------------------------------------

        rec_Y_node_micro = rec_F_micro[:, self.d:self.d+1]
        logits_Y_micro = self.M_dag(rec_Y_node_micro)
        
        if args.dataset in ('twitch', 'elliptic'):
            if y_local.shape[1] == 1 and logits_Y_micro.shape[1] > 1:
                true_label_micro = F.one_hot(y_local.squeeze().long(), logits_Y_micro.shape[1]).float()
            else:
                true_label_micro = y_local.float()
            loss_rec_Y_micro = criterion(logits_Y_micro, true_label_micro).mean()
        else:
            loss_rec_Y_micro = criterion(logits_Y_micro, y_local.squeeze().long()).mean()
            
        loss_micro = loss_rec_X_micro + loss_rec_Y_micro

        # 注意：这里的宏观重构由于 proto 的维度现在是 64，为了让 DAG 引擎维持原始骨干维度重构，
        # 建议仅依赖微观损失进行 DAG 图结构发现（这通常足够）。
        # 这里将其简化为仅利用 loss_micro，避免原型维度的不匹配干扰 A 矩阵的原始物理意义。
        loss_macro = 0.0

        alpha = torch.sigmoid(self.dag_gate) 
        loss_rec_fused = (1.0 - alpha) * loss_micro

        A_sq = A_no_diag * A_no_diag
        h_A = torch.trace(torch.matrix_exp(A_sq)) - (self.d + 1)
        h_A_clipped = torch.clamp(h_A, min=-10.0, max=10.0)
        loss_dag_reg = 0.5 * (h_A_clipped ** 2) + self.lambda_l1 * torch.norm(A_no_diag, p=1)

        return loss_rec_fused + loss_dag_reg

    def forward(self, x, edge_index, training=False):
        self.training = training
        
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.pre_fc(x))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.act_fn(self.conv1(h, edge_index))
        h = F.dropout(h, self.dropout, training=self.training)
        z = self.conv2(h, edge_index) 

        Ca_norm = self.get_causal_effect()
        
        # 1. 给阶段一 DAG 使用：原维度的分离特征
        z_inv_dag = z.detach() * Ca_norm.unsqueeze(0)
        
        # 2. 获取完整的因果图 A
        A_full = self.get_masked_A()
        
        # 3. 🌟 喂给 Causal Transformer，通过 `.detach()` 防止主任务梯度污染 A
        z_inv_cls = self.causal_transformer(z, A_full.detach())
        z_inv_cls = F.dropout(z_inv_cls, self.dropout, training=self.training)
        
        # 4. 把 Transformer 找出的 64 维精炼致因组合交给 Classifier 判决
        logits = self.classifier(z_inv_cls)
        
        if self.training:
            return logits, z, z_inv_dag, z_inv_cls
        else:
            return logits

    def loss_compute(self, data, criterion, args):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx
        envs = data.env if hasattr(data, 'env') else None
        
        logits, z_local, z_inv_dag, z_inv_cls = self.forward(x, edge_index, training=True)
        
        z_tr = z_local[train_idx]
        y_tr = y[train_idx]
        inv_cls_tr = z_inv_cls[train_idx]
        env_tr = envs[train_idx] if envs is not None else None
        
        # 【对齐原型】利用 Transformer 加工出的 64 维高阶输出进行对比学习，需对齐维度
        z_tr_proj = F.linear(z_tr, self.causal_transformer.feat_emb.t()) # 投影到 64 维做对比
        self.update_prototypes(z_tr_proj.detach(), inv_cls_tr.detach(), y_tr, env_tr)
        loss_cl_in, loss_cl_cr = self.compute_contrastive_loss(z_tr_proj, inv_cls_tr, y_tr, env_tr)
        
        # 【约束因果】用原始的 z_tr 去重构矩阵 A
        loss_dag = self.dag_reconstruction_loss(z_tr.detach(), y_tr, criterion, args)
        
        # 分类损失计算（极简平均版）
        if args.dataset in ('twitch', 'elliptic'):
            if y_tr.shape[1] == 1 and logits.shape[1] > 1:
                true_label = F.one_hot(y_tr.squeeze().long(), logits.shape[1]).float()
            else:
                true_label = y_tr.float()
            sup_loss_raw = criterion(logits[train_idx], true_label)
            if len(sup_loss_raw.shape) > 1:
                sup_loss_raw = sup_loss_raw.mean(dim=1)
        else:
            sup_loss_raw = criterion(logits[train_idx], y_tr.squeeze().long())

        loss_cls = sup_loss_raw.mean()
        
        # 双轨联合优化 Loss 打包
        loss_model_opt = loss_cls + self.lambda_cl * loss_cl_in
        loss_dag_opt = self.lambda_dag * loss_dag + self.lambda_cl * loss_cl_cr
        
        return loss_model_opt, loss_dag_opt, loss_cls.item(), (self.lambda_dag * loss_dag).item(), (self.lambda_cl * loss_cl_cr).item()