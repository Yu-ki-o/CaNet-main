import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class CausalFeatureTransformer(nn.Module):
    """
    融合了 DAG-aware Transformer 思想的特征因果提纯器。
    引入了 DAG 硬掩码 (Hard Mask) 与 原始输入残差连接 (Residual Raw-Input Connection)。
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
        
        # 控制 Transformer 输出与原始输入加权比例的参数 alpha
        self.alpha_res = nn.Parameter(torch.tensor(0.5)) 
        
        # DAG 掩码阈值：低于此因果概率的边将被彻底切断
        self.mask_threshold = 0.1 
        
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
        nn.init.constant_(self.alpha_res, 0.5)

    def _forward_chunk(self, Z_chunk, A_no_diag):
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
        
        # ==========================================================
        # 🌟 核心抢救：连续可导的 DAG-aware Soft-Mask 
        # ==========================================================
        causal_matrix = torch.abs(A_no_diag)
        
        # 🌟 致命 Bug 修复：转置因果矩阵！
        # 标签 Token(d) 关注 特征(j)，对应的是边 j -> d，即 A[j, d]
        # 转置后，scores[d, j] 正确对齐到边 A[j, d]
        causal_matrix = causal_matrix.t() 
        
        # 🌟 打破冷启动死锁：动态 Min-Max 归一化
        # 保证无论初期 A 有多小，当前最大的“疑似因果边”一定等于 1.0，强行打通信息流让梯度流转
        c_max = causal_matrix.max()
        if c_max > 1e-6:
            causal_matrix = causal_matrix / c_max
        else:
            causal_matrix = causal_matrix + 1e-3 # 防止全 0 崩溃
            
        # 必须允许绝对的自注意力
        causal_matrix.fill_diagonal_(1.0) 
        
        # 🌟 映射到对数空间作为掩码 (连续松弛版)
        # 概率接近 0 的边，log 值为极大负数 (如 -20)，等效于论文的硬掩码
        # 概率接近 1 的边，log 值接近 0，信息无损通过，全程丝滑可导！
        dag_mask = torch.log(causal_matrix + 1e-9)
        
        scores = scores + dag_mask.unsqueeze(0).unsqueeze(0)
        # ==========================================================
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(N, seq_len, self.d_emb)
        out = self.o_proj(out)
        
        H = out + self.ffn(self.norm2(out))
        
        # 残差连接，保留原始信息底色
        Z_final = self.alpha_res * H + X_norm
        
        return Z_final[:, -1, :]

    def forward(self, Z, A_no_diag, chunk_size=1024):
        N = Z.size(0)
        if N <= chunk_size:
            return self._forward_chunk(Z, A_no_diag)
            
        Z_causal_chunks = []
        for i in range(0, N, chunk_size):
            Z_causal_chunks.append(self._forward_chunk(Z[i : i + chunk_size], A_no_diag))
        return torch.cat(Z_causal_chunks, dim=0)


class GraphCIW(nn.Module):
    def __init__(self, d_in, c, args, device):
        super(GraphCIW, self).__init__()
        self.device = device
        self.d = args.hidden_channels  
        self.c = c                     
        self.num_envs = getattr(args, 'train_env_num', 1)
        
        from torch_geometric.nn import GCNConv
        self.pre_fc = nn.Linear(d_in, self.d)
        self.conv1 = GCNConv(self.d, self.d)
        self.conv2 = GCNConv(self.d, self.d)
        self.act_fn = nn.GELU()
        
        self.causal_transformer = CausalFeatureTransformer(self.d, d_emb=64, num_heads=4)
        self.classifier = nn.Linear(64, c) 

        self.A = Parameter(torch.zeros(self.d + 1, self.d + 1))
        self.M_dag = nn.Linear(1, self.c, bias=False)
        self.log_vars = Parameter(torch.zeros(self.d)) 
        self.dag_gate = Parameter(torch.tensor([0.0]))

        self.gamma = getattr(args, 'gamma', 0.99)
        self.tau = getattr(args, 'tau', 0.5)
        self.dropout = getattr(args, 'dropout', 0.0)
        self.lambda_l1 = getattr(args, 'lambda_l1', 1e-5)
        self.lambda_cl = getattr(args, 'lambda_cl', 0.1) 
        self.lambda_dag = getattr(args, 'lambda_dag', 0.1) 
        self.vrex_weight = getattr(args, 'vrex_weight', 10.0) 
        
        self.register_buffer('proto_in', torch.zeros(self.num_envs, c, 64))
        self.register_buffer('proto_cr', torch.zeros(c, 64))
        
        # ==========================================================
        # 🌟 增广拉格朗日参数注册区
        # ==========================================================
        self.register_buffer('lagrange_multiplier', torch.tensor(0.0))
        self.register_buffer('rho', torch.tensor(1.0))                 
        self.register_buffer('h_A_last', torch.tensor(float('inf')))   
        self.register_buffer('current_h_A', torch.tensor(0.0))   

        self.reset_parameters()

    def get_masked_A(self):
        # 🌟 修复：防止 in-place operation 破坏计算图
        mask = torch.ones(self.d + 1, self.d + 1, device=self.A.device, requires_grad=False)
        mask[self.d:, :] = 0.0 
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
        nn.init.constant_(self.log_vars, 0.0)
        nn.init.constant_(self.dag_gate, 0.0)
        self.proto_in.zero_()
        self.proto_cr.zero_()

    def get_causal_effect(self):
        A_no_diag = self.get_masked_A()
        A_sq = A_no_diag * A_no_diag
        #这里原本是e的指数次幂，这里简化成两跳影响
        C_tot = A_sq + (torch.matmul(A_sq, A_sq) / 2.0)
        Ca = C_tot[:self.d, self.d] 
        Ca_min, Ca_max = Ca.min(), Ca.max()
        
        if Ca_max - Ca_min < 1e-4:
            return (Ca - Ca.detach()) + 0.5 
        else:
            Ca_norm = (Ca - Ca_min) / (Ca_max - Ca_min + 1e-8)
            return torch.sigmoid((Ca_norm - 0.5) / 0.05)

    def update_prototypes(self, z_local, z_invariant, y, envs):
        unique_classes = torch.unique(y)
        for c in unique_classes:
            c_idx = int(c.item())
            # 🌟 修复：使用 view(-1) 防止 0 维坍缩
            mask_c = (y.view(-1) == c)
            
            if envs is not None and self.num_envs > 1:
                env_means_for_c = []
                for e in range(self.num_envs):
                    mask_ce = mask_c & (envs.view(-1) == e)
                    if mask_ce.any():
                        env_means_for_c.append(z_invariant[mask_ce].mean(dim=0))
                
                if len(env_means_for_c) > 0:
                    z_inv_c_mean = torch.stack(env_means_for_c).mean(dim=0) 
                else:
                    z_inv_c_mean = z_invariant[mask_c].mean(dim=0)
            else:
                z_inv_c_mean = z_invariant[mask_c].mean(dim=0)
            
            if self.proto_cr[c_idx].abs().sum() < 1e-6:
                self.proto_cr[c_idx] = F.normalize(z_inv_c_mean.detach(), dim=0)
            else:
                self.proto_cr[c_idx] = F.normalize(self.gamma * self.proto_cr[c_idx] + (1 - self.gamma) * z_inv_c_mean.detach(), dim=0)
            
            if envs is not None:
                for e in range(self.num_envs):
                    mask_ce = mask_c & (envs.view(-1) == e)
                    if mask_ce.any():
                        z_loc_ce_mean = z_local[mask_ce].mean(dim=0)
                        if self.proto_in[e, c_idx].abs().sum() < 1e-6:
                            self.proto_in[e, c_idx] = F.normalize(z_loc_ce_mean.detach(), dim=0)
                        else:
                            self.proto_in[e, c_idx] = F.normalize(self.gamma * self.proto_in[e, c_idx] + (1 - self.gamma) * z_loc_ce_mean.detach(), dim=0)

    def compute_contrastive_loss(self, z_local, z_invariant, y, envs):
        y_idx = y.view(-1).long() # 🌟 修复：防止坍缩
        z_inv_norm = F.normalize(z_invariant, dim=1)
        z_loc_norm = F.normalize(z_local, dim=1)

        logits_cr = torch.matmul(z_inv_norm, self.proto_cr.t()) / self.tau
        loss_cl_cr = F.cross_entropy(logits_cr, y_idx)

        loss_cl_in = torch.tensor(0.0).to(self.device)
        if envs is not None:
            e_idx = envs.view(-1).long()
            protos_e = self.proto_in[e_idx] 
            logits_in = torch.bmm(protos_e, z_loc_norm.unsqueeze(2)).squeeze(2) / self.tau 
            loss_cl_in = F.cross_entropy(logits_in, y_idx)

        return loss_cl_in, loss_cl_cr

    def dag_reconstruction_loss(self, z_local, y_local, envs, criterion, args):
        A_no_diag = self.get_masked_A()
        z_norm = F.layer_norm(z_local, (self.d,))
        y_factor_micro = y_local.float().view(-1, 1) / max(1.0, float(self.c - 1))
        F_micro = torch.cat([z_norm, y_factor_micro], dim=1)
        rec_F_micro = torch.matmul(F_micro, A_no_diag)
        
        diff = F_micro[:, :self.d] - rec_F_micro[:, :self.d]
        clamped_log_vars = torch.clamp(self.log_vars, min=-5.0, max=5.0)
        precision = torch.exp(-clamped_log_vars)
        nll_loss = 0.5 * precision * (diff**2) + 0.5 * clamped_log_vars
        loss_rec_X_node = nll_loss.mean(dim=1) 
        
        rec_Y_node_micro = rec_F_micro[:, self.d:self.d+1]
        logits_Y_micro = self.M_dag(rec_Y_node_micro)
        
        # 🌟 修复：兼容 1D [N] 和 2D [N, 1] 的标签形状
        is_single_col = (len(y_local.shape) == 1) or (y_local.shape[1] == 1)
        
        if args.dataset in ('twitch', 'elliptic'):
            true_label = F.one_hot(y_local.view(-1).long(), self.c).float() if is_single_col else y_local.float()
            loss_rec_Y_node = F.binary_cross_entropy_with_logits(logits_Y_micro, true_label, reduction='none').mean(dim=1)
        else:
            loss_rec_Y_node = F.cross_entropy(logits_Y_micro, y_local.view(-1).long(), reduction='none')
            
        loss_micro_node = loss_rec_X_node + loss_rec_Y_node 
        var_reg = 1e-4 * torch.mean(clamped_log_vars ** 2)

        # 环境平权与方差惩罚 (V-REx)
        if envs is not None and self.num_envs > 1:
            env_losses = []
            for e in range(self.num_envs):
                mask_e = (envs.view(-1) == e)
                if mask_e.any():
                    env_losses.append(loss_micro_node[mask_e].mean()) 
            
            if len(env_losses) > 1:
                env_losses = torch.stack(env_losses)
                loss_micro_fused = env_losses.mean() + self.vrex_weight * env_losses.var()
            else:
                loss_micro_fused = env_losses[0]
        else:
            loss_micro_fused = loss_micro_node.mean()
            
        loss_micro_fused = loss_micro_fused + var_reg
        loss_rec_final = (1.0 - torch.sigmoid(self.dag_gate)) * loss_micro_fused

        # ==========================================================
        # 🌟 DAG-GNN 标准的增广拉格朗日与多项式迹约束
        # ==========================================================
        A_sq = A_no_diag * A_no_diag
        alpha_poly = 1.0 / (self.d + 1) 
        M = torch.eye(self.d + 1, device=self.device) + alpha_poly * A_sq
        M_pow = torch.matrix_power(M, self.d + 1)
        h_A = torch.trace(M_pow) - (self.d + 1)
        
        loss_dag_reg = self.lagrange_multiplier * h_A + 0.5 * self.rho * (h_A ** 2) 
        loss_dag_reg = loss_dag_reg + self.lambda_l1 * torch.norm(A_no_diag, p=1)
        
        # 记录当前环状态供 main.py 中的外部调度器读取
        self.current_h_A = h_A.detach().clone()

        return loss_rec_final + loss_dag_reg

    def forward(self, x, edge_index, target_idx=None, training=False):
        self.training = training
        
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.pre_fc(x))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.act_fn(self.conv1(h, edge_index))
        h = F.dropout(h, self.dropout, training=self.training)
        z = self.conv2(h, edge_index) 

        if target_idx is not None:
            z_compute = z[target_idx]
        else:
            z_compute = z 

        Ca_norm = self.get_causal_effect()
        z_inv_dag = z_compute.detach() * Ca_norm.unsqueeze(0)
        
        # 🌟 修复：获取带掩码的真实 DAG 矩阵 A_no_diag，并切断梯度送入 Transformer
        A_no_diag = self.get_masked_A().detach()
        z_inv_cls = self.causal_transformer(z_compute, A_no_diag)
        
        z_inv_cls = F.dropout(z_inv_cls, self.dropout, training=self.training)
        logits = self.classifier(z_inv_cls)
        
        if self.training:
            return logits, z_compute, z_inv_dag, z_inv_cls
        else:
            return logits
        
    def loss_compute(self, data, criterion, args):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx
        envs = data.env if hasattr(data, 'env') else None
        
        sample_size = getattr(args, 'sub_sample_size', 4096) 
        if train_idx.size(0) > sample_size:
            perm = torch.randperm(train_idx.size(0), device=self.device)[:sample_size]
            sub_idx = train_idx[perm]
        else:
            sub_idx = train_idx
            
        y_sub = y[sub_idx]
        env_sub = envs[sub_idx] if envs is not None else None

        logits_sub, z_local_sub, _, z_inv_cls_sub = self.forward(x, edge_index, target_idx=sub_idx, training=True)
        
        z_tr_proj_sub = F.linear(z_local_sub, self.causal_transformer.feat_emb.t())
        self.update_prototypes(z_tr_proj_sub.detach(), z_inv_cls_sub.detach(), y_sub, env_sub)
        
        loss_cl_in, _ = self.compute_contrastive_loss(z_tr_proj_sub, z_inv_cls_sub, y_sub, env_sub)
        
        is_single_col = (len(y_sub.shape) == 1) or (y_sub.shape[1] == 1)
        if args.dataset in ('twitch', 'elliptic'):
            true_label = F.one_hot(y_sub.view(-1).long(), self.c).float() if is_single_col else y_sub.float()
            loss_cls = criterion(logits_sub, true_label)
            if len(loss_cls.shape) > 1: loss_cls = loss_cls.mean(dim=1)
            loss_cls = loss_cls.mean()
        else:
            loss_cls = criterion(logits_sub, y_sub.view(-1).long())
            # 🌟 修复：均值操作防止标量解包错误
            loss_cls = loss_cls.mean()
            
        loss_dag = self.dag_reconstruction_loss(z_local_sub.detach(), y_sub, env_sub, criterion, args)
        
        # 🌟 修复：专属通道传入带有梯度的真实 DAG 矩阵 A
        A_no_diag_with_grad = self.get_masked_A()
        z_inv_cls_for_dag = self.causal_transformer(z_local_sub.detach(), A_no_diag_with_grad)
        
        _, loss_cl_cr_for_dag = self.compute_contrastive_loss(z_tr_proj_sub.detach(), z_inv_cls_for_dag, y_sub, env_sub)

        loss_model_opt = loss_cls + self.lambda_cl * loss_cl_in
        loss_dag_opt = self.lambda_dag * loss_dag + self.lambda_cl * loss_cl_cr_for_dag
        
        return loss_model_opt, loss_dag_opt, loss_cls.item(), (self.lambda_dag * loss_dag).item(), (self.lambda_cl * loss_cl_cr_for_dag).item()