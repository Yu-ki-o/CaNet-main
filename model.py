import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv 
from torch_geometric.nn import SAGEConv
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


class NodeWeightGenerator(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(NodeWeightGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1) 
        )

    def reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        nn.init.constant_(self.net[2].bias, 1.0)

    def forward(self, x):
        return torch.sigmoid(self.net(x))

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

        self.classifier = nn.Linear(self.d, c)
        self.weight_net = NodeWeightGenerator(self.d, self.d // 2)

        # 严格遵循 d+1 维度（d 个特征节点，1 个标签节点）
        self.A = Parameter(torch.zeros(self.d + 1, self.d + 1))
        nn.init.uniform_(self.A, a=-0.05, b=0.05)
        
        # M_dag 的任务是将 1 维的重构标签节点映射到 c 维分类逻辑
        self.M_dag = nn.Linear(1, self.c, bias=False)

        if args.dataset in ['cora', 'citeseer', 'pubmed', 'twitch']:
            self.rff_dim = 16  
        elif args.dataset in ['arxiv', 'elliptic']:
            self.rff_dim = 128 
        self.omega = Parameter(torch.randn(1, self.d, self.rff_dim), requires_grad=False)
        self.phi = Parameter(torch.rand(1, self.d, self.rff_dim) * 2 * math.pi, requires_grad=False)

        self.alpha = args.alpha if hasattr(args, 'alpha') else 0.9
        self.global_size = args.global_size if hasattr(args, 'global_size') else 2048
        self.register_buffer('z_global', torch.zeros(self.global_size, self.d))
        self.register_buffer('w_global', torch.ones(self.global_size, 1) / self.global_size)
        self.global_ptr = 0 
        self.queue_full = False 

        self.gamma = args.gamma if hasattr(args, 'gamma') else 0.99
        self.tau = args.tau if hasattr(args, 'tau') else 0.5 
        self.register_buffer('proto_in', torch.zeros(self.num_envs, c, self.d))
        self.register_buffer('proto_cr', torch.zeros(c, self.d))

        self.dropout = getattr(args, 'dropout', 0.0)
        self.lambda_l1 = getattr(args, 'lambda_l1', 1e-5)
        self.lambda_ind = getattr(args, 'lambda_ind', 0.1)  
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
        self.weight_net.reset_parameters()
        self.M_dag.reset_parameters()
        nn.init.uniform_(self.A, -0.01, 0.01)

        self.z_global.zero_()
        self.w_global.fill_(1.0 / self.global_size)
        self.global_ptr = 0
        self.queue_full = False
        self.proto_in.zero_()
        self.proto_cr.zero_()

        if hasattr(self, 'omega') and hasattr(self, 'phi'):
            nn.init.normal_(self.omega)
            nn.init.uniform_(self.phi, 0, 2 * math.pi)
            
        if hasattr(self, 'dag_gate'):
            nn.init.constant_(self.dag_gate, 0.0)

    def get_causal_effect_and_mask(self):
        A_no_diag = self.get_masked_A()
        A_sq = A_no_diag * A_no_diag
        
        dim = A_sq.size(0)
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
            
        Ca_expand = Ca_mask.unsqueeze(1)
        S = 1.0 - torch.matmul(Ca_expand, Ca_expand.t())
        
        return Ca_mask, S

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
        
        # -----------------------------------------
        # 引擎 A：微观节点级别 (Micro)
        # -----------------------------------------
        z_norm = F.layer_norm(z_local, (self.d,))
        y_factor_micro = y_local.float().view(-1, 1) / max(1.0, float(self.c - 1))
        
        F_micro = torch.cat([z_norm, y_factor_micro], dim=1)
        rec_F_micro = torch.matmul(F_micro, A_no_diag)
        
        loss_rec_X_micro = F.mse_loss(rec_F_micro[:, :self.d], F_micro[:, :self.d])
        
        rec_Y_node_micro = rec_F_micro[:, self.d:self.d+1]
        logits_Y_micro = self.M_dag(rec_Y_node_micro)
        
        # 自适应 Criterion 计算并强制 .mean() 降维为标量
        if args.dataset in ('twitch', 'elliptic'):
            if y_local.shape[1] == 1 and logits_Y_micro.shape[1] > 1:
                true_label_micro = F.one_hot(y_local.squeeze().long(), logits_Y_micro.shape[1]).float()
            else:
                true_label_micro = y_local.float()
            loss_rec_Y_micro = criterion(logits_Y_micro, true_label_micro).mean()
        else:
            loss_rec_Y_micro = criterion(logits_Y_micro, y_local.squeeze().long()).mean()
            
        loss_micro = loss_rec_X_micro + loss_rec_Y_micro

        # -----------------------------------------
        # 引擎 B：宏观原型级别 (Macro)
        # -----------------------------------------
        proto_features = self.proto_in.detach().view(-1, self.d)
        proto_norm = F.layer_norm(proto_features, (self.d,))
        
        labels_seq = torch.arange(self.c, device=self.device).repeat(self.num_envs).view(-1, 1)
        y_factor_macro = labels_seq.float() / max(1.0, float(self.c - 1))
        
        F_macro = torch.cat([proto_norm, y_factor_macro], dim=1)
        rec_F_macro = torch.matmul(F_macro, A_no_diag)
        
        loss_rec_X_macro = F.mse_loss(rec_F_macro[:, :self.d], F_macro[:, :self.d])
        
        rec_Y_node_macro = rec_F_macro[:, self.d:self.d+1]
        logits_Y_macro = self.M_dag(rec_Y_node_macro)
        
        # 修复 Macro 引擎：同样自适应 criterion 并强制 .mean()
        if args.dataset in ('twitch', 'elliptic'):
            if labels_seq.shape[1] == 1 and logits_Y_macro.shape[1] > 1:
                true_label_macro = F.one_hot(labels_seq.squeeze().long(), logits_Y_macro.shape[1]).float()
            else:
                true_label_macro = labels_seq.float()
            loss_rec_Y_macro = criterion(logits_Y_macro, true_label_macro).mean()
        else:
            loss_rec_Y_macro = criterion(logits_Y_macro, labels_seq.squeeze().long()).mean()
        
        loss_macro = loss_rec_X_macro + loss_rec_Y_macro

        # -----------------------------------------
        # 融合与无环约束
        # -----------------------------------------
        alpha = torch.sigmoid(self.dag_gate) 
        loss_rec_fused = alpha * loss_macro + (1.0 - alpha) * loss_micro

        A_sq = A_no_diag * A_no_diag
        h_A = torch.trace(torch.matrix_exp(A_sq)) - (self.d + 1)
        h_A_clipped = torch.clamp(h_A, min=-10.0, max=10.0)
        loss_dag_reg = 0.5 * (h_A_clipped ** 2) + self.lambda_l1 * torch.norm(A_no_diag, p=1)

        return loss_rec_fused + loss_dag_reg

    def compute_weighted_independence_loss(self, z_local, w_local, S):
        if z_local.size(0) >= 256:
            z_concat, w_concat = z_local, w_local
        else:
            valid_global = self.global_size if self.queue_full else self.global_ptr
            if valid_global > 0:
                z_concat = torch.cat([z_local, self.z_global[:valid_global]], dim=0)
                w_concat = torch.cat([w_local, self.w_global[:valid_global]], dim=0)
            else:
                z_concat, w_concat = z_local, w_local
                
        w_concat_smooth = w_concat + 0.05
        w_concat = w_concat_smooth / torch.sum(w_concat_smooth)
        
        z_mean = torch.mean(z_concat, dim=0, keepdim=True)
        z_std = torch.std(z_concat, dim=0, keepdim=True) + 1e-5
        z_concat_norm = (z_concat - z_mean) / z_std
        
        # 恢复 RFF 高维非线性映射
        z_expand = z_concat_norm.unsqueeze(-1)
        z_rff = math.sqrt(2) * torch.cos(z_expand * self.omega + self.phi) 
        
        N, D, R = z_rff.shape
        z_rff_flat = z_rff.view(N, D * R)
        
        mean = torch.sum(w_concat * z_rff_flat, dim=0, keepdim=True)
        z_centered = z_rff_flat - mean
        
        cov = torch.matmul((w_concat * z_centered).t(), z_centered) 
        cov_sq = cov ** 2
        cov_blocks_sq = cov_sq.view(D, R, D, R).sum(dim=(1, 3)) / (R * R)
        
        ind_loss = torch.sum(torch.triu(S * cov_blocks_sq, diagonal=1))
        num_pairs = (self.d * (self.d - 1)) / 2.0
        return ind_loss / num_pairs

    def update_global_queue(self, z_local, w_local):
        batch_size = z_local.size(0)
        if batch_size == 0: return

        if batch_size >= self.global_size:
            self.z_global.copy_(z_local[-self.global_size:].detach())
            self.w_global.copy_(w_local[-self.global_size:].detach())
            self.global_ptr = 0
            self.queue_full = True
            return

        ptr = self.global_ptr
        if ptr + batch_size <= self.global_size:
            self.z_global[ptr:ptr+batch_size].copy_(z_local.detach())
            self.w_global[ptr:ptr+batch_size].copy_(w_local.detach())
            self.global_ptr = ptr + batch_size
            if self.global_ptr == self.global_size:
                self.global_ptr = 0
                self.queue_full = True
        else:
            remain = self.global_size - ptr
            self.z_global[ptr:].copy_(z_local[:remain].detach())
            self.w_global[ptr:].copy_(w_local[:remain].detach())
            
            wrap = batch_size - remain
            self.z_global[:wrap].copy_(z_local[remain:].detach())
            self.w_global[:wrap].copy_(w_local[remain:].detach())
            self.global_ptr = wrap
            self.queue_full = True

    def forward(self, x, edge_index, training=False):
        self.training = training
        
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.pre_fc(x))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.act_fn(self.conv1(h, edge_index))
        h = F.dropout(h, self.dropout, training=self.training)
        z = self.conv2(h, edge_index) 

        w = self.weight_net(z.detach()) 
        Ca_norm, S = self.get_causal_effect_and_mask()
        
        z_inv_dag = z.detach() * Ca_norm.unsqueeze(0)
        z_inv_cls = z * Ca_norm.detach().unsqueeze(0)
        z_inv_cls = F.dropout(z_inv_cls, self.dropout, training=self.training)
        
        logits = self.classifier(z_inv_cls)
        
        if self.training:
            return logits, z, w, S, z_inv_dag, z_inv_cls
        else:
            return logits

    def loss_compute(self, data, criterion, args):
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx
        envs = data.env if hasattr(data, 'env') else None
        
        logits, z_local, w_local, S, z_inv_dag, z_inv_cls = self.forward(x, edge_index, training=True)
        
        z_tr = z_local[train_idx]
        w_tr = w_local[train_idx]
        y_tr = y[train_idx]
        inv_dag_tr = z_inv_dag[train_idx]
        env_tr = envs[train_idx] if envs is not None else None
        
        loss_ind = self.compute_weighted_independence_loss(z_tr.detach(), w_tr, S.detach())
        
        self.update_prototypes(z_tr.detach(), inv_dag_tr.detach(), y_tr, env_tr)
        
        # 【关键修正】传入带梯度的 z_tr，允许骨干网络接收域内对比学习梯度
        loss_cl_in, loss_cl_cr = self.compute_contrastive_loss(z_tr, inv_dag_tr, y_tr, env_tr)
        
        loss_dag = self.dag_reconstruction_loss(z_tr.detach(), y_tr, criterion, args)
        
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

        w_tr_norm = w_tr.squeeze() / torch.clamp(w_tr.squeeze().mean(), min=1e-3)
        loss_cls = torch.mean(w_tr_norm.detach() * sup_loss_raw)
        
        self.update_global_queue(z_tr, w_tr)
        
        # 拆分并打包优化目标
        loss_model_opt = loss_cls + self.lambda_ind * loss_ind + self.lambda_cl * loss_cl_in
        loss_dag_opt = self.lambda_dag * loss_dag + self.lambda_cl * loss_cl_cr
        
        return loss_model_opt, loss_dag_opt, loss_cls.item(), (self.lambda_ind * loss_ind).item(), (self.lambda_dag * loss_dag).item(), (self.lambda_cl * loss_cl_cr).item()