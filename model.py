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
        # 使用 add 聚合，配合我们在 forward 中手算的归一化系数，完美等价于原版 gcn_conv
        super(CaNetBasicConv, self).__init__(aggr='add') 
        self.residual = residual
        
        # 严格对齐 CaNet: 输入维度是 in_channels * 2，且原版没有使用 bias
        self.lin = nn.Linear(in_channels * 2, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # 严格对齐 CaNet 的均匀分布初始化方式
        stdv = 1. / math.sqrt(self.lin.out_features)
        self.lin.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        # 1. 严格复刻 CaNet 的 gcn_conv 对称度归一化 (d_norm_in * d_norm_out)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 2. 邻居信息聚合 (完全等价于原版 hi = gcn_conv(x, adj))
        hi = self.propagate(edge_index, x=x, norm=norm)

        # 3. 显式拼接：严格按照原版 torch.cat([hi, x], 1) 的顺序
        cat_x = torch.cat([hi, x], dim=1)

        # 4. 线性映射
        out = self.lin(cat_x)

        # 5. 残差连接 (对应原版的 if self.residual: output = output + x)
        # if self.residual:
        #     out = out + x

        return out

    def message(self, x_j, norm):
        # 在传播时乘以归一化系数
        return norm.view(-1, 1) * x_j


class NodeWeightGenerator(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(NodeWeightGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1) # 移除 Sigmoid，移到 forward 中
        )

    def reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        # 【抗方差修复 3】: 给最后一层加上 1.0 的正向偏置。
        # 这样 Sigmoid 初始输出在 0.73 左右（接近1），让初期训练更稳定，类似标准ERM平滑起步
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
        
        # self.conv1 = GCNConv(d_in, self.d)
        # self.conv2 = GCNConv(self.d, self.d)
        # self.conv1 = SAGEConv(d_in, self.d)
        # self.conv2 = SAGEConv(self.d, self.d)

        # 【重要同步】：复刻 CaNet 的 fcs[0] 前置投影！
        # 先把维度从数据集的输入维度 (d_in) 统一投射到隐藏层维度 (self.d)
        self.pre_fc = nn.Linear(d_in, self.d)
        
        # 装载原汁原味的 CaNet 骨干卷积 (输入输出都是 self.d)
        self.conv1 = CaNetBasicConv(self.d, self.d, residual=True)
        self.conv2 = CaNetBasicConv(self.d, self.d, residual=True)
        
        # 激活函数也对齐 CaNet 的使用方式
        self.act_fn = nn.GELU()

        self.classifier = nn.Linear(self.d, c)
        self.weight_net = NodeWeightGenerator(self.d, self.d // 2)

        #有向无环图(关系树)的构建，可以通过该矩阵A的幂次方(m)研究一个特征在经过m个特征(连接边)后对结果label的影响程度
        # self.A = Parameter(torch.zeros(self.d + 1, self.d + 1))
        # nn.init.uniform_(self.A, a=-0.05, b=0.05)
        # self.M_dag = nn.Linear(self.d + 1, self.d + 1, bias=False)

        # 扩展 DAG 矩阵维度以容纳所有类别节点并彻底删除 self.M_dag
        self.A = Parameter(torch.zeros(self.d + self.c, self.d + self.c))
        nn.init.uniform_(self.A, a=-0.05, b=0.05)
        self.M_dag = nn.Linear(self.d + c, self.d + c, bias=False)

        #self.rff_dim = args.rff_dim if hasattr(args, 'rff_dim') else 128
        # 在 main.py 解析参数后，根据数据集自动分配最优 rff_dim 策略
        if args.dataset in ['cora', 'citeseer', 'pubmed', 'twitch']:
            self.rff_dim = 16  # 轻量级图，强力降噪
        elif args.dataset in ['arxiv', 'elliptic']:
            self.rff_dim = 128 # 重量级图，火力全开
        self.omega = Parameter(torch.randn(1, self.d, self.rff_dim), requires_grad=False)
        self.phi = Parameter(torch.rand(1, self.d, self.rff_dim) * 2 * math.pi, requires_grad=False)

        self.alpha = args.alpha if hasattr(args, 'alpha') else 0.9
        #这里等会看一下，好像没有传入global_size，那就默认2048了
        self.global_size = args.global_size if hasattr(args, 'global_size') else 2048
        self.register_buffer('z_global', torch.zeros(self.global_size, self.d))
        self.register_buffer('w_global', torch.ones(self.global_size, 1) / self.global_size)
        self.global_ptr = 0 
        self.queue_full = False 

        self.gamma = args.gamma if hasattr(args, 'gamma') else 0.99
        self.tau = args.tau if hasattr(args, 'tau') else 0.5 
        self.register_buffer('proto_in', torch.zeros(self.num_envs, c, self.d))
        self.register_buffer('proto_cr', torch.zeros(c, self.d))

        self.dropout = getattr(args, 'dropout',0.0 )
        self.tau = getattr(args, 'tau', 1) 
        
        # 【关键修复】：彻底交出控制权，默认值降至安全线 1e-5
        self.lambda_l1 = getattr(args, 'lambda_l1', 1e-5)
        
        # 【关键修复】：名字必须和 argparse 以及 grid_search.py 里完全一致！
        self.lambda_ind = getattr(args, 'lambda_ind', 0.1)  
        self.lambda_cl = getattr(args, 'lambda_cl', 0.1) 
        self.lambda_dag = getattr(args, 'lambda_dag', 0.1) 

        # 创新点：自适应粒度门控 (Adaptive Granularity Gate)
        # 初始化为 0.0，经过 sigmoid 后刚好是 0.5，代表初始状态下宏观和微观各占一半
        self.dag_gate = Parameter(torch.tensor([0.0]))
        self.reset_parameters()

    def get_masked_A(self):
        mask = torch.ones_like(self.A)
        mask[self.d:, :] = 0.0
        mask.fill_diagonal_(0.0)
        
        A_masked = self.A * mask
        return A_masked
    
    def reset_parameters(self):
        self.pre_fc.reset_parameters()  # 别忘了重置前置层
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

        # 🚨 修复泄露点一：重置 RFF 基底空间
        if hasattr(self, 'omega') and hasattr(self, 'phi'):
            nn.init.normal_(self.omega)
            nn.init.uniform_(self.phi, 0, 2 * math.pi)
            
        # 🚨 修复泄露点二：重置自适应 DAG 门控
        if hasattr(self, 'dag_gate'):
            nn.init.constant_(self.dag_gate, 0.0)
    #从因果图A中生成掩码
    # def get_causal_effect_and_mask(self):
    #     #每一个特征或者结果label不能自己导致自己，故将对角线元素转为0
    #     A_no_diag = self.A - torch.diag(torch.diag(self.A))
    #     #这里是矩阵逐元素的乘法，目的是为了非负性，防止计算多跳影响时正负抵消导致特征对于最终label的影响为0
    #     A_sq = A_no_diag * A_no_diag
    #     #e^M = I + M + \frac{M^2}{2!} + \frac{M^3}{3!} + \dots + \frac{M^k}{k!} + \dots
    #     #这里才是计算多跳影响，用e的幂次方展开累加特征在每一跳(对其余特征或最终label的)影响，可以看到一个特征距离影响目标越远时，影响力越小
    #     C_tot = torch.matrix_exp(A_sq)
        
        
    #     # 提取所有特征对标签的因果效应强度
    #     Ca = C_tot[:self.d, self.d] 
    #     Ca_min, Ca_max = Ca.min(), Ca.max()
    #     # 严格对齐原文：“因果效应为处理后的数值，范围0～1”
    #     Ca_norm = (Ca - Ca_min) / (Ca_max - Ca_min + 1e-8)
    #     # 【极其重要】：绝对不要加 0.5 或 0.05 的偏移！
    #     # 让非因果特征彻底归 0，让分类器彻底断奶！
    #     Ca_mask = Ca_norm


    #     # # 获取矩阵维度
    #     # dim = A_sq.size(0)
    #     # I = torch.eye(dim, device=A_sq.device)
        
    #     # # 预计算 2 跳和 3 跳的矩阵乘法
    #     # A_sq_2 = torch.matmul(A_sq, A_sq)
    #     # A_sq_3 = torch.matmul(A_sq_2, A_sq)
        
    #     # # 严格按照泰勒展开式前三阶累加：I + M + M^2/2! + M^3/3!
    #     # #C_tot = I + A_sq + (A_sq_2 / 2.0) + (A_sq_3 / 6.0)
    #     # C_tot = A_sq + (A_sq_2 / 2.0)


    #     # #提取所有特征对标签的因果效应强度
    #     # Ca = C_tot[:self.d, self.d] 
    #     # Ca_min, Ca_max = Ca.min(), Ca.max()
    #     # # 软掩码机制：如果效应太弱，就当全1；否则把效应缩放到 0.5 到 1.0 之间。
    #     # # 0.5 代表这个特征大概率是垃圾，只保留一半威力；1.0 代表绝对是因果特征，满血保留。
    #     # if Ca_max - Ca_min < 1e-4:
    #     #     # Ca_mask = torch.ones_like(Ca) 
    #     #     # 加上 1.0 后，数值全为 1.0，且不会切断计算图。
    #     #     Ca_mask = (Ca - Ca.detach()) + 1.0
    #     # else:
    #     #     Ca_norm = (Ca - Ca_min) / (Ca_max - Ca_min + 1e-8)
    #     #     # 【抗方差修复 1】: 软掩码机制。将其缩放至 [0.5, 1.0] 之间。
    #     #     # 保证因果效应最弱的特征也能保留 50% 的信息，防止某次 Run 运气差误杀关键特征导致崩溃。
    #     #     Ca_mask = 0.5 + 0.5 * Ca_norm
        
    #     # if Ca_max - Ca_min < 1e-4:
    #     #     # 这里可以用我们之前聊过的 0.5 激活机制，打破僵局
    #     #     Ca_mask = (Ca - Ca.detach()) + 0.5 
    #     # else:
    #     #     Ca_norm = (Ca - Ca_min) / (Ca_max - Ca_min + 1e-8)
            
    #     #     # 【终极锐化修复】：撤销 50% 的低保！
    #     #     # 将缩放范围改为 [0.05, 1.0]。
    #     #     # 0.05 的极小值只为了维持 PyTorch 的反向传播梯度不断裂，
    #     #     # 但在正向传播的物理意义上，它已经几乎被彻底“物理静音”了。
    #     #     Ca_mask = 0.05 + 0.95 * Ca_norm
            
    #     #     # 进阶玩法：如果你想更加极致，甚至可以套一个可导的 Sigmoid 锐化
    #     #     # temperature = 0.1
    #     #     # Ca_mask = torch.sigmoid((Ca_norm - 0.5) / temperature)    
            
    #     Ca_expand = Ca_mask.unsqueeze(1)
    #     #如果两个特征中至少一个与结果标签无关，则消除该标签与其余标签(因果特征或其余不相关标签)的联系
    #     S = 1.0 - torch.matmul(Ca_expand, Ca_expand.t())
    #     return Ca_mask, S

    def get_causal_effect_and_mask(self):
        # A_no_diag = self.A - torch.diag(torch.diag(self.A))
        A_no_diag = self.get_masked_A()
        A_sq = A_no_diag * A_no_diag
        
        dim = A_sq.size(0)
        I = torch.eye(dim, device=A_sq.device)
        A_sq_2 = torch.matmul(A_sq, A_sq)
        C_tot = A_sq + (A_sq_2 / 2.0)

        # # 提取特征节点指向所有类别节点的效应子矩阵
        # Ca_matrix = C_tot[:self.d, self.d:] 
        
        # # 坍缩聚合为一维的总因果强度向量
        # Ca = Ca_matrix.sum(dim=1) 
        
        # Ca_min, Ca_max = Ca.min(), Ca.max()
        
        # if Ca_max - Ca_min < 1e-4:
        #     Ca_mask = (Ca - Ca.detach()) + 0.5 
        # else:
        #     # 绝对的 0 到 1 缩放，彻底剔除虚假特征泄露
        #     Ca_norm = (Ca - Ca_min) / (Ca_max - Ca_min + 1e-8)
        #     Ca_mask = Ca_norm 

        # 修正 Ca 的提取
        Ca_matrix = C_tot[:self.d, self.d:] # 特征到类别的子矩阵
        Ca = torch.max(Ca_matrix, dim=1)[0] # 提取对任意类别最强的因果连接
        # 锐化处理 (Sharpening)
        temperature = 0.05 
        Ca_norm = (Ca - Ca.min()) / (Ca.max() - Ca.min() + 1e-8)
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

        return loss_cl_in + loss_cl_cr

    # def dag_reconstruction_loss_on_prototypes(self):
    #     #这里的proto_in维度：(环境数，类别数，特征维度)，是每个环境中每个类别样本的平均特征
    #     proto_features = self.proto_in.detach().view(-1, self.d)  #展成(环境数*类别数，特征维度)
    #     #将每行对应的标签都设为1/类别数并拼接到proto_feature的最后一列后
    #     labels = torch.arange(self.c, device=self.device).repeat(self.num_envs)
    #     labels_onehot = F.one_hot(labels, num_classes=self.c).float()
    #     label_factor = torch.mean(labels_onehot, dim=1, keepdim=True)
    #     F_factors = torch.cat([proto_features, label_factor], dim=1) 
    #     #这里防止了A是单位矩阵，在单位矩阵的情形下，下述的构造的mse_loss最小，这里防止了这种情形
    #     A_no_diag = self.A - torch.diag(torch.diag(self.A))
    #     #这里用线性近似拟合来计算误差构建矩阵A，如果特征间存在非线性关系效果是不是更差？ 可以改进
    #     reconstructed_F = self.M_dag(torch.matmul(F_factors, A_no_diag))
    #     loss_rec = F.mse_loss(reconstructed_F, F_factors)
    #     #下面这段是无环的约束
    #     A_sq = A_no_diag * A_no_diag
    #     h_A = torch.trace(torch.matrix_exp(A_sq)) - (self.d + 1)
    #     h_A_clipped = torch.clamp(h_A, min=-10.0, max=10.0)
    #     loss_dag_reg = 0.5 * (h_A_clipped ** 2) + self.lambda_l1 * torch.norm(A_no_diag, p=1)

    #     return loss_rec + loss_dag_reg

    # def dag_reconstruction_loss_on_prototypes(self):
    #     proto_features = self.proto_in.detach().view(-1, self.d) 

    #     # 1. 必须拉平特征方差！
    #     proto_features = F.layer_norm(proto_features, (self.d,))

    #     labels = torch.arange(self.c, device=self.device).repeat(self.num_envs)
        
    #     # 生成全尺寸 One-hot 标签矩阵
    #     label_factor = F.one_hot(labels, num_classes=self.c).float()
        
    #     # 拼接后的特征矩阵维度严格对齐 self.d + self.c
    #     F_factors = torch.cat([proto_features, label_factor], dim=1) 
        
    #     # A_no_diag = self.A - torch.diag(torch.diag(self.A))

    #     # 2. 使用受限掩码
    #     A_no_diag = self.get_masked_A()
    #     # 跃过 M_dag，直接执行纯粹的 NOTEARS 线性重构
    #     reconstructed_F = torch.matmul(F_factors, A_no_diag)

    #     # 3. 切分误差，强制放大标签的权重 (d/c 倍)！
    #     loss_rec_X = F.mse_loss(reconstructed_F[:, :self.d], F_factors[:, :self.d])
    #     loss_rec_Y = F.mse_loss(reconstructed_F[:, self.d:], F_factors[:, self.d:])
    #     loss_rec = loss_rec_X + (self.d / self.c) * loss_rec_Y

    #     #loss_rec = F.mse_loss(reconstructed_F, F_factors)
        
    #     # 无环约束与稀疏正则化保持标准计算
    #     A_sq = A_no_diag * A_no_diag
    #     h_A = torch.trace(torch.matrix_exp(A_sq)) - (self.d + self.c)
    #     h_A_clipped = torch.clamp(h_A, min=-10.0, max=10.0)
    #     loss_dag_reg = 0.5 * (h_A_clipped ** 2) + self.lambda_l1 * torch.norm(A_no_diag, p=1)

    #     return loss_rec + loss_dag_reg

    # def dag_reconstruction_loss(self, z_local, y_local):
    #     # 1. 必须拉平特征方差！
    #     z_norm = F.layer_norm(z_local, (self.d,))
        
    #     # 2. 直接使用当前 batch 的真实标签
    #     label_factor = F.one_hot(y_local.squeeze().long(), num_classes=self.c).float()
        
    #     F_factors = torch.cat([z_norm, label_factor], dim=1) 
    #     A_no_diag = self.get_masked_A()
        
    #     # 3. 线性重构
    #     reconstructed_F = torch.matmul(F_factors, A_no_diag)

    #     # 切分误差，强制放大标签的权重 (d/c 倍)！
    #     loss_rec_X = F.mse_loss(reconstructed_F[:, :self.d], F_factors[:, :self.d])
    #     loss_rec_Y = F.mse_loss(reconstructed_F[:, self.d:], F_factors[:, self.d:])
    #     loss_rec = loss_rec_X + (self.d / self.c) * loss_rec_Y
        
    #     # 无环约束
    #     A_sq = A_no_diag * A_no_diag
    #     h_A = torch.trace(torch.matrix_exp(A_sq)) - (self.d + self.c)
    #     h_A_clipped = torch.clamp(h_A, min=-10.0, max=10.0)
    #     loss_dag_reg = 0.5 * (h_A_clipped ** 2) + self.lambda_l1 * torch.norm(A_no_diag, p=1)

    #     return loss_rec + loss_dag_reg


    # def dag_reconstruction_loss(self, z_local, y_local):
    #     # -----------------------------------------
    #     # 引擎 A：计算微观节点级别的重构误差 (Micro)
    #     # -----------------------------------------
    #     z_norm = F.layer_norm(z_local, (self.d,))
    #     y_onehot = F.one_hot(y_local.squeeze().long(), num_classes=self.c).float()
    #     F_micro = torch.cat([z_norm, y_onehot], dim=1)
        
    #     A_no_diag = self.get_masked_A()
    #     rec_micro = torch.matmul(F_micro, A_no_diag)
        
    #     loss_rec_X_micro = F.mse_loss(rec_micro[:, :self.d], F_micro[:, :self.d])
    #     loss_rec_Y_micro = F.mse_loss(rec_micro[:, self.d:], F_micro[:, self.d:])
    #     loss_micro = loss_rec_X_micro + (self.d / self.c) * loss_rec_Y_micro

    #     # -----------------------------------------
    #     # 引擎 B：计算宏观原型级别的重构误差 (Macro)
    #     # -----------------------------------------
    #     proto_features = self.proto_in.detach().view(-1, self.d)
    #     proto_norm = F.layer_norm(proto_features, (self.d,))
    #     labels_seq = torch.arange(self.c, device=self.device).repeat(self.num_envs)
    #     proto_y_onehot = F.one_hot(labels_seq, num_classes=self.c).float()
    #     F_macro = torch.cat([proto_norm, proto_y_onehot], dim=1)
        
    #     rec_macro = torch.matmul(F_macro, A_no_diag)
        
    #     loss_rec_X_macro = F.mse_loss(rec_macro[:, :self.d], F_macro[:, :self.d])
    #     loss_rec_Y_macro = F.mse_loss(rec_macro[:, self.d:], F_macro[:, self.d:])
    #     loss_macro = loss_rec_X_macro + (self.d / self.c) * loss_rec_Y_macro

    #     # -----------------------------------------
    #     # 核心创新：可学习的动态加权融合
    #     # -----------------------------------------
    #     # 将无界的参数通过 sigmoid 压缩到 [0, 1] 之间
    #     alpha = torch.sigmoid(self.dag_gate) 
        
    #     # 自动平衡：alpha 控制原型权重，(1-alpha) 控制节点权重
    #     loss_rec_fused = alpha * loss_macro + (1.0 - alpha) * loss_micro

    #     # -----------------------------------------
    #     # 共享的无环约束与稀疏正则化
    #     # -----------------------------------------
    #     A_sq = A_no_diag * A_no_diag
    #     h_A = torch.trace(torch.matrix_exp(A_sq)) - (self.d + self.c)
    #     h_A_clipped = torch.clamp(h_A, min=-10.0, max=10.0)
    #     loss_dag_reg = 0.5 * (h_A_clipped ** 2) + self.lambda_l1 * torch.norm(A_no_diag, p=1)

    #     return loss_rec_fused + loss_dag_reg

    
    def dag_reconstruction_loss(self, z_local, y_local):
        # 获取无对角线的邻接矩阵 A
        A_no_diag = self.get_masked_A()
        
        # -----------------------------------------
        # 引擎 A：微观节点级别 (Micro)
        # -----------------------------------------
        z_norm = F.layer_norm(z_local, (self.d,))
        y_onehot = F.one_hot(y_local.squeeze().long(), num_classes=self.c).float()
        F_micro = torch.cat([z_norm, y_onehot], dim=1)
        
        # 【改造点 1】：引入 M_dag 进行映射，模拟特征与标签间的映射函数 G [cite: 211, 217]
        rec_micro = self.M_dag(torch.matmul(F_micro, A_no_diag))
        
        # 特征重构保持 MSE (L2 Norm) [cite: 222, 224]
        loss_rec_X_micro = F.mse_loss(rec_micro[:, :self.d], F_micro[:, :self.d])
        # 【改造点 2】：标签重构改用交叉熵损失 (L_ce) 
        logits_Y_micro = rec_micro[:, self.d:]
        loss_rec_Y_micro = F.cross_entropy(logits_Y_micro, y_local.squeeze().long())
        
        # 结合损失，这里可以沿用你之前的维度平衡系数
        loss_micro = loss_rec_X_micro + (self.d / self.c) * loss_rec_Y_micro

        # -----------------------------------------
        # 引擎 B：宏观原型级别 (Macro) - 论文推荐的稳定优化方式 [cite: 251, 253]
        # -----------------------------------------
        proto_features = self.proto_in.detach().view(-1, self.d)
        proto_norm = F.layer_norm(proto_features, (self.d,))
        # 原型对应的真实标签索引
        labels_seq = torch.arange(self.c, device=self.device).repeat(self.num_envs)
        proto_y_onehot = F.one_hot(labels_seq, num_classes=self.c).float()
        F_macro = torch.cat([proto_norm, proto_y_onehot], dim=1)
        
        # 同样应用 M_dag
        rec_macro = self.M_dag(torch.matmul(F_macro, A_no_diag))
        
        loss_rec_X_macro = F.mse_loss(rec_macro[:, :self.d], F_macro[:, :self.d])
        # 标签重构同样改用交叉熵
        logits_Y_macro = rec_macro[:, self.d:]
        loss_rec_Y_macro = F.cross_entropy(logits_Y_macro, labels_seq.long())
        
        loss_macro = loss_rec_X_macro + (self.d / self.c) * loss_rec_Y_macro

        # -----------------------------------------
        # 动态融合与约束 (保持你原有的创新逻辑)
        # -----------------------------------------
        alpha = torch.sigmoid(self.dag_gate) 
        loss_rec_fused = alpha * loss_macro + (1.0 - alpha) * loss_micro

        # 共享的无环约束 (h(A)) 与稀疏正则化 (L1) [cite: 230, 232]
        A_sq = A_no_diag * A_no_diag
        # h(A) = Tr(e^(A*A)) - (d + c)
        h_A = torch.trace(torch.matrix_exp(A_sq)) - (self.d + self.c)
        h_A_clipped = torch.clamp(h_A, min=-10.0, max=10.0)
        loss_dag_reg = 0.5 * (h_A_clipped ** 2) + self.lambda_l1 * torch.norm(A_no_diag, p=1)

        return loss_rec_fused + loss_dag_reg

    
    
    def compute_weighted_independence_loss(self, z_local, w_local, S):
        # 【抗方差修复 2】: 动态缓存绕过。
        # 图数据一次前向传播如果节点足够多(>256)，当前的协方差估计已经足够精准。
        # 此时强行使用旧 Epoch 的全局缓存只会引入滞后的噪声，破坏稳定性。
        if z_local.size(0) >= 256:
            z_concat, w_concat = z_local, w_local
        else:
            valid_global = self.global_size if self.queue_full else self.global_ptr
            if valid_global > 0:
                z_concat = torch.cat([z_local, self.z_global[:valid_global]], dim=0)
                w_concat = torch.cat([w_local, self.w_global[:valid_global]], dim=0)
            else:
                z_concat, w_concat = z_local, w_local

        # w_concat = w_concat / (torch.sum(w_concat) + 1e-8)
        # 【救命神药：强制标签平滑】
        w_concat_smooth = w_concat + 0.05
        w_concat = w_concat_smooth / torch.sum(w_concat_smooth)
        if z_local.size(0) >= 256: # 随便挑个条件限制打印频率
            print(f"DEBUG w_concat: max={w_concat.max().item():.6f}, min={w_concat.min().item():.6f}")
        z_mean = torch.mean(z_concat, dim=0, keepdim=True)
        z_std = torch.std(z_concat, dim=0, keepdim=True) + 1e-5
        z_concat_norm = (z_concat - z_mean) / z_std
        z_expand = z_concat_norm.unsqueeze(-1)
        # z_expand = z_concat.unsqueeze(-1) 
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


    # def compute_weighted_independence_loss(self, z_local, w_local, S):
    #     # ... (保留你之前的全局队列逻辑) ...
    #     if z_local.size(0) >= 256:
    #         z_concat, w_concat = z_local, w_local
    #     else:
    #         valid_global = self.global_size if self.queue_full else self.global_ptr
    #         if valid_global > 0:
    #             z_concat = torch.cat([z_local, self.z_global[:valid_global]], dim=0)
    #             w_concat = torch.cat([w_local, self.w_global[:valid_global]], dim=0)
    #         else:
    #             z_concat, w_concat = z_local, w_local
    #     w_concat_smooth = w_concat + 0.05
    #     w_concat = w_concat_smooth / torch.sum(w_concat_smooth)
        
    #     z_mean = torch.mean(z_concat, dim=0, keepdim=True)
    #     z_std = torch.std(z_concat, dim=0, keepdim=True) + 1e-5
    #     z_concat_norm = (z_concat - z_mean) / z_std
        
    #     # 🚨 删除 RFF 逻辑，直接计算线性加权协方差！🚨
    #     # z_centered 维度: [N, D]
    #     z_centered = z_concat_norm - torch.sum(w_concat * z_concat_norm, dim=0, keepdim=True)
        
    #     # cov 维度: [D, D] (例如 64 x 64，极其稳定)
    #     cov = torch.matmul((w_concat * z_centered).t(), z_centered) 
    #     cov_sq = cov ** 2
        
    #     # 直接使用 S 掩码屏蔽因果特征间的惩罚
    #     ind_loss = torch.sum(torch.triu(S * cov_sq, diagonal=1))
    #     num_pairs = (self.d * (self.d - 1)) / 2.0
    #     return ind_loss / num_pairs


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
        # x = F.dropout(x, self.dropout, training=self.training)
        
        # z = F.relu(self.conv1(x, edge_index))
        # z = F.dropout(z, self.dropout, training=self.training)
        # z = self.conv2(z, edge_index)
        # 1. 严格对齐 CaNet 的第一步：Dropout -> 投影 -> ReLU
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.pre_fc(x))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.act_fn(self.conv1(h, edge_index))
        h = F.dropout(h, self.dropout, training=self.training)
        z = self.conv2(h, edge_index) 
        # 注意：在进入 CIW 核心因果逻辑前，z 已经是经过残差和拼接的强大特征了！


        w = self.weight_net(z.detach()) 
        
        Ca_norm, S = self.get_causal_effect_and_mask()
        
        z_invariant = z * Ca_norm.detach().unsqueeze(0)
        #这里原ciw的流程中好像有残差连接？后面看一下
        z_invariant = F.dropout(z_invariant, self.dropout, training=self.training)
        
        logits = self.classifier(z_invariant)
        
        if self.training:
            return logits, z, w, S, z_invariant
        else:
            return logits

    def loss_compute(self, data, criterion, args):
        # 插在 loss_compute 函数的最前面
        print(f"--- DEBUG EPOCH ---")
        print(f"Lambdas: ind={self.lambda_ind}, dag={self.lambda_dag}")
        print(f"A_matrix_sum: {self.A.abs().sum().item():.6f}")
        if hasattr(self, 'z_global'):
            print(f"z_global_sum: {self.z_global.abs().sum().item():.6f}")
            print(f"-------------------")
        
        x, edge_index, y = data.x, data.edge_index, data.y
        train_idx = data.train_idx
        envs = data.env if hasattr(data, 'env') else None
        
        logits, z_local, w_local, S, z_invariant = self.forward(x, edge_index, training=True)
        
        z_tr = z_local[train_idx]
        w_tr = w_local[train_idx]
        y_tr = y[train_idx]
        inv_tr = z_invariant[train_idx]
        env_tr = envs[train_idx] if envs is not None else None
        
        z_tr_norm = F.normalize(z_tr, p=2, dim=1)
        # loss_ind = self.compute_weighted_independence_loss(z_tr_norm.detach(), w_tr, S)

        loss_ind = self.compute_weighted_independence_loss(z_tr.detach(), w_tr, S.detach())
        
        self.update_prototypes(z_tr, inv_tr, y_tr, env_tr)
        loss_cl = self.compute_contrastive_loss(z_tr, inv_tr, y_tr, env_tr)
        loss_dag = self.dag_reconstruction_loss(z_tr.detach(), y_tr)
        
        # if args.dataset in ('twitch', 'elliptic'):
        #     if y_tr.shape[1] == 1 and logits.shape[1] > 1:
        #         true_label = F.one_hot(y_tr.squeeze().long(), logits.shape[1]).float()
        #     else:
        #         true_label = y_tr.float()
        #     sup_loss_raw = F.binary_cross_entropy_with_logits(logits[train_idx], true_label, reduction='none')
        #     if len(sup_loss_raw.shape) > 1:
        #         sup_loss_raw = sup_loss_raw.mean(dim=1)
        # else:
        #     sup_loss_raw = F.cross_entropy(logits[train_idx], y_tr.squeeze().long(), reduction='none')
        # 在 loss_compute 函数中修改：
        
        if args.dataset in ('twitch', 'elliptic'):
            if y_tr.shape[1] == 1 and logits.shape[1] > 1:
                true_label = F.one_hot(y_tr.squeeze().long(), logits.shape[1]).float()
            else:
                true_label = y_tr.float()
            # 删掉写死的 F.binary_... ，直接用外部带 pos_weight 的 criterion！
            sup_loss_raw = criterion(logits[train_idx], true_label)
            if len(sup_loss_raw.shape) > 1:
                sup_loss_raw = sup_loss_raw.mean(dim=1)
        else:
            sup_loss_raw = criterion(logits[train_idx], y_tr.squeeze().long())


        w_tr_norm = w_tr.squeeze() / torch.clamp(w_tr.squeeze().mean(), min=1e-3)
        loss_cls = torch.mean(w_tr_norm.detach() * sup_loss_raw)
        
        self.update_global_queue(z_tr, w_tr)
        
        total_loss = loss_cls + self.lambda_ind * loss_ind + self.lambda_dag * loss_dag + self.lambda_cl * loss_cl
        return total_loss, loss_cls.item(), (self.lambda_ind * loss_ind).item(), (self.lambda_dag * loss_dag).item(), (self.lambda_cl * loss_cl).item()
        # return total_loss