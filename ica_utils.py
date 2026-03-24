import torch
import time
from sklearn.decomposition import FastICA
import warnings

def apply_ica_disentanglement(data, n_components=None, random_state=42):
    """
    通用的图节点特征 ICA 解耦模块
    :param data: PyG 的 Data 对象 (需包含 data.x)
    :param n_components: 降维后的独立成分数。如果为 None，则自适应保留一定维度。
    """
    print("\n[INFO] 启动 FastICA 因果特征解耦 (Causal Feature Disentanglement)...")
    start_time = time.time()
    
    original_x = data.x.cpu().numpy()
    num_nodes, num_features = original_x.shape
    
    # 自适应维度策略：如果未指定，且原始特征维度极大(如Citeseer的3700+)，则进行降维去噪
    # 如果原始特征维度较小，则保持维度不变，仅做正交解耦
    if n_components is None:
        n_components = min(num_features, 512) 
        
    # 初始化 FastICA (使用 unit-variance 白化以符合因果独立假设)
    # 忽略 sklearn 的收敛警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=n_components, 
                      random_state=random_state, 
                      max_iter=1000, 
                      whiten='unit-variance')
        
        # 1. 拟合并转换出解耦后的独立因果特征
        disentangled_x = ica.fit_transform(original_x)
    
    # 2. 将纯净特征重新挂载回 PyG 的 data 对象
    data.x = torch.tensor(disentangled_x, dtype=torch.float32)
    
    print(f"[INFO] 解耦完成！耗时: {time.time() - start_time:.2f} 秒")
    print(f"[INFO] 特征纠缠解除：维度从 {num_features} 映射至 {data.x.shape[1]} 个独立因果成分\n")
    
    return data