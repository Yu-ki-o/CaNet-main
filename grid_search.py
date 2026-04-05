import itertools
import subprocess
import time
import os

# ==========================================
# 1. 定义你要扫描的超参数搜索空间
# ==========================================
# 这里放入我们之前讨论过的“突破 47% 瓶颈”的核心破局参数
param_grid = {
    # 🆕 新增的因果图约束参数：
    'lambda_dag': [0.1, 0.5, 1.0],      # DAG 无环约束：通常需要稍微大一点以保证生成的是 DAG
    'lambda_ind': [0.01, 0.1, 1.0],          # 独立性约束：类似于之前的 ciw_loss，控制在 0.1 左右防坍塌
    'lambda_cl':  [0.05, 0.1, 1.0],     # 对比/聚类损失：促进特征的区分度
    
    
    # 'tau': [0.1, 0.5, 1.0],            # Gumbel 温度：0.1 强制专家分化，1.0 平均分配
    # 'weight_decay': [5e-5, 5e-4],      # 权重衰减：防止在训练集上过拟合
    # 'dropout': [0.2, 0.5]              # 随机失活：进一步提升泛化能力
}

# 固定的基础命令
base_cmd = [
    "python", "main.py",
    "--dataset", "twitch",
    "--backbone", "gcn",
    "--env_type", "graph",
    "--combine_result",
    "--store",
    "--epochs","250",
    "--weight_decay","5e-5",
    "--tau","3",
    "--dropout","0",
    "--result_name","ciw6"
]

# 创建日志文件夹
log_dir = "search_logs"
os.makedirs(log_dir, exist_ok=True)

# ==========================================
# 2. 生成所有超参数组合并执行
# ==========================================
keys = list(param_grid.keys())
values = list(param_grid.values())
# 生成笛卡尔积（所有可能的组合）
combinations = list(itertools.product(*values))

print(f"🚀 开始执行超参数扫描，共计 {len(combinations)} 组实验...")
print("-" * 50)

for i, combination in enumerate(combinations):
    # 将当前的参数组合打包成字典
    current_params = dict(zip(keys, combination))
    
    # 构建当前组合的完整命令
    cmd = base_cmd.copy()
    result_name_parts = ["ciw"]
    
    for key, value in current_params.items():
        cmd.extend([f"--{key}", str(value)])
        # 把参数拼接到 result_name 里，方便你以后看保存的文件名就能知道是哪组参数
        result_name_parts.append(f"{key}{value}")
        
    # 拼接 result_name
    result_name = "_".join(result_name_parts)
    cmd.extend(["--result_name", result_name])
    
    # 打印当前正在运行的命令
    cmd_str = " ".join(cmd)
    print(f"\n[{i+1}/{len(combinations)}] 正在运行: \n{cmd_str}")
    
    # 设置日志文件，把这组参数的终端输出保存下来，免得刷屏丢失
    log_file_path = os.path.join(log_dir, f"exp_{i+1}.log")
    
    start_time = time.time()
    
    # 执行命令并将输出重定向到日志文件
    with open(log_file_path, "w") as log_file:
        log_file.write(f"执行命令: {cmd_str}\n")
        log_file.write("=" * 50 + "\n")
        
        # 启动子进程运行 main.py
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        process.wait() # 等待当前实验跑完
        
    cost_time = time.time() - start_time
    print(f"✅ 实验 {i+1} 完成! 耗时: {cost_time:.2f} 秒. 日志已保存至: {log_file_path}")

print("\n🎉 所有网格搜索实验执行完毕！请去 check 你的最优结果吧！")