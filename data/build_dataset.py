#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据集预处理脚本
将真实的传播数据转换为IVGD项目所需的.SG格式

数据输入：
- {DATASET_NAME}/edges.txt: 网络拓扑结构
- {DATASET_NAME}/cascades.txt: 传播级联数据

数据输出：
- {DATASET_NAME}_25c.SG: IVGD可用的数据文件（SparseGraph对象）

数据格式说明：
- SparseGraph对象包含：
  1. adj_matrix: 邻接矩阵 (N, N) - 无向图，无权重
  2. prob_matrix: 概率矩阵 (N, N) - 边激活概率，随机生成在[0.0001, 0.15]
  3. influ_mat_list: 影响力快照列表 (M, N, T) - M个样本，N个节点，T个时间步
     - influ_mat_list[:, :, 0]: t=0时刻的种子节点
     - influ_mat_list[:, :, 1:T-1]: t=1到t=T-2的中间扩散过程
     - influ_mat_list[:, :, T-1]: t=T-1时刻的最终影响范围
"""

import numpy as np
import pickle
import scipy.sparse as sp
from collections import defaultdict
import os
import sys

# 添加项目根目录到路径，以便导入SparseGraph类
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data.sparsegraph import SparseGraph

# ============================================================================
# 配置参数 - 所有可调参数集中在此处
# ============================================================================

# 数据集名称（修改此处来处理不同数据集：android, twitter, christianity, douban等）
DATASET_NAME = 'android'

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))   # 脚本所在目录
DATASET_DIR = os.path.join(SCRIPT_DIR, DATASET_NAME)   # 数据集文件夹
EDGES_FILE = os.path.join(DATASET_DIR, 'edges.txt')   # 边文件
CASCADES_FILE = os.path.join(DATASET_DIR, 'cascades.txt')   # 级联文件
OUTPUT_FILE = os.path.join(SCRIPT_DIR, f'{DATASET_NAME}_25c.SG')   # 输出.SG文件
U2IDX_FILE = os.path.join(DATASET_DIR, 'u2idx.pickle')   # 用户ID映射文件
IDX2U_FILE = os.path.join(DATASET_DIR, 'idx2u.pickle')   # 索引到用户映射文件

# 概率矩阵参数（与karate数据集保持一致）
PROB_MIN = 0.0001  # 最小概率值
PROB_MAX = 0.15    # 最大概率值
RANDOM_SEED = 42   # 随机种子，保证可重复性

# 级联处理参数
NUM_TIMESTEPS = 25  # 时间步数（T=25，包括t=0到t=24）
SOURCE_RATIO = 0.05  # 源节点时间比例（前5%时间内出现的节点作为源节点）
MIN_CASCADE_SIZE = 10  # 最小级联大小（节点数）
MIN_SOURCE_NODES = 1  # 最小源节点数
MAX_SAMPLES = None  # 最大样本数（None表示不限制）

# 图类型参数
DIRECTED_GRAPH = False  # 是否为有向图（False表示无向图，与karate保持一致）

# ============================================================================
# 脚本开始
# ============================================================================

print("=" * 70)
print(f"{DATASET_NAME.upper()} 数据集预处理脚本")
print("=" * 70)
print(f"\n配置参数:")
print(f"  - 数据集名称: {DATASET_NAME}")
print(f"  - 数据集目录: {DATASET_DIR}")
print(f"  - 输出文件: {OUTPUT_FILE}")
print(f"  - 源节点比例: {SOURCE_RATIO * 100}%")
print(f"  - 基础传播概率: {BASE_PROB}")
print(f"  - 图类型: {'有向图' if DIRECTED_GRAPH else '无向图'}")


# ============================================================================
# 步骤1: 读取边文件，构建用户ID映射
# ============================================================================
print("\n[步骤1] 读取网络拓扑结构...")

unique_users = set()

# 读取所有边，收集所有唯一用户
edges_list = []
with open(EDGES_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(',')
            if len(parts) == 2:
                user1, user2 = parts[0].strip(), parts[1].strip()
                unique_users.add(user1)
                unique_users.add(user2)
                edges_list.append((user1, user2))

print(f"  - 读取边数: {len(edges_list)}")
print(f"  - 唯一用户数: {len(unique_users)}")

# 构建用户ID到索引的映射 (从0开始)
sorted_users = sorted(list(unique_users), key=lambda x: int(x))
u2idx = {user: idx for idx, user in enumerate(sorted_users)}
idx2u = {idx: user for user, idx in u2idx.items()}

print(f"  - 用户ID映射完成: {len(u2idx)} 个用户")

# 保存映射关系
with open(U2IDX_FILE, 'wb') as f:
    pickle.dump(u2idx, f)
with open(IDX2U_FILE, 'wb') as f:
    pickle.dump(idx2u, f)

# ============================================================================
# 步骤2: 构建邻接矩阵
# ============================================================================
print("\n[步骤2] 构建邻接矩阵...")

n_nodes = len(u2idx)
adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)

# 构建邻接矩阵 (有向图或无向图)
for user1, user2 in edges_list:
    idx1, idx2 = u2idx[user1], u2idx[user2]
    adj_matrix[idx1, idx2] = 1
    if not DIRECTED_GRAPH:
        adj_matrix[idx2, idx1] = 1  # 无向图需要对称

# 转换为稀疏矩阵
adj_sparse = sp.csr_matrix(adj_matrix)
print(f"  - 邻接矩阵形状: {adj_sparse.shape}")
print(f"  - 非零边数: {adj_sparse.nnz}")

# ============================================================================
# 步骤3: 构建传播概率矩阵
# ============================================================================
print("\n[步骤3] 构建传播概率矩阵...")

# 基于度数的概率分配策略
degree = adj_matrix.sum(axis=1)  # 每个节点的度数
prob_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)

# 根据配置的衰减方式分配概率
for i in range(n_nodes):
    if degree[i] > 0:
        for j in range(n_nodes):
            if adj_matrix[i, j] == 1:
                if PROB_DECAY == 'sqrt':
                    # 概率与度数的平方根成反比
                    prob_matrix[i, j] = BASE_PROB / np.sqrt(degree[i])
                elif PROB_DECAY == 'linear':
                    # 概率与度数成反比
                    prob_matrix[i, j] = BASE_PROB / degree[i]
                else:  # 'const'
                    # 常数概率
                    prob_matrix[i, j] = BASE_PROB

prob_sparse = sp.csr_matrix(prob_matrix)
print(f"  - 传播概率矩阵形状: {prob_sparse.shape}")
print(f"  - 概率范围: [{prob_matrix[prob_matrix > 0].min():.4f}, {prob_matrix[prob_matrix > 0].max():.4f}]")

# ============================================================================
# 步骤4: 处理级联数据，构建训练样本
# ============================================================================
print("\n[步骤4] 处理传播级联数据...")

all_samples = []
valid_cascades = 0
skipped_cascades = 0

with open(CASCADES_FILE, 'r') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        
        # 解析级联数据
        chunks = line.split(',')
        userlist = []
        timestamplist = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            try:
                parts = chunk.split()
                if len(parts) == 2:
                    user, timestamp = parts
                elif len(parts) == 3:
                    root, user, timestamp = parts
                    # 添加root节点
                    if root in u2idx:
                        userlist.append(u2idx[root])
                        timestamplist.append(float(timestamp))
                else:
                    continue
                
                # 添加当前用户节点
                if user in u2idx:
                    userlist.append(u2idx[user])
                    timestamplist.append(float(timestamp))
            except Exception as e:
                # print(f"  警告: 第{line_num}行解析错误 - {chunk}: {e}")
                continue
        
        # 检查级联是否有效
        if len(userlist) < MIN_CASCADE_SIZE:
            skipped_cascades += 1
            continue
        
        # 去重，保留第一次出现的节点
        seen_users = set()
        unique_userlist = []
        unique_timestamplist = []
        for user, timestamp in zip(userlist, timestamplist):
            if user not in seen_users:
                seen_users.add(user)
                unique_userlist.append(user)
                unique_timestamplist.append(timestamp)
        
        userlist = unique_userlist
        timestamplist = unique_timestamplist
        
        if len(userlist) < MIN_CASCADE_SIZE:
            skipped_cascades += 1
            continue
        
        # 提取源节点向量 (前SOURCE_RATIO时间内出现的节点)
        # 计算时间窗口：从最早时间到 (最早时间 + 时间范围 * SOURCE_RATIO)
        min_timestamp = min(timestamplist)
        max_timestamp = max(timestamplist)
        time_range = max_timestamp - min_timestamp
        
        if time_range > 0:
            # 前5%时间内的节点作为源节点
            source_time_threshold = min_timestamp + time_range * SOURCE_RATIO
            source_nodes = [user for user, ts in zip(userlist, timestamplist) 
                          if ts <= source_time_threshold]
        else:
            # 如果所有节点时间戳相同，取前MIN_SOURCE_NODES个节点
            source_nodes = userlist[:MIN_SOURCE_NODES]
        
        # 确保至少有MIN_SOURCE_NODES个源节点
        if len(source_nodes) < MIN_SOURCE_NODES:
            source_nodes = userlist[:MIN_SOURCE_NODES]
        
        # 扩散结果向量 (所有节点)
        influence_nodes = userlist
        
        # 构建二值向量
        seed_vector = np.zeros(n_nodes, dtype=np.float32)
        influence_vector = np.zeros(n_nodes, dtype=np.float32)
        
        for node in source_nodes:
            seed_vector[node] = 1.0
        
        for node in influence_nodes:
            influence_vector[node] = 1.0
        
        # 检查是否有有效的源节点和扩散节点
        if seed_vector.sum() > 0 and influence_vector.sum() > seed_vector.sum():
            all_samples.append((seed_vector, influence_vector))
            valid_cascades += 1

        print(f"  - 总级联数: {line_num}")
        print(f"  - 有效级联数: {valid_cascades}")
        print(f"  - 跳过级联数: {skipped_cascades}")

        if valid_cascades == 0:
            print("\n错误: 没有有效的级联数据!")
            exit(1)

# ============================================================================
# 步骤5: 构建inverse_pairs张量
# ============================================================================
print("\n[步骤5] 构建训练样本张量...")

# 按照原始数据格式构建: [n_batches, batch_size, n_nodes, 2]
n_samples = len(all_samples)
n_batches = n_samples  # 每个样本一个batch

# 构建inverse_pairs
inverse_pairs_list = []
for i, (seed_vec, influ_vec) in enumerate(all_samples):
    # 每个样本: [n_nodes, 2]
    sample = np.stack([seed_vec, influ_vec], axis=1)  # [n_nodes, 2]
    inverse_pairs_list.append(sample)

# 转换为torch张量: [n_samples, n_nodes, 2]
inverse_pairs = np.array(inverse_pairs_list)  # [n_samples, n_nodes, 2]

# 为了匹配原始格式 [n_batches, batch_size, n_nodes, 2]
# 我们在第1维添加一个维度
inverse_pairs = inverse_pairs[:, np.newaxis, :, :]  # [n_samples, 1, n_nodes, 2]

inverse_pairs_tensor = torch.FloatTensor(inverse_pairs)

print(f"  - inverse_pairs形状: {inverse_pairs_tensor.shape}")
print(f"  - 样本数: {inverse_pairs_tensor.shape[0]}")
print(f"  - 批次大小: {inverse_pairs_tensor.shape[1]}")
print(f"  - 节点数: {inverse_pairs_tensor.shape[2]}")

# ============================================================================
# 步骤6: 保存为.SG文件
# ============================================================================
print("\n[步骤6] 保存为.SG文件...")

output_data = {
    'adj': adj_sparse,
    'prob': prob_sparse,
    'inverse_pairs': inverse_pairs_tensor
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(output_data, f)

print(f"  - 数据已保存到: {OUTPUT_FILE}")

# ============================================================================
# 验证数据
# ============================================================================
print("\n[步骤7] 验证生成的数据...")

with open(OUTPUT_FILE, 'rb') as f:
    loaded_data = pickle.load(f)

print(f"  - adj形状: {loaded_data['adj'].shape}")
print(f"  - prob形状: {loaded_data['prob'].shape}")
print(f"  - inverse_pairs形状: {loaded_data['inverse_pairs'].shape}")

# 检查数据样本
sample_idx = 0
sample = loaded_data['inverse_pairs'][sample_idx, 0, :, :]
seed_vec = sample[:, 0]
influ_vec = sample[:, 1]
print(f"\n  样本 #{sample_idx}:")
print(f"    - 源节点数: {int(seed_vec.sum())}")
print(f"    - 扩散节点数: {int(influ_vec.sum())}")
print(f"    - 扩散率: {influ_vec.sum() / n_nodes * 100:.2f}%")

