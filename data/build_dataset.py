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

# 概率矩阵参数（基于度中心性）
PROB_HIGH_DEGREE = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]  # 高度节点的概率选项
PROB_LOW_DEGREE = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]   # 低度节点的概率选项
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
print(f"  - 时间步数: {NUM_TIMESTEPS}")
print(f"  - 源节点时间比例: 前10%")
print(f"  - 高度节点概率选项: {PROB_HIGH_DEGREE}")
print(f"  - 低度节点概率选项: {PROB_LOW_DEGREE}")
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

# 基于度中心性的概率分配策略
np.random.seed(RANDOM_SEED)
prob_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)

# 计算每个节点的度数
degree = adj_matrix.sum(axis=1)  # 每个节点的度数
avg_degree = np.mean(degree)  # 平均度数

print(f"  - 平均度数: {avg_degree:.2f}")

# 为每条边分配概率（基于源节点的度中心性）
for i in range(n_nodes):
    for j in range(n_nodes):
        if adj_matrix[i, j] == 1:
            # 根据节点i的度数决定传播概率
            if degree[i] > avg_degree:
                # 高度节点：从较高概率范围随机选择
                p = np.random.choice(PROB_HIGH_DEGREE)
            else:
                # 低度节点：从较低概率范围随机选择
                p = np.random.choice(PROB_LOW_DEGREE)
            prob_matrix[i, j] = p

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
        
        # 归一化时间戳到[0, 1]范围
        min_timestamp = min(timestamplist)
        max_timestamp = max(timestamplist)
        time_range = max_timestamp - min_timestamp
        
        if time_range > 0:
            # 归一化时间戳
            normalized_timestamps = [(ts - min_timestamp) / time_range for ts in timestamplist]
        else:
            # 如果所有节点时间戳相同，均匀分配
            normalized_timestamps = [i / len(timestamplist) for i in range(len(timestamplist))]
        
        # 构建时间序列影响力矩阵 (N, T)，T=25个时间步
        influ_mat = np.zeros((n_nodes, NUM_TIMESTEPS), dtype=np.float32)
        
        # 确定源节点：前5%时间内出现的节点作为t=0的种子节点
        source_time_threshold = 0.05  # 5%的时间
        source_nodes = [user for user, norm_ts in zip(userlist, normalized_timestamps) 
                       if norm_ts <= source_time_threshold]
        
        # 确保至少有MIN_SOURCE_NODES个源节点
        if len(source_nodes) < MIN_SOURCE_NODES:
            source_nodes = userlist[:MIN_SOURCE_NODES]
        
        # 设置t=0时刻的种子节点
        for node in source_nodes:
            influ_mat[node, 0] = 1.0
        
        # 将剩余时间（10%-100%）的级联过程分配到24个时间步（t=1到t=24）
        remaining_nodes = [(user, norm_ts) for user, norm_ts in zip(userlist, normalized_timestamps) 
                          if norm_ts > source_time_threshold]
        
        if len(remaining_nodes) > 0:
            # 将时间范围[0.1, 1.0]映射到时间步[1, 24]
            for user, norm_ts in remaining_nodes:
                # 映射到时间步: 0.1 -> 1, 1.0 -> 24
                time_step = int((norm_ts - source_time_threshold) / (1.0 - source_time_threshold) * 23) + 1
                time_step = min(time_step, NUM_TIMESTEPS - 1)  # 确保不超过24
                
                # 该节点在time_step时刻及之后都被激活
                influ_mat[user, time_step:] = 1.0
        else:
            # 如果没有剩余节点，最后一个时间步与第一个时间步相同
            influ_mat[:, 1:] = influ_mat[:, 0:1]
        
        # 如果某些时间步没有新增节点，用前一个时间步的状态填充
        for t in range(1, NUM_TIMESTEPS):
            # 保证单调性：后面的时间步至少包含前面时间步的所有激活节点
            influ_mat[:, t] = np.maximum(influ_mat[:, t], influ_mat[:, t-1])
        
        # 检查是否有有效的级联（源节点数 > 0 且 最终影响节点数 > 源节点数）
        seed_count = influ_mat[:, 0].sum()
        final_count = influ_mat[:, -1].sum()
        
        if seed_count > 0 and final_count > seed_count:
            all_samples.append(influ_mat)
            valid_cascades += 1
            
            # 限制最大样本数
            if MAX_SAMPLES is not None and valid_cascades >= MAX_SAMPLES:
                break

print(f"  - 总级联数: {line_num}")
print(f"  - 有效级联数: {valid_cascades}")
print(f"  - 跳过级联数: {skipped_cascades}")

if valid_cascades == 0:
    print("\n错误: 没有有效的级联数据!")
    exit(1)

# ============================================================================
# 步骤5: 构建influ_mat_list张量
# ============================================================================
print("\n[步骤5] 构建训练样本张量...")

# 将所有样本堆叠成三维数组: [M, N, T]
# M: 样本数量
# N: 节点数量
# T: 时间步数（25）
influ_mat_list = np.array(all_samples, dtype=np.float32)  # [M, N, T]

print(f"  - influ_mat_list形状: {influ_mat_list.shape}")
print(f"  - 样本数 (M): {influ_mat_list.shape[0]}")
print(f"  - 节点数 (N): {influ_mat_list.shape[1]}")
print(f"  - 时间步数 (T): {influ_mat_list.shape[2]}")

# 验证数据的有效性
print("\n  样本统计:")
for i in range(min(3, len(all_samples))):
    seed_count = int(influ_mat_list[i, :, 0].sum())
    final_count = int(influ_mat_list[i, :, -1].sum())
    print(f"    样本 #{i}: 源节点数={seed_count}, 最终影响节点数={final_count}")


# ============================================================================
# 步骤6: 保存为.SG文件
# ============================================================================
print("\n[步骤6] 保存为.SG文件...")

# 构建SparseGraph对象（只传入adj_matrix）
graph = SparseGraph(adj_sparse)

# 手动设置prob_matrix和influ_mat_list属性
graph.prob_matrix = prob_sparse
graph.influ_mat_list = influ_mat_list

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(graph, f)

print(f"  - 数据已保存到: {OUTPUT_FILE}")


# ============================================================================
# 验证数据
# ============================================================================
print("\n[步骤7] 验证生成的数据...")

with open(OUTPUT_FILE, 'rb') as f:
    loaded_graph = pickle.load(f)

print(f"  - adj_matrix形状: {loaded_graph.adj_matrix.shape}")
print(f"  - prob_matrix形状: {loaded_graph.prob_matrix.shape}")
print(f"  - influ_mat_list形状: {loaded_graph.influ_mat_list.shape}")

# 检查数据样本
print("\n  前3个样本详情:")
for sample_idx in range(min(3, loaded_graph.influ_mat_list.shape[0])):
    influ_mat = loaded_graph.influ_mat_list[sample_idx]  # [N, T]
    seed_vec = influ_mat[:, 0]
    influ_vec = influ_mat[:, -1]
    
    # 统计中间扩散过程
    intermediate_counts = [influ_mat[:, t].sum() for t in range(1, NUM_TIMESTEPS-1)]
    
    print(f"\n  样本 #{sample_idx}:")
    print(f"    - 源节点数 (t=0): {int(seed_vec.sum())}")
    print(f"    - 最终影响节点数 (t=24): {int(influ_vec.sum())}")
    print(f"    - 扩散率: {influ_vec.sum() / n_nodes * 100:.2f}%")
    print(f"    - 中间时间步节点数范围: [{int(min(intermediate_counts))}, {int(max(intermediate_counts))}]")

print("\n" + "=" * 70)
print("数据集构建完成!")
print("=" * 70)
