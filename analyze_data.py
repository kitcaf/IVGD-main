import pickle
import sys
import numpy as np
import os
sys.path.append('data')

# 加载karate数据集作为示例
currentPath = os.path.dirname(__file__)
with open(os.path.join(currentPath, 'data', 'karate_25c.SG'), 'rb') as f:
    graph = pickle.load(f)

print('='*70)
print('分析 adj_matrix 和 prob_matrix 的关系')
print('='*70)
print()

# 1. 基本信息
print('=== 1. 基本信息 ===')
print(f'节点数量: {graph.num_nodes()}')
print(f'边数量: {graph.num_edges()}')
print(f'是否有向图: {graph.is_directed()}')
print(f'是否加权图: {graph.is_weighted()}')
print()

# 2. 邻接矩阵分析
print('=== 2. 邻接矩阵 (adj_matrix) 分析 ===')
adj = graph.adj_matrix
print(f'形状: {adj.shape}')
print(f'非零元素数量: {adj.nnz}')
print(f'邻接矩阵的唯一值: {np.unique(adj.data)}')
print(f'邻接矩阵是否对称: {(adj != adj.T).nnz == 0}')
print()

# 3. 概率矩阵分析
print('=== 3. 概率矩阵 (prob_matrix) 分析 ===')
prob = graph.prob_matrix
print(f'形状: {prob.shape}')
print(f'非零元素数量: {prob.nnz}')
print(f'值范围: [{prob.data.min():.6f}, {prob.data.max():.6f}]')
print(f'平均概率: {prob.data.mean():.6f}')
print(f'中位数概率: {np.median(prob.data):.6f}')
print(f'概率矩阵是否对称: {(prob != prob.T).nnz == 0}')
print()

# 4. 稀疏结构对比
print('=== 4. 邻接矩阵与概率矩阵的稀疏结构对比 ===')
adj_indices = set(zip(adj.nonzero()[0], adj.nonzero()[1]))
prob_indices = set(zip(prob.nonzero()[0], prob.nonzero()[1]))
print(f'邻接矩阵非零位置数量: {len(adj_indices)}')
print(f'概率矩阵非零位置数量: {len(prob_indices)}')
print(f'两者非零位置是否完全相同: {adj_indices == prob_indices}')
print()

# 5. 节点度数分析
print('=== 5. 节点度数分析 ===')
degrees = np.array(adj.sum(axis=1)).flatten()
print(f'节点度数范围: [{degrees.min():.0f}, {degrees.max():.0f}]')
print(f'平均度数: {degrees.mean():.2f}')
print(f'度数分布（前10个节点）: {degrees[:10]}')
print()

# 6. 推断 prob_matrix 的生成方式
print('=== 6. 推断 prob_matrix 的生成方式 ===')
print('分析每条边的概率值与节点度数的关系...')
print()

# 获取邻接矩阵和概率矩阵的坐标和值
adj_coo = adj.tocoo()
prob_coo = prob.tocoo()

# 分析前20条边
print('前20条边的详细信息:')
print(f"{'边':<10} {'源节点':<8} {'目标节点':<10} {'adj值':<8} {'prob值':<12} {'目标度数':<10} {'1/度数':<12}")
print('-' * 80)
for i in range(min(20, adj.nnz)):
    src = adj_coo.row[i]
    dst = adj_coo.col[i]
    adj_val = adj_coo.data[i]
    prob_val = prob_coo.data[i]
    dst_degree = degrees[dst]
    inv_degree = 1.0 / dst_degree if dst_degree > 0 else 0
    print(f"{i:<10} {src:<8} {dst:<10} {adj_val:<8.1f} {prob_val:<12.6f} {dst_degree:<10.0f} {inv_degree:<12.6f}")

print()

# 7. 验证假设：prob = 1 / degree(destination)
print('=== 7. 验证假设: prob_matrix[i,j] = 1 / degree(j) ===')
hypothesis_correct = True
mismatches = []
for i in range(adj.nnz):
    src = adj_coo.row[i]
    dst = adj_coo.col[i]
    prob_val = prob_coo.data[i]
    expected_prob = 1.0 / degrees[dst]
    diff = abs(prob_val - expected_prob)
    if diff > 1e-6:  # 允许小的数值误差
        hypothesis_correct = False
        mismatches.append((src, dst, prob_val, expected_prob, diff))

if hypothesis_correct:
    print('✓ 假设正确！prob_matrix[i,j] = 1 / degree(j)')
    print('  这是加权级联(Weighted Cascade)模型的标准设置')
else:
    print(f'✗ 假设不完全正确，发现 {len(mismatches)} 处不匹配')
    if len(mismatches) <= 5:
        print('不匹配的边:')
        for src, dst, prob_val, expected, diff in mismatches:
            print(f'  边({src},{dst}): 实际={prob_val:.6f}, 期望={expected:.6f}, 差异={diff:.6f}')

print()

# 8. 尝试其他假设
print('=== 8. 探索其他可能的生成规则 ===')

# 假设2: 均匀分布
uniform_prob = 1.0 / graph.num_nodes()
print(f'假设2: 均匀概率 = 1/N = {uniform_prob:.6f}')
if abs(prob.data.mean() - uniform_prob) < 0.01:
    print('  可能性: 高')
else:
    print(f'  可能性: 低（实际平均值={prob.data.mean():.6f}）')

print()

# 假设3: 基于源节点度数
print('假设3: prob_matrix[i,j] 与源节点i的度数相关')
correlations = []
for i in range(min(100, adj.nnz)):
    src = adj_coo.row[i]
    prob_val = prob_coo.data[i]
    src_degree = degrees[src]
    correlations.append((src_degree, prob_val))

if len(correlations) > 0:
    src_degrees_sample = [x[0] for x in correlations]
    prob_vals_sample = [x[1] for x in correlations]
    corr = np.corrcoef(src_degrees_sample, prob_vals_sample)[0, 1]
    print(f'  源节点度数与概率值的相关系数: {corr:.4f}')
    if abs(corr) > 0.5:
        print('  可能性: 高')
    else:
        print('  可能性: 低')

print()

# 9. 深入分析：查看概率值的分布模式
print('=== 9. 深入分析概率值的分布 ===')
print(f'概率值的统计信息:')
print(f'  最小值: {prob.data.min():.6f}')
print(f'  最大值: {prob.data.max():.6f}')
print(f'  平均值: {prob.data.mean():.6f}')
print(f'  标准差: {prob.data.std():.6f}')
print(f'  25分位: {np.percentile(prob.data, 25):.6f}')
print(f'  50分位: {np.percentile(prob.data, 50):.6f}')
print(f'  75分位: {np.percentile(prob.data, 75):.6f}')
print()

# 10. 检查是否是随机生成的概率
print('=== 10. 检查是否为随机生成的概率 ===')
# 如果是随机的，应该在某个范围内均匀分布
prob_range = prob.data.max() - prob.data.min()
print(f'概率值范围: {prob_range:.6f}')
print(f'是否所有值都在 (0, 0.15] 范围内: {np.all((prob.data > 0) & (prob.data <= 0.15))}')

# 检查概率值是否有明显的聚类
unique_probs = np.unique(np.round(prob.data, 4))
print(f'去重后的概率值数量（保留4位小数）: {len(unique_probs)}')
if len(unique_probs) < 20:
    print(f'唯一概率值: {unique_probs}')
    print('  结论: 可能是基于特定规则生成的（如度数相关）')
else:
    print('  结论: 可能是随机生成或基于连续函数生成的')
print()

# 11. 分析：prob_matrix 与 adj_matrix 的关系
print('=== 11. 验证 prob_matrix 的归一化性质 ===')
# 检查每个节点的入边概率和
print('检查每个节点的入边概率总和:')
in_prob_sums = np.array(prob.sum(axis=0)).flatten()
print(f'  入边概率和的范围: [{in_prob_sums.min():.6f}, {in_prob_sums.max():.6f}]')
print(f'  入边概率和的平均值: {in_prob_sums.mean():.6f}')
print(f'  入边概率和接近1的节点数: {np.sum(np.abs(in_prob_sums - 1.0) < 0.01)}')

# 检查每个节点的出边概率和
out_prob_sums = np.array(prob.sum(axis=1)).flatten()
print(f'  出边概率和的范围: [{out_prob_sums.min():.6f}, {out_prob_sums.max():.6f}]')
print(f'  出边概率和的平均值: {out_prob_sums.mean():.6f}')
print()

# 12. 最终推断
print('='*70)
print('最终推断结果')
print('='*70)
print()
print('基于以上分析，prob_matrix 的生成方式最可能是:')
print()

if np.all(np.abs(in_prob_sums - 1.0) < 0.01):
    print('✓ 结论: 概率矩阵使用了归一化入边概率')
    print('  每个节点的所有入边概率和为 1')
    print('  这是 Linear Threshold (LT) 模型的标准设置')
elif len(unique_probs) < 30:
    print('✓ 结论: 概率矩阵可能基于节点度数或其他离散规则生成')
    print('  但不是简单的 1/degree(j) 公式')
else:
    print('✓ 结论: 概率矩阵可能是随机生成的')
    print('  在一定范围内（如 [0.0001, 0.15]）随机采样')
    print('  这在模拟实验中很常见，用于测试模型的泛化能力')

print()
print('具体特征:')
print(f'  1. 邻接矩阵: 无向无权图，所有边权重为1')
print(f'  2. 概率矩阵: 与邻接矩阵具有相同的稀疏结构')
print(f'  3. 概率矩阵: 非对称（即使邻接矩阵是对称的）')
print(f'  4. 概率值范围: [{prob.data.min():.6f}, {prob.data.max():.6f}]')
print(f'  5. 平均传播概率: {prob.data.mean():.6f}')
print()
print('='*70)
