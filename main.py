"""
主训练脚本
该脚本实现了IVGD (Inverse Influence Graph Detection) 模型的训练流程
主要步骤：
1. 加载预训练的i-DeepIS模型（初步估计）
2. 使用ALM网络对预测结果进行校正
3. 评估模型在训练集和测试集上的性能

这里主要是训练 ALM 网络，也就是通过约束机制来提高预测准确率
alm_net: 种子节点数量约束、拉格朗日乘子项（软约束）、二次惩罚项、校正网络约束、标签约束（监督信号）
"""

import logging
import numpy as np
import copy
import torch.nn as nn
import matplotlib.pyplot as plt

from main.training import FeatureCons, get_idx_new_seeds, get_predictions_new_seeds
from main.utils import load_dataset
from main.alm_net import alm_net
from main.metrics_utils import compute_all_metrics, precompute_shortest_paths
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, mean_squared_error

# 配置日志
logging.basicConfig(
    format='%(asctime)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    pass
import torch.optim as optim

# ==================== 关键参数配置 ====================
dataset = 'android'  # 数据集名称: 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid'
model_name = 'deepis'  # 模型名称: 'deepis'

# 加载数据集
graph = load_dataset(dataset)
print(graph)

# 复制影响力矩阵列表
influ_mat_list = copy.copy(graph.influ_mat_list)

# 划分训练集和测试集（80%训练，20%测试）
num_training = int(len(graph.influ_mat_list) * 0.8)
graph.influ_mat_list = graph.influ_mat_list[:num_training]
print(graph.influ_mat_list.shape), print(influ_mat_list.shape)

# ==================== 模型配置 ====================
ndim = 5  # 特征维度
fea_constructor = FeatureCons(model_name, ndim=ndim)
fea_constructor.prob_matrix = graph.prob_matrix
device = 'cuda:0'  # 使用第二块GPU（GPU 1）

# 加载预训练的i-DeepIS模型
model = torch.load("i-deepis_" + dataset + ".pt", weights_only=False)
model = model.to(device)  # 将模型移到GPU

# 获取第一个样本的预测结果（用于测试）
influ_pred = get_predictions_new_seeds(model, fea_constructor, graph.influ_mat_list[0, :, 0], 
                                      np.arange(len(graph.influ_mat_list[0, :, 0])))

# ==================== ALM网络参数 ====================
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
alpha = 1      # 正则化参数
tau = 10       # 校正网络权重
rho = 1e-3     # 拉格朗日乘子更新步长
lamda = 0      # 拉格朗日乘子初始值
threshold = 0.5  # 二分类阈值

# 初始化拉格朗日乘子
nu = torch.zeros(size=(graph.influ_mat_list.shape[1], 1)).to(device)

# 创建ALM网络
net = alm_net(alpha=alpha, tau=tau, rho=rho)
net = net.to(device)  # 将ALM网络移到GPU
optimizer = optim.SGD(net.parameters(), lr=1e-2)

# ==================== 训练阶段 ====================
net.train()
for i, influ_mat in enumerate(graph.influ_mat_list):
    print("shampe: i={:d}\n".format(i))
    # 提取种子节点向量和影响力向量
    seed_vec = influ_mat[:, 0]  # 第0列是种子节点标记
    seed_idx = np.argwhere(seed_vec == 1)  # 找到种子节点索引
    influ_vec = influ_mat[:, -1]  # 最后一列是影响力传播结果
    
    # 设置概率矩阵
    fea_constructor.prob_matrix = graph.prob_matrix
    
    # 根据影响力向量反推种子节点
    seed_preds = get_idx_new_seeds(model, influ_vec)
    seed_preds = torch.tensor(seed_preds).unsqueeze(-1).float().to(device)
    influ_vec = torch.tensor(influ_vec).unsqueeze(-1).float().to(device)
    seed_vec = torch.tensor(seed_vec).unsqueeze(-1).float().to(device)
    
    # 训练ALM网络（每个样本训练10个epoch）
    for epoch in range(10):
        print("epoch:" + str(epoch) + "\n")
        optimizer.zero_grad()
        # 使用ALM网络校正预测结果
        seed_correction = net(seed_preds, seed_vec, lamda)
        # 计算交叉熵损失
        loss = criterion(seed_correction, seed_vec.squeeze(-1).long())
        print("loss:{:0.6f}".format(loss))
        loss.backward(retain_graph=True)
        optimizer.step()

# ==================== 评估阶段 ====================
net.eval()
graph = load_dataset(dataset)
influ_mat_list = copy.copy(graph.influ_mat_list)
print(graph)

# 预计算最短路径矩阵（用于AED指标计算）
print("预计算最短路径矩阵...")
dist_matrix = precompute_shortest_paths(graph.adj_matrix)
print("✓ 最短路径矩阵计算完成")

# 初始化评估指标
train_acc = 0   # 训练集准确率
test_acc = 0    # 测试集准确率
train_pr = 0    # 训练集精确率
test_pr = 0     # 测试集精确率
train_re = 0    # 训练集召回率
test_re = 0     # 测试集召回率
train_f1 = 0    # 训练集F1分数
test_f1 = 0     # 测试集F1分数
train_auc = 0   # 训练集AUC
test_auc = 0    # 测试集AUC

# 初始化概率序列指标
train_map = 0   # 训练集MAP
test_map = 0    # 测试集MAP
train_p_at_k = 0  # 训练集P@K_true
test_p_at_k = 0   # 测试集P@K_true
train_aed = 0   # 训练集AED
test_aed = 0    # 测试集AED

# 遍历所有样本进行评估
for i, influ_mat in enumerate(influ_mat_list):
    print("i={:d}".format(i))
    seed_vec = influ_mat[:, 0]
    seed_idx = np.argwhere(seed_vec == 1)
    influ_vec = influ_mat[:, -1]
    positive = np.where(seed_vec == 1)
    
    fea_constructor.prob_matrix = graph.prob_matrix
    
    # 获取预测结果
    seed_preds = get_idx_new_seeds(model, influ_vec)
    seed_preds = torch.tensor(seed_preds).unsqueeze(-1).float().to(device)
    influ_vec = torch.tensor(influ_vec).unsqueeze(-1).float().to(device)
    seed_vec = torch.tensor(seed_vec).unsqueeze(-1).float().to(device)
    
    # 使用ALM网络校正预测
    seed_correction = net(seed_preds, seed_preds, lamda)
    seed_correction = torch.softmax(seed_correction, dim=1)
    
    # 转换为numpy数组（先拷贝到CPU）
    seed_preds = seed_preds.squeeze(-1).detach().cpu().numpy()
    seed_correction = seed_correction[:, 1].squeeze(-1).detach().cpu().numpy()
    seed_vec = seed_vec.squeeze(-1).detach().cpu().numpy()
    influ_vec_np = influ_vec.detach().cpu().numpy()
    
    # 计算感染节点数和源点数
    num_infected = int(np.sum(influ_vec_np))
    num_sources = int(np.sum(seed_vec))
    
    # 计算各项指标
    if i < num_training:  # 训练集
        train_acc += accuracy_score(seed_vec, seed_correction >= threshold)
        train_pr += precision_score(seed_vec, seed_correction >= threshold, zero_division=1)
        train_re += recall_score(seed_vec, seed_correction >= threshold)
        train_f1 += f1_score(seed_vec, seed_correction >= threshold)
        train_auc += roc_auc_score(seed_vec, seed_correction)
        
        # 计算概率序列指标
        metrics = compute_all_metrics(
            y_true_prob=seed_correction,
            y_true_binary=seed_vec,
            map_top_k=num_infected,  # 感染节点数
            pk_top_k=num_sources,    # 真实源点数
            aed_top_k=num_sources,   # 真实源点数
            dist_matrix=dist_matrix,
            adj_matrix=None
        )
        train_map += metrics['MAP']
        train_p_at_k += metrics['P@K']
        train_aed += metrics['AED']
    else:  # 测试集
        test_acc += accuracy_score(seed_vec, seed_correction >= threshold)
        test_pr += precision_score(seed_vec, seed_correction >= threshold, zero_division=1)
        test_re += recall_score(seed_vec, seed_correction >= threshold)
        test_f1 += f1_score(seed_vec, seed_correction >= threshold)
        test_auc += roc_auc_score(seed_vec, seed_preds)
        
        # 计算概率序列指标
        metrics = compute_all_metrics(
            y_true_prob=seed_correction,
            y_true_binary=seed_vec,
            map_top_k=num_infected,  # 感染节点数
            pk_top_k=num_sources,    # 真实源点数
            aed_top_k=num_sources,   # 真实源点数
            dist_matrix=dist_matrix,
            adj_matrix=None
        )
        test_map += metrics['MAP']
        test_p_at_k += metrics['P@K']
        test_aed += metrics['AED']

# ==================== 输出结果 ====================
num_test = len(influ_mat_list) - num_training

print('\n' + '=' * 80)
print('评估结果统计')
print('=' * 80)

print('\n【训练集指标】')
print(f'  Accuracy:  {train_acc / num_training:.6f}')
print(f'  Precision: {train_pr / num_training:.6f}')
print(f'  Recall:    {train_re / num_training:.6f}')
print(f'  F1-Score:  {train_f1 / num_training:.6f}')
print(f'  AUC:       {train_auc / num_training:.6f}')
print(f'  MAP:       {train_map / num_training:.6f}')
print(f'  P@K_true:  {train_p_at_k / num_training:.6f}')
print(f'  AED:       {train_aed / num_training:.6f}')

print('\n【测试集指标】')
print(f'  Accuracy:  {test_acc / num_test:.6f}')
print(f'  Precision: {test_pr / num_test:.6f}')
print(f'  Recall:    {test_re / num_test:.6f}')
print(f'  F1-Score:  {test_f1 / num_test:.6f}')
print(f'  AUC:       {test_auc / num_test:.6f}')
print(f'  MAP:       {test_map / num_test:.6f}')
print(f'  P@K_true:  {test_p_at_k / num_test:.6f}')
print(f'  AED:       {test_aed / num_test:.6f}')

print('\n' + '=' * 80)
print('指标说明：')
print('  - Accuracy/Precision/Recall/F1-Score/AUC: 二分类评估指标')
print('  - MAP (Mean Average Precision): K值为感染区域节点数')
print('  - P@K_true (Precision@K_true): K值为真实源点数')
print('  - AED (Average Euclidean Distance): 基于真实源点数计算拓扑容错性')
print('=' * 80)
