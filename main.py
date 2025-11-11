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
device = 'cuda:1'  # 使用第二块GPU（GPU 1）

# 加载预训练的i-DeepIS模型
model = torch.load("i-deepis_" + dataset + ".pt")

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
nu = torch.zeros(size=(graph.influ_mat_list.shape[1], 1))

# 创建ALM网络
net = alm_net(alpha=alpha, tau=tau, rho=rho)
optimizer = optim.SGD(net.parameters(), lr=1e-2)

# ==================== 训练阶段 ====================
net.train()
for i, influ_mat in enumerate(graph.influ_mat_list):
    print("i={:d}".format(i))
    # 提取种子节点向量和影响力向量
    seed_vec = influ_mat[:, 0]  # 第0列是种子节点标记
    seed_idx = np.argwhere(seed_vec == 1)  # 找到种子节点索引
    influ_vec = influ_mat[:, -1]  # 最后一列是影响力传播结果
    
    # 设置概率矩阵
    fea_constructor.prob_matrix = graph.prob_matrix
    
    # 根据影响力向量反推种子节点
    seed_preds = get_idx_new_seeds(model, influ_vec)
    seed_preds = torch.tensor(seed_preds).unsqueeze(-1).float()
    influ_vec = torch.tensor(influ_vec).unsqueeze(-1).float()
    seed_vec = torch.tensor(seed_vec).unsqueeze(-1).float()
    
    # 训练ALM网络（每个样本训练10个epoch）
    for epoch in range(10):
        print("epoch:" + str(epoch))
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
    seed_preds = torch.tensor(seed_preds).unsqueeze(-1).float()
    influ_vec = torch.tensor(influ_vec).unsqueeze(-1).float()
    seed_vec = torch.tensor(seed_vec).unsqueeze(-1).float()
    
    # 使用ALM网络校正预测
    seed_correction = net(seed_preds, seed_preds, lamda)
    seed_correction = torch.softmax(seed_correction, dim=1)
    
    # 转换为numpy数组
    seed_preds = seed_preds.squeeze(-1).detach().numpy()
    seed_correction = seed_correction[:, 1].squeeze(-1).detach().numpy()
    seed_vec = seed_vec.squeeze(-1).detach().numpy()
    
    # 计算各项指标
    if i < num_training:  # 训练集
        train_acc += accuracy_score(seed_vec, seed_correction >= threshold)
        train_pr += precision_score(seed_vec, seed_correction >= threshold, zero_division=1)
        train_re += recall_score(seed_vec, seed_correction >= threshold)
        train_f1 += f1_score(seed_vec, seed_correction >= threshold)
        train_auc += roc_auc_score(seed_vec, seed_correction)
    else:  # 测试集
        test_acc += accuracy_score(seed_vec, seed_correction >= threshold)
        test_pr += precision_score(seed_vec, seed_correction >= threshold, zero_division=1)
        test_re += recall_score(seed_vec, seed_correction >= threshold)
        test_f1 += f1_score(seed_vec, seed_correction >= threshold)
        test_auc += roc_auc_score(seed_vec, seed_preds)

# ==================== 输出结果 ====================
print('training acc:', train_acc / num_training)
print('training pr:', train_pr / num_training)
print('training re:', train_re / num_training)
print('training fs:', train_f1 / num_training)
print('training auc:', train_auc / num_training)
print('test acc:', test_acc / (len(influ_mat_list) - num_training))
print('test pr:', test_pr / (len(influ_mat_list) - num_training))
print('test re:', test_re / (len(influ_mat_list) - num_training))
print('test fs:', test_f1 / (len(influ_mat_list) - num_training))
print('test auc:', test_auc / (len(influ_mat_list) - num_training))
