
"""
预训练脚本
该脚本实现了i-DeepIS模型的预训练过程
网络中观测到的影响范围（哪些节点被激活）？预测哪些节点是信息扩散的源头（种子节点）
主要步骤：
1. 加载数据集并划分训练集/验证集/测试集
2. 构建i-DeepIS模型（包含MLP和扩散传播模块）
学习节点特征，预测每个节点是种子节点的概率

初步估计
"""

import logging
import numpy as np
from pathlib import Path
import copy
import torch
from main.i_deepis import i_DeepIS, DiffusionPropagate
from main.models.MLP import MLPTransform
from main.training import train_model, FeatureCons, get_predictions_new_seeds
from main.utils import load_dataset

# 配置日志
logging.basicConfig(
    format='%(asctime)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

# 定义评估指标函数
me_op = lambda x, y: np.mean(np.abs(x - y))  # 平均绝对误差
te_op = lambda x, y: np.abs(np.sum(x) - np.sum(y))  # 总体误差

# ==================== 关键参数配置 ====================
dataset = 'android'  # 数据集名称: 'android, 'karate','dolphins','jazz','netscience','cora_ml', 'power_grid'
model_name = 'deepis'  # 模型名称: 'deepis'

# 加载数据集
graph = load_dataset(dataset)
print(graph)

# 复制影响力矩阵列表 - 快照列表
influ_mat_list = copy.copy(graph.influ_mat_list)
num_node = influ_mat_list.shape[1]  # 节点数量

# 划分训练集和测试集（80%训练，20%测试）
num_training = int(len(graph.influ_mat_list) * 0.8)
graph.influ_mat_list = graph.influ_mat_list[:num_training]
print(graph.influ_mat_list.shape), print(influ_mat_list.shape)

# ==================== 模型配置 ====================
ndim = 5  # 特征维度

# 创建扩散传播模块（迭代2次）
propagate_model = DiffusionPropagate(graph.prob_matrix, niter=2)

# 创建特征构造器
fea_constructor = FeatureCons(model_name, ndim=ndim)
fea_constructor.prob_matrix = graph.prob_matrix

device = 'cuda:1'  # 使用第二块GPU（GPU 1）

# ==================== 训练参数配置 ====================
# idx_split_args 需要根据不同数据集调整
args_dict = {
    'learning_rate': 1e-4,  # 学习率
    'λ': 0,  # 影响扩散误差权重
    'γ': 0,  # 正则化权重
    'ckpt_dir': Path('.'),  # 检查点保存目录
    'idx_split_args': {  # 数据集划分参数
        'ntraining': int(num_node/3),    # 训练集节点数
        'nstopping': int(num_node/3),    # 早停验证集节点数
        'nval': int(num_node/3),         # 测试集节点数
        'seed': 2413340114               # 随机种子
    },
    'test': False,  # 是否为测试模式
    'device': device,
    'print_interval': 1,  # 打印间隔
    'batch_size': None,   # 批量大小（None表示使用全部数据）
}

# ==================== 构建模型 ====================
if model_name == 'deepis':
    # 创建MLP模型：输入维度=ndim，隐藏层=[ndim, ndim]，输出维度=1
    gnn_model = MLPTransform(input_dim=ndim, hiddenunits=[ndim, ndim], num_classes=1, device=device)
else:
    pass

# 创建i-DeepIS模型（结合GNN和扩散传播）
model = i_DeepIS(gnn_model=gnn_model, propagate=propagate_model)

# ==================== 训练模型 ====================
model, result = train_model(model_name + '_' + dataset, model, fea_constructor, graph, **args_dict)

# ==================== 评估预训练模型 ====================
# 使用第一个训练样本测试扩散预测的准确性
influ_pred = get_predictions_new_seeds(
    model, 
    fea_constructor, 
    graph.influ_mat_list[0, :, 0],  # 种子节点向量
    np.arange(len(graph.influ_mat_list[0, :, 0]))  # 所有节点索引
)
print("diffusion mae:" + str(me_op(influ_pred, graph.influ_mat_list[0, :, 1])))

# ==================== 保存模型 ====================
torch.save(model, "i-deepis_" + dataset + ".pt")