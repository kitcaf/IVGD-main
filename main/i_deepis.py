import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import ipdb
import scipy.sparse as sp


class Identity(nn.Module):
    """
    恒等传播模块 - 不进行任何传播，直接返回预测值
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, preds, seed_idx, idx):
        """
        前向传播
        参数:
            preds: 节点预测概率
            seed_idx: 种子节点索引
            idx: 需要返回的节点索引
        返回:
            指定节点的预测概率
        """
        return preds[idx]


class DiffusionPropagate(nn.Module):
    """
    扩散传播模块 - 基于概率矩阵进行迭代扩散传播
    模拟信息在网络中的传播过程
    """
    def __init__(self, prob_matrix, niter):
        """
        初始化扩散传播模块
        参数:
            prob_matrix: 概率传播矩阵，表示节点间的影响概率
            niter: 迭代次数
        """
        super(DiffusionPropagate, self).__init__()
        self.niter = niter
        # 如果输入是稀疏矩阵，转换为稠密数组
        if sp.isspmatrix(prob_matrix):
            prob_matrix = prob_matrix.toarray()
        # 将概率矩阵注册为buffer（不参与梯度计算的张量）
        self.register_buffer('prob_matrix', torch.FloatTensor(prob_matrix))

    def forward(self, preds, seed_idx, idx):
        """
        前向传播 - 执行扩散传播过程
        参数:
            preds: 初始节点预测概率
            seed_idx: 种子节点索引（这些节点的值固定为1）
            idx: 需要返回的节点索引
        返回:
            经过扩散传播后指定节点的概率
        """
        temp = preds
        temp = temp.flatten()
        device = preds.device
        # 执行niter次迭代传播
        for i in range(self.niter):
            # P2: 每条边传播失败的概率矩阵
            P2 = self.prob_matrix.T * preds.view((1, -1)).expand(self.prob_matrix.shape)
            # P3: 每条边不传播的概率 (1 - 传播概率)
            P3 = torch.ones(self.prob_matrix.shape).to(device) - P2
            # 计算每个节点不被激活的概率（所有入边都不传播）
            # 最终概率 = 1 - 所有入边都不传播的概率
            preds = torch.ones((self.prob_matrix.shape[0],)).to(device) - torch.prod(P3, dim=1)
            # 种子节点概率固定为1
            preds[seed_idx] = 1
        # 将传播结果与初始预测取平均
        preds = (preds + temp) / 2
        return preds[idx]
    
    def backward(self, preds):
        """
        反向传播 - 通过迭代反向推断初始状态
        参数:
            preds: 观测到的最终节点概率
        返回:
            推断的初始节点概率
        """
        device = preds.device
        res = preds
        temp = preds
        # 执行10次外层迭代优化
        for j in range(10):
            # 执行niter次内层迭代传播
            for i in range(self.niter):
                P2 = self.prob_matrix.T * res.view((1, -1)).expand(self.prob_matrix.shape)
                P3 = torch.ones(self.prob_matrix.shape).to(device) - P2
                temp = torch.ones((self.prob_matrix.shape[0],)).to(device) - torch.prod(P3, dim=1)
                # 已知为种子的节点保持为1
                temp[preds == 1] = 1
            # 反向推断：2*观测值 - 传播后的值
            res = 2 * preds - temp
            # 将结果裁剪到[0, 1]区间
            res = torch.maximum(torch.minimum(res, torch.tensor(1)), torch.tensor(0))
        return res


class i_DeepIS(nn.Module):
    """
    i-DeepIS模型：逆向深度影响扩散模型
    本质上是一个节点回归任务，用于根据观测到的影响范围反推种子节点
    
    模型结构：
    1. GNN模型：学习节点特征表示并预测节点概率
    2. 传播模块：模拟影响在网络中的扩散过程
    """

    def __init__(self, gnn_model: nn.Module, propagate: nn.Module):
        """
        初始化i-DeepIS模型
        参数:
            gnn_model: 图神经网络模型，用于节点特征学习和预测
            propagate: 传播模块，用于模拟影响扩散过程
        """
        super(i_DeepIS, self).__init__()
        self.gnn_model = gnn_model
        self.propagate = propagate

        # 获取需要正则化的参数（只包括需要梯度的参数）
        self.reg_params = list(filter(lambda x: x.requires_grad, self.gnn_model.parameters()))

    def forward(self, idx: torch.LongTensor):
        """
        前向传播函数
        参数:
            idx: 需要获取预测值的节点索引
        返回:
            指定节点的影响概率预测值
        
        注意：实际上会预测所有节点的值，idx只是指定获取哪些节点的预测
        """
        device = next(self.gnn_model.parameters()).device
        total_node_nums = self.gnn_model.features.weight.shape[0]
        total_nodes = torch.LongTensor(np.arange(total_node_nums)).to(device)
        
        # 获取种子节点信息（特征矩阵的第一列）
        seed = self.gnn_model.features.weight[:, 0]
        seed_idx = torch.LongTensor(np.argwhere(seed.detach().cpu().numpy() == 1)).to(device)
        seed = torch.unsqueeze(seed, 1)
        
        # 使用GNN模型预测所有节点的概率
        predictions = self.gnn_model(total_nodes)
        # 将预测值与种子信息结合（取平均）
        predictions = (predictions + seed) / 2

        # 通过传播模块进行扩散，然后选择指定节点的预测值
        predictions = self.propagate(predictions, seed_idx, idx)

        return predictions.flatten()

    def backward(self, prediction: torch.LongTensor):
        """
        反向推断函数：根据观测到的影响范围反推初始种子节点
        参数:
            prediction: 观测到的节点影响概率
        返回:
            推断的初始种子节点概率分布
        """
        device = next(self.gnn_model.parameters()).device
        total_node_nums = self.gnn_model.features.weight.shape[0]
        total_nodes = torch.LongTensor(np.arange(total_node_nums)).to(device)
        
        # 通过传播模块的反向过程推断初始状态
        res = self.propagate.backward(prediction)
        self.gnn_model.features.weight[:, 0] = res
        
        # 执行10次迭代优化
        for i in range(10):
            # 使用当前参数进行前向预测
            temp = self.gnn_model(total_nodes).squeeze()
            # 反向计算：2*观测值 - 预测值
            res = 2 * prediction - temp
            # 更新特征矩阵的第一列（种子信息）
            self.gnn_model.features.weight[:, 0] = res.float()
        return res
    
    def loss(self, predictions, labels, λ, γ):
        """
        计算损失函数
        参数:
            predictions: 模型预测值
            labels: 真实标签
            λ: 影响扩散误差的权重系数
            γ: 正则化项的权重系数
        返回:
            总损失值 = 节点级误差 + λ*影响扩散误差 + γ*正则化项
        """
        # L1: 节点级别的平均绝对误差
        L1 = torch.sum(torch.abs(predictions - labels)) / len(labels)
        
        # L2: 影响扩散误差（预测的总影响范围与真实总影响范围的差异）
        L2 = torch.abs(torch.sum(predictions) - torch.sum(labels)) / (
                torch.sum(labels) + 1e-5)
        
        # Reg: L2正则化项（所有可训练参数的平方和）
        Reg = sum(torch.sum(param ** 2) for param in self.reg_params)
        
        # 总损失
        Loss = L1 + λ * L2 + γ * Reg
        return Loss
