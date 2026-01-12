"""
多层感知机(MLP)变换模型
带谱归一化(Spectral Normalization)的MLP实现，用于i-DeepIS模型
"""

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import ipdb

import numpy as np
import time
import random
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
from typing import List
from torch.nn.functional import normalize


class MLPTransform(nn.Module):
    """
    MLP变换网络
    多层感知机，带dropout和谱归一化，用于节点特征学习和影响概率预测
    """
    def __init__(self, 
                input_dim,
                hiddenunits: List[int],
                num_classes,
                bias=True,
                drop_prob=0.5,
                n_power_iterations=10,
                eps=1e-12,
                coeff=0.9,
                device='cuda'):
        """
        初始化MLP变换网络
        
        参数:
            input_dim: 输入特征维度
            hiddenunits: 隐藏层神经元数量列表，例如[64, 32]表示两个隐藏层
            num_classes: 输出类别数（对于回归任务为1）
            bias: 是否使用偏置项
            drop_prob: Dropout概率
            n_power_iterations: 谱归一化的幂迭代次数
            eps: 数值稳定性的小常数
            coeff: 谱归一化系数，控制Lipschitz常数的上界
            device: 计算设备（'cuda'或'cpu'）
        """
        super(MLPTransform, self).__init__()
        
        # features是一个占位符，每次前向传播前会被替换为所需的节点特征矩阵
        # 保存模型参数时会先删除self.features.weight
        self.features = None

        # 构建多层全连接网络
        fcs = [nn.Linear(input_dim, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i-1], hiddenunits[i]))
        fcs.append(nn.Linear(hiddenunits[-1], num_classes))

        self.fcs = nn.ModuleList(fcs)

        # Dropout层配置
        if drop_prob is 0:
            self.dropout = lambda x: x  # 不使用dropout
        else:
            self.dropout = nn.Dropout(drop_prob)
        
        self.act_fn = nn.ReLU()  # 激活函数
        
        # 谱归一化参数
        self.n_power_iterations = n_power_iterations  # 幂迭代次数
        self.eps = eps  # 数值稳定性
        self.coeff = coeff  # Lipschitz常数上界
        self.device = device

    def forward(self, nodes: torch.LongTensor):
        """
        前向传播
        
        参数:
            nodes: 节点索引张量
        
        返回:
            节点的影响概率预测（经过sigmoid激活，范围在0-1之间）
        """
        # 第一层：特征嵌入 -> 第一个隐藏层
        layer_inner = self.act_fn(self.fcs[0](self.dropout(self.features(nodes))))
        
        # 中间隐藏层：应用谱归一化
        for fc in self.fcs[1:-1]:
            # 对权重进行谱归一化
            weight = self.compute_weight(fc)
            fc.weight.data.copy_(weight)
            layer_inner = self.act_fn(fc(layer_inner))

        # 输出层：使用sigmoid激活得到概率
        res = torch.sigmoid(self.fcs[-1](self.dropout(layer_inner)))
        return res
    
    def compute_weight(self, module):
        """
        计算谱归一化后的权重
        使用幂迭代方法近似计算权重矩阵的最大奇异值，并对权重进行归一化
        
        参数:
            module: 需要进行谱归一化的神经网络层
        
        返回:
            归一化后的权重矩阵
        
        注意:
            谱归一化可以限制模型的Lipschitz常数，提高训练稳定性和泛化能力
            这对于生成对抗网络(GAN)等模型特别重要
        """
        # 重要说明：
        # 如果设置了`do_power_iteration`，u和v向量会在幂迭代中原地更新
        # 这在DataParallel模式下非常重要，因为：
        # 1. DataParallel会将模块复制到多个设备，向量（作为buffer）会被广播
        # 2. 每个副本在自己的设备上运行谱归一化幂迭代
        # 3. 简单赋值会导致更新丢失
        # 
        # 因此依赖两个重要行为：
        # 1. DataParallel在张量已在正确设备上时不会克隆存储
        # 2. 如果out张量形状正确，会直接填充值
        # 
        # 所有设备上执行相同的幂迭代，原地更新张量会确保device[0]上的模块
        # 副本通过共享存储更新并行模块的u向量
        # 
        # 但是，在使用u和v归一化权重之前需要克隆它们
        # 这是为了支持两次前向传播的反向传播（例如GAN训练中的常见模式：
        # loss = D(real) - D(fake)），否则引擎会抱怨第一次前向传播所需的
        # 变量（即u和v向量）在第二次前向传播中被改变
        
        weight = module.weight.clone()
        # 初始化左右奇异向量
        u = torch.rand(weight.shape[0], device=weight.device)
        v = torch.rand(weight.shape[1], device=weight.device)
        
        with torch.no_grad():
            # 幂迭代方法：迭代计算最大奇异值对应的左右奇异向量
            for _ in range(self.n_power_iterations):
                # 权重矩阵的谱范数等于 u^T W v，其中u和v是第一左右奇异向量
                # 幂迭代产生u和v的近似值
                v = normalize(torch.mv(weight.t(), u), dim=0, eps=self.eps, out=v)
                u = normalize(torch.mv(weight, v), dim=0, eps=self.eps, out=u)
            
            if self.n_power_iterations > 0:
                # 克隆以支持反向传播（见上述说明）
                u = u.clone()
                v = v.clone()

        # 计算谱范数（最大奇异值）
        sigma = torch.dot(u, torch.mv(weight, v))
        
        # 软归一化：只有当sigma大于coeff时才进行归一化
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor

        return weight


