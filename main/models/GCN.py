"""
图卷积网络(GCN)模型实现
"""

import torch
import torch.nn.functional as F
import torch
from torch_sparse import SparseTensor, spmm
import time
# from torch_geometric.utils import add_remaining_self_loops
from scipy.sparse.linalg import inv
from scipy.sparse import coo_matrix
import numpy as np


def azw(adj, z, w):
    """
    高效计算 A * Z * W（邻接矩阵 * 特征矩阵 * 权重矩阵）
    
    参数:
        adj: 邻接矩阵（可以是SparseTensor或普通tensor）
        z: 特征矩阵
        w: 权重矩阵
    
    返回:
        A * Z * W 的计算结果
    
    优化策略:
        先计算 Z * W，然后计算 A * (Z * W)，这样比先计算 A * Z 再计算结果 * W 更高效
    """
    if isinstance(adj, SparseTensor):
        # 获取稀疏矩阵的坐标格式
        row, col, adj_value = adj.coo()
        edge_index = torch.stack([row, col], dim=0)

        # 方法1: A*Z then (A*Z)*W (注释掉的旧方法)
        # pre_spmm = time.time()
        # temp = spmm(edge_index, adj_value, z.size()[0], z.size()[0], z)
        # print('time for spmm:', time.time() - pre_spmm)
        # return temp.matmul(w)

        # 方法2: Z*W THEN A*(Z*W) - 更高效！
        temp = z.matmul(w)
        return spmm(edge_index, adj_value, temp.size()[0], temp.size()[0], temp)
    else:
        # 普通矩阵直接计算
        return adj.matmul(z.matmul(w))


class GCN(torch.nn.Module):
    """
    图卷积网络(Graph Convolutional Network)
    实现两层GCN结构
    """
    def __init__(self, w1, w2):
        """
        初始化GCN
        参数:
            w1: 第一层的权重矩阵
            w2: 第二层的权重矩阵
        """
        super(GCN, self).__init__()
        self.w1 = w1
        self.w2 = w2
    
    def forward(self, adj, x):
        """
        前向传播
        参数:
            adj: 邻接矩阵
            x: 输入特征矩阵
        返回:
            经过两层GCN处理后的特征
        """
        # 第一层GCN: adj * x * w1，然后通过ReLU激活
        z1 = azw(adj, x, self.w1)
        z1 = F.relu(z1)
        # 第二层GCN: adj * z1 * w2（输出层，无激活函数）
        z2 = azw(adj, z1, self.w2)
        return z2




