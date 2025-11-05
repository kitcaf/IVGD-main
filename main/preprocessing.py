from typing import List, Tuple, Dict
import copy
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def gen_seeds(size: int = None) -> np.ndarray:
    """
    生成随机种子数组
    参数:
        size: 生成的种子数量，如果为None则生成单个种子
    返回:
        uint32类型的随机种子数组
    """
    max_uint32 = np.iinfo(np.uint32).max
    return np.random.randint(
            max_uint32 + 1, size=size, dtype=np.uint32)


def exclude_idx(idx: np.ndarray, idx_exclude_list: List[np.ndarray]) -> np.ndarray:
    """
    从索引数组中排除指定的索引
    参数:
        idx: 原始索引数组
        idx_exclude_list: 需要排除的索引列表
    返回:
        排除指定索引后的数组
    """
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])


def gen_splits_(array, train_size, stopping_size, val_size):
    """
    将数据集划分为训练集、早停验证集和测试集
    参数:
        array: 需要划分的数组
        train_size: 训练集大小
        stopping_size: 早停验证集大小
        val_size: 测试集大小
    返回:
        train_idx: 训练集索引
        stopping_idx: 早停验证集索引
        val_idx: 测试集索引
    """
    assert train_size + stopping_size + val_size <= len(array), '划分大小总和不能超过数组长度'
    from sklearn.model_selection import train_test_split
    # 首先分离出训练集
    train_idx, tmp = train_test_split(array, train_size=train_size, test_size=stopping_size+val_size)
    # 然后将剩余数据分为早停验证集和测试集
    stopping_idx, val_idx = train_test_split(tmp, train_size=stopping_size, test_size=val_size)
    
    return train_idx, stopping_idx, val_idx


def normalize_attributes(attr_matrix):
    """
    归一化属性矩阵，使用L1范数进行行归一化
    参数:
        attr_matrix: 属性矩阵，可以是稀疏矩阵或稠密矩阵
    返回:
        归一化后的属性矩阵
    """
    epsilon = 1e-12  # 防止除零的小常数
    if isinstance(attr_matrix, sp.csr_matrix):
        # 稀疏矩阵处理
        attr_norms = spla.norm(attr_matrix, ord=1, axis=1)  # 计算每行的L1范数
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)  # 计算归一化系数
        attr_mat_norm = attr_matrix.multiply(attr_invnorms[:, np.newaxis])  # 按行归一化
    else:
        # 稠密矩阵处理
        attr_norms = np.linalg.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix * attr_invnorms[:, np.newaxis]
    return attr_mat_norm
