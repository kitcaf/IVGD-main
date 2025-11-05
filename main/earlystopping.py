from typing import List
import copy
import operator
from enum import Enum, auto
import numpy as np

from torch.nn import Module


class StopVariable(Enum):
    """
    早停监控变量的枚举类型
    定义了可以用于早停判断的指标类型
    """
    LOSS = auto()      # 损失函数
    ACCURACY = auto()  # 准确率
    NONE = auto()      # 不使用早停


class Best(Enum):
    """
    最佳模型保存策略的枚举类型
    """
    RANKED = auto()  # 按优先级排序：只有当按优先级排序的所有指标都改善时才保存
    ALL = auto()     # 所有指标：所有监控指标都需要改善才保存模型


# 早停默认参数配置
stopping_args = dict(
        stop_varnames=[StopVariable.LOSS],  # 监控的变量列表
        patience=30,                        # 耐心值：多少个epoch没有改善就停止
        max_epochs=1000,                    # 最大训练轮数
        remember=Best.RANKED)               # 最佳模型保存策略


class EarlyStopping:
    """
    早停机制类
    监控训练过程中的指标变化，在指标不再改善时提前停止训练
    并保存训练过程中的最佳模型
    """
    def __init__(
            self, model: Module, stop_varnames: List[StopVariable],
            patience: int = 30, max_epochs: int = 1000, remember: Best = Best.ALL):
        """
        初始化早停机制
        参数:
            model: 需要监控的模型
            stop_varnames: 监控的指标变量列表
            patience: 耐心值，连续多少个epoch没有改善就停止训练
            max_epochs: 最大训练轮数
            remember: 最佳模型保存策略
        """
        self.model = model
        self.comp_ops = []       # 比较操作符列表
        self.stop_vars = []      # 监控变量名称列表
        self.best_vals = []      # 最佳值列表
        
        # 根据监控变量类型设置相应的比较操作和初始值
        for stop_varname in stop_varnames:
            if stop_varname is StopVariable.LOSS:
                self.stop_vars.append('loss')
                self.comp_ops.append(operator.le)  # 损失越小越好，使用<=比较
                self.best_vals.append(np.inf)       # 初始值为正无穷
            elif stop_varname is StopVariable.ACCURACY:
                self.stop_vars.append('acc')
                self.comp_ops.append(operator.ge)  # 准确率越大越好，使用>=比较
                self.best_vals.append(-np.inf)      # 初始值为负无穷
        
        self.remember = remember
        self.remembered_vals = copy.copy(self.best_vals)  # 记忆的最佳值（用于保存模型）
        self.max_patience = patience
        self.patience = self.max_patience
        self.max_epochs = max_epochs
        self.best_epoch = None      # 最佳epoch编号
        self.best_state = None      # 最佳模型状态字典

    def check(self, values: List[np.floating], epoch: int) -> bool:
        """
        检查是否应该停止训练
        参数:
            values: 当前epoch的监控指标值列表
            epoch: 当前epoch编号
        返回:
            True表示应该停止训练，False表示继续训练
        """
        # 检查每个监控指标是否有改善
        checks = [self.comp_ops[i](val, self.best_vals[i])
                  for i, val in enumerate(values)]
        
        if any(checks):  # 如果有任何一个指标改善
            # 更新最佳值：改善的指标使用新值，未改善的保持原值
            self.best_vals = np.choose(checks, [self.best_vals, values])  # False=0, True=1
            self.patience = self.max_patience  # 重置耐心值
            
            # 检查是否需要保存模型
            comp_remembered = [
                    self.comp_ops[i](val, self.remembered_vals[i])
                    for i, val in enumerate(values)]
            
            if self.remember is Best.ALL:
                # ALL策略：所有指标都改善才保存模型
                if all(comp_remembered):
                    self.best_epoch = epoch
                    self.remembered_vals = copy.copy(values)
                    # 保存模型状态（转移到CPU以节省显存）
                    self.best_state = {
                            key: value.cpu() for key, value
                            in self.model.state_dict().items()}
            elif self.remember is Best.RANKED:
                # RANKED策略：按优先级顺序，第一个改善的指标触发保存
                for i, comp in enumerate(comp_remembered):
                    if comp:
                        if not(self.remembered_vals[i] == values[i]):
                            self.best_epoch = epoch
                            self.remembered_vals = copy.copy(values)
                            self.best_state = {
                                    key: value.cpu() for key, value
                                    in self.model.state_dict().items()}
                            break
                    else:
                        break
        else:
            # 没有改善，减少耐心值
            self.patience -= 1
        
        # 当耐心值降为0时，返回True表示应该停止训练
        return self.patience == 0
