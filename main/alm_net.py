import torch
import torch.nn.functional as F
from main.correction import correction


class alm_net(torch.nn.Module):
    """
    增广拉格朗日乘子法(Augmented Lagrangian Method, ALM)网络
    该网络通过迭代优化的方式来校正节点的预测概率
    使用5层校正网络进行迭代更新，每层都包含一个校正模块
    """
    def __init__(self, alpha, tau, rho):
        """
        初始化ALM网络
        参数:
            alpha: 正则化参数，控制原始预测的权重
            tau: 校正网络的权重参数
            rho: 拉格朗日乘子的更新步长
        """
        super(alm_net, self).__init__()
        self.number_layer = 5  # 网络层数
        
        # 为每一层设置alpha参数（正则化权重）
        self.alpha1 = alpha
        self.alpha2 = alpha
        self.alpha3 = alpha
        self.alpha4 = alpha
        self.alpha5 = alpha

        # 为每一层设置tau参数（校正网络权重）
        self.tau1 = tau
        self.tau2 = tau
        self.tau3 = tau
        self.tau4 = tau
        self.tau5 = tau

        # 为每一层创建独立的校正网络
        self.net1 = correction()
        self.net2 = correction()
        self.net3 = correction()
        self.net4 = correction()
        self.net5 = correction()

        # 为每一层设置rho参数（约束惩罚权重）
        self.rho1 = rho
        self.rho2 = rho
        self.rho3 = rho
        self.rho4 = rho
        self.rho5 = rho

    def forward(self, x, label, lamda):
        """
        前向传播函数，通过5层迭代优化来校正预测结果
        参数:
            x: 初始预测概率，shape为(N, 1)
            label: 真实标签，shape为(N, 1)
            lamda: 拉格朗日乘子
        返回:
            校正后的预测结果，shape为(N, 2)
        """
        # 计算正样本总数，用于约束优化
        sum = torch.sum(label)
        
        # 将标签和预测都转换为二维表示: [1-p, p]
        label = torch.cat((1 - label, label), dim=1)
        x = torch.cat((1 - x, x), dim=1)
        
        # 第一层迭代更新
        prob = x[:, 1].unsqueeze(-1)  # 提取正类概率
        # ALM更新公式：结合校正网络输出、标签信息、拉格朗日乘子和约束惩罚
        x = (self.tau1 * self.net1(prob) - label * torch.softmax(x, dim=1) / label.shape[0] - lamda
             - self.rho1 * (torch.sum(x) - sum) + self.alpha1 * x) / (
                    self.tau1 + self.alpha1)
        prob = x[:, 1].unsqueeze(-1)
        # 更新拉格朗日乘子
        lamda = lamda + self.rho1 * (torch.sum(prob) - sum)
        
        # 第二层迭代更新
        x = (self.tau2 * self.net2(prob) - label * torch.softmax(x, dim=1) / label.shape[0] - lamda
             - self.rho2 * (torch.sum(x) - sum) + self.alpha2 * x) / (
                    self.tau2 + self.alpha2)
        prob = x[:, 1].unsqueeze(-1)
        lamda = lamda + self.rho2 * (torch.sum(prob) - sum)
        
        # 第三层迭代更新
        x = (self.tau3 * self.net3(prob) - label * torch.softmax(x, dim=1) / label.shape[0] - lamda
            - self.rho3 * (torch.sum(x) - sum) + self.alpha3 * x) / (
                   self.tau3 + self.alpha3)
        prob = x[:, 1].unsqueeze(-1)
        lamda = lamda + self.rho3 * (torch.sum(prob) - sum)
        
        # 第四层迭代更新
        x = (self.tau4 * self.net4(prob) - label * torch.softmax(x, dim=1) / label.shape[0] - lamda
             - self.rho4 * (torch.sum(x) - sum) + self.alpha4 * x) / (
                    self.tau4 + self.alpha4)
        prob = x[:, 1].unsqueeze(-1)
        lamda = lamda + self.rho4 * (torch.sum(prob) - sum)
        
        # 第五层迭代更新
        x = (self.tau5 * self.net5(prob) - label * torch.softmax(x, dim=1) / label.shape[0] - lamda
             - self.rho5 * (torch.sum(x) - sum) + self.alpha5 * x) / (
                    self.tau5 + self.alpha5)
        return x

    def correction(self, pred):
        """
        综合所有层的校正结果，取平均值作为最终校正输出
        参数:
            pred: 预测概率，shape为(N, 2)
        返回:
            平均校正结果，shape为(N, 2)
        """
        temp = pred[:, 0].unsqueeze(-1)  # 提取负类概率
        # 将5个校正网络的输出取平均
        return (self.net1(temp) + self.net2(temp) + self.net3(temp) + self.net4(temp) + self.net5(temp)) / self.number_layer
