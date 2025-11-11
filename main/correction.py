import torch
import torch.nn.functional as F

class correction(torch.nn.Module):
    """
    校正网络模块
    用于校正节点的预测概率，通过一个三层的多层感知机(MLP)网络来学习校正函数
    """
    def __init__(self):
        """
        初始化校正网络
        网络结构: 2 -> 1000 -> 1000 -> 2
        """
        super(correction, self).__init__()
        number_of_neurons = 1000  # 隐藏层神经元数量
        # 第一层：输入层到第一个隐藏层 (2 -> 1000)
        self.fc1 = torch.nn.Linear(2, number_of_neurons)
        # 第二层：隐藏层到隐藏层 (1000 -> 1000)
        self.fc2 = torch.nn.Linear(number_of_neurons, number_of_neurons)
        # 第三层：隐藏层到输出层 (1000 -> 2)
        self.fc3 = torch.nn.Linear(number_of_neurons, 2)
    
    def forward(self, x):
        """
        前向传播函数
        参数:
            x: 输入概率向量，shape为(N, 1)，其中N是节点数
        返回:
            校正后的概率向量，shape为(N, 2)，表示[1-p, p]的概率分布
        """
        # 将输入概率x转换为二维表示: [1-x, x]
        x = torch.cat((1-x, x), dim=1)
        # 第一层全连接 + ReLU激活
        temp = F.relu(self.fc1(x))
        # 第二层全连接 + ReLU激活
        temp = F.relu(self.fc2(temp))
        # 第三层全连接（线性输出）
        temp = self.fc3(temp)
        # 残差连接：将原始输入加到输出上
        temp = (temp + x)
        # 将结果裁剪到[0, 1]区间，确保输出是有效的概率值
        # 使用与temp相同的设备和数据类型
        temp = torch.minimum(torch.maximum(torch.zeros(temp.shape, device=temp.device, dtype=temp.dtype), temp), 
                            torch.ones(temp.shape, device=temp.device, dtype=temp.dtype))
        return temp