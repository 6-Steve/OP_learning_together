import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # 输入层到隐藏层的全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 隐藏层到输出层的全连接层
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 通过隐藏层，并应用ReLU激活函数
        x = torch.relu(self.fc1(x))
        # 通过输出层（这里没有应用激活函数，通常用于分类任务的最后一层会应用softmax，但这里为了简单起见省略了）
        x = self.fc2(x)
        return x

    # 设置网络参数


input_size = 10  # 输入特征的数量
hidden_size = 5  # 隐藏层神经元的数量
output_size = 1  # 输出特征的数量（例如，回归任务的一个连续值，或二分类任务的一个概率值）

# 创建网络实例
model = SimpleNN(input_size, hidden_size, output_size)

# 打印网络结构
print(model)

# 定义一个损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失，通常用于回归任务
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 生成一些随机数据作为示例输入和目标
inputs = torch.randn(32, input_size)  # 批次大小为32的随机输入数据
targets = torch.randn(32, output_size)  # 对应的随机目标数据

# 前向传播
outputs = model(inputs)

# 计算损失
loss = criterion(outputs, targets)

# 反向传播和优化
optimizer.zero_grad()  # 清空梯度
loss.backward()  # 计算梯度
optimizer.step()  # 更新参数

# 打印损失值
print('Loss:', loss.item())