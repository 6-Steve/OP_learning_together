import torch
import torch.nn as nn
import torch.nn.functional as F

'''
作为示例仅实现了两层denseblock与一层transition,
输出每经过一层后的矩阵大小，最后输出loss
'''


class DenseLayer(nn.Module):
    #Dense层
    def __init__(self, in_channels, growth_rate, dropout_rate=0.0):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        #归一化
        self.relu = nn.ReLU(inplace=True)
        #激活
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        #卷积 3*3
        self.dropout = nn.Dropout(dropout_rate)
        #dropout_rate=0  不进行dropout

    def forward(self, x):
        #向前传播
        new_features = self.conv(self.relu(self.bn(x)))
        # print(new_features.shape)
        new_features = self.dropout(new_features)
        # print(x.shape,'and',new_features.shape)
        return torch.cat([x,new_features],1)
        #   返回拼接后的tensor,只进行一次卷积,拼接


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, dropout_rate=0.0):
        '''

        :param num_layers: Dense layers层数
        :param in_channels: 输入通道数
        :param growth_rate: 学习率
        :param dropout_rate: dropout参数

        '''
        super(DenseBlock, self).__init__()
        layers = []

        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, dropout_rate))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class SimpleDenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleDenseNet, self).__init__()
        # Initial convolution
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # First Dense Block
        self.dense_block1 = DenseBlock(num_layers=3, in_channels=64, growth_rate=32)

        self.transition1 = TransitionLayer(in_channels=160 , out_channels=160)

        # Second Dense Block
        self.dense_block2 = DenseBlock(num_layers=3, in_channels=160,
                                       growth_rate=32)  # Adjust input channels accordingly

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Fully Connected Layer
        self.fc = nn.Linear(256, num_classes)  # Adjust input features accordingly


    def forward(self, x):
        '''
        定义了包含两个Dense块的简单DenseNet架构。初始卷积层将输入图像转换为64通道的特征图，
        然后经过两个Denseblock，最后通过全局平均池化和全连接层进行分类。
        '''
        x = self.initial_conv(x)
        # 初始卷积
        x = self.initial_pool(F.relu(x))
        # 第一个denseblock
        x = self.dense_block1(x)
        print("After Dense Block 1: ", x.shape)  # Output shape after first Dense Block
        # Hypothetical Transition Layer (Uncomment if you want to use it)
        x = self.transition1(x)
        print("After Transition Layer: ", x.shape)

        # Second Dense Block
        x = self.dense_block2(x)
        print("After Dense Block 2: ", x.shape)  # Output shape after second Dense Block

        # Global Average Pooling
        x = self.global_pool(x)
        print("After pooling: ", x.shape)
        x = x.view(x.size(0), -1)
        # Fully Connected Layer
        x = self.fc(x)
        return x

    # Example usage


if __name__ == "__main__":
    model = SimpleDenseNet(num_classes=10)
    criterion = nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Optimizer
    x = torch.randn(1, 3, 256, 256)
    #输入一个随机的测试值
    output = model(x)
    print("Final Output: ", output.shape)  # Should be [1, 10] for 10 classes

    targets = torch.randint(0, 10, (1,))
    loss = criterion(output, targets)
    print("Loss before one step of optimization:", loss.item())
    #第一次正向传播

    # # 反向传播
    # optimizer.zero_grad()  # Clear gradients
    # loss.backward()  # Compute gradients
    # optimizer.step()  # Update parameters
    # output = model(x)
    # loss = criterion(output, targets)
    # print("Loss after one step of optimization:", loss.item())
    # #可以输出经过反向传播过后的loss值


    # # 第二次反向传播
    # optimizer.zero_grad()  # Clear gradients
    # loss.backward()  # Compute gradients
    # optimizer.step()  # Update parameters
    # output = model(x)
    # loss = criterion(output, targets)
    # print("Loss after one step of optimization:", loss.item())