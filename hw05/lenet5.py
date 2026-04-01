import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeNet-5 神经网络结构
    参考：Gradient-based learning applied to document recognition (LeCun et al., 1998)
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # 卷积层 C1: 输入1通道，输出6通道，核5×5，padding=2 使输出为28×28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # 池化层 S2: 2×2 平均池化
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 卷积层 C3: 输入6通道，输出16通道，核5×5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # 池化层 S4: 2×2 平均池化
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 卷积层 C5: 输入16通道，输出120通道，核5×5
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)

        # 全连接层 F6: 120 → 84
        self.fc1 = nn.Linear(120, 84)
        # 输出层: 84 → 10
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        # C1 + tanh + S2
        x = self.pool1(torch.tanh(self.conv1(x)))
        # C3 + tanh + S4
        x = self.pool2(torch.tanh(self.conv2(x)))
        # C5
        x = torch.tanh(self.conv3(x))
        # 展平
        x = x.view(x.size(0), -1)
        # F6
        x = torch.tanh(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LeNet5()
    print(model)
    print(f"参数量: {count_parameters(model):,}")
