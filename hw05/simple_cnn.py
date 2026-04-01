import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['font.family'] = 'SimHei'  # 使用支持中文的字体


class SimpleCNN(nn.Module):
    """简单的卷积神经网络模型，用于MNIST手写数字分类"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层：输入1通道(灰度图)，输出16通道，3x3卷积核，步长1，填充1
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 最大池化层：2x2窗口，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层：将特征映射到10个类别
        self.fc = nn.Linear(in_features=16 * 14 * 14, out_features=10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc(x)
        return x


def load_data(batch_size=64):
    """加载MNIST数据集并进行预处理"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


def train(model, train_loader, criterion, optimizer, device, epochs=5):
    """训练模型"""
    model.train()
    train_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return train_losses


def test(model, test_loader, criterion, device):
    """测试模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    train_loader, test_loader = load_data(batch_size=64)

    model = SimpleCNN().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = train(model, train_loader, criterion, optimizer, device, epochs=5)
    test_loss, test_accuracy = test(model, test_loader, criterion, device)

    torch.save(model.state_dict(), 'simple_cnn_mnist.pth')
    print('模型已保存为 simple_cnn_mnist.pth')


if __name__ == '__main__':
    main()
