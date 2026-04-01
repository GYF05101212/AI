import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from lenet5 import LeNet5

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9

# 数据预处理（MNIST 是 28x28，需要 padding 到 32x32）
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整为 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型
model = LeNet5().to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# 训练
print("\n开始训练...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 200 == 199:
            print(f'Epoch: {epoch+1}/{EPOCHS} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {running_loss/200:.4f} | Acc: {100.*correct/total:.2f}%')
            running_loss = 0.0

    # 每个 epoch 结束后测试
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()

    test_acc = 100. * test_correct / test_total
    print(f'Epoch {epoch+1} 完成 | 测试准确率: {test_acc:.2f}%')

end_time = time.time()
print(f"\n训练完成！总耗时: {(end_time - start_time)/60:.2f} 分钟")

# 保存模型
torch.save(model.state_dict(), 'lenet5_mnist.pth')
print("模型已保存为 lenet5_mnist.pth")
