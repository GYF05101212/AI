import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 1. 数据准备 ====================

# 图像预处理（修复：单通道归一化）
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # 确保是单通道灰度图
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道归一化
])

transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # 确保是单通道灰度图
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道归一化
])

# 修正后的数据路径
data_dir = "/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray/chest_xray"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# 验证路径
print(f"训练集路径: {train_dir}")
print(f"训练集存在: {os.path.exists(train_dir)}")
print(f"测试集路径: {test_dir}")
print(f"测试集存在: {os.path.exists(test_dir)}")

# 加载训练数据
full_train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)

# 按 8:2 划分训练集和验证集
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# 验证集使用无数据增强的预处理
val_dataset.dataset.transform = transform_val_test

# 加载测试集
test_dataset = datasets.ImageFolder(test_dir, transform=transform_val_test)

# 数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 打印数据集信息
print(f"\n训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"类别: {full_train_dataset.classes}")
print(f"类别映射: {full_train_dataset.class_to_idx}")

# ==================== 2. 模型构建 ====================

class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()
        
        # 卷积块1: 输入1通道 -> 32通道
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 卷积块2: 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 卷积块3: 64 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 卷积块4: 128 -> 256
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 全连接层（输入维度：256 * 14 * 14 = 50176）
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = PneumoniaCNN(num_classes=2).to(device)
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# ==================== 3. 训练配置 ====================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# 训练历史记录
train_losses = []
val_losses = []
train_accs = []
val_accs = []

num_epochs = 15
best_val_acc = 0

print("\n开始训练...")
start_time = time.time()

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_epoch_loss = train_loss / len(train_loader)
    train_epoch_acc = 100 * train_correct / train_total
    train_losses.append(train_epoch_loss)
    train_accs.append(train_epoch_acc)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_acc = 100 * val_correct / val_total
    val_losses.append(val_epoch_loss)
    val_accs.append(val_epoch_acc)
    
    # 调整学习率
    scheduler.step(val_epoch_loss)
    
    # 保存最佳模型
    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.2f}% | "
          f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%")

end_time = time.time()
print(f"\n训练完成！总耗时: {(end_time - start_time)/60:.2f} 分钟")
print(f"最佳验证准确率: {best_val_acc:.2f}%")

# ==================== 4. 测试评估 ====================

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算评估指标
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

print("\n========== 测试集评估结果 ==========")
print(f"准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"精确率 (Precision): {precision:.4f} ({precision*100:.2f}%)")
print(f"召回率 (Recall): {recall:.4f} ({recall*100:.2f}%)")
print(f"F1分数: {f1:.4f} ({f1*100:.2f}%)")

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
print("\n混淆矩阵:")
print("          预测正常  预测肺炎")
print(f"实际正常     {cm[0,0]}        {cm[0,1]}")
print(f"实际肺炎     {cm[1,0]}        {cm[1,1]}")

# ==================== 5. 绘制曲线 ====================

plt.figure(figsize=(12, 4))

# Loss曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Accuracy曲线
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# 混淆矩阵热力图
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['NORMAL', 'PNEUMONIA'])
plt.yticks(tick_marks, ['NORMAL', 'PNEUMONIA'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                 color='white' if cm[i, j] > cm.max()/2 else 'black')

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print("\n模型已保存为 best_model.pth")
print("曲线图已保存为 training_curves.png")
print("混淆矩阵已保存为 confusion_matrix.png")
