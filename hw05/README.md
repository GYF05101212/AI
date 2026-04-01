# HW05 卷积神经网络实现

## 项目结构
hw05/
├── README.md # 项目说明
├── requirements.txt # Python 依赖
├── report.md # 学习摘要 + LeNet-5结构 + 对比
├── debug_notes.md # 调试记录
├── simple_cnn.py # 任务一：极简CNN（公众号文章代码）
├── lenet5.py # LeNet-5 模型定义
├── train_lenet.py # LeNet-5 训练脚本
└── models/ # 保存的模型文件（运行时生成）

## 环境配置

```bash
pip install -r requirements.txt

## 实验结果

| 模型 | 参数量 | 测试准确率 | 训练耗时 |
|-----|-------|-----------|---------|
| 极简CNN | 31,530 | 98.08% | ~2-3 分钟 |
| LeNet-5 | 91,946 | 99.04% | 4.69 分钟 |

## 调试记录

详见 [debug_notes.md](debug_notes.md)

## 报告

详见 [report.md](report.md)
