# HW07 胸部X光肺炎图像分类

## 项目结构
hw07/
├── README.md # 项目说明
├── requirements.txt # Python 依赖
├── report.md # 实验报告
├── pneumonia_cnn.py # 完整训练代码
├── training_curves.png # 训练曲线图
├── confusion_matrix.png # 混淆矩阵图
└── best_model.pth # 最佳模型权重（运行后生成）

## 数据集

- **名称**: Chest X-Ray Images (Pneumonia)
- **来源**: Kaggle
- **规模**: 约 5800 张儿童胸部X光影像
- **类别**: NORMAL（正常）、PNEUMONIA（肺炎）

## 环境配置

```bash
pip install -r requirements.txt
