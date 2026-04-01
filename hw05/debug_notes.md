# 调试记录

## 环境信息
- Python 版本：3.13（conda base 环境）
- PyTorch 版本：2.11.0
- 设备：CPU

## 遇到的问题

### 问题1：运行 simple_cnn.py 时报错 ImportError: DLL load failed
- **现象**：PIL.Image 导入失败
- **原因分析**：Python 3.13 与 Pillow 最新版本不兼容
- **解决方案**：执行 `pip uninstall pillow -y` 然后 `pip install pillow==10.4.0`

### 问题2：运行 train_lenet.py 时报错 mat1 and mat2 shapes cannot be multiplied
- **现象**：期望 120x84 矩阵乘法，但输入是 64x480
- **原因分析**：LeNet-5 中 C5 卷积层输出尺寸计算错误，展平后的维度应该是 480 而不是 120
- **解决方案**：修改 `lenet5.py` 中的全连接层输入维度从 `120` 改为 `120 * 2 * 2 = 480`

## 运行结果

### 极简CNN (simple_cnn.py)
- 测试准确率：98.08%
- 训练耗时：约 2-3 分钟

### LeNet-5 (train_lenet.py)
- 模型参数量：91,946
- 测试准确率：99.04%
- 训练耗时：4.69 分钟
