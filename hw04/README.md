# HW04 大模型文稿生成 + 声音克隆 + 语音识别

## 项目结构
hw04/
├── README.md # 项目总览
├── text_gen.md # 任务一：大模型生成文稿
├── jianying.md # 任务二：剪映声音克隆说明
├── asr_report.md # 任务三：ASR 调研报告
├── asr_demo.py # ASR 识别代码
├── requirements.txt # ASR 依赖
└── voice_clone.mp3 # 配音音频（需自行添加）

## 任务说明

### 任务一：大模型生成文稿
- **模型**：DeepSeek
- **标题**：大模型如何改变编程教育
- **详见**：[text_gen.md](text_gen.md)

### 任务二：剪映声音克隆
- 使用剪映声音克隆功能，以任务一文稿为脚本生成配音
- **详见**：[jianying.md](jianying.md)

### 任务三：ASR 调研与实现
- **对比方案**：OpenAI Whisper、Vosk、FunASR
- **选型**：Vosk（轻量、支持实时、中文友好）
- **实现功能**：音频文件语音识别
- **详见**：[asr_report.md](asr_report.md)

## 运行方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
