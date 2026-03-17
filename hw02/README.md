# 实验二：论文导读 + DeepSeek Chatbot
# HW02 大模型论文导读与Chatbot示例

## 任务一：论文导读

- **论文题目**：大模型赋能碳审计框架构建——以DeepSeek为例
- **作者**：朱雯
- **出处**：《商业经济》2026年第5期
- **导读生成模型**：DeepSeek
- **配图来源**：从论文PDF中截取图1、图2、图3、图4，保存于`images`文件夹
- **导读文件**：[导读_大模型赋能碳审计框架构建.md](./导读_大模型赋能碳审计框架构建.md)

## 任务二：Chatbot示例代码

### 实现方式
使用DeepSeek官方API（兼容OpenAI接口格式），实现简单的对话功能。

### 环境要求
- Python 3.8+
- 安装依赖：`pip install openai python-dotenv`

### 配置说明
1. 访问 https://platform.deepseek.com/ 注册并获取API Key
2. 在项目根目录创建`.env`文件：
