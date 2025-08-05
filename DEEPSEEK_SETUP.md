# 🤖 DeepSeek API 配置指南

DeepSeek是国内优秀的大语言模型服务商，提供兼容OpenAI格式的API接口，性价比很高。

## 🚀 快速开始

### 1. 获取DeepSeek API密钥

1. 访问 [DeepSeek官网](https://platform.deepseek.com/)
2. 注册账号并登录
3. 进入控制台，创建API密钥
4. 复制API密钥备用

### 2. 配置环境变量

复制环境变量模板文件：
```bash
cp env.example .env
```

编辑`.env`文件，配置DeepSeek API：
```bash
# 注释掉OpenAI配置，启用DeepSeek配置
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_BASE_URL=https://api.openai.com/v1

# DeepSeek API (推荐)
OPENAI_API_KEY=your_deepseek_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com/v1
```

### 3. 运行DeepSeek示例

```bash
# 运行DeepSeek专门示例
python3 examples/08_deepseek_demo.py

# 或使用统一运行器
python3 run_examples.py
# 然后选择 "7. DeepSeek API 演示"
```

## 🎯 DeepSeek模型选择

### DeepSeek-Chat
- **用途**: 通用对话、创作、分析、翻译
- **特点**: 中文理解优秀，支持长上下文
- **模型名**: `deepseek-chat`

### DeepSeek-Coder  
- **用途**: 代码生成、调试、解释、重构
- **特点**: 专门为编程任务优化
- **模型名**: `deepseek-coder`

## 💰 费用说明

DeepSeek的定价相比OpenAI更加经济实惠：

- **DeepSeek-Chat**: ¥1/百万tokens (输入) / ¥2/百万tokens (输出)
- **DeepSeek-Coder**: ¥1/百万tokens (输入) / ¥2/百万tokens (输出)

## 🔧 在项目中切换模型

### 方法1: 修改配置文件
编辑`config.py`文件：
```python
# 切换到DeepSeek模型
DEFAULT_MODEL = "deepseek-chat"
# 或编程任务使用
# DEFAULT_MODEL = "deepseek-coder"
```

### 方法2: 在代码中直接指定
```python
from langchain_openai import ChatOpenAI

# 使用DeepSeek Chat
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7
)

# 使用DeepSeek Coder
llm_coder = ChatOpenAI(
    model="deepseek-coder", 
    temperature=0.3  # 编程任务建议使用较低温度
)
```

## 🌟 DeepSeek优势

1. **中文支持**: 原生中文训练，理解中国文化背景
2. **性价比高**: 价格仅为OpenAI的1/10左右
3. **无需代理**: 国内服务器，访问速度快
4. **专业编程**: DeepSeek-Coder在编程任务上表现优秀
5. **API兼容**: 完全兼容OpenAI API格式

## 🛠️ 常见问题

### Q: 如何在同一个项目中使用多个模型？
A: 可以创建不同的ChatOpenAI实例：
```python
# OpenAI模型（需要代理）
openai_llm = ChatOpenAI(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your_openai_key"
)

# DeepSeek模型（国内直连）
deepseek_llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1", 
    api_key="your_deepseek_key"
)
```

### Q: 嵌入模型怎么办？
A: 目前DeepSeek主要提供对话模型，嵌入模型可以使用：
- OpenAI的text-embedding-ada-002 (需要代理)
- 本地嵌入模型(如sentence-transformers)
- 其他国内嵌入服务

### Q: 如何处理上下文长度限制？
A: DeepSeek支持较长的上下文，但仍需注意：
```python
# 对于长文本，可以分段处理
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,  # DeepSeek支持的上下文长度
    chunk_overlap=200
)
```

## 📚 更多资源

- [DeepSeek官方文档](https://platform.deepseek.com/docs)
- [API价格说明](https://platform.deepseek.com/pricing)
- [模型能力对比](https://platform.deepseek.com/models)

---

**开始你的DeepSeek AI之旅！🚀**