# LangChain 学习 Demo

这是一个全面的 LangChain 学习项目，包含了多个实用示例，帮助你快速掌握 LangChain 的核心概念和应用。

## 🚀 项目特色

- **完整的示例集合**: 从基础到高级，涵盖 LangChain 的主要功能
- **实用的代码示例**: 每个示例都可以独立运行，便于学习和实验
- **详细的注释说明**: 代码中包含丰富的中文注释，易于理解
- **Web 应用演示**: 包含 Streamlit 构建的交互式 Web 应用

## 📁 项目结构

```
langchain-demo/
├── config.py              # 配置文件
├── requirements.txt        # 依赖包列表
├── .env.example           # 环境变量示例
├── README.md              # 项目说明
└── examples/              # 示例代码目录
    ├── 01_basic_llm.py          # 基础 LLM 调用
    ├── 02_prompt_templates.py   # 提示词模板
    ├── 03_chains.py             # 链式调用
    ├── 04_memory.py             # 记忆功能
    ├── 05_rag.py                # RAG 检索增强生成
    ├── 06_agents.py             # 智能代理
    ├── 07_streamlit_app.py      # Streamlit Web 应用
    └── 08_deepseek_demo.py      # DeepSeek API 演示
```

## 🛠️ 安装配置

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd langchain-demo
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 文件并重命名为 `.env`，然后配置你的 API 密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# 如果使用代理
# HTTP_PROXY=http://127.0.0.1:7890
# HTTPS_PROXY=http://127.0.0.1:7890
```

### 4. 验证配置

```bash
python config.py
```

如果配置正确，你会看到 "✅ 配置检查通过" 的消息。

## 📚 示例说明

### 1. 基础 LLM 调用 (`01_basic_llm.py`)

学习如何：
- 初始化和配置 LLM 模型
- 进行单轮和多轮对话
- 使用系统提示词
- 实现流式输出

```bash
python examples/01_basic_llm.py
```

### 2. 提示词模板 (`02_prompt_templates.py`)

学习如何：
- 创建和使用提示词模板
- 实现动态参数替换
- 使用少样本学习模板
- 构建复杂的聊天模板

```bash
python examples/02_prompt_templates.py
```

### 3. 链式调用 (`03_chains.py`)

学习如何：
- 创建简单和复杂的处理链
- 实现顺序和并行处理
- 使用自定义函数和分支逻辑
- 构建动态路由系统

```bash
python examples/03_chains.py
```

### 4. 记忆功能 (`04_memory.py`)

学习如何：
- 实现对话上下文记忆
- 使用不同类型的记忆策略
- 管理长对话的内存限制
- 创建自定义记忆系统

```bash
python examples/04_memory.py
```

### 5. RAG 检索增强生成 (`05_rag.py`)

学习如何：
- 构建向量数据库
- 实现文档检索和排序
- 结合检索和生成功能
- 优化检索质量和效果

```bash
python examples/05_rag.py
```

### 6. 智能代理 (`06_agents.py`)

学习如何：
- 创建具有工具使用能力的代理
- 实现多步推理和决策
- 构建自定义工具
- 管理代理的记忆和状态

```bash
python examples/06_agents.py
```

### 7. Streamlit Web 应用 (`07_streamlit_app.py`)

启动交互式 Web 应用：

```bash
streamlit run examples/07_streamlit_app.py
```

Web 应用包含：
- 💬 智能对话：多轮对话，支持记忆
- 📚 文档问答：上传文档进行问答
- 🔍 RAG 检索：可视化检索过程
- 🤖 代理助手：工具使用演示

## 🎯 学习路径建议

### 初学者路径
1. 从 `01_basic_llm.py` 开始，了解基础概念
2. 学习 `02_prompt_templates.py`，掌握提示词工程
3. 尝试 `07_streamlit_app.py`，体验完整应用

### 进阶学习路径
1. 深入学习 `03_chains.py`，理解链式处理
2. 掌握 `04_memory.py`，实现状态管理
3. 探索 `05_rag.py`，构建知识问答系统

### 高级应用路径
1. 研究 `06_agents.py`，开发智能代理
2. 结合所有概念，构建复杂应用
3. 根据具体需求定制和优化

## 🔧 常见问题

### Q: API 调用失败怎么办？
A: 检查以下几点：
- API 密钥是否正确配置
- 网络连接是否正常
- 是否需要设置代理
- API 账户是否有足够余额

### Q: 向量数据库相关错误？
A: 确保安装了所需的向量数据库依赖：
```bash
pip install chromadb faiss-cpu
```

### Q: Streamlit 应用无法启动？
A: 检查 Streamlit 是否正确安装：
```bash
pip install streamlit
```

### Q: 如何使用其他 LLM 模型？
A: 项目支持多种LLM服务：

**DeepSeek API (推荐，国内用户)**：
```bash
# 在 .env 文件中配置
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
```
详见：[DeepSeek配置指南](DEEPSEEK_SETUP.md)

**智谱AI GLM**：
```bash
OPENAI_API_KEY=your_zhipu_api_key  
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

**其他模型**: 修改 `config.py` 中的模型配置。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

感谢 LangChain 社区提供的优秀框架和文档。

---

**Happy Learning! 🎉**

如果这个项目对你有帮助，请给个 ⭐️ 支持一下！