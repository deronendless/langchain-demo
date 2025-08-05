"""
示例7: Streamlit Web应用
一个完整的LangChain Web应用示例
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import check_config, DEFAULT_MODEL

# 页面配置
st.set_page_config(
    page_title="LangChain Demo",
    page_icon="🦜",
    layout="wide"
)

# 侧边栏配置
st.sidebar.title("🦜 LangChain Demo")
st.sidebar.markdown("这是一个综合性的LangChain学习应用")

# 检查配置
try:
    check_config()
    config_ok = True
except Exception as e:
    config_ok = False
    st.sidebar.error(f"配置错误: {e}")

if config_ok:
    # 选择功能模块
    demo_mode = st.sidebar.selectbox(
        "选择功能模块",
        ["💬 智能对话", "📚 文档问答", "🔍 RAG检索", "🤖 代理助手"]
    )
    
    # 模型参数设置
    st.sidebar.subheader("模型参数")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 100, 2000, 1000, 100)

    # 初始化模型
    @st.cache_resource
    def get_llm():
        return ChatOpenAI(model=DEFAULT_MODEL, temperature=temperature, max_tokens=max_tokens)

    @st.cache_resource
    def get_embeddings():
        return OpenAIEmbeddings()

    # 智能对话模块
    if demo_mode == "💬 智能对话":
        st.title("💬 智能对话")
        st.markdown("与AI进行多轮对话，支持上下文记忆")
        
        # 初始化对话记忆
        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferWindowMemory(
                k=5, return_messages=True
            )
        
        # 系统提示词设置
        system_prompt = st.text_area(
            "系统提示词",
            value="你是一个友善且博学的AI助手，请用中文回答问题。",
            height=100
        )
        
        # 对话历史显示
        st.subheader("对话历史")
        chat_container = st.container()
        
        # 显示历史消息
        with chat_container:
            messages = st.session_state.memory.chat_memory.messages
            for msg in messages:
                if hasattr(msg, 'content'):
                    if msg.__class__.__name__ == 'HumanMessage':
                        st.chat_message("user").markdown(msg.content)
                    else:
                        st.chat_message("assistant").markdown(msg.content)
        
        # 用户输入
        if user_input := st.chat_input("请输入你的问题..."):
            # 显示用户消息
            st.chat_message("user").markdown(user_input)
            
            # 创建提示词模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            # 获取历史消息
            history = st.session_state.memory.chat_memory.messages
            
            # 调用LLM
            llm = get_llm()
            chain = prompt | llm | StrOutputParser()
            
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    response = chain.invoke({
                        "history": history,
                        "input": user_input
                    })
                    st.markdown(response)
            
            # 更新记忆
            st.session_state.memory.chat_memory.add_user_message(user_input)
            st.session_state.memory.chat_memory.add_ai_message(response)
        
        # 清空对话按钮
        if st.button("🗑️ 清空对话历史"):
            st.session_state.memory.clear()
            st.experimental_rerun()

    # 文档问答模块
    elif demo_mode == "📚 文档问答":
        st.title("📚 文档问答")
        st.markdown("上传文档，基于文档内容进行问答")
        
        # 文档上传
        uploaded_file = st.file_uploader(
            "上传文档",
            type=['txt'],
            help="目前支持txt格式文件"
        )
        
        if uploaded_file:
            # 读取文档内容
            content = uploaded_file.read().decode('utf-8')
            st.text_area("文档内容预览", content[:500] + "...", height=150)
            
            # 处理文档
            @st.cache_data
            def process_document(content):
                # 分割文档
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                texts = text_splitter.split_text(content)
                
                # 创建文档对象
                documents = [Document(page_content=text) for text in texts]
                return documents
            
            documents = process_document(content)
            st.success(f"文档已处理，分割为 {len(documents)} 个片段")
            
            # 创建向量存储
            @st.cache_resource
            def create_vectorstore(_documents):
                embeddings = get_embeddings()
                vectorstore = Chroma.from_documents(_documents, embeddings)
                return vectorstore
            
            vectorstore = create_vectorstore(documents)
            
            # 问答功能
            st.subheader("向文档提问")
            question = st.text_input("请输入你的问题：")
            
            if question:
                with st.spinner("搜索相关内容..."):
                    # 检索相关文档
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    relevant_docs = retriever.get_relevant_documents(question)
                    
                    # 显示检索到的文档片段
                    st.subheader("相关文档片段")
                    for i, doc in enumerate(relevant_docs):
                        with st.expander(f"片段 {i+1}"):
                            st.write(doc.page_content)
                    
                    # 生成回答
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    prompt = ChatPromptTemplate.from_template(
                        """基于以下文档内容回答问题。如果文档中没有相关信息，请说明无法回答。

文档内容:
{context}

问题: {question}

回答:"""
                    )
                    
                    llm = get_llm()
                    chain = prompt | llm | StrOutputParser()
                    
                    with st.spinner("生成回答..."):
                        answer = chain.invoke({
                            "context": context,
                            "question": question
                        })
                    
                    st.subheader("AI回答")
                    st.markdown(answer)

    # RAG检索模块
    elif demo_mode == "🔍 RAG检索":
        st.title("🔍 RAG检索演示")
        st.markdown("演示检索增强生成的工作流程")
        
        # 预定义文档库
        sample_docs = [
            "LangChain是一个用于开发由语言模型驱动的应用程序的框架。",
            "向量数据库是存储和检索向量嵌入的专门数据库。",
            "RAG技术结合了信息检索和文本生成，可以提供更准确的答案。",
            "Python是一种流行的编程语言，广泛用于AI开发。",
            "Streamlit是一个用于构建数据科学Web应用的Python库。"
        ]
        
        # 创建向量数据库
        @st.cache_resource
        def setup_vectordb():
            documents = [Document(page_content=text) for text in sample_docs]
            embeddings = get_embeddings()
            vectorstore = Chroma.from_documents(documents, embeddings)
            return vectorstore
        
        vectorstore = setup_vectordb()
        
        # RAG参数设置
        col1, col2 = st.columns(2)
        with col1:
            num_docs = st.slider("检索文档数量", 1, 5, 2)
        with col2:
            similarity_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.5)
        
        # 查询输入
        query = st.text_input("输入查询:")
        
        if query:
            # 检索过程可视化
            st.subheader("🔍 检索过程")
            
            with st.spinner("检索相关文档..."):
                # 相似度搜索
                docs_with_scores = vectorstore.similarity_search_with_score(
                    query, k=num_docs
                )
                
                # 显示检索结果
                for i, (doc, score) in enumerate(docs_with_scores):
                    with st.expander(f"文档 {i+1} (相似度: {1-score:.3f})"):
                        st.write(doc.page_content)
                        st.progress(1-score)
            
            # 生成回答
            st.subheader("🤖 生成回答")
            if docs_with_scores:
                context = "\n".join([doc.page_content for doc, score in docs_with_scores])
                
                prompt = ChatPromptTemplate.from_template(
                    """基于以下上下文回答问题:

上下文:
{context}

问题: {question}

回答:"""
                )
                
                llm = get_llm()
                chain = prompt | llm | StrOutputParser()
                
                with st.spinner("生成回答..."):
                    answer = chain.invoke({
                        "context": context,
                        "question": query
                    })
                
                st.markdown(f"**回答:** {answer}")

    # 代理助手模块
    elif demo_mode == "🤖 代理助手":
        st.title("🤖 代理助手")
        st.markdown("具有工具使用能力的智能代理")
        
        # 模拟工具
        class SimpleCalculator:
            @staticmethod
            def calculate(expression):
                try:
                    result = eval(expression, {"__builtins__": {}}, {})
                    return f"计算结果: {result}"
                except:
                    return "计算错误"
        
        class WeatherAPI:
            @staticmethod
            def get_weather(city):
                weather_data = {
                    "北京": "晴天, 15°C",
                    "上海": "多云, 18°C", 
                    "广州": "小雨, 25°C"
                }
                return weather_data.get(city, f"暂无{city}的天气信息")
        
        # 工具选择
        st.subheader("可用工具")
        col1, col2 = st.columns(2)
        with col1:
            use_calculator = st.checkbox("计算器", value=True)
        with col2:
            use_weather = st.checkbox("天气查询", value=True)
        
        # 代理对话
        st.subheader("与代理对话")
        
        if 'agent_history' not in st.session_state:
            st.session_state.agent_history = []
        
        # 显示对话历史
        for msg in st.session_state.agent_history:
            if msg['role'] == 'user':
                st.chat_message("user").markdown(msg['content'])
            else:
                st.chat_message("assistant").markdown(msg['content'])
        
        # 用户输入
        if user_input := st.chat_input("请输入你的需求..."):
            st.chat_message("user").markdown(user_input)
            st.session_state.agent_history.append({"role": "user", "content": user_input})
            
            # 简单的工具路由逻辑
            response = ""
            
            # 检查是否需要计算
            if use_calculator and any(op in user_input for op in ['+', '-', '*', '/', '计算']):
                # 提取数学表达式（简化版）
                import re
                math_pattern = r'[\d+\-*/().\s]+'
                match = re.search(math_pattern, user_input)
                if match:
                    expression = match.group().strip()
                    calc_result = SimpleCalculator.calculate(expression)
                    response += f"🧮 {calc_result}\n\n"
            
            # 检查是否需要天气查询
            if use_weather and '天气' in user_input:
                cities = ['北京', '上海', '广州']
                for city in cities:
                    if city in user_input:
                        weather_result = WeatherAPI.get_weather(city)
                        response += f"🌤️ {city}天气: {weather_result}\n\n"
                        break
            
            # 如果没有匹配到工具，使用普通LLM回复
            if not response:
                llm = get_llm()
                prompt = ChatPromptTemplate.from_template(
                    "你是一个有用的助手。用户说: {input}\n请简洁地回复:"
                )
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({"input": user_input})
            
            # 显示回复
            st.chat_message("assistant").markdown(response)
            st.session_state.agent_history.append({"role": "assistant", "content": response})
        
        # 清空对话
        if st.button("🗑️ 清空对话"):
            st.session_state.agent_history = []
            st.experimental_rerun()

# 页面底部信息
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 使用说明")
st.sidebar.markdown(
    """
1. 确保已正确配置API密钥
2. 选择不同的功能模块进行体验
3. 调整模型参数以获得不同效果
4. 尝试不同类型的问题和任务
"""
)

st.sidebar.markdown("### 🛠️ 技术栈")
st.sidebar.markdown(
    """
- **LangChain**: AI应用框架
- **OpenAI**: 语言模型
- **Streamlit**: Web界面
- **Chroma**: 向量数据库
"""
)