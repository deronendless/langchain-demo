"""
示例5: RAG (检索增强生成)
演示如何使用LangChain实现检索增强生成系统
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
# 如果FAISS安装失败，可以只使用Chroma
try:
    from langchain.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️  FAISS未安装，将只使用ChromaDB作为向量数据库")
    FAISS_AVAILABLE = False
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from config import check_config, DEFAULT_MODEL

def create_sample_documents():
    """创建示例文档"""
    documents = [
        Document(
            page_content="LangChain是一个用于开发由语言模型驱动的应用程序的框架。它提供了模块化的组件，可以轻松构建复杂的AI应用。",
            metadata={"source": "langchain_intro", "topic": "基础介绍"}
        ),
        Document(
            page_content="向量数据库是存储和检索向量嵌入的专门数据库。常见的向量数据库包括Chroma、FAISS、Pinecone等。",
            metadata={"source": "vector_db", "topic": "向量数据库"}
        ),
        Document(
            page_content="RAG（检索增强生成）是一种结合信息检索和生成的技术，可以让AI模型访问外部知识库来提供更准确的答案。",
            metadata={"source": "rag_concept", "topic": "RAG技术"}
        ),
        Document(
            page_content="Python是一种高级编程语言，以其简洁的语法和强大的库生态系统而闻名。它广泛用于Web开发、数据科学和人工智能。",
            metadata={"source": "python_intro", "topic": "编程语言"}
        ),
        Document(
            page_content="机器学习是人工智能的一个分支，通过算法和统计模型让计算机从数据中学习，无需显式编程即可执行特定任务。",
            metadata={"source": "ml_intro", "topic": "机器学习"}
        )
    ]
    return documents

def basic_rag_example():
    """基础RAG示例"""
    print("=== 基础RAG示例 ===\n")
    
    check_config()
    
    # 初始化模型和嵌入
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    embeddings = OpenAIEmbeddings()
    
    # 创建示例文档
    documents = create_sample_documents()
    
    # 创建向量存储
    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 创建RAG提示词模板
    template = """基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法回答。

上下文:
{context}

问题: {question}

回答:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 创建RAG链
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 测试问题
    questions = [
        "什么是LangChain？",
        "什么是RAG技术？", 
        "向量数据库有哪些？",
        "Python有什么特点？"
    ]
    
    for question in questions:
        print(f"问题: {question}")
        
        # 检索相关文档
        relevant_docs = retriever.get_relevant_documents(question)
        print("检索到的文档:")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"  {i}. {doc.page_content[:50]}... (来源: {doc.metadata['source']})")
        
        # 生成回答
        answer = rag_chain.invoke(question)
        print(f"回答: {answer}")
        print("-" * 80)

def advanced_rag_with_reranking():
    """带重新排序的高级RAG示例"""
    print("=== 带重新排序的高级RAG ===\n")
    
    check_config()
    
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    embeddings = OpenAIEmbeddings()
    
    # 创建更多示例文档
    documents = create_sample_documents()
    
    # 添加更多文档
    additional_docs = [
        Document(
            page_content="深度学习是机器学习的一个子集，使用多层神经网络来学习数据的复杂模式。",
            metadata={"source": "deep_learning", "topic": "深度学习"}
        ),
        Document(
            page_content="自然语言处理（NLP）是人工智能的一个领域，专注于计算机理解和生成人类语言。",
            metadata={"source": "nlp", "topic": "自然语言处理"}
        )
    ]
    documents.extend(additional_docs)
    
    # 创建向量存储 - 使用ChromaDB替代FAISS
    if FAISS_AVAILABLE:
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("使用FAISS向量数据库")
    else:
        vectorstore = Chroma.from_documents(documents, embeddings)
        print("使用ChromaDB向量数据库")
    
    # 创建检索器（获取更多候选文档）
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 重新排序函数
    def rerank_documents(docs, question, llm):
        """使用LLM对检索到的文档进行重新排序"""
        if not docs:
            return docs
        
        # 为每个文档生成相关性评分
        rerank_prompt = ChatPromptTemplate.from_template(
            """给以下文档相对于问题的相关性打分（0-10分）：

问题: {question}
文档: {document}

只返回数字分数："""
        )
        
        scored_docs = []
        for doc in docs:
            score_response = llm.invoke(
                rerank_prompt.format_messages(question=question, document=doc.page_content)
            )
            try:
                score = float(score_response.content.strip())
            except:
                score = 5.0  # 默认分数
            scored_docs.append((doc, score))
        
        # 按分数排序并返回前2个
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:2]]
    
    # 创建带重新排序的RAG链
    def retrieve_and_rerank(question):
        docs = retriever.get_relevant_documents(question)
        return rerank_documents(docs, question, llm)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    template = """基于以下精选的上下文信息回答问题：

上下文:
{context}

问题: {question}

请提供详细而准确的回答:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": RunnableLambda(retrieve_and_rerank) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 测试
    question = "深度学习和机器学习有什么关系？"
    print(f"问题: {question}")
    
    # 显示重新排序过程
    initial_docs = retriever.get_relevant_documents(question)
    print(f"\n初始检索到 {len(initial_docs)} 个文档:")
    for i, doc in enumerate(initial_docs, 1):
        print(f"  {i}. {doc.page_content[:60]}...")
    
    reranked_docs = retrieve_and_rerank(question)
    print(f"\n重新排序后选择了 {len(reranked_docs)} 个文档:")
    for i, doc in enumerate(reranked_docs, 1):
        print(f"  {i}. {doc.page_content[:60]}...")
    
    answer = rag_chain.invoke(question)
    print(f"\n最终回答: {answer}")
    print()

def rag_with_metadata_filtering():
    """带元数据过滤的RAG示例"""
    print("=== 带元数据过滤的RAG ===\n")
    
    check_config()
    
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    embeddings = OpenAIEmbeddings()
    
    # 创建文档
    documents = create_sample_documents()
    
    # 创建向量存储
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # 按主题过滤的检索函数
    def retrieve_by_topic(question, topic_filter=None):
        if topic_filter:
            # 使用元数据过滤
            docs = vectorstore.similarity_search(
                question, 
                k=3,
                filter={"topic": topic_filter}
            )
        else:
            docs = vectorstore.similarity_search(question, k=3)
        return docs
    
    def format_docs(docs):
        return "\n\n".join(f"[{doc.metadata['topic']}] {doc.page_content}" for doc in docs)
    
    template = """基于以下分类的上下文信息回答问题：

上下文:
{context}

问题: {question}

回答:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 测试不同的过滤条件
    test_cases = [
        ("什么是向量数据库？", None),
        ("什么是向量数据库？", "向量数据库"),
        ("编程语言相关的内容有哪些？", "编程语言")
    ]
    
    for question, topic_filter in test_cases:
        print(f"问题: {question}")
        if topic_filter:
            print(f"过滤条件: 主题='{topic_filter}'")
        
        docs = retrieve_by_topic(question, topic_filter)
        context = format_docs(docs)
        
        print("检索到的文档:")
        for doc in docs:
            print(f"  - [{doc.metadata['topic']}] {doc.page_content[:50]}...")
        
        # 生成回答
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        print(f"回答: {answer}")
        print("-" * 80)

def multi_query_rag():
    """多查询RAG示例"""
    print("=== 多查询RAG ===\n")
    
    check_config()
    
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    embeddings = OpenAIEmbeddings()
    
    # 创建文档
    documents = create_sample_documents()
    vectorstore = Chroma.from_documents(documents, embeddings)
    
    # 生成多个查询的函数
    def generate_queries(question, llm):
        """为一个问题生成多个不同的查询表述"""
        query_prompt = ChatPromptTemplate.from_template(
            """你是一个AI助手，需要为给定的问题生成3个不同的查询表述，以便更好地检索相关信息。

原问题: {question}

请生成3个不同角度的查询（每行一个）："""
        )
        
        response = llm.invoke(query_prompt.format_messages(question=question))
        queries = [q.strip() for q in response.content.split('\n') if q.strip()]
        return queries[:3]  # 取前3个
    
    # 多查询检索函数
    def multi_query_retrieve(question):
        # 生成多个查询
        queries = generate_queries(question, llm)
        print(f"生成的查询:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        
        # 对每个查询进行检索
        all_docs = []
        for query in queries:
            docs = vectorstore.similarity_search(query, k=2)
            all_docs.extend(docs)
        
        # 去重（基于内容）
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        return unique_docs[:3]  # 返回前3个唯一文档
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    template = """基于以下通过多查询检索到的上下文信息回答问题：

上下文:
{context}

问题: {question}

回答:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 测试
    question = "如何使用AI技术处理文本？"
    print(f"原问题: {question}\n")
    
    docs = multi_query_retrieve(question)
    print(f"\n最终检索到 {len(docs)} 个唯一文档:")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc.page_content}")
    
    context = format_docs(docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    print(f"\n最终回答: {answer}")

if __name__ == "__main__":
    try:
        basic_rag_example()
        advanced_rag_with_reranking()
        rag_with_metadata_filtering()
        multi_query_rag()
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查你的API密钥配置")