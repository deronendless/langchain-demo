"""
示例1: 基础LLM调用
演示如何使用LangChain调用大语言模型
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from config import check_config, DEFAULT_MODEL, TEMPERATURE

def basic_chat():
    """基础聊天示例"""
    print("=== 基础LLM调用示例 ===\n")
    
    # 检查配置
    check_config()
    
    # 初始化模型
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=TEMPERATURE
    )
    
    # 1. 简单的单轮对话
    print("1. 简单对话:")
    response = llm.invoke([HumanMessage(content="你好，请介绍一下自己")])
    print(f"AI: {response.content}\n")
    
    # 2. 带系统提示的对话
    print("2. 带系统提示的对话:")
    messages = [
        SystemMessage(content="你是一个友善的Python编程助手"),
        HumanMessage(content="如何在Python中读取文件？")
    ]
    response = llm.invoke(messages)
    print(f"AI: {response.content}\n")
    
    # 3. 多轮对话
    print("3. 多轮对话:")
    conversation = [
        SystemMessage(content="你是一个数学老师"),
        HumanMessage(content="什么是二次方程？"),
    ]
    
    response = llm.invoke(conversation)
    print(f"AI: {response.content}\n")
    
    # 继续对话
    conversation.append(AIMessage(content=response.content))
    conversation.append(HumanMessage(content="能给我一个例子吗？"))
    
    response = llm.invoke(conversation)
    print(f"AI: {response.content}\n")

def streaming_example():
    """流式输出示例"""
    print("=== 流式输出示例 ===\n")
    
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=TEMPERATURE,
        streaming=True
    )
    
    print("AI正在思考...")
    for chunk in llm.stream([HumanMessage(content="请写一首关于程序员的短诗")]):
        print(chunk.content, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    try:
        basic_chat()
        streaming_example()
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查你的API密钥配置和网络连接")