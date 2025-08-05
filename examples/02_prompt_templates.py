"""
示例2: 提示词模板
演示如何使用LangChain的提示词模板功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate, 
    ChatPromptTemplate, 
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    FewShotPromptTemplate
)
from langchain.schema import HumanMessage
from config import check_config, DEFAULT_MODEL

def basic_prompt_template():
    """基础提示词模板示例"""
    print("=== 基础提示词模板 ===\n")
    
    # 创建简单的提示词模板
    template = """你是一个{role}，请回答以下问题：
问题：{question}
请用{language}回答，并保持{tone}的语调。"""
    
    prompt = PromptTemplate(
        input_variables=["role", "question", "language", "tone"],
        template=template
    )
    
    # 格式化提示词
    formatted_prompt = prompt.format(
        role="Python专家",
        question="如何优化Python代码性能？",
        language="中文",
        tone="专业而友善"
    )
    
    print("格式化后的提示词:")
    print(formatted_prompt)
    print()

def chat_prompt_template():
    """聊天提示词模板示例"""
    print("=== 聊天提示词模板 ===\n")
    
    check_config()
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    
    # 创建聊天提示词模板
    system_template = "你是一个{expertise}专家，具有{years}年的经验。"
    human_template = "请解释一下{topic}，并给出实际的例子。"
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    # 格式化并调用
    messages = chat_prompt.format_messages(
        expertise="机器学习",
        years="10",
        topic="什么是过拟合"
    )
    
    print("发送的消息:")
    for msg in messages:
        print(f"{msg.__class__.__name__}: {msg.content}")
    print()
    
    response = llm.invoke(messages)
    print(f"AI回复: {response.content}\n")

def few_shot_template():
    """少样本提示词模板示例"""
    print("=== 少样本提示词模板 ===\n")
    
    # 定义示例
    examples = [
        {
            "question": "如何在Python中创建列表？",
            "answer": "使用方括号创建列表：my_list = [1, 2, 3, 'hello']"
        },
        {
            "question": "如何在Python中定义函数？",
            "answer": "使用def关键字：def my_function(param): return param * 2"
        }
    ]
    
    # 创建示例模板
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="问题: {question}\n答案: {answer}"
    )
    
    # 创建少样本提示词模板
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="以下是一些Python编程的问答示例：",
        suffix="问题: {input}\n答案:",
        input_variables=["input"]
    )
    
    # 使用模板
    formatted = few_shot_prompt.format(input="如何在Python中读取文件？")
    print("少样本提示词:")
    print(formatted)
    print()

def advanced_chat_template():
    """高级聊天模板示例"""
    print("=== 高级聊天模板 ===\n")
    
    check_config()
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    
    # 创建包含历史消息的模板
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "你是{character}，请保持角色设定进行对话。"
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    # 模拟对话历史
    history = [
        HumanMessage(content="你好"),
        HumanMessage(content="今天天气怎么样？")
    ]
    
    # 格式化消息
    messages = prompt.format_messages(
        character="一个幽默的天气播报员",
        history=history,
        input="明天会下雨吗？"
    )
    
    print("发送的完整对话:")
    for i, msg in enumerate(messages):
        print(f"{i+1}. {msg.__class__.__name__}: {msg.content}")
    print()
    
    response = llm.invoke(messages)
    print(f"AI回复: {response.content}\n")

if __name__ == "__main__":
    try:
        basic_prompt_template()
        chat_prompt_template()
        few_shot_template()
        advanced_chat_template()
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查你的API密钥配置")