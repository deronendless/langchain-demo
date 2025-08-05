"""
示例3: 链式调用
演示如何使用LangChain的链式调用功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from config import check_config, DEFAULT_MODEL

def simple_chain():
    """简单链式调用示例"""
    print("=== 简单链式调用 ===\n")
    
    check_config()
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    
    # 创建提示词模板
    prompt = ChatPromptTemplate.from_template(
        "给我讲一个关于{topic}的有趣故事，不超过100字。"
    )
    
    # 创建输出解析器
    output_parser = StrOutputParser()
    
    # 使用LCEL (LangChain Expression Language) 创建链
    chain = prompt | llm | output_parser
    
    # 执行链
    result = chain.invoke({"topic": "程序员"})
    print(f"故事: {result}\n")

def sequential_chain_example():
    """顺序链示例"""
    print("=== 顺序链示例 ===\n")
    
    check_config()
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    
    # 第一个链：生成故事大纲
    outline_prompt = PromptTemplate(
        input_variables=["topic"],
        template="为{topic}创建一个简短的故事大纲（3-4个要点）："
    )
    outline_chain = LLMChain(llm=llm, prompt=outline_prompt, output_key="outline")
    
    # 第二个链：基于大纲写故事
    story_prompt = PromptTemplate(
        input_variables=["outline"],
        template="根据以下大纲写一个200字的故事：\n{outline}\n\n故事："
    )
    story_chain = LLMChain(llm=llm, prompt=story_prompt, output_key="story")
    
    # 第三个链：给故事评分
    review_prompt = PromptTemplate(
        input_variables=["story"],
        template="请为以下故事打分（1-10分）并给出简短评价：\n{story}\n\n评分和评价："
    )
    review_chain = LLMChain(llm=llm, prompt=review_prompt, output_key="review")
    
    # 创建顺序链
    overall_chain = SequentialChain(
        chains=[outline_chain, story_chain, review_chain],
        input_variables=["topic"],
        output_variables=["outline", "story", "review"],
        verbose=True
    )
    
    # 执行链
    result = overall_chain({"topic": "太空探险"})
    
    print("生成的大纲:")
    print(result["outline"])
    print("\n生成的故事:")
    print(result["story"])
    print("\n故事评价:")
    print(result["review"])
    print()

def custom_chain_with_functions():
    """带自定义函数的链示例"""
    print("=== 带自定义函数的链 ===\n")
    
    check_config()
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    
    def word_count(text):
        """计算文本字数"""
        return len(text.split())
    
    def format_output(data):
        """格式化输出"""
        return f"文本: {data['text']}\n字数: {data['word_count']} 字"
    
    # 创建链
    prompt = ChatPromptTemplate.from_template("用简洁的语言解释什么是{concept}")
    
    chain = (
        prompt 
        | llm 
        | StrOutputParser()
        | RunnableLambda(lambda x: {"text": x, "word_count": word_count(x)})
        | RunnableLambda(format_output)
    )
    
    result = chain.invoke({"concept": "人工智能"})
    print(result)
    print()

def branching_chain():
    """分支链示例"""
    print("=== 分支链示例 ===\n")
    
    check_config()
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    
    # 创建不同的处理分支
    technical_prompt = ChatPromptTemplate.from_template(
        "用技术术语解释{topic}，面向专业人员："
    )
    
    simple_prompt = ChatPromptTemplate.from_template(
        "用简单易懂的语言解释{topic}，面向普通用户："
    )
    
    # 创建分支处理函数
    def route_question(data):
        """根据难度级别路由到不同的处理分支"""
        if data.get("level") == "technical":
            return technical_prompt
        else:
            return simple_prompt
    
    # 创建动态路由链
    chain = (
        RunnableLambda(route_question)
        | llm
        | StrOutputParser()
    )
    
    # 测试技术级别
    print("技术级别解释:")
    result1 = chain.invoke({"topic": "区块链", "level": "technical"})
    print(result1)
    print()
    
    # 测试简单级别
    print("简单级别解释:")
    result2 = chain.invoke({"topic": "区块链", "level": "simple"})
    print(result2)
    print()

def parallel_chain():
    """并行链示例"""
    print("=== 并行链示例 ===\n")
    
    check_config()
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    
    # 创建多个并行处理链
    summary_chain = (
        ChatPromptTemplate.from_template("总结{text}的主要内容：")
        | llm
        | StrOutputParser()
    )
    
    sentiment_chain = (
        ChatPromptTemplate.from_template("分析{text}的情感倾向（积极/消极/中性）：")
        | llm
        | StrOutputParser()
    )
    
    keywords_chain = (
        ChatPromptTemplate.from_template("提取{text}的关键词（用逗号分隔）：")
        | llm
        | StrOutputParser()
    )
    
    # 使用RunnableParallel创建并行链
    from langchain.schema.runnable import RunnableParallel
    
    parallel_chain = RunnableParallel(
        summary=summary_chain,
        sentiment=sentiment_chain,
        keywords=keywords_chain
    )
    
    # 测试文本
    text = "今天是个美好的一天，我学会了使用LangChain创建强大的AI应用程序。这个工具真的很棒，让开发变得更加简单高效。"
    
    result = parallel_chain.invoke({"text": text})
    
    print("原文:")
    print(text)
    print("\n分析结果:")
    print(f"摘要: {result['summary']}")
    print(f"情感: {result['sentiment']}")
    print(f"关键词: {result['keywords']}")
    print()

if __name__ == "__main__":
    try:
        simple_chain()
        sequential_chain_example()
        custom_chain_with_functions()
        branching_chain()
        parallel_chain()
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查你的API密钥配置")