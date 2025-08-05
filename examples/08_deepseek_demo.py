"""
示例8: DeepSeek API调用演示
演示如何使用DeepSeek API进行各种AI任务
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from config import check_config

def deepseek_basic_chat():
    """基础DeepSeek对话示例"""
    print("=== DeepSeek基础对话 ===\n")
    
    try:
        check_config()
    except Exception as e:
        print(f"❌ 配置错误: {e}")
        print("请确保在.env文件中配置了DeepSeek API:")
        print("OPENAI_API_KEY=your_deepseek_api_key")
        print("OPENAI_BASE_URL=https://api.deepseek.com/v1")
        return
    
    # 使用DeepSeek模型
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=1000
    )
    
    print("1. 简单对话:")
    response = llm.invoke([HumanMessage(content="你好，请介绍一下DeepSeek")])
    print(f"DeepSeek: {response.content}\n")
    
    print("2. 代码生成:")
    code_prompt = [
        SystemMessage(content="你是一个专业的Python程序员"),
        HumanMessage(content="请写一个Python函数来计算斐波那契数列的第n项")
    ]
    response = llm.invoke(code_prompt)
    print(f"DeepSeek: {response.content}\n")

def deepseek_coder_demo():
    """DeepSeek Coder模型演示"""
    print("=== DeepSeek Coder编程助手 ===\n")
    
    # 使用DeepSeek Coder模型（专门为编程优化）
    llm = ChatOpenAI(
        model="deepseek-coder",
        temperature=0.3,  # 编程任务使用较低温度
        max_tokens=1500
    )
    
    tasks = [
        "写一个Python类来实现栈数据结构",
        "解释什么是装饰器，并给出一个例子",
        "如何在Python中实现单例模式？",
        "写一个函数来检查字符串是否是回文"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"{i}. 任务: {task}")
        response = llm.invoke([
            SystemMessage(content="你是一个经验丰富的Python开发者，请提供清晰、简洁的代码示例和解释。"),
            HumanMessage(content=task)
        ])
        print(f"DeepSeek Coder: {response.content}")
        print("-" * 80)

def deepseek_chain_example():
    """DeepSeek链式处理示例"""
    print("=== DeepSeek链式处理 ===\n")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.7
    )
    
    # 创建一个分析-总结-建议的处理链
    analysis_prompt = ChatPromptTemplate.from_template(
        "分析以下问题的关键要点：{problem}"
    )
    
    summary_prompt = ChatPromptTemplate.from_template(
        "根据以下分析，提供一个简洁的总结：{analysis}"
    )
    
    suggestion_prompt = ChatPromptTemplate.from_template(
        "基于以下总结，给出3个具体的建议：{summary}"
    )
    
    # 创建处理链
    analysis_chain = analysis_prompt | llm | StrOutputParser()
    summary_chain = summary_prompt | llm | StrOutputParser()
    suggestion_chain = suggestion_prompt | llm | StrOutputParser()
    
    # 测试问题
    problem = "如何提高团队的工作效率和沟通协作？"
    
    print(f"原问题: {problem}\n")
    
    # 步骤1: 分析
    analysis = analysis_chain.invoke({"problem": problem})
    print(f"1. 分析结果:\n{analysis}\n")
    
    # 步骤2: 总结
    summary = summary_chain.invoke({"analysis": analysis})
    print(f"2. 总结:\n{summary}\n")
    
    # 步骤3: 建议
    suggestions = suggestion_chain.invoke({"summary": summary})
    print(f"3. 建议:\n{suggestions}\n")

def deepseek_chinese_tasks():
    """DeepSeek中文任务演示"""
    print("=== DeepSeek中文任务处理 ===\n")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.8
    )
    
    tasks = [
        {
            "name": "诗词创作",
            "prompt": "请写一首关于程序员生活的现代诗",
            "system": "你是一位富有创意的现代诗人"
        },
        {
            "name": "文言文翻译",
            "prompt": "请将以下现代汉语翻译成文言文：人工智能技术正在快速发展，深刻地改变着我们的生活方式。",
            "system": "你是一位精通古代汉语的学者"
        },
        {
            "name": "商业计划",
            "prompt": "为一个基于AI的在线教育平台制定简要的商业计划",
            "system": "你是一位经验丰富的商业顾问"
        },
        {
            "name": "技术解释",
            "prompt": "用通俗易懂的语言解释什么是大语言模型",
            "system": "你是一位善于科普的技术专家"
        }
    ]
    
    for task in tasks:
        print(f"任务: {task['name']}")
        print(f"要求: {task['prompt']}")
        
        messages = [
            SystemMessage(content=task['system']),
            HumanMessage(content=task['prompt'])
        ]
        
        response = llm.invoke(messages)
        print(f"DeepSeek回复:\n{response.content}")
        print("=" * 60)

def deepseek_streaming_demo():
    """DeepSeek流式输出演示"""
    print("=== DeepSeek流式输出 ===\n")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.7,
        streaming=True
    )
    
    prompt = "请讲一个关于AI和人类合作的有趣故事，大约200字。"
    
    print(f"问题: {prompt}")
    print("\nDeepSeek正在思考并回答...")
    print("-" * 40)
    
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        print(chunk.content, end="", flush=True)
    
    print("\n" + "-" * 40)
    print("流式输出完成！")

def deepseek_model_comparison():
    """DeepSeek不同模型对比"""
    print("=== DeepSeek模型对比 ===\n")
    
    models = ["deepseek-chat", "deepseek-coder"]
    task = "写一个Python函数来实现二分查找算法"
    
    for model in models:
        print(f"模型: {model}")
        print(f"任务: {task}")
        
        llm = ChatOpenAI(
            model=model,
            temperature=0.3,
            max_tokens=800
        )
        
        response = llm.invoke([
            SystemMessage(content="请提供清晰的代码实现和解释"),
            HumanMessage(content=task)
        ])
        
        print(f"回复:\n{response.content}")
        print("=" * 80)

if __name__ == "__main__":
    print("🤖 DeepSeek API 演示")
    print("=" * 60)
    
    try:
        deepseek_basic_chat()
        deepseek_coder_demo()
        deepseek_chain_example()
        deepseek_chinese_tasks()
        deepseek_streaming_demo()
        deepseek_model_comparison()
        
        print("\n🎉 DeepSeek演示完成！")
        print("\n💡 使用提示:")
        print("1. DeepSeek-Chat: 适合通用对话、创作、分析任务")
        print("2. DeepSeek-Coder: 专门优化用于编程任务")
        print("3. 支持中文对话，理解中国文化背景")
        print("4. API兼容OpenAI格式，使用简单")
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        print("\n🔧 解决方案:")
        print("1. 检查.env文件中的DeepSeek API配置")
        print("2. 确认API密钥有效且有足够余额")
        print("3. 检查网络连接")