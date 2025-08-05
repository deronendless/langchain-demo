"""
示例6: 智能代理 (Agents)
演示如何使用LangChain创建能够使用工具的智能代理
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
import json
import requests
from datetime import datetime
import math
from config import check_config, DEFAULT_MODEL

# 定义自定义工具
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "用于进行数学计算。输入应该是一个数学表达式。"
    
    def _run(self, query: str) -> str:
        try:
            # 安全的数学计算
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round})
            
            result = eval(query, {"__builtins__": {}}, allowed_names)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂不支持异步操作")

class WeatherTool(BaseTool):
    name = "weather"
    description = "获取指定城市的天气信息。输入应该是城市名称。"
    
    def _run(self, query: str) -> str:
        # 模拟天气API调用
        weather_data = {
            "北京": {"temperature": "15°C", "condition": "晴天", "humidity": "45%"},
            "上海": {"temperature": "18°C", "condition": "多云", "humidity": "60%"},
            "广州": {"temperature": "25°C", "condition": "小雨", "humidity": "80%"},
            "深圳": {"temperature": "26°C", "condition": "晴天", "humidity": "55%"}
        }
        
        city = query.strip()
        if city in weather_data:
            data = weather_data[city]
            return f"{city}当前天气: 温度{data['temperature']}, {data['condition']}, 湿度{data['humidity']}"
        else:
            return f"抱歉，暂无{city}的天气信息"
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂不支持异步操作")

class TimeTool(BaseTool):
    name = "current_time"
    description = "获取当前时间。不需要任何输入参数。"
    
    def _run(self, query: str) -> str:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"当前时间: {current_time}"
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("暂不支持异步操作")

def simple_agent_example():
    """简单代理示例"""
    print("=== 简单代理示例 ===\n")
    
    check_config()
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
    
    # 创建工具
    tools = [
        CalculatorTool(),
        WeatherTool(),
        TimeTool()
    ]
    
    # 创建代理提示词模板
    template = """你是一个有用的助手，可以使用以下工具来回答问题：

{tools}

使用以下格式：

Question: 需要回答的问题
Thought: 我需要思考如何回答这个问题
Action: 要使用的工具名称
Action Input: 工具的输入
Observation: 工具返回的结果
... (这个思考/行动/观察的过程可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终答案

开始！

Question: {input}
Thought: {agent_scratchpad}"""
    
    prompt = PromptTemplate.from_template(template)
    
    # 创建ReAct代理
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)
    
    # 测试问题
    questions = [
        "现在几点了？",
        "计算 25 * 4 + 10",
        "北京的天气怎么样？",
        "如果我在上海，温度是多少度？计算一下华氏度是多少？"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        print("代理执行过程:")
        try:
            result = agent_executor.invoke({"input": question})
            print(f"最终答案: {result['output']}")
        except Exception as e:
            print(f"执行错误: {e}")
        print("-" * 80)

def multi_step_agent_example():
    """多步骤代理示例"""
    print("=== 多步骤代理示例 ===\n")
    
    check_config()
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
    
    # 增加更多工具
    class TextAnalyzerTool(BaseTool):
        name = "text_analyzer"
        description = "分析文本的字数、词数和字符数。输入应该是要分析的文本。"
        
        def _run(self, query: str) -> str:
            text = query.strip()
            char_count = len(text)
            word_count = len(text.split())
            line_count = len(text.splitlines())
            
            return f"文本分析结果: 字符数={char_count}, 词数={word_count}, 行数={line_count}"
        
        async def _arun(self, query: str) -> str:
            raise NotImplementedError("暂不支持异步操作")
    
    class UnitConverterTool(BaseTool):
        name = "unit_converter"
        description = "进行单位转换。支持温度(C2F, F2C)、长度(m2ft, ft2m)等。格式: '数值 单位1 to 单位2'"
        
        def _run(self, query: str) -> str:
            try:
                parts = query.strip().split()
                if len(parts) != 4 or parts[2].lower() != 'to':
                    return "格式错误，请使用: '数值 单位1 to 单位2'"
                
                value = float(parts[0])
                from_unit = parts[1].lower()
                to_unit = parts[3].lower()
                
                conversions = {
                    ('c', 'f'): lambda x: x * 9/5 + 32,
                    ('f', 'c'): lambda x: (x - 32) * 5/9,
                    ('m', 'ft'): lambda x: x * 3.28084,
                    ('ft', 'm'): lambda x: x / 3.28084,
                    ('kg', 'lb'): lambda x: x * 2.20462,
                    ('lb', 'kg'): lambda x: x / 2.20462
                }
                
                if (from_unit, to_unit) in conversions:
                    result = conversions[(from_unit, to_unit)](value)
                    return f"转换结果: {value} {from_unit} = {result:.2f} {to_unit}"
                else:
                    return f"不支持从 {from_unit} 到 {to_unit} 的转换"
            except Exception as e:
                return f"转换错误: {str(e)}"
        
        async def _arun(self, query: str) -> str:
            raise NotImplementedError("暂不支持异步操作")
    
    tools = [
        CalculatorTool(),
        WeatherTool(),
        TimeTool(),
        TextAnalyzerTool(),
        UnitConverterTool()
    ]
    
    # 创建代理
    template = """你是一个多功能助手，可以使用多种工具来解决复杂问题。

可用工具:
{tools}

请按照以下格式思考和行动：

Question: 用户的问题
Thought: 分析问题，决定需要使用哪些工具
Action: 选择工具
Action Input: 工具输入
Observation: 工具输出
... (根据需要重复)
Thought: 基于所有观察结果，我现在可以给出最终答案
Final Answer: 综合回答

Question: {input}
Thought: {agent_scratchpad}"""
    
    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)
    
    # 复杂问题测试
    complex_questions = [
        "今天几点了？如果现在是15度摄氏度，换算成华氏度是多少？",
        "分析这段文本'人工智能是未来科技发展的重要方向'，然后计算它有多少个字符",
        "上海今天的温度是多少？如果我要换算成华氏度，应该怎么计算？"
    ]
    
    for question in complex_questions:
        print(f"\n复杂问题: {question}")
        print("代理执行过程:")
        try:
            result = agent_executor.invoke({"input": question})
            print(f"最终答案: {result['output']}")
        except Exception as e:
            print(f"执行错误: {e}")
        print("-" * 80)

def custom_agent_with_memory():
    """带记忆的自定义代理"""
    print("=== 带记忆的自定义代理 ===\n")
    
    check_config()
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
    
    class MemoryTool(BaseTool):
        name = "memory"
        description = "存储或检索信息。格式: 'STORE key value' 或 'GET key'"
        
        def __init__(self):
            super().__init__()
            self.storage = {}
        
        def _run(self, query: str) -> str:
            parts = query.strip().split()
            if not parts:
                return "请提供操作指令"
            
            action = parts[0].upper()
            
            if action == "STORE" and len(parts) >= 3:
                key = parts[1]
                value = " ".join(parts[2:])
                self.storage[key] = value
                return f"已存储: {key} = {value}"
            elif action == "GET" and len(parts) == 2:
                key = parts[1]
                if key in self.storage:
                    return f"检索到: {key} = {self.storage[key]}"
                else:
                    return f"未找到键: {key}"
            elif action == "LIST":
                if self.storage:
                    items = [f"{k} = {v}" for k, v in self.storage.items()]
                    return f"存储的信息:\n" + "\n".join(items)
                else:
                    return "暂无存储的信息"
            else:
                return "无效操作。使用 'STORE key value', 'GET key' 或 'LIST'"
        
        async def _arun(self, query: str) -> str:
            raise NotImplementedError("暂不支持异步操作")
    
    # 创建包含记忆工具的代理
    memory_tool = MemoryTool()
    tools = [
        CalculatorTool(),
        WeatherTool(),
        memory_tool
    ]
    
    template = """你是一个智能助手，具有记忆能力。你可以存储和检索信息。

可用工具:
{tools}

格式:
Question: 问题
Thought: 思考
Action: 动作
Action Input: 输入
Observation: 观察
...
Final Answer: 最终答案

Question: {input}
Thought: {agent_scratchpad}"""
    
    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 测试记忆功能
    memory_tests = [
        "请记住我的名字是张三",
        "请记住我住在北京",
        "我的名字是什么？",
        "我住在哪里？",
        "北京的天气怎么样？顺便告诉我我是谁？",
        "列出你记住的所有信息"
    ]
    
    for question in memory_tests:
        print(f"\n问题: {question}")
        print("代理执行:")
        try:
            result = agent_executor.invoke({"input": question})
            print(f"回答: {result['output']}")
        except Exception as e:
            print(f"错误: {e}")
        print("-" * 60)

if __name__ == "__main__":
    try:
        simple_agent_example()
        multi_step_agent_example()
        custom_agent_with_memory()
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查你的API密钥配置")