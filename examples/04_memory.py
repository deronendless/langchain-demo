"""
示例4: 记忆功能
演示如何使用LangChain的记忆功能来维持对话上下文
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory
)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from config import check_config, DEFAULT_MODEL

def buffer_memory_example():
    """缓冲区记忆示例"""
    print("=== 缓冲区记忆示例 ===\n")
    
    check_config()
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    
    # 创建缓冲区记忆
    memory = ConversationBufferMemory(return_messages=True)
    
    # 创建包含记忆的提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手，请根据对话历史进行回答。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # 创建链
    chain = prompt | llm
    
    # 模拟多轮对话
    conversations = [
        "我的名字是张三",
        "我是一名软件工程师",
        "我最喜欢的编程语言是Python",
        "请问我的名字是什么？",
        "我的职业是什么？"
    ]
    
    for i, user_input in enumerate(conversations):
        print(f"轮次 {i+1}:")
        print(f"用户: {user_input}")
        
        # 从记忆中获取历史消息
        history = memory.chat_memory.messages
        
        # 调用链
        response = chain.invoke({
            "history": history,
            "input": user_input
        })
        
        print(f"AI: {response.content}")
        
        # 将对话添加到记忆中
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response.content)
        print()

def window_memory_example():
    """滑动窗口记忆示例"""
    print("=== 滑动窗口记忆示例 ===\n")
    
    # 创建窗口大小为3的记忆（只保留最近3轮对话）
    memory = ConversationBufferWindowMemory(k=3, return_messages=True)
    
    # 添加一些对话历史
    conversations = [
        ("用户1", "AI回复1"),
        ("用户2", "AI回复2"), 
        ("用户3", "AI回复3"),
        ("用户4", "AI回复4"),
        ("用户5", "AI回复5")
    ]
    
    for user_msg, ai_msg in conversations:
        memory.chat_memory.add_user_message(user_msg)
        memory.chat_memory.add_ai_message(ai_msg)
    
    # 查看记忆中保存的消息（应该只有最近3轮）
    print("滑动窗口记忆中的消息:")
    messages = memory.chat_memory.messages
    for i, msg in enumerate(messages):
        msg_type = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"{i+1}. {msg_type}: {msg.content}")
    print(f"\n总共保存了 {len(messages)} 条消息")
    print()

def summary_memory_example():
    """摘要记忆示例"""
    print("=== 摘要记忆示例 ===\n")
    
    check_config()
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    
    # 创建摘要记忆
    memory = ConversationSummaryMemory(llm=llm, return_messages=True)
    
    # 添加一些长对话
    long_conversation = [
        ("你好，我想了解机器学习", "你好！我很乐意帮你了解机器学习。机器学习是人工智能的一个分支..."),
        ("什么是监督学习？", "监督学习是机器学习的一种类型，它使用标记的训练数据来学习..."),
        ("能给我举个例子吗？", "当然可以！一个典型的监督学习例子是邮件分类..."),
        ("那无监督学习呢？", "无监督学习是另一种类型，它处理没有标签的数据..."),
        ("强化学习是什么？", "强化学习是通过与环境交互来学习最优行为策略的方法...")
    ]
    
    for user_msg, ai_msg in long_conversation:
        memory.chat_memory.add_user_message(user_msg)
        memory.chat_memory.add_ai_message(ai_msg)
    
    # 获取对话摘要
    summary = memory.predict_new_summary(
        memory.chat_memory.messages, 
        ""
    )
    
    print("对话摘要:")
    print(summary)
    print()

def summary_buffer_memory_example():
    """摘要缓冲区记忆示例"""
    print("=== 摘要缓冲区记忆示例 ===\n")
    
    check_config()
    llm = ChatOpenAI(model=DEFAULT_MODEL)
    
    # 创建摘要缓冲区记忆（token限制为100）
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=100,
        return_messages=True
    )
    
    # 添加对话
    conversations = [
        "我正在学习Python编程",
        "我已经掌握了基础语法", 
        "现在想学习Web开发框架",
        "推荐学习Django还是Flask？",
        "我更偏向于轻量级的框架"
    ]
    
    ai_responses = [
        "很好！Python是一门优秀的编程语言。",
        "掌握基础语法是很重要的第一步。",
        "Web开发是Python的重要应用领域。",
        "Django和Flask都是优秀的框架，各有特点。",
        "基于你的偏好，我推荐从Flask开始学习。"
    ]
    
    for user_msg, ai_msg in zip(conversations, ai_responses):
        memory.chat_memory.add_user_message(user_msg)
        memory.chat_memory.add_ai_message(ai_msg)
        
        print(f"用户: {user_msg}")
        print(f"AI: {ai_msg}")
        
        # 查看当前记忆状态
        print("当前记忆状态:")
        if hasattr(memory, 'moving_summary_buffer') and memory.moving_summary_buffer:
            print(f"摘要: {memory.moving_summary_buffer}")
        
        messages = memory.chat_memory.messages
        print(f"缓冲区消息数: {len(messages)}")
        print("-" * 50)

def custom_memory_example():
    """自定义记忆示例"""
    print("=== 自定义记忆示例 ===\n")
    
    class UserProfileMemory:
        """自定义用户档案记忆"""
        def __init__(self):
            self.profile = {}
            self.conversation_history = []
        
        def add_user_info(self, key, value):
            """添加用户信息"""
            self.profile[key] = value
        
        def add_conversation(self, user_msg, ai_msg):
            """添加对话"""
            self.conversation_history.append({
                "user": user_msg,
                "ai": ai_msg
            })
        
        def get_profile_summary(self):
            """获取用户档案摘要"""
            if not self.profile:
                return "暂无用户信息"
            
            summary = "用户档案: "
            for key, value in self.profile.items():
                summary += f"{key}: {value}; "
            return summary
        
        def get_recent_conversations(self, n=3):
            """获取最近n轮对话"""
            return self.conversation_history[-n:]
    
    # 使用自定义记忆
    memory = UserProfileMemory()
    
    # 模拟收集用户信息的对话
    print("模拟对话过程:")
    
    # 添加用户信息
    memory.add_user_info("姓名", "李四")
    memory.add_user_info("职业", "数据科学家")
    memory.add_user_info("兴趣", "机器学习")
    
    # 添加对话历史
    memory.add_conversation("我想学习深度学习", "基于你的背景，我建议从神经网络基础开始")
    memory.add_conversation("推荐一些学习资源", "我推荐deeplearning.ai的课程")
    memory.add_conversation("需要什么先决条件？", "你已经有数据科学背景，应该没问题")
    
    print("用户档案:")
    print(memory.get_profile_summary())
    print("\n最近对话:")
    for conv in memory.get_recent_conversations():
        print(f"用户: {conv['user']}")
        print(f"AI: {conv['ai']}")
        print()

if __name__ == "__main__":
    try:
        buffer_memory_example()
        window_memory_example()
        summary_memory_example()
        summary_buffer_memory_example()
        custom_memory_example()
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查你的API密钥配置")