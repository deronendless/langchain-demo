#!/usr/bin/env python3
"""
示例运行器
提供一个统一的入口来运行所有示例
"""
import sys
import os
import subprocess
from config import check_config

def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def run_example(script_name, description):
    """运行单个示例"""
    script_path = os.path.join("examples", script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ 文件不存在: {script_path}")
        return False
    
    print(f"\n🚀 运行示例: {description}")
    print(f"📁 文件: {script_name}")
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True)
        if result.returncode == 0:
            print(f"✅ {description} 运行完成")
            return True
        else:
            print(f"❌ {description} 运行失败")
            return False
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        return False

def main():
    """主函数"""
    print_header("LangChain Demo 示例运行器")
    
    # 检查配置
    try:
        check_config()
        print("✅ 配置检查通过")
    except Exception as e:
        print(f"❌ 配置错误: {e}")
        print("请先配置好 .env 文件中的 API 密钥")
        return
    
    # 定义所有示例
    examples = [
        ("01_basic_llm.py", "基础 LLM 调用"),
        ("02_prompt_templates.py", "提示词模板"),
        ("03_chains.py", "链式调用"),
        ("04_memory.py", "记忆功能"),
        ("05_rag.py", "RAG 检索增强生成"),
        ("06_agents.py", "智能代理"),
        ("08_deepseek_demo.py", "DeepSeek API 演示")
    ]
    
    print("\n📋 可用示例:")
    for i, (script, desc) in enumerate(examples, 1):
        print(f"  {i}. {desc} ({script})")
    print(f"  8. 运行 Streamlit Web 应用")
    print(f"  9. 运行所有示例")
    print(f"  0. 退出")
    
    while True:
        try:
            choice = input("\n请选择要运行的示例 (0-9): ").strip()
            
            if choice == "0":
                print("👋 再见!")
                break
            elif choice == "8":
                print("\n🌐 启动 Streamlit Web 应用...")
                print("🔗 应用将在浏览器中打开: http://localhost:8501")
                print("💡 按 Ctrl+C 停止应用")
                subprocess.run([sys.executable, "-m", "streamlit", "run", 
                              "examples/07_streamlit_app.py"])
            elif choice == "9":
                print("\n🔄 运行所有示例...")
                success_count = 0
                for script, desc in examples:
                    if run_example(script, desc):
                        success_count += 1
                    print()  # 空行分隔
                
                print(f"\n📊 运行结果: {success_count}/{len(examples)} 个示例成功运行")
            elif choice.isdigit() and 1 <= int(choice) <= 7:
                idx = int(choice) - 1
                script, desc = examples[idx]
                run_example(script, desc)
            else:
                print("❌ 无效选择，请输入 0-9 之间的数字")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()