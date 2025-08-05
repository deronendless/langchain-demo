"""
配置文件
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# OpenAI配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 模型配置 - 根据使用的API服务商选择模型
# OpenAI模型
# DEFAULT_MODEL = "gpt-3.5-turbo"
# DEFAULT_MODEL = "gpt-4"

# DeepSeek模型 (推荐)
DEFAULT_MODEL = "deepseek-chat"
# DEFAULT_MODEL = "deepseek-coder"

# 智谱AI模型
# DEFAULT_MODEL = "glm-4"

# 嵌入模型 (目前只有OpenAI提供)
EMBEDDING_MODEL = "text-embedding-ada-002"

# 其他配置
MAX_TOKENS = 1000
TEMPERATURE = 0.7

def check_config():
    """检查配置是否正确"""
    if not OPENAI_API_KEY:
        raise ValueError("请在.env文件中设置OPENAI_API_KEY")
    print("✅ 配置检查通过")
    return True