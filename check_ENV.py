

# 文件路径：check_install.py
import os
from dotenv import load_dotenv

load_dotenv()

# 检查依赖包能否正常导入
import langchain
import langchain_deepseek
import langchain_openai
import langchain_chroma
import chromadb
import langgraph

print(f"langchain 版本: {langchain.__version__}")

# 检查密钥是否已配置
assert os.getenv("DEEPSEEK_API_KEY"), "未检测到 DEEPSEEK_API_KEY，请检查 .env 文件"
# assert os.getenv("OPENAI_API_KEY"), "未检测到 OPENAI_API_KEY，请检查 .env 文件"
assert os.getenv("DASHSCOPE_API_KEY"), "未检测到 DASHSCOPE_API_KEY，请检查 .env 文件"

print("环境配置成功~可以开始写客服机器人了！")