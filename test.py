# import os

# from dotenv import load_dotenv

# # 加载当前目录 .env 文件
# load_dotenv()

# # 获取 API Key
# api_key = os.getenv("DEEPSEEK_API_KEY")

# print(api_key)


import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# 加载 .env
load_dotenv()

# 获取 API KEY
api_key = os.getenv("DEEPSEEK_API_KEY")

# 创建模型
llm = ChatDeepSeek(
    api_key=api_key,
    model="deepseek-v4-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# 调用模型
response = llm.invoke("你好，请介绍 LangChain")

print(response.content)