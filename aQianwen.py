import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 加载环境变量
load_dotenv()

# 使用 ChatOpenAI 对接 DashScope（千问兼容 OpenAI 协议）
model = ChatOpenAI(
    #model="qwen3.8-max",
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # ⚠️ 注意：兼容模式端点
    temperature=0,
)

response = model.invoke("你好，请介绍 LangChain, 100字以内，中文回答。")

# ✅ AIMessage 的内容字段是 .content
print(response.content)