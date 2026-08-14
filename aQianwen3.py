import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "openai:qwen3.8-max",
    api_key=os.getenv("DASHSCOPE_API_KEY"),          # ✅ 提取为显式参数
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # ✅ 提取为显式参数
    temperature=0,
)

response = model.invoke("你好，请介绍下你是什么模型，不少于100字")
print(response.content)