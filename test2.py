
import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# 加载 .env
load_dotenv()

# 获取 API KEY
api_key = os.getenv("DEEPSEEK_API_KEY")


llm = ChatDeepSeek(
    api_key=api_key,    # 设置你的 DeepSeek API Key
    model="deepseek-v4-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to Chinese."
    ),
    (
        "human",
        "I love programming."
    ),
]

if __name__ == '__main__':
    ai_msg = llm.invoke(messages)
    print(ai_msg.content)