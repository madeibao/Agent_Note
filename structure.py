from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import json

load_dotenv()

class SentimentResult(BaseModel):
    """情感分析结果"""
    sentiment: str = Field(description="积极/消极/中性")
    score: float = Field(description="情感强度 0~1")
    keywords: list[str] = Field(description="关键情感词")

model = ChatDeepSeek(model="deepseek-chat", temperature=0)


# 不使用 with_structured_output()，而是通过提示词让模型返回 JSON 格式，然后手动解析：
# deepseek 的支持度不够，改用其他的方式来实现 


# 使用 JSON 模式提示词
prompt_template = """请分析以下文本的情感，返回 JSON 格式结果，包含以下字段：
- sentiment: 积极/消极/中性
- score: 情感强度 0~1
- keywords: 关键情感词列表

文本: {text}

只返回 JSON，不要有其他内容。"""

texts = [
    "菜鸟教程 RUNOOB 真的太好用了，强烈推荐！",
    "这个教程内容太少了，不太值。",
    "今天天气不错。",
]

for text in texts:
    prompt = prompt_template.format(text=text)
    response = model.invoke(prompt)
    
    try:
        # 尝试解析 JSON
        result_data = json.loads(response.content)
        result = SentimentResult(**result_data)
        print(f"文本: {text[:30]}...")
        print(f"  情感: {result.sentiment}, 强度: {result.score}, 关键词: {result.keywords}")
    except json.JSONDecodeError:
        print(f"解析失败: {response.content}")