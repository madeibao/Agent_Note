from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
import json

load_dotenv()

class Analysis(BaseModel):
    """分析结果"""
    summary: str = Field(description="一句话总结")
    score: int = Field(description="评分 1~10")
    pros: list[str] = Field(description="优点列表")
    cons: list[str] = Field(description="缺点列表")


from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

# model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

# 千问通过 OpenAI 兼容协议接入，provider 前缀为 "openai:"
model = init_chat_model(
    "openai:qwen-plus",                      # ← 模型名改为千问系列
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # ← 阿里云 DashScope API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # ← 千问兼容端点
    temperature=0
)

# 构建提示词，要求返回 JSON 格式
system_prompt = """你是课程评估专家，评估用户描述的课程。
请以JSON格式返回分析结果，格式如下：
{
    "summary": "一句话总结",
    "score": 评分(1-10的整数),
    "pros": ["优点1", "优点2"],
    "cons": ["缺点1", "缺点2"]
}"""

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content="菜鸟教程 RUNOOB 的 Python 课程：内容系统全面，实例丰富，而且完全免费。但视频教程较少，高级内容覆盖不够。")
]

response = model.invoke(messages)
content = response.content

# 提取 JSON（处理可能的markdown包裹）
if "```json" in content:
    content = content.split("```json")[1].split("```")[0]
elif "```" in content:
    content = content.split("```")[1].split("```")[0]

# 解析并验证
try:
    data = json.loads(content)
    analysis = Analysis(**data)
    print(f"总结: {analysis.summary}")
    print(f"评分: {analysis.score}/10")
    print(f"优点: {', '.join(analysis.pros)}")
    print(f"缺点: {', '.join(analysis.cons)}")
except (json.JSONDecodeError, ValueError) as e:
    print(f"解析失败: {e}")
    print(f"原始响应: {content}")
    # 可以尝试重新生成或使用备用方案