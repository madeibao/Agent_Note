
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# 加载 .env
load_dotenv()
    
# 获取 API KEY
api_key = os.getenv("DEEPSEEK_API_KEY")


# 1. 定义模型 (Model)
llm = ChatDeepSeek(
    api_key=api_key,
    model="deepseek-v4-flash"
)

# 2. 定义提示词模板 (Prompt)
# system: 设定 AI 的角色
# user: 用户的具体输入
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的技术专家，擅长用通俗易懂的'大白话'解释复杂的科技概念。你的解释应该包含一个生动的比喻。"),
    ("user", "{concept}")
])

# 3. 定义输出解析器 (Output Parser)
# 将模型的 Message 对象直接转为纯字符串
parser = StrOutputParser()

# 4. 构建链 (Chain) - 使用管道符 | 连接
# 流程：输入字典 -> 填充 Prompt -> 发给 LLM -> 解析输出
chain = prompt | llm | parser

# 5. 调用链
concept_to_explain = "量子纠缠"
print(f"正在解释概念：{concept_to_explain} (流式输出)...\n")

# 这里的 chunk 是每次生成的片段
for chunk in chain.stream({"concept": "递归神经网络"}):
    print(chunk, end="", flush=True)