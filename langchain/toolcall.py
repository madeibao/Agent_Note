
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

@tool
def add(a: Annotated[int, "First integer"], b: Annotated[int, "Second integer"]) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(
    a: Annotated[int, "First integer"],
    b: Annotated[int, "Second integer"],
) -> int:
    """Multiply two integers."""
    return a * b


def build_model() -> ChatOpenAI:
    load_dotenv(Path(__file__).parent.with_name(".env"), encoding="utf-8-sig")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY. Please fill it in D:\\code\\langchain\\.env")

    return ChatOpenAI(
        model="qwen-plus",
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
    )


def main() -> None:
    model = build_model()

    tools = [add, multiply]
    tool_map = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)

    messages = [HumanMessage(content="9乘6等于多少？5加3等于多少？")]

    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    print("模型第一次响应：")
    print("content:", ai_msg.content or "<empty because the model requested tool calls>")
    print("tool_calls:", ai_msg.tool_calls)
    print()

    if not ai_msg.tool_calls:
        print("模型没有发起工具调用，直接回答：")
        print(ai_msg.content)
        return

    print("开始执行工具：")
    for tool_call in ai_msg.tool_calls:
        selected_tool = tool_map[tool_call["name"]]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
        print(f"- {tool_call['name']}({tool_call['args']}) -> {tool_msg.content}")
    print()

    final_msg = model.invoke(messages)
    print("模型最终回答：")
    print(final_msg.content)


if __name__ == "__main__":
    main()
