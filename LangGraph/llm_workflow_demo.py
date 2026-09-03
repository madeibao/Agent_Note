import os
from pathlib import Path
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


ROOT = Path(__file__).parent
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class WritingState(TypedDict):
    """State: data shared across the writing workflow."""

    topic: str
    audience: str
    outline: str
    draft: str
    final_article: str


def build_model() -> ChatOpenAI:
    load_dotenv(ROOT.parent / ".env", encoding="utf-8-sig")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY. Please fill it in D:\\code\\agent\\LangGraph\\.env")

    return ChatOpenAI(
        model="qwen-plus",
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
    )


model = build_model()


def generate_outline(state: WritingState) -> dict[str, str]:
    """Node 1: generate an outline for the article."""
    response = model.invoke([
        SystemMessage(content="你是一个擅长教学型技术文章规划的中文编辑。"),
        HumanMessage(
            content=(
                f"请为主题《{state['topic']}》生成一个简洁大纲。\n"
                f"目标读者：{state['audience']}。\n"
                "要求：只输出 4 个小标题，每个小标题后附一句说明。"
            )
        ),
    ])
    return {"outline": str(response.content)}


def write_draft(state: WritingState) -> dict[str, str]:
    """Node 2: write a short draft based on the outline."""
    response = model.invoke([
        SystemMessage(content="你是一个讲解清晰、循序渐进的中文技术作者。"),
        HumanMessage(
            content=(
                f"请根据下面的大纲写一篇短文初稿。\n\n"
                f"主题：{state['topic']}\n"
                f"目标读者：{state['audience']}\n\n"
                f"大纲：\n{state['outline']}\n\n"
                "要求：控制在 500 字以内，语言适合初学者。"
            )
        ),
    ])
    return {"draft": str(response.content)}


def polish_article(state: WritingState) -> dict[str, str]:
    """Node 3: polish the draft into the final article."""
    response = model.invoke([
        SystemMessage(content="你是一个严格但友好的中文技术文章润色编辑。"),
        HumanMessage(
            content=(
                "请润色下面的初稿，让它更自然、更适合博客发布。\n"
                "要求：保留核心意思，增加小标题，避免过度营销。\n\n"
                f"初稿：\n{state['draft']}"
            )
        ),
    ])
    return {"final_article": str(response.content)}


def build_graph():
    builder = StateGraph(WritingState)

    builder.add_node("生成大纲", generate_outline)
    builder.add_node("写初稿", write_draft)
    builder.add_node("润色文章", polish_article)

    builder.add_edge(START, "生成大纲")
    builder.add_edge("生成大纲", "写初稿")
    builder.add_edge("写初稿", "润色文章")
    builder.add_edge("润色文章", END)

    return builder.compile()


def main() -> None:
    workflow = build_graph()

    initial_state = {
        "topic": "LangGraph 为什么适合构建 AI 工作流",
        "audience": "刚学完 LangChain、准备学习 LangGraph 的初学者",
        "outline": "",
        "draft": "",
        "final_article": "",
    }

    result = workflow.invoke(initial_state)

    print("\n=== 1. 大纲 ===")
    print(result["outline"])

    print("\n=== 2. 初稿 ===")
    print(result["draft"])

    print("\n=== 3. 最终润色稿 ===")
    print(result["final_article"])


if __name__ == "__main__":
    main()
