import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


ROOT = Path(__file__).parent
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class ChatState(TypedDict):
    """State shared by all nodes in this graph."""
    messages: Annotated[list[BaseMessage], add_messages]


def build_model() -> ChatOpenAI:
    # Prefer this project's .env, then fall back to the LangChain practice project.
    env_path = ROOT.parent / ".env"
    if not env_path.exists():
        env_path = Path(r"D:\code\langchain\.env")

    load_dotenv(env_path, encoding="utf-8-sig")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing DASHSCOPE_API_KEY. Put it in D:\\code\\agent\\LangGraph\\.env "
            "or D:\\code\\langchain\\.env."
        )

    return ChatOpenAI(
        model="qwen-plus",
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
    )


model = build_model()


def chatbot(state: ChatState) -> dict[str, list[AIMessage]]:
    """A single node: read messages from state, call the model, return new messages."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def build_graph():
    graph_builder = StateGraph(ChatState)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    return graph_builder.compile()


def main() -> None:
    app = build_graph()

    initial_state = {
        "messages": [
            HumanMessage(content="你好，我正在学习 LangGraph。请用三句话解释 State、Node、Edge。")
        ]
    }

    final_state = app.invoke(initial_state)

    print("最终 State 里的 messages：")
    for index, message in enumerate(final_state["messages"], start=1):
        print(f"\n{index}. {message.type}")
        print(message.content)


if __name__ == "__main__":
    main()
