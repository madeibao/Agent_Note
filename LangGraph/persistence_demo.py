import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


ROOT = Path(__file__).parent
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


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


def chatbot(state: MessagesState) -> dict:
    """A normal chat node. MessagesState will save and restore messages."""
    messages = [
        SystemMessage(
            content=(
                "你是一个 LangGraph 学习助教。请用中文回答。"
                "如果用户问到之前说过的信息，要根据历史消息回答。"
            )
        ),
        *state["messages"],
    ]
    response = model.invoke(messages)
    return {"messages": [response]}


def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def print_latest_answer(title: str, result: MessagesState) -> None:
    print("\n" + "=" * 80)
    print(title)
    print(result["messages"][-1].content)


def main() -> None:
    graph = build_graph()

    same_thread_config = {
        "configurable": {
            "thread_id": "student-xiaoming"
        }
    }

    other_thread_config = {
        "configurable": {
            "thread_id": "student-other"
        }
    }

    print("=" * 80)
    print("示例 1：同一个 thread_id，会保存并读取历史状态")

    first_result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="我叫小明，我现在正在学习 LangGraph 的持久化。")
            ]
        },
        config=same_thread_config,
    )
    print_latest_answer("第一次调用，用户告诉模型自己的名字：", first_result)

    second_result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="我刚才说我叫什么？我正在学什么？")
            ]
        },
        config=same_thread_config,
    )
    print_latest_answer("第二次调用，使用同一个 thread_id，模型可以读取之前状态：", second_result)

    print("\n" + "=" * 80)
    print("示例 2：换一个 thread_id，相当于新会话，不会读取小明的历史")

    third_result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="我刚才说我叫什么？我正在学什么？")
            ]
        },
        config=other_thread_config,
    )
    print_latest_answer("第三次调用，换了 thread_id，模型通常不知道之前的信息：", third_result)

    print("\n" + "=" * 80)
    print("当前 same_thread_config 里的消息数量：", len(second_result["messages"]))
    print("你可以理解为：MemorySaver 根据 thread_id 保存了这条会话的 MessagesState。")


if __name__ == "__main__":
    main()
