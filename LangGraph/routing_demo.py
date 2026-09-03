import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


ROOT = Path(__file__).parent
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class CustomerServiceState(TypedDict):
    """State shared by the routing workflow."""

    user_input: str
    route: str
    reason: str
    output: str


class RouteDecision(BaseModel):
    """Structured routing decision returned by the model."""

    route: Literal["pre_sale", "after_sale", "technical"] = Field(
        description="Route user question to pre_sale, after_sale, or technical."
    )
    reason: str = Field(description="Brief reason for the routing decision.")


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
router_model = model.with_structured_output(RouteDecision)


def classify_question(state: CustomerServiceState) -> dict[str, str]:
    """Node 1: classify the user input and decide which branch to enter."""
    response = router_model.invoke([
        SystemMessage(
            content=(
                "你是智能客服路由器。请把用户问题分类到以下三类之一：\n"
                "1. pre_sale：购买前咨询、价格、套餐、功能、试用、购买建议。\n"
                "2. after_sale：退款、退货、订单、物流、发票、售后政策、已购买产品问题。\n"
                "3. technical：安装、报错、配置、接口、数据库、代码、系统故障。\n"
                "必须根据用户问题给出结构化路由结果。"
            )
        ),
        HumanMessage(content=state["user_input"]),
    ])
    return {
        "route": response.route,
        "reason": response.reason,
    }


def route_decision(state: CustomerServiceState) -> str:
    """Conditional edge router."""
    if state["route"] == "pre_sale":
        return "pre_sale_handler"
    if state["route"] == "after_sale":
        return "after_sale_handler"
    if state["route"] == "technical":
        return "technical_handler"
    raise ValueError(f"Unknown route: {state['route']}")


def pre_sale_handler(state: CustomerServiceState) -> dict[str, str]:
    """Node 2A: handle pre-sale questions."""
    return {
        "output": (
            "【售前咨询】已进入售前处理流程。\n"
            f"路由原因：{state['reason']}\n"
            "建议回复：可以向用户介绍产品功能、套餐价格、适用场景，并引导预约演示或试用。"
        )
    }


def after_sale_handler(state: CustomerServiceState) -> dict[str, str]:
    """Node 2B: handle after-sale questions."""
    return {
        "output": (
            "【售后问题】已进入售后处理流程。\n"
            f"路由原因：{state['reason']}\n"
            "建议回复：先核实订单号、购买时间和问题描述，再根据售后政策处理退款、退货、换货或物流问题。"
        )
    }


def technical_handler(state: CustomerServiceState) -> dict[str, str]:
    """Node 2C: handle technical questions."""
    return {
        "output": (
            "【技术支持】已进入技术处理流程。\n"
            f"路由原因：{state['reason']}\n"
            "建议回复：收集系统环境、报错信息、复现步骤和配置文件，再给出排查建议或升级到工程师。"
        )
    }


def build_graph():
    builder = StateGraph(CustomerServiceState)

    builder.add_node("classify_question", classify_question)
    builder.add_node("pre_sale_handler", pre_sale_handler)
    builder.add_node("after_sale_handler", after_sale_handler)
    builder.add_node("technical_handler", technical_handler)

    builder.add_edge(START, "classify_question")
    builder.add_conditional_edges(
        "classify_question",
        route_decision,
        ["pre_sale_handler", "after_sale_handler", "technical_handler"],
    )
    builder.add_edge("pre_sale_handler", END)
    builder.add_edge("after_sale_handler", END)
    builder.add_edge("technical_handler", END)

    return builder.compile()


def main() -> None:
    graph = build_graph()

    test_cases = [
        "你们这个产品多少钱？个人版和企业版有什么区别？",
        "我上周买的产品有质量问题，想申请退货。",
        "软件安装后启动失败，提示数据库连接超时，应该怎么配置？",
        "我的订单已经发货三天了，为什么还没收到？",
    ]

    for question in test_cases:
        print("\n" + "=" * 80)
        print("用户问题：", question)
        result = graph.invoke({
            "user_input": question,
            "route": "",
            "reason": "",
            "output": "",
        })
        print("路由结果：", result["route"])
        print("路由原因：", result["reason"])
        print("处理输出：")
        print(result["output"])


if __name__ == "__main__":
    main()
