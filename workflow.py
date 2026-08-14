
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated

import operator
import os

load_dotenv()

# ─── 定义状态结构（在节点间传递的数据）────────────────────────
class ResearchState(TypedDict):
    topic: str                             # 研究主题
    research_notes: str                    # 研究笔记
    draft: str                             # 草稿
    review_feedback: str                   # 审阅意见
    final_report: str                      # 最终报告
    revision_count: Annotated[int, operator.add]   # 修改次数（累加）

# llm = ChatOpenAI(model="gpt-4o")

llm = ChatOpenAI(
    #model="qwen3.8-max",
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # ⚠️ 注意：兼容模式端点
    temperature=0.7,
)
# ─── 定义节点函数 ─────────────────────────────────────────────
def research_node(state: ResearchState) -> dict:
    """节点1：调研阶段"""
    response = llm.invoke(
        f"请对以下主题进行简要调研，列出 5 个关键要点：{state['topic']}"
    )
    return {"research_notes": response.content}

def write_node(state: ResearchState) -> dict:
    """节点2：撰写草稿"""
    prompt = f"""
    主题：{state['topic']}
    调研笔记：{state['research_notes']}
    {'上次审阅意见：' + state.get('review_feedback', '') if state.get('review_feedback') else ''}

    请根据以上内容撰写一篇 100 字的分析报告草稿。
    """
    response = llm.invoke(prompt)
    return {"draft": response.content, "revision_count": 1}

def review_node(state: ResearchState) -> dict:
    """节点3：审阅草稿"""
    response = llm.invoke(
        f"审阅以下报告，如果质量达标回复 'APPROVED'，否则给出具体修改意见：\n\n{state['draft']}"
    )
    return {"review_feedback": response.content}

def finalize_node(state: ResearchState) -> dict:
    """节点4：最终定稿"""
    return {"final_report": state["draft"]}

# ─── 路由函数：决定审阅后走哪条路 ─────────────────────────────
def should_revise(state: ResearchState) -> str:
    if "APPROVED" in state["review_feedback"]:
        return "finalize"           # → 最终定稿
    elif state["revision_count"] >= 3:
        return "finalize"           # → 超过3次修改，强制结束
    else:
        return "revise"             # → 返回写作节点修改

# ─── 构建工作流图 ─────────────────────────────────────────────
workflow = StateGraph(ResearchState)

# 添加节点
workflow.add_node("research",  research_node)
workflow.add_node("write",     write_node)
workflow.add_node("review",    review_node)
workflow.add_node("finalize",  finalize_node)

# 设置入口
workflow.set_entry_point("research")

# 添加边（定义流转逻辑）
workflow.add_edge("research", "write")       # 调研 → 写作
workflow.add_edge("write",    "review")      # 写作 → 审阅

# 条件边：审阅后根据结果选择路径
workflow.add_conditional_edges(
    "review",
    should_revise,
    {
        "revise":   "write",    # 需要修改 → 回到写作
        "finalize": "finalize"  # 通过审核 → 最终定稿
    }
)

workflow.add_edge("finalize", END)

# 编译并运行
app = workflow.compile()

result = app.invoke({"topic": "生成式 AI 对软件开发行业的影响", "revision_count": 0})
print(result["final_report"])