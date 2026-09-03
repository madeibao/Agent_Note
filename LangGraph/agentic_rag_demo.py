import hashlib
import math
import os
import re
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


ROOT = Path(__file__).parent
DOC_PATH = ROOT / "sample_docs" / "langgraph_rag_notes.md"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_REWRITES = 1


class HashEmbeddings(Embeddings):
    """Small local embedding model for demos. Not for production."""

    def __init__(self, dimensions: int = 96) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0
        length = math.sqrt(sum(value * value for value in vector))
        if length == 0:
            return vector
        return [value / length for value in vector]

    def _tokenize(self, text: str) -> list[str]:
        english_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", text.lower())
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        return english_tokens + chinese_chars


class AgenticRagState(TypedDict):
    """Graph state.

    messages uses add_messages, so every node can append new messages instead of
    manually copying the full history.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    rewrite_count: int


class GradeDocuments(BaseModel):
    """Binary relevance score for retrieved documents."""

    score: Literal["yes", "no"] = Field(
        description="Return yes if retrieved context can answer the question, otherwise no."
    )


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


def load_and_split_documents() -> list[Document]:
    markdown_text = DOC_PATH.read_text(encoding="utf-8-sig")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
        ]
    )
    section_docs = header_splitter.split_text(markdown_text)

    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=260,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    return chunk_splitter.split_documents(section_docs)


def build_retriever_tool():
    chunks = load_and_split_documents()
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=HashEmbeddings(dimensions=96),
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return create_retriever_tool(
        retriever,
        "retrieve_langgraph_notes",
        "搜索并返回 LangGraph、RAG、Agentic RAG、Workflow、State、Node、Edge、持久化等学习笔记。",
    )


model = build_model()
retriever_tool = build_retriever_tool()


def generate_query_or_respond(state: AgenticRagState) -> dict[str, list[AIMessage]]:
    """Node 1: decide whether to answer directly or call the retriever tool."""
    system_prompt = SystemMessage(
        content=(
            "你是一个严谨的 LangGraph 学习助手。"
            "如果问题涉及 LangGraph、RAG、Agent、Workflow、State、Node、Edge、持久化等课程知识，"
            "优先调用 retrieve_langgraph_notes 工具检索资料。"
            "如果问题明显与这些知识无关，可以直接回答或说明资料范围不包含。"
        )
    )
    response = model.bind_tools([retriever_tool]).invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_retrieve(state: AgenticRagState) -> Literal["retrieve", "end"]:
    """Route after the decision node."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "retrieve"
    return "end"


def grade_documents(state: AgenticRagState) -> Literal["generate_answer", "rewrite_question"]:
    """Node router: grade whether retrieved docs are relevant enough."""
    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "generate_answer"

    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = (
        "你是一个检索结果相关性评分器。判断检索到的资料是否足以回答用户问题。\n"
        "只要资料中有明确相关信息，就返回 yes；如果资料明显无关或不足，就返回 no。\n\n"
        f"用户问题：{question}\n\n"
        f"检索资料：{context}"
    )
    response = model.with_structured_output(GradeDocuments).invoke([HumanMessage(content=prompt)])
    print(f"\n[grade_documents] relevance score: {response.score}")
    return "generate_answer" if response.score == "yes" else "rewrite_question"


def rewrite_question(state: AgenticRagState) -> dict[str, object]:
    """Node 3: rewrite the original question to improve retrieval."""
    question = state["messages"][0].content
    prompt = (
        "请把用户问题改写成更适合检索知识库的查询。\n"
        "要求：保留原意，补充关键术语，只输出改写后的问题。\n\n"
        f"原始问题：{question}"
    )
    response = model.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [HumanMessage(content=str(response.content))],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


def generate_answer(state: AgenticRagState) -> dict[str, list[AIMessage]]:
    """Node 4: generate final answer from retrieved context."""
    original_question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = (
        "你是负责回答 LangGraph 学习问题的助手。\n"
        "请只根据给定资料回答。资料不足时，必须说：资料中没有足够信息。\n"
        "回答要中文、清晰、分点说明。\n\n"
        f"用户原始问题：{original_question}\n\n"
        f"资料：\n{context}"
    )
    response = model.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}


def build_graph():
    workflow = StateGraph(AgenticRagState)

    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
        },
    )
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


def print_messages(messages: list[BaseMessage]) -> None:
    print("\n--- Graph Messages ---")
    for index, message in enumerate(messages, start=1):
        content = str(message.content).replace("\n", " ")
        if len(content) > 350:
            content = content[:350] + "..."
        print(f"{index}. {message.type}: {content}")
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            print(f"   tool_calls: {tool_calls}")


def run_question(graph, question: str) -> None:
    print("\n" + "=" * 90)
    print("用户问题：", question)
    result = graph.invoke({
        "messages": [HumanMessage(content=question)],
        "rewrite_count": 0,
    })
    print_messages(result["messages"])
    print("\n最终回答：")
    print(result["messages"][-1].content)


def main() -> None:
    graph = build_graph()
    questions = [
        "代理式 RAG 的流程是什么？",
        "LangGraph 的 State、Node、Edge 分别是什么？",
        "这个知识库里有没有教我怎么烤蛋糕？",
    ]
    for question in questions:
        run_question(graph, question)


if __name__ == "__main__":
    main()
