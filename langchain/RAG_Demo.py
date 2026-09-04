


import hashlib
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


ROOT = Path(__file__).parent
DOC_PATH = ROOT / "sample" / "langchain_notes.md"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class HashEmbeddings(Embeddings):
    """Tiny local embeddings for learning RAG without calling an embedding API."""

    def __init__(self, dimensions: int = 64) -> None:
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


def build_model() -> ChatOpenAI:
    load_dotenv(ROOT.parent / ".env", encoding="utf-8-sig")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY. Please fill it in D:\\code\\langchain\\.env")

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
        chunk_size=220,
        chunk_overlap=40,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    return chunk_splitter.split_documents(section_docs)


def build_retriever():
    chunks = load_and_split_documents()
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=HashEmbeddings(dimensions=64),
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})


def format_docs(docs: list[Document]) -> str:
    formatted = []
    for index, doc in enumerate(docs, start=1):
        section = doc.metadata.get("section", "无章节")
        source = doc.metadata.get("title", doc.metadata.get("source", "未知来源"))
        formatted.append(
            f"[资料 {index}]\n来源: {source}\n章节: {section}\n内容: {doc.page_content}"
        )
    return "\n\n".join(formatted)


def main() -> None:
    model = build_model()
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一个严谨的 LangChain 学习助手。只能根据给定资料回答。"
            "如果资料里没有答案，就自己查找相关资料回答我。回答要用中文。",
        ),
        (
            "human",
            "问题：{question}\n\n资料：\n{context}",
        ),
    ])

    chain = prompt | model | StrOutputParser()

    questions = [
        "RAG 的典型流程是什么？",
        "为什么文本切分需要 chunk_overlap？",
        "LangChain 能不能帮我订机票？",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print("用户问题：", question)

        docs = retriever.invoke(question)
        print("\n检索到的资料：")
        for index, doc in enumerate(docs, start=1):
            print(f"{index}. section={doc.metadata.get('section', '无章节')}, preview={doc.page_content[:80].replace(chr(10), ' ')}")

        answer = chain.invoke({
            "question": question,
            "context": format_docs(docs),
        })
        print("\n模型回答：")
        print(answer)


if __name__ == "__main__":
    main()
