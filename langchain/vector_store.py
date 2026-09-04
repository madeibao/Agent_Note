
import hashlib
import math
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


ROOT = Path(__file__).parent
DOC_PATH = ROOT / "sample" / "langchain_notes.md"

class HashEmbeddings(Embeddings):
    """Tiny local embeddings for learning vector stores without calling an API.

    This is not a production embedding model. It only maps tokens into a fixed-size
    vector so we can demonstrate indexing, similarity search, and retrievers.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = self._tokenize(text)
        for token in tokens:
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
        chunk_size=180,
        chunk_overlap=30,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    return chunk_splitter.split_documents(section_docs)


def demo_embeddings(embeddings: Embeddings) -> None:
    print("\n=== 1. Embeddings: 文本 -> 向量 ===")
    query = "什么是 RAG 检索增强生成？"
    vector = embeddings.embed_query(query)

    print("query:", query)
    print("vector type:", type(vector).__name__)
    print("vector dimensions:", len(vector))
    print("first 8 values:", [round(value, 3) for value in vector[:8]])


def demo_vector_store(chunks: list[Document], embeddings: Embeddings) -> InMemoryVectorStore:
    print("\n=== 2. Vector Store: chunks + vectors -> searchable index ===")
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    print("indexed chunks:", len(chunks))
    print("first chunk metadata:", chunks[0].metadata)
    print("first chunk preview:", chunks[0].page_content[:120].replace("\n", " "))
    return vector_store


def demo_similarity_search(vector_store: InMemoryVectorStore) -> None:
    print("\n=== 3. Similarity Search: 查询 -> 相似文档 ===")
    query = "RAG 的流程是什么？"
    docs = vector_store.similarity_search(query, k=3)

    print("query:", query)
    for index, doc in enumerate(docs, start=1):
        print(f"\nresult {index}")
        print("metadata:", doc.metadata)
        print("preview:", doc.page_content[:140].replace("\n", " "))


def demo_similarity_search_with_score(vector_store: InMemoryVectorStore) -> None:
    print("\n=== 4. Similarity Search With Score: 查看相似度分数 ===")
    query = "PromptTemplate 有什么用？"
    results = vector_store.similarity_search_with_score(query, k=3)

    print("query:", query)
    for index, (doc, score) in enumerate(results, start=1):
        print(f"\nresult {index}, score={score:.4f}")
        print("metadata:", doc.metadata)
        print("preview:", doc.page_content[:140].replace("\n", " "))


def demo_retriever(vector_store: InMemoryVectorStore) -> None:
    print("\n=== 5. Retriever: 向量库 -> 统一检索接口 ===")
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("为什么要切分文本？")

    print("retriever output type:", type(docs).__name__)
    print("retrieved documents:", len(docs))
    for index, doc in enumerate(docs, start=1):
        print(f"\nretrieved {index}")
        print("metadata:", doc.metadata)
        print("preview:", doc.page_content[:140].replace("\n", " "))


def main() -> None:
    chunks = load_and_split_documents()
    embeddings = HashEmbeddings(dimensions=64)

    demo_embeddings(embeddings)
    vector_store = demo_vector_store(chunks, embeddings)
    demo_similarity_search(vector_store)
    demo_similarity_search_with_score(vector_store)
    demo_retriever(vector_store)


if __name__ == "__main__":
    main()
