



from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv  # 读取 .env 文件中的配置
import os

load_dotenv()  # 从项目根目录的 .env 加载密钥等环境变量

# 创建嵌入模型实例，通过 OpenAI 兼容协议对接百炼
# chunk_size=10：百炼 Embedding 接口单次请求最多接受 10 条文本，
# OpenAIEmbeddings 默认一次打包 1000 条，知识库稍大就会超限报错。
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",  # 嵌入模型名称（必填），此处以 v4 为例
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量读取百炼 API Key（必填）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 百炼 OpenAI 兼容端点（必填）
    check_embedding_ctx_length=False,
    chunk_size=10,
)

# 对单个查询文本进行向量化
text = "This is a test document."
query_result = embeddings.embed_query(text)
print("文本向量长度：", len(query_result), sep='')

# 批量对多个文档进行向量化
doc_results = embeddings.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ])
print("文本向量数量：", len(doc_results), "，文本向量长度：", len(doc_results[0]), sep='')