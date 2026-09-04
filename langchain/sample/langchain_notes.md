# LangChain 核心组件学习笔记

LangChain 是一个用于开发大语言模型应用的框架。它的核心思想不是让模型单独工作，而是把模型、提示词、工具、文档、检索器和输出解析器组合成可靠的应用流程。

## Messages

Messages 是聊天模型的输入和输出单位。常见的消息包括 SystemMessage、HumanMessage、AIMessage 和 ToolMessage。SystemMessage 用来设置规则，HumanMessage 表示用户输入，AIMessage 表示模型输出，ToolMessage 表示工具执行结果。

在多轮对话中，模型不是只看最新一句话，而是看一组消息。历史消息越多，模型能理解的上下文越完整，但也会消耗更多 token。因此真实应用中需要裁剪、过滤和合并历史消息。

## Prompt Templates

Prompt Templates 用来把提示词结构和变量分开。开发者可以先定义模板，再在运行时填入变量。这样做的好处是提示词可以复用，也更容易维护。

ChatPromptTemplate 适合聊天模型。它可以生成 system、human、assistant 等不同角色的消息。MessagesPlaceholder 可以把历史消息插入模板，是多轮对话和 Agent 的常用组件。

## Output Parsers

Output Parsers 用来把模型输出转换成程序更容易处理的格式。StrOutputParser 可以把 AIMessage 转成字符串。JsonOutputParser 可以把模型输出解析成字典。Pydantic 结构化输出可以让模型结果直接变成对象。

输出解析器的价值在于减少后续程序处理的不确定性。如果模型输出要进入数据库、API 或前端页面，就应该尽量使用结构化输出。

## RAG

RAG 是 Retrieval-Augmented Generation，也就是检索增强生成。它先从知识库中检索相关文档，再把这些文档和用户问题一起交给模型回答。

RAG 的典型流程是：加载文档，切分文本，生成向量，存入向量库，检索相关片段，构造提示词，调用模型，解析输出。

## Text Splitters

Text Splitters 用来把长文档切成较小的文本块。切块太大，检索不精确，也容易超过上下文窗口。切块太小，语义可能被切碎。

chunk_size 控制每个文本块的大致大小。chunk_overlap 控制相邻文本块之间的重叠内容。重叠可以避免关键上下文刚好被切断。
