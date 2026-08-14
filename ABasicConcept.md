
# LLM 
LLM (Large Language Model) 大语言模型，


# Vibe Coding

     Vibe Coding（氛围编程）是一种全新的软件开发方式。
     
     Vibe Coding 是一种编程方式，它将 LLM 引入到编程中，从而实现更智能的生成。

     Vibe Coding 的核心思想是：让 LLM 在回答问题时，先从外部知识库中检索相关内容，再基于检索结果生成回答，而不是仅依赖模型训练时记住的知识。

     Vibe Coding 的核心是知识库，它将 LLM 与知识库进行结合，从而实现更智能的生成。


# Token (词元) 
    
     AI 能理解的最小文本单位

# Prompt 提示

     提示，就是给 LLM 输入的文本，用于指导 LLM 的行为。

# Agent 

    Agent = LLM (大脑) + Planning (规划) + Tool use (执行) + Memory (记忆)。

* LLM (大脑)： 作为核心推理机，负责理解意图、生成文本和进行逻辑判断。

* Planning (规划)： 能够将复杂的目标（如"帮我策划一场技术沙龙"）拆解成可执行的步骤。

* Memory (记忆)： 记录对话历史（短期）和存储专业知识库（长期）。

* Tool Use (工具使用)： 能够根据需求去查谷歌搜索、读数据库、甚至跑 Python 代码。


## 普通的 LLM 只是 One-shot（一次性） 的响应，而 Agent 的核心在于 Iterative（迭代）。

    ReAct 模式 (Reason + Act) 是目前最主流的 Agent 推理逻辑：

* Thought (思考)： 模型描述当前要做什么，为什么要这么做。

* Action (行动)： 模型选择一个工具（如：Google Search）。

* Observation (观察)： 模型读取工具返回的结果。

* Repeat (循环)： 重复上述步骤，直到得出最终答案。


# RAG (Retrieval Augmented Generation) 检索增强生成


RAG (Retrieval Augmented Generation) 是一种 LLM 架构，它将 LLM 与知识库进行结合，从而实现更智能的生成。

RAG 的核心思想是：让 LLM 在回答问题时，先从外部知识库中检索相关内容，再基于检索结果生成回答，而不是仅依赖模型训练时记住的知识。

这解决了 LLM 的两个核心痛点：知识截止日期（模型不知道训练后发生的事）和幻觉问题（模型在不确定时会编造答案）。

RAG 的核心是知识库，它将 LLM 与知识库进行结合，从而实现更智能的生成。



# MCP  

* 连接任意外部系统：实时数据库、API、本地文件，既可以读也可以写，也可以对外提供 RAG 检索能力；MCP 不等于 RAG，MCP 可以承载 RAG 能力
* MCP = Model Context Protocol，模型上下文协议**，Anthropic 2024‑11 开源的开放标准协议，被比喻为**AI 领域的 USB‑C 接口**CSDN博...。

 
- 一句话本质：**一套标准化通信规范，解决大模型如何安全、统一对接外部数据源、工具、业务系统**，不再每个系统写一套定制对接代码。
- 底层基于 JSON‑RPC 2.0，不是大模型，不是库，是通信协议标准，类似 HTTP 对于 Web 的地位。