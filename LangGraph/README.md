# LangGraph Demo 01

This is the first minimal LangGraph example:

```text
START -> chatbot -> END
```

It demonstrates four core concepts:

- `State`: shared data carried through the graph
- `Node`: a Python function that reads and updates state
- `Edge`: the route from one node to another
- `Graph`: the compiled executable workflow

## Run

```powershell
conda activate agent-dev
cd D:\code\agent\LangGraph
python .\hello_graph.py
```

The script reads `DASHSCOPE_API_KEY` from this folder's `.env` first. If it does not exist, it falls back to:

```text
D:\code\langchain\.env
```
