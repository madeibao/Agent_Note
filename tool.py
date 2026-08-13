

from langchain.tools import tool

# 最简单的工具：一个普通函数 + @tool 装饰器
@tool
def hello_tool(name: str) -> str:
    """向指定的人打招呼。

    Args:
        name: 要打招呼的人的名字
    """
    return f"你好，{name}！欢迎来到菜鸟教程 RUNOOB。"

# 工具也是普通的 Python 函数，可以直接调用
result = hello_tool.invoke({"name": "小明"})
print(result)

# 工具包含自动生成的描述信息
print(f"\n工具名称: {hello_tool.name}")
print(f"工具描述: {hello_tool.description}")