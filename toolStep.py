
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。

    Args:
        city: 城市名称
    """
    weather_data = {
        "杭州": "晴，25°C",
        "北京": "多云，18°C",
    }
    return weather_data.get(city, f"未找到 {city} 的天气数据")


@tool
def get_time(city: str) -> str:
    """查询指定城市的当前时间。

    Args:
        city: 城市名称
    """
    time_data = {
        "杭州": "14:30",
        "北京": "14:30",
        "纽约": "02:30",
    }
    return time_data.get(city, f"未找到 {city} 的时间数据")


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[get_weather, get_time],
    system_prompt="你是一个乐于助人的助手。",
)

# 使用 stream_mode="updates" 可以看到每一个步骤
print("=== Agent 执行过程追踪 ===\n")
step = 0
for chunk in agent.stream(
    {"messages": [HumanMessage(content="杭州现在天气怎么样？几点了？")]},
    stream_mode="updates",
):
    step += 1
    print(f"--- 步骤 {step} ---")
    for node_name, update in chunk.items():
        print(f"节点: {node_name}")
        if "messages" in update:
            for msg in update["messages"]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # AI 消息包含工具调用
                    for tc in msg.tool_calls:
                        print(f"  → 请求调用工具: {tc['name']}({tc['args']})")
                elif msg.type == "tool":
                    print(f"  → 工具结果 [{msg.name}]: {msg.content}")
                elif msg.type == "ai" and msg.content:
                    print(f"  → AI 回复: {msg.content[:100]}")