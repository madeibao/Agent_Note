

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_agent, after_agent,
    before_model, after_model,
)

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool

@before_agent
def start_log(state, runtime):
    """Agent 开始前"""
    print(">>> [before_agent] Agent 开始 <<<")
    runtime.stream_writer({"type": "lifecycle", "phase": "start"})
    return None

@before_model
def pre_model(state, runtime):
    """每次模型调用前"""
    msg_count = len(state.get("messages", []))
    print(f"  -> [before_model] 第 {msg_count} 条消息")
    return None

@after_model
def post_model(state, runtime):
    """每次模型调用后"""
    last = state["messages"][-1] if state.get("messages") else None
    if hasattr(last, 'tool_calls') and last.tool_calls:
        tools = [tc['name'] for tc in last.tool_calls]
        print(f"  <- [after_model] 请求工具: {tools}")
    else:
        content = str(last.content)[:50] if last and hasattr(last, 'content') else ""
        print(f"  <- [after_model] 直接回复: {content}...")
    return None

@after_agent
def end_log(state, runtime):
    """Agent 结束后"""
    total_msgs = len(state.get("messages", []))
    print(f"<<< [after_agent] Agent 结束，共 {total_msgs} 条消息 <<<")
    return None


@tool
def get_weather(city: str) -> str:
    """查询天气"""
    return f"{city}: 晴，25°C"


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[start_log, pre_model, post_model, end_log],
    system_prompt="你是助手。",
)

print("\n========== 第一个问题（需要工具） ==========")
result = agent.invoke({
    "messages": [HumanMessage(content="杭州天气？")]
})

print(f"\n最终回复: {result['messages'][-1].content}")

print("\n========== 第二个问题（无需工具） ==========")
result = agent.invoke({
    "messages": [HumanMessage(content="你好")]
})
print(f"\n最终回复: {result['messages'][-1].content}")