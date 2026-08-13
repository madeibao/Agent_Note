
import os

from langchain.tools import tool
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# 加载 .env
load_dotenv()

# 获取 API KEY
api_key = os.getenv("DEEPSEEK_API_KEY")

@tool
def get_stock_price(symbol: str) -> str:
    """查询股票当前价格。

    Args:
        symbol: 股票代码，如 AAPL、GOOGL、TSLA
    """
    # 模拟股票数据
    prices = {"AAPL": 185.50, "GOOGL": 142.30, "TSLA": 245.80}
    price = prices.get(symbol.upper())
    if price is None:
        return f"未找到股票代码 {symbol}"
    return f"{symbol.upper()} 当前价格：${price}"


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """货币换算。

    Args:
        amount: 金额
        from_currency: 原货币代码，如 USD、CNY
        to_currency: 目标货币代码，如 CNY、USD
    """
    # 模拟汇率
    rates = {
        ("USD", "CNY"): 7.25,
        ("CNY", "USD"): 0.138,
    }
    rate = rates.get((from_currency.upper(), to_currency.upper()))
    if rate is None:
        return f"不支持 {from_currency} → {to_currency} 的换算"
    result = round(amount * rate, 2)
    return f"{amount} {from_currency} = {result} {to_currency}"


model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)
agent = create_agent(
    model=model,
    tools=[get_stock_price, convert_currency],
    system_prompt="你是一个金融助手，帮助用户查询股价和换算货币。",
)

# Agent 会自动决定调用哪个工具
result = agent.invoke({
    "messages": [HumanMessage(content="苹果股价是多少？换算成人民币是多少？")]
})

for msg in result["messages"]:
    if msg.type == "tool":
        print(f"[调用工具 {msg.name}] {msg.content}")
    elif msg.type == "ai" and msg.content:
        print(f"\n最终回答: {msg.content}")