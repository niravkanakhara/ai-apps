from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph.message import add_messages, Annotated
from langgraph.types import interrupt, Command

load_dotenv()

from langchain_azure_ai.chat_models import AzureChatOpenAI

myendpoint = "https://nirav-mkv49s8o-westus3.cognitiveservices.azure.com/"
mydeployment_name = "gpt-4.1"
myapi_version = "2024-12-01-preview"

llm = AzureChatOpenAI(
    model_name=mydeployment_name,
    api_version=myapi_version,
    azure_endpoint=myendpoint
    # api_key=api_key,
)

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

from langchain_core.tools import tool

@tool
def get_stock_price(symbol: str) -> float:
    """Retrun the current price of a stock for the given stick symbol.
    :param symbol: The stock symbol to look up.
    :return: The current stock price.

    AAPL is Apple Inc.
    GOOGL is Alphabet Inc. (Google)
    MSFT is Microsoft Corporation.
    RIL is Reliance Industries Limited.
    AMZN is Amazon, Inc.
    """

    stock_prices = {
        "AAPL": 150.25,
        "GOOGL": 2750.50,
        "MSFT": 299.00,
        "RIL": 2200.75,
        "AMZN": 135.00
    }

    return stock_prices.get(symbol.upper(), 0.0)

@tool
def buy_stocks(symbol: str, quantity: int, total_price: float) -> str:
    """
    Buy stocks for the given symbol and quantity
    """
    decision = interrupt(f"Approve buying {quantity} of {symbol} stocks for ${total_price:.2f}")

    if decision == 'yes':
        return f"you have {quantity} shares of {symbol} for a total price of {total_price}."
    else:
        return "Buying declined"

tools = [get_stock_price, buy_stocks]

llm_with_tools = llm.bind_tools(tools)

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

def chatbot(state: State) -> State:
    return {"messages": [llm_with_tools.invoke(state["messages"])]} 

builder = StateGraph(State)

builder.add_node("chatbot_node", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_conditional_edges("chatbot_node", tools_condition)
builder.add_edge("tools", "chatbot_node")

builder.set_entry_point("chatbot_node")

memory_saver = MemorySaver()

graph = builder.compile(checkpointer=memory_saver)

config = {"configurable" : {"thread_id" : "1"}}

state = graph.invoke({"messages": "What is the price of 10 Amazon stock?" }, config = config)

print(state["messages"][-1].content)

state = graph.invoke({"messages": "Buy 10  Amazon stocks at current price" }, config = config)

print(state.get("__interrupt__"))

decision = input("Approve (yes/no):")

state = graph.invoke(Command(resume=decision), config=config )

print(state["messages"][-1].content)

