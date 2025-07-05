from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt import create_react_agent

from langchain.chat_models import init_chat_model
model = init_chat_model("ollama/qwen3:4b")

client = MultiServerMCPClient(
    {
        "mcp": {
            "command": "python",
            "args": ["./agent/mcp_server.py"],
            "transport": "stdio",
        },
        "webhook": {
            "url": "http://localhost:8080",
            "transport": "streamable_http",
        }
    }
)

tools = await client.get_tools()

def call_model(state: MessagesState):
    response = model.bind_tools(tools).invoke(state["messages"])
    return {"messages": response}

builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_node(ToolNode(tools))
builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model",
    tools_condition,
)
builder.add_edge("tools", "call_model")
graph = builder.compile()
math_response = await graph.ainvoke({"messages": "what's (3 + 5) x 12?"})
weather_response = await graph.ainvoke({"messages": "what is the weather in nyc?"})
