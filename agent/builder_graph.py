#!/usr/bin/env python3
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama

import asyncio

model = ChatOllama(model = "ollama/qwen3:4b")

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

async def main():

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
    response = await graph.ainvoke({"messages": "what's (3 + 5) x 12?"})
    print(response

if __name__ == "__main__":
    asyncio.run(main())
