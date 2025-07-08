#!/usr/bin/env python3
from typing import TypedDict

import httpx
import asyncio

import logging
from typing_extensions import TypedDict, Annotated

from langchain.agents.chat.prompt import HUMAN_MESSAGE
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, force=True)

async def build_client(llm_provider:str = "ollama"):
    """
    Building LangGraph client that uses MCP Servers
    :param llm_provider: ollama (for development); gemini, gpt-4o, grok, etc. (for production)
    """

    # Selecting LLM providers
    try:
        if llm_provider == "ollama":
            llm = ChatOllama(model="qwen3:8b")
        elif llm_provider == "gemini":
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        elif llm_provider == "openai":
            llm = ChatOpenAI(model="gpt-4o",
                temperature=0,
                max_retries=3,
                timeout=60)
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")
    except httpx.ConnectError as e:
        print(f"Ollama not running in background!. {str(e)}")
    except Exception as e:
        print(f"LLM initialization failed! {str(e)}")

    # Initializing MCP Client with MCP servers connections (e.g. mcp_server.py)
    client = MultiServerMCPClient(
        {
            "mcp": {
                "command": "python",
                "args": ["mcp_server.py"],
                "transport": "stdio",
            }
        }
    )

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    builder = StateGraph(State)

    tools = await client.get_tools()
    for tool in tools:
        logging.info(f"{tool}\n")

    # Creating LangGraph Nodes

    async def assistant(state: State):
        llm_response = await llm.bind_tools(tools).ainvoke(state["messages"])
        return {"messages": llm_response}

    builder.add_node(assistant)
    builder.add_node(ToolNode(tools))

    # Adding LangGraph Edges

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    builder.add_edge("assistant", END)

    # Compiling the graph
    return builder.compile()

async def main():
    builder_graph = await build_client(llm_provider="ollama")
    # question = "what are the tool names from the mcp servers?"
    # message = [HumanMessage(content=question)]
    # graph_response = await builder_graph.ainvoke({"messages": message})
    # print(graph_response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
