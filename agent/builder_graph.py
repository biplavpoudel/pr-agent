#!/usr/bin/env python3

import httpx
import asyncio

import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

with open("./prompts/system_prompt.txt", "r", encoding="utf-8") as f:
    ASSISTANT_SYSTEM_PROMPT_BASE = f.read()

class GraphProcessingState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    prompts: str = Field(default_factory=str, description="Prompts to be used by the agent")
    # tools_enabled: dict = Field(default_factory=dict, description="Tools enabled for the agent")


async def build_workflow(llm_provider="ollama") -> CompiledStateGraph:
    """
    Building LangGraph client that uses MCP Servers
    :param llm_provider: ollama (for development); gemini, gpt-4o, grok, etc. (for production)
    """

    # Selecting LLM providers
    try:
        if llm_provider == "ollama":
            llm = ChatOllama(model="qwen3:8b", temperature=0.2)
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
                "args": ["./agent/mcp_server.py"],
                "transport": "stdio",
            }
        }
    )

    assistant_tools = await client.get_tools()
    for tool in assistant_tools:
        logging.info(f"{tool}\n")

    # Creating LangGraph Nodes
    async def assistant_node(state: GraphProcessingState, config=None):
        assistant_model = llm.bind_tools(assistant_tools)
        if state.prompts:
            final_prompt = "\n".join([state.prompt, ASSISTANT_SYSTEM_PROMPT_BASE])
        else:
            final_prompt = ASSISTANT_SYSTEM_PROMPT_BASE
        # creating a chat prompt template with system messages along with user messages using placeholder
        prompts = ChatPromptTemplate.from_messages(
            [
                ("system", final_prompt), MessagesPlaceholder(variable_name="messages")
            ]
        )
        # Runnable sequence to pipe the output of prompt( i.e. list of messages) as input to the assistant_model
        sequence = prompts | assistant_model
        response = await sequence.ainvoke({"messages": state.messages}, config=config)
        return {"messages": response}

    def tools_condition_edge(state: GraphProcessingState):
        # similar to tools_condition from langgraph.prebuilt.tool_node, but added logger for debugging
        # routes to ToolNode if tools called in the last message, else ENDS
        last_message = state.messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info(f"Tools call detected: {last_message.tool_calls}")
            return "tools"
        return END

    # async def graph_workflow() -> CompiledStateGraph:
    # Initializing graph with state
    builder = StateGraph(GraphProcessingState)
    # Adding nodes to the graph
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(assistant_tools))
    # Adding LangGraph edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition_edge,
    )
    builder.add_edge("tools", "assistant")
    builder.add_edge("assistant", END)
    # Compiling the graph
    return builder.compile()

async def main():
    builder_graph = await build_workflow(llm_provider="ollama")
    question = "what are the tool names from the mcp servers?"
    # Create an initial state instance that conforms to GraphProcessingState
    initial_state = GraphProcessingState(
        messages=[HumanMessage(content=question)],
        prompts=""
    )
    graph_response = await builder_graph.ainvoke(initial_state)
    print(graph_response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
