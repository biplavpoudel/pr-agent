#!/usr/bin/env python3

import httpx
import asyncio
from pathlib import Path
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

SYSTEM_PROMPT_PATH = Path("./prompts/system_prompt.txt")


class GraphProcessingState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    prompts: str = Field(default_factory=str, description="Prompts to be used by the agent")

class AssistantAgent:
    def __init__(self, llm_provider:str, system_prompt_path: Path = SYSTEM_PROMPT_PATH):
        """
        Attributes:
            llm_provider(str): ollama (for development); gemini, gpt-4o, grok, etc. (for production)
            system_prompt_path(Path): Path object representing the system prompt
        """
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt_path.read_text(encoding="utf-8")
        self.llm = self.init_model()
        self.client = self.init_mcp_client()
        self.tools = []

    def init_model(self):
        """Initializing the LLM model for agentic workflow
        """

        provider = self.llm_provider
        try:
            if provider == "ollama":
                return ChatOllama(model="qwen3:8b", temperature=0.3)
            elif provider == "gemini":
                return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
            elif provider == "openai":
                return ChatOpenAI(model="gpt-4o",
                                 temperature=0,
                                 max_retries=3,
                                 timeout=60)
            else:
                raise ValueError(f"Unknown LLM provider: {provider}")
        except httpx.ConnectError as e:
            print(f"Ollama not running in background!. {str(e)}")
        except Exception as e:
            print(f"LLM initialization failed! {str(e)}")

    @staticmethod
    def init_mcp_client() -> MultiServerMCPClient:
        """Initializing MCP Client with MCP servers connections (e.g. mcp_server.py)
        """
        client = MultiServerMCPClient(
            {
                "mcp": {
                    "command": "python",
                    "args": ["./agent/mcp_server.py"],
                    "transport": "stdio",
                }
            }
        )
        return client

    async def setup_tools(self):
        try:
            self.tools = await self.client.get_tools()
            if not self.tools:
                logger.warning("No tools loaded from MCP!")
            # for tool in self.tools:
            #     logger.info(f"Loaded Tool: {tool}")
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")
            self.tools = []


    async def assistant_node(self, state: GraphProcessingState, config=None):
        """ Creating Assistant LangGraph Nodes
        """
        if not self.llm:
            raise RuntimeError("Failed to initialize LLM. Please check the provider or service.")

        assistant_model = self.llm.bind_tools(self.tools)
        if state.prompts:
            final_prompt = "\n".join([state.prompts, self.system_prompt])
        else:
            final_prompt = self.system_prompt
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

    @staticmethod
    def tools_condition_edge(state: GraphProcessingState) -> str:
        # similar to tools_condition from langgraph.prebuilt.tool_node, but added logger for debugging
        # routes to ToolNode if tools called in the last message, else ENDS
        last_message = state.messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info(f"Tools call detected: {last_message.tool_calls}")
            return "tools"
        return END

    async def build_workflow(self) -> CompiledStateGraph:
        """ Building Language Workflow
        """
        await self.setup_tools()
        builder = StateGraph(GraphProcessingState)
        # Adding nodes to the graph
        builder.add_node("assistant", self.assistant_node)
        builder.add_node("tools", ToolNode(self.tools))
        # Adding LangGraph edges
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            self.tools_condition_edge,
        )
        builder.add_edge("tools", "assistant")
        # Compiling the graph
        return builder.compile()


    async def ask(self, question:str):
        builder_graph = await self.build_workflow()

        # Create an initial state instance that conforms to GraphProcessingState
        initial_state = GraphProcessingState(
            messages=[HumanMessage(content=question)],
            prompts=""
        )
        return await builder_graph.ainvoke(initial_state)


async def main():
    agent = AssistantAgent(llm_provider="ollama")
    tools = await agent.init_mcp_client().get_tools()
    # response = await agent.ask("create me an example of bugfix template.")
    # print("\nFinal Response:\n", response["messages"][-1].content)
    mcp_tools = {tool.name: tool.description.split(".")[0] for tool in tools}
    print("\nMCP Tools:\n", mcp_tools)

if __name__ == "__main__":
    asyncio.run(main())
