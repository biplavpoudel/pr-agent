import re

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()


def build_graph(llm_provider: str = "gemma"):
    # Adding LLM/VLM Model
    if llm_provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                endpoint_url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
                temperature=0,
            ),
            verbose=True,
        )
    elif llm_provider == "openai":
        llm = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=3, timeout=60)
    elif llm_provider == "gemma":
        llm = ChatGoogleGenerativeAI(
            model="gemini/gemini-2.0-flash-lite-001", temperature=0.1
        )
        # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif llm_provider == "ollama":
        try:
            # model = "qwen3:4b"
            model = "qwen3:8b"
            # httpx.ConnectError when Ollama not running in background
            llm = ChatOllama(
                model=model,
                temperature=0.6,
                repeat_penalty=1,
                top_k=20,
                top_p=0.95,
                stop=["<|im_start|>", "<|im_end|>"],
            )
        except httpx.ConnectError as e:
            print(f"Ollama not running in background: {e}")
        except Exception as e:
            print(f"Ollama not working: {e}")
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")

    # Let's add system prompt from system_prompt.txt
    with open("system_prompt.txt", encoding="utf-8") as f:
        sys_message = f.read()
    system_prompt = SystemMessage(content=sys_message)

    # Adding Embedding Model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Adding Vector Store
    vector_store = InMemoryVectorStore(embeddings)
    # client = QdrantClient(url="https://e75c4d7b-d8c8-4f7b-9e54-892bc1583cb9.europe-west3-0.gcp.cloud.qdrant.io:6333",
    #                       api_key=os.environ["QDRANT_API_KEY"])
    # client.create_collection(
    #     collection_name="langchain_assistant",
    #     vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    # )
    # vector_store = QdrantVectorStore(
    #     client=client,
    #     collection_name="langchain_assistant",
    #     embedding=embeddings,
    # )

    # Initialize Logger

    graph_builder = StateGraph(MessagesState)
    retriever = vector_store.as_retriever()

    # Creating Node
    def assistant(state: MessagesState):
        response_message = llm.invoke(state["messages"] + [system_prompt])
        print(f"\nDEBUG: LLM Raw Response Type: {type(response_message)}")
        print(f"DEBUG: LLM Raw Response Content: {response_message.content}")
        if isinstance(response_message, AIMessage) and response_message.tool_calls:
            print(f"DEBUG: LLM identified tool calls: {response_message.tool_calls}")
        else:
            print("DEBUG: LLM did NOT identify tool calls.")
        # return {"messages": [response_message]}

        return {
            "messages": [response_message],
        }

    def retrieve_documents(state: MessagesState):
        latest_user_msg = [
            msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)
        ][-1]
        retrieved_docs = retriever.invoke(latest_user_msg)
        return {
            "messages": [
                SystemMessage(
                    content="\n\n".join([doc.content for doc in retrieved_docs])
                )
            ]
        }

    # Add Nodes to Graph
    graph_builder.add_node("assistant", assistant)
    # graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_node("document_retriever", retrieve_documents)

    # Add Edges to Graph
    graph_builder.add_edge(START, "document_retriever")
    graph_builder.add_edge("document_retriever", "assistant")
    graph_builder.add_conditional_edges(
        "assistant",
        # If the latest message requires a tool, route to tools
        # Otherwise, provide a direct response
        tools_condition,
    )
    # graph_builder.add_edge("tools", "assistant")

    # Build Graph
    return graph_builder.compile()


if __name__ == "__main__":
    # question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    # question = "Search Wikipedia for 'Python programming language'"
    # question = "What is 200 power 8.99999?"
    # question = ".rewsna eht sa 'tfel' drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI"
    api_url = "https://agents-course-unit4-scoring.hf.space"
    # url = api_url + "/files/" + "cca530fc-4052-43b2-b130-b30968d8aa44"
    # question = f"Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation. Attached filename is: 'cca530fc-4052-43b2-b130-b30968d8aa44.png' and url is: {url} "
    # question = "Given this table defining * on the set S = {a, b, c, d, e}\n\n|*|a|b|c|d|e|\n|---|---|---|---|---|---|\n|a|a|b|c|b|d|\n|b|b|c|a|e|c|\n|c|c|a|b|b|a|\n|d|b|e|b|e|d|\n|e|d|b|a|d|c|\n\nprovide the subset of S involved in any possible counter-examples that prove * is not commutative. Provide your answer as a comma separated list of the elements in the set in alphabetical order."
    # question = "Examine the video at https://www.youtube.com/watch?v=1htKBjuUWec.\n\nWhat does Teal\'c say in response to the question : Isn\'t that hot?"
    question = "Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
    graph = build_graph(llm_provider="openai")

    messages = [HumanMessage(content=question)]
    response = graph.invoke({"messages": messages})
    answer = response["messages"][-1].content
    pattern = r"<think>(.*?)</think>"
    print(f"DEBUG: Agent is returning response from the graph: {answer}\n")
    answer = re.sub(pattern=pattern, repl="", string=answer, flags=re.DOTALL)
    # Strip everything before 'FINAL ANSWER:'
    cleaned_answer = re.sub(r".*?FINAL ANSWER:\s*", "", answer, flags=re.DOTALL)
    print(f"DEBUG: Cleaned answer is: {cleaned_answer}\n\n\n\n\n")
    print(cleaned_answer)
    # for m in response["messages"]:
    #     pprint(m)
