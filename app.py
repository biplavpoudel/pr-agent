#!/usr/bin/env python3
import asyncio
import json
import os
import logging
from pathlib import Path
from typing import AsyncGenerator, List, Tuple
from uuid import uuid4
import threading

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import RunnableConfig

load_dotenv()

# Import your existing components
from agent.builder_graph import AssistantAgent, GraphProcessingState

# Configuration
USER_INPUT_MAX_LENGTH = 8000
TRIM_MESSAGE_LENGTH = 12
RECURSION_LIMIT = 15

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for modern, intuitive design
css_path = Path("./static/ui.css")
CSS = css_path.read_text() if os.path.exists(css_path) else " "

# Load system prompt
SYSTEM_PROMPT_PATH = Path("./prompts/system_prompt.txt")
if SYSTEM_PROMPT_PATH.exists():
    DEFAULT_PROMPT = SYSTEM_PROMPT_PATH.read_text()
else:
    DEFAULT_PROMPT = "You are a helpful GitHub PR assistant."

# Initialize agent globally
agent = None
graph = None
mcp_tools = {}

# Global stop flag with thread-safe access
stop_generation = threading.Event()


async def initialize_agent():
    """Initialize the agent and graph once"""
    global agent, graph, mcp_tools
    if agent is None:
        agent = AssistantAgent(llm_provider="gemini")
        graph = await agent.build_workflow()
        tools = await agent.init_mcp_client().get_tools()
        mcp_tools = {tool.name: tool.description.split(".")[0] for tool in tools}
    return agent, graph, mcp_tools


class ChatState:
    """Simple state management for chat"""

    def __init__(self):
        self.messages = []
        self.graph_state = {"messages": [], "prompts": "", "project_dir": None}
        self.session_id = str(uuid4())

    def add_message(self, message):
        self.messages.append(message)
        self.graph_state["messages"].append(message)
        # Keep only recent messages to save tokens
        if len(self.graph_state["messages"]) > TRIM_MESSAGE_LENGTH:
            self.graph_state["messages"] = self.graph_state["messages"][-TRIM_MESSAGE_LENGTH:]

    def set_prompt(self, prompt):
        self.graph_state["prompts"] = prompt

    def set_project_dir(self, project_dir: str):
        """Set the project directory in the graph state"""
        if project_dir:
            self.graph_state["project_dir"] = project_dir
        else:
            self.graph_state["project_dir"] = os.getcwd()

    def clear(self):
        self.messages = []
        self.graph_state = {"messages": [], "prompts": "", "project_dir": None}
        self.session_id = str(uuid4())


# Global chat state
chat_state = ChatState()


async def chat_response(message: str, history: List, system_prompt: str) -> AsyncGenerator[str, None]:
    """Main chat function with streaming response"""
    try:
        # Initialize agent if needed
        await initialize_agent()

        # Configure the graph run
        config = RunnableConfig(
            recursion_limit=RECURSION_LIMIT,
            run_name="pr-agent-chat",
            configurable={
                "thread_id": chat_state.session_id,
                "project_dir": chat_state.graph_state.get("project_dir")}
        )
        logger.info(f"Running tools with project_dir: {chat_state.graph_state['project_dir']}")

        # Update system prompt if changed
        if system_prompt.strip():
            system_prompt = f"{DEFAULT_PROMPT}\nCurrent project directory: {chat_state.graph_state['project_dir']}"
            chat_state.set_prompt(system_prompt)

        logger.info(f"System Prompt is: {chat_state.graph_state['prompts']}")

        # Add user message
        user_msg = HumanMessage(content=message[:USER_INPUT_MAX_LENGTH])
        chat_state.add_message(user_msg)

        # Stream the response
        full_response = ""
        tool_calls_made = []

        async for stream_mode, chunk in graph.astream(
                chat_state.graph_state,
                config=config,
                stream_mode=["values", "messages"]
        ):
            # Check stop flag
            if stop_generation.is_set():
                yield full_response + "\n\n⏹️ Generation stopped by user."
                break

            if stream_mode == "values":
                # Check for tool calls
                if chunk.get("messages"):
                    last_message = chunk["messages"][-1]
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            tool_name = tool_call.get('name', 'unknown')
                            if tool_name not in tool_calls_made:
                                tool_calls_made.append(tool_name)
                                full_response += f"🔧 Running tool: {tool_name}...\n\n"
                                yield full_response

                # Update chat state with final values
                chat_state.graph_state = chunk

            elif stream_mode == "messages":
                msg, metadata = chunk
                # Only stream assistant messages
                if (metadata.get('langgraph_node') == "assistant" and
                        hasattr(msg, 'content') and msg.content):
                    full_response += msg.content
                    yield full_response

        # Add assistant response to state
        if full_response and not stop_generation.is_set():
            assistant_msg = AIMessage(content=full_response)
            chat_state.add_message(assistant_msg)

    except Exception as e:
        logger.error(f"Error in chat_response: {e}")
        yield "❌ An error occurred. Please try again or check the logs."


def clear_chat():
    """Clear chat history"""
    stop_generation.set()  # Stop any ongoing generation
    chat_state.clear()
    return []


def get_project_directories():
    """Get list of project directories - you can customize this"""
    dirs_path = Path("./project_dirs.json")
    if not dirs_path.exists():
        return ["Current Directory"]
    else:
        with open(dirs_path, 'r') as f:
            directories = json.load(f)
        return ["Current Directory"] + directories.get("directories", [])

def add_project_directory(new_path: str):
    """Add a new project directory to the JSON file"""
    dirs_path = Path("./project_dirs.json")
    if not dirs_path.exists():
        # Create the file if it doesn't exist
        with open(dirs_path, 'w') as f:
            json.dump({"directories": []}, f)

    with open(dirs_path, 'r') as f:
        directories = json.load(f)

    # Add the new path if it's not already in the list
    if new_path not in directories.get("directories", []):
        directories["directories"].append(new_path)
        with open(dirs_path, 'w') as f:
            json.dump(directories, f, indent=2)



def create_quick_actions():
    """Create quick action buttons for common tasks"""
    return [
        ("🧲 Create Pull Request", "Create a pull request with the current changes using the suggested template"),
        ("📝 Suggest PR Template", "Analyze my current changes and suggest the most appropriate PR template"),
        ("🔍 Analyze Changes", "Use analyze_file_changes to show me what files have been modified"),
        ("📝 Deployment Summary", "Create a summary of recent deployments and CI/CD status"),
        ("📊 Generate Report", "Create a comprehensive PR status report with CI/CD status"),
        ("🚨 Check CI Status", "Get the current GitHub Actions workflow status"),
        ("📢 Send Slack Update", "Check recent CI events and send a team notification to Slack"),
        ("🔧 Troubleshoot Failures", "Help me troubleshoot any failing GitHub Actions workflows"),
    ]


def handle_quick_action(action_text: str) -> str:
    """Handle quick action button clicks"""
    action_map = {
        "🧲 Create Pull Request": "Create a pull request with the current changes using the suggested template",
        "📝 Suggest PR Template": "Analyze my current changes and suggest the most appropriate PR template",
        "🔍 Analyze Changes": "Use analyze_file_changes to show me what files have been modified in my current branch",
        "📝 Deployment Summary": "Create a summary of recent deployments and CI/CD status",
        "📊 Generate Report": "Generate a comprehensive PR status report including CI/CD results and file changes",
        "🚨 Check CI Status": "Check the current status of all GitHub Actions workflows",
        "📢 Send Slack Update": "Check recent CI events and send a summary notification to our team Slack channel",
        "🔧 Troubleshoot Failures": "Help me troubleshoot any failing GitHub Actions workflows with specific recommendations",
    }
    return action_map.get(action_text, action_text)

async def process_chat_response(message: str, history: List[dict], system_prompt: str, project_dir: str):
    """Wrapper to handle async chat response with proper event loop"""
    if project_dir == "Current Directory":
        project_dir = os.getcwd()

    if not message.strip():
        yield history

    stop_generation.clear()
    chat_state.graph_state["project_dir"] = Path(os.path.expanduser(project_dir)) if project_dir else os.getcwd()

    # Directly append user input
    history.append({"role": "user", "content": message})

    response = ""
    try:
        async for chunk in chat_response(message, history, system_prompt):
            # Update assistant response in history
            history.append({"role": "assistant", "content": chunk})
            yield history
    except Exception as e:
        logger.exception(f"❌ Error in process_chat_response.")
        history.append({"role": "assistant", "content": "❌ An error occurred. Please try again."})
        yield history


def stop_chat():
    """Stop the current generation"""
    stop_generation.set()
    return "⏹️ Stopping generation..."

async def create_interface():
    """Create the main Gradio interface"""
    # Ensure agent and tools are initialized
    await initialize_agent()

    with gr.Blocks(title="GitHub PR Agent", css=CSS, theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🤖 GitHub PR Agent")
        gr.Markdown("Your AI assistant for GitHub Pull Requests, CI/CD monitoring, and team notifications.")

        with gr.Row():
            with gr.Column(scale=3):
                # Quick Actions
                gr.Markdown("### Quick Actions")
                quick_actions = create_quick_actions()

                action_buttons = []
                with gr.Row():
                    for i, (label, _) in enumerate(quick_actions[:4]):
                        btn = gr.Button(label, elem_classes=["quick-action-btn"], scale=1)
                        action_buttons.append(btn)

                with gr.Row():
                    for i, (label, _) in enumerate(quick_actions[-4:]):
                        btn = gr.Button(label, elem_classes=["quick-action-btn"], scale=1)
                        action_buttons.append(btn)

                # Chat Interface
                with gr.Column(elem_classes=["chat-container"]):
                    chatbot = gr.Chatbot(
                        height=500,
                        show_copy_button=True,
                        avatar_images=("./static/human.png", "./static/robot.png"),
                        type="messages",
                        # layout="panel", # gives modern look
                    )

                    with gr.Row():
                        with gr.Column(scale=8):
                            msg_input = gr.Textbox(
                                placeholder="Ask me about PR templates, CI status, or team notifications...",
                                show_label=False,
                                lines=3,
                            )
                        with gr.Column(scale=2):
                            with gr.Row(equal_height=True):
                                submit_btn = gr.Button("Send", variant="primary", min_width=2)
                                stop_btn = gr.Button("Stop", variant="stop", min_width=2)
                            with gr.Row():
                                clear_btn = gr.Button("Clear", variant="huggingface", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### Project Directory")
                project_dir_dropdown = gr.Dropdown(
                    choices=get_project_directories(),
                    value="Current Directory",
                    label="Select Project Directory",
                    interactive=True,
                    container=True
                )

                gr.Markdown("### Add Custom Project Directory")
                custom_path_input = gr.Textbox(
                    placeholder="Enter custom project directory path...",
                    label="Custom Path",
                    interactive=True
                )
                add_path_btn = gr.Button("Add Path")

                def update_dropdown(new_path):
                    add_project_directory(new_path)
                    return get_project_directories()

                add_path_btn.click(
                    fn=update_dropdown,
                    inputs=[custom_path_input],
                    outputs=[project_dir_dropdown]
                )

                gr.Markdown("### System Prompt")
                system_prompt = gr.Textbox(
                    value=DEFAULT_PROMPT,
                    placeholder="System prompt for the agent...",
                    lines=8,
                    show_label=False,
                    elem_classes=["system-prompt"]
                )

                # gr.Markdown("### 🔧 Available Tools")
                # # Dynamically generate the markdown string
                # tools_md = "\n".join(
                #     f"- `{name}`: {desc}" for name, desc in mcp_tools.items()
                # )
                # gr.Markdown(f"""{tools_md}
                # """)

        # Event handlers
        submit_btn.click(
            fn=process_chat_response,
            inputs=[msg_input, chatbot, system_prompt, project_dir_dropdown],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        )

        msg_input.submit(
            fn=process_chat_response,
            inputs=[msg_input, chatbot, system_prompt, project_dir_dropdown],
            outputs=[chatbot],
            queue=True
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        )

        stop_btn.click(
            fn=stop_chat,
            outputs=[msg_input]
        )

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot]
        )

        # Quick action handlers
        for i, (label, action) in enumerate(quick_actions):
            action_buttons[i].click(
                fn=lambda action=action: action,
                outputs=[msg_input]
            )

        # Examples
        gr.Examples(
            examples=[
                ["What PR template should I use for my bug fix?"],
                ["Check the status of our CI/CD pipelines"],
                ["Send a summary of recent deployments to Slack"],
                ["Help me troubleshoot the failing test workflow"],
                ["Generate a comprehensive PR report"],
                ["What files have changed in my current branch?"],
                ["Create a pull request with the suggested template"],
            ],
            inputs=[msg_input]
        )

        # # Footer with instructions
        # gr.Markdown("""
        # ---
        # **💡 Tips:**
        # - Select your project directory from the dropdown above
        # - Use quick actions for common tasks
        # - The agent can analyze your git repository and suggest appropriate PR templates
        # - System prompt can be customized for specific team workflows
        # - Use the Stop button to halt generation if needed
        # - All tools work with your selected git repository and GitHub webhooks
        # """)


        # Markdown to showcase available tools
        gr.Markdown("""
        ---
        ### 🔧 Available Tools""")
        # Dynamically generate the markdown string
        tools_md = "\n".join(
            f"- `{name}`: {desc}" for name, desc in mcp_tools.items()
        )
        gr.Markdown(f"""{tools_md}
        """)


    return app


async def main():
    """Main function to run the app"""

    # Initialize agent on startup
    # No need to initialize agent manually, it's lazy-loaded
    # await initialize_agent()

    # Create and launch interface
    app = await create_interface()

    # Launch with custom settings
    app.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    asyncio.run(main())