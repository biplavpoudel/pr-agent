# 🤖 PR-Agent

This repository is inspired by the excellent [mcp-course](https://github.com/huggingface/mcp-course) project from Hugging Face.

The primary goal of this repository is to reimplement the system using the **LangGraph** framework and allow the use of **custom LLMs** (e.g., OpenAI, Mistral, Gemini, etc.), rather than being restricted to **Claude Code** as in the original implementation.

---

### 🧩 Core Components

1. **MCP Server**
   - Implements a flexible, modular toolchain for automated PR template generation and analysis.
   - Supports multiple LLM backends as interchangeable agents.

2. **GitHub Actions Integration**
   - Enables real-time CI/CD monitoring and PR analysis through GitHub Webhooks.
   - Automates feedback loops via GitHub comments and checks.

3. **Slack Webhook Notifications**
   - Sends automatic team notifications on PR events, analysis results, or errors.
   - Useful for tracking updates in collaborative workflows.

---