# PR-Agent

This repository is inspired from the repo from HuggingFace: [mcp-course](https://github.com/huggingface/mcp-course).

The main aim of this repo is to implement the repository to use LangGraph framework and user desired LLMs instead of Claude Code, which the original repo enforces.

### Core Components
1. MCP Server
   - Contains sets of tools for PR template suggestions 
2. Github Actions Integration
   - Realtime CI/CD monitoring using webhooks 
3. Slack Webhook Notification
    - Automated team notifications for updates and changes