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
### 🚂 Initialize Cloudflared Tunnel
1. Download and install [cloudflared](http://github.com/cloudflare/cloudflared/releases/latest/) package.
2. Login to Cloudflare: `cloudflared tunnel login`
3. Create a tunnel. e.g. ` cloudflared tunnel create webhook-tunnel` 
4. A credential json file is created at `~/.cloudflared/[tunnel-id].json`
5. Create CNAME DNS record on Cloudflare for webhook-tunnel :
   ```
   cloudflared tunnel route dns webhook-tunnel webhook.example.com
   ``` 
6. Create a config.yml file in `~/.cloudflared` with custom subdomain. For example:
   ```
   tunnel: [tunnel_id]
   credentials-file: /home/biplav/.cloudflared/[tunnel_id].json
   
   ingress:
     - hostname: webhook.example.com
       service: http://localhost:8080
     - service: http_status:404
   ```

---
### 🧑‍💻 Using the agent
1. Head to agent directory: `cd agent`
2. Start the MCP server: `python mcp_server.py`
3. Start the Webhook server: `python agent/webhook_server.py`
4. In next terminal, expose webhook_server using Cloudflare Tunnel:
   `cloudflared tunnel run webhook-tunnel`
5. To test the local webhook server, you can use:
   `curl http://localhost:8080/webhook`
6. To test the webhook server using tunnel, you can use:
   `curl https://webhook.example.com/webhook`
