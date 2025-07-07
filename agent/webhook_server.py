#!/usr/bin/env python3
"""Simple Webhook Server for GitHub Actions events.
Stores events in a JSON file for the MCP server.
"""
import json
from datetime import datetime
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, Header, HTTPException, Response
from fastapi.responses import JSONResponse
import hmac
import hashlib
import os
# Initializing FastAPI application to receive POST request from GitHub webhook
app = FastAPI()
# Path to store git events
EVENTS_FILE = Path(__file__).parent / "events_git.json"
@app.get("/")
async def root():
    return JSONResponse({"message": "GitHub Actions Webhook Server is running."})
# GitHub Webhook Payload Authentication
async def is_valid_signature(raw_payload: bytes, signature: str=Header(None)) -> bool:
    """
    Validate the GitHub HMAC SHA256 signature against the provided signature.
    Args:
        raw_payload (bytes): Raw payload from GitHub webhook request.
        signature (str): The GitHub HMAC SHA256 signature from incoming payload.
    """
    mac = hmac.new(bytes(os.getenv("WEBHOOK_SECRET"), encoding="utf-8"), msg=raw_payload, digestmod=hashlib.sha256)
    expected_signature = "sha256=" + mac.hexdigest()
    return hmac.compare_digest(expected_signature, signature)
@app.post("/webhook/github/")
async def handle_webhook(request:Request) -> Response:
    """Handles incoming GitHub webhook."""
    raw_payload = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not await is_valid_signature(raw_payload, signature):
        return JSONResponse(content={"detail": "Invalid signature"}, status_code=401)
    try:
        data = json.loads(raw_payload)
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": request.headers.get("X-GitHub-Event", "unknown"),
            "action": data.get("action"),
            "workflow_run": data.get("workflow_run"),
            "check_run": data.get("check_run"),
            "repository": data.get("repository", {}).get("full_name"),
            "sender": data.get("sender", {}).get("login"),
        }
        # Load existing events
        events = []
        if EVENTS_FILE.exists():
            with open(EVENTS_FILE, "r") as f:
                events = json.load(f)
        # Add new event and keep last 100
        events.append(event)
        events = events[-100:]
        # Save appended events
        with open(EVENTS_FILE, "w") as f:
            json.dump(events, f, indent=2)
        return JSONResponse(content = {"status": "Payload Received"}, status_code=200)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
if __name__ == "__main__":
    print("🚀 Starting webhook server on http://localhost:8080")
    print("📝 Events will be saved to:", EVENTS_FILE)
    print("🔗 Webhook URL: http://localhost:8080/webhook/github")
    uvicorn.run(app, host="localhost", port=8080)