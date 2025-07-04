#!/usr/bin/env python3
import asyncio
import json
import os
import subprocess
from datetime import datetime
import textwrap
import aiofiles

import requests
from typing import Optional
from typing import Dict, Any
from pathlib import Path


from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import logging

logging.basicConfig(level=logging.ERROR, force=True)

# Initializing the FastMCP server
mcp = FastMCP("pr_agent")

load_dotenv(verbose=True)

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
EVENTS_FILE = Path(__file__).parent.parent / "events_git.json"

# Type mapping for PR templates
TYPE_MAPPING = {
    "Bug fix": ["bug", "fix"],
    "Feature": ["feature", "enhancement"],
    "Documentation": ["docs", "documentation"],
    "Refactor": ["refactor", "cleanup"],
    "Test": ["test", "testing"],
    "Performance": ["performance", "optimization"],
    "Security": ["security"],
}


@mcp.tool()
async def analyze_file_changes(
    base_branch: str = "main",
    include_diff: bool = True,
    max_diff_lines: int = 500,
    working_directory: Optional[str] = None,
) -> str:
    """Gets the full diff and list of changed files in the current git repository.

    Args:
        base_branch: Base branch to compare against (default: main)
        include_diff: Include the full diff content (default: true)
        max_diff_lines: Maximum number of diff lines to include (default: 500)
        working_directory: Directory to run git commands in (default: current directory)
    """
    try:
        # Trying to get working directory from roots
        if working_directory is None:
            try:
                context = mcp.get_context()
                roots_result = await context.session.list_roots()
                root = roots_result.roots[0]
                working_directory = root.uri.path
            except (Exception, ValueError, RuntimeError) as e:
                logging.error(
                    f"Context unavailable outside of request. Context fallback triggered! {str(e)}"
                )
                pass

        # Using provided working directory else current directory
        cwd = working_directory if working_directory else os.getcwd()

        # List of changed files
        diff_files = subprocess.run(
            ["git", "diff", "--name-status", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        )

        # Diff statistics
        stat_result = subprocess.run(
            ["git", "diff", "--stat", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
        )

        # Complete diff if requested
        diff_content = ""
        truncated = False
        if include_diff:
            diff_result = subprocess.run(
                ["git", "diff", f"{base_branch}...HEAD"],
                capture_output=True,
                text=True,
                cwd=cwd,
            )
            diff_lines = diff_result.stdout.split("\n")

            # MCP tools have a token limit of 25k. Git diffs for larger projects can exceed this limit.
            # So using truncation to limit the `git diff` content.
            if len(diff_lines) > max_diff_lines:
                diff_content = "\n".join(diff_lines[:max_diff_lines])
                diff_content += (
                    f"\n\nShowing {max_diff_lines} of {len(diff_lines)} lines ..."
                )
                diff_content += (
                    "\nIncrease the value of max_diff_lines parameter for more ..."
                )
                truncated = True
            else:
                diff_content = diff_result.stdout

        # Commit messages for context
        commits_result = subprocess.run(
            ["git", "log", "--oneline", f"{base_branch}..HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
        )

        git_result = {
            "base_branch": base_branch,
            "files_changed": diff_files.stdout,
            "statistics": stat_result.stdout,
            "commits": commits_result.stdout,
            "diff": diff_content if include_diff else "Diff not included",
            "truncated": truncated,
            "total_diff_lines": len(diff_lines) if include_diff else 0,
        }

        return json.dumps(git_result, indent=2)

    except subprocess.CalledProcessError as e:
        return json.dumps({"error": f"Git error: {e.stderr}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_pr_templates() -> str:
    """List all available PR templates and their contents."""

    if not any(file.endswith(".md") for file in os.listdir(TEMPLATES_DIR)):
        await create_default_templates(TEMPLATES_DIR)

    default_templates = {
        file.split(".")[0].capitalize().replace("_", " "): file
        for file in os.listdir(TEMPLATES_DIR)
        if file.endswith(".md")
    }
    templates = [
        {
            "filename": file,
            "type": template_type,
            "content": (TEMPLATES_DIR / file).read_text(),
        }
        for template_type, file in default_templates.items()
    ]

    return json.dumps(templates, indent=2)

@mcp.tool()
async def create_default_templates():
    """Create default PR templates and their contents if empty."""
    templates = {
        "bug_fix" : "## 🐞 Bug Fix\n\n### 📄 Description\n\n### 🔍 Root Cause\n\n### 🛠️ Solution\n\n### ✅ Testing Done\n\n### 🔗 Related Issues",
        "feature" : "## ✨ New Feature\n\n### 📄 Description\n\n### 🚀 Motivation\n\n### 🛠️ Implementation Details\n\n### ✅ Testing Checklist\n\n### 📚 Documentation### ⚠️ Breaking Changes",
        "documentation" : "## 📝 Documentation Update\n\n### 📄 Description\n\n### ✏️ Changes Made\n\n### ✅ Review Checklist",
        "refactor" : "## ♻️ Code Refactoring\n\n### 📄 Description\n\n### 🎯 Motivation\n\n### 🔧 Changes Made\n\n### ✅ Testing Checklist\n\n### ✅ Testing Checklist",
        "test" : "## 🧪 Test Update\n\n### 📄 Description\n\n### 📈 Coverage Impact\n\n### 🧷 Test Types Added/Updated\n\n### 🧩 Related Features/Components",
        "performance" : "## 🚀 Performance Improvement\n\n### 📄 Description\n\n### 📊 Metrics (Before/After)\n\n### 🔧 Changes Made\n\n### ✅ Testing Checklist",
        "security" : "## 🔐 Security Update\n\n### 📄 Description\n\n### 🎯 Impact\n\n### 🛠️ Solution\n\n### ✅ Testing Checklist\n\n### 🔗 References",
    }

    for template_type in templates.values():
        file_path = TEMPLATES_DIR / f"{template_type}.md"
        file_path.write_text(templates[template_type], encoding="utf-8")

@mcp.tool()
async def create_default_specific_template(template_path):
    """Create a specified default PR template and its content if empty."""
    templates = {
        "bug_fix" : "## 🐞 Bug Fix\n\n### 📄 Description\n\n### 🔍 Root Cause\n\n### 🛠️ Solution\n\n### ✅ Testing Done\n\n### 🔗 Related Issues",
        "feature" : "## ✨ New Feature\n\n### 📄 Description\n\n### 🚀 Motivation\n\n### 🛠️ Implementation Details\n\n### ✅ Testing Checklist\n\n### 📚 Documentation### ⚠️ Breaking Changes",
        "documentation" : "## 📝 Documentation Update\n\n### 📄 Description\n\n### ✏️ Changes Made\n\n### ✅ Review Checklist",
        "refactor" : "## ♻️ Code Refactoring\n\n### 📄 Description\n\n### 🎯 Motivation\n\n### 🔧 Changes Made\n\n### ✅ Testing Checklist\n\n### ✅ Testing Checklist",
        "test" : "## 🧪 Test Update\n\n### 📄 Description\n\n### 📈 Coverage Impact\n\n### 🧷 Test Types Added/Updated\n\n### 🧩 Related Features/Components",
        "performance" : "## 🚀 Performance Improvement\n\n### 📄 Description\n\n### 📊 Metrics (Before/After)\n\n### 🔧 Changes Made\n\n### ✅ Testing Checklist",
        "security" : "## 🔐 Security Update\n\n### 📄 Description\n\n### 🎯 Impact\n\n### 🛠️ Solution\n\n### ✅ Testing Checklist\n\n### 🔗 References",
    }

    if not os.path.exists(template_path):
        default_templates = {
            file.split(".")[0].capitalize().replace("_", " "): file
            for file in os.listdir(TEMPLATES_DIR)
            if file.endswith(".md")
        }

        file_name = next(filename for filename in default_templates.values() if filename == template_path)
        content = templates[file_name.split(".")[0]]
        template_path.write_text(content)

@mcp.tool()
async def suggest_template(changes_summary: str, change_type: str) -> str:
    """Let LLM analyze the changes and suggest the most appropriate PR template.

    Args:
        changes_summary: Your analysis of what the changes do
        change_type: Type of change you've identified (bug, feature, docs, refactor, test, security, performance)
    """

    # Get available templates
    templates_response = await get_pr_templates()
    templates = json.loads(templates_response)

    # Find matching template using generator function
    matching_template = next(
        iter(
            templates
            for templates, aliases in TYPE_MAPPING.items()
            if change_type.lower() in aliases
        ),
        "Feature",
    )
    template_file = DEFAULT_TEMPLATES.get(matching_template, "feature.md")
    selected_template = next(
        (t for t in templates if t["filename"] == template_file),
        templates[0],  # Defaults to first template (i.e. Bug Fix)
    )

    suggestion = {
        "recommended_template": selected_template,
        "reasoning": f"Based on your analysis: '{changes_summary}', this appears to be a {change_type} change.",
        "template": selected_template["content"],
        "usage_hint": "LLM can fill out this template based on the specific changes in your pull request.",
    }

    return json.dumps(suggestion, indent=2)


@mcp.tool()
async def get_recent_actions_events(limit: int = 10) -> str:
    """Get recent GitHub Actions events received via webhook.

    Args:
        limit: Maximum number of events to return (default: 10)
    """
    # Read events from file
    if not EVENTS_FILE.exists():
        return json.dumps([])

    with open(EVENTS_FILE, "r") as f:
        events = json.load(f)

    # Return most recent events
    recent = events[-limit:]
    return json.dumps(recent, indent=2)


@mcp.tool()
async def get_workflow_status(workflow_name: Optional[str] = None) -> str:
    """Get the current status of GitHub Actions workflows.

    Args:
        workflow_name: Optional specific workflow name to filter by
    """

    if not EVENTS_FILE.exists():
        return json.dumps({"message": "No GitHub Actions events received yet"})

    with open(EVENTS_FILE, "r") as f:
        events = json.load(f)

    if not events:
        return json.dumps({"message": "GitHub Actions events empty!"})

    # Filtering out workflow events
    workflow_events = [e for e in events if e.get("workflow_run") is not None]

    # if specific name given, filter it out
    if workflow_name:
        workflow_events = [
            e for e in workflow_events if e["workflow_run"].get("name") == workflow_name
        ]

    # Group by workflow and get latest status
    workflows = {}
    for event in workflow_events:
        run = event["workflow_run"]
        run_name = run["name"]
        # comparing dates between event with same names to get the last updated one
        if run_name not in workflows or datetime.fromisoformat(
            run["updated_at"]
        ) > datetime.fromisoformat(workflows[run_name]["updated_at"]):
            workflows[run_name] = {
                "name": run_name,
                "status": run["status"],
                "conclusion": run.get("conclusion"),
                "run_number": run["run_number"],
                "updated_at": run["updated_at"],
                "html_url": run["html_url"],
            }

    return json.dumps(list(workflows.values()), indent=2)


@mcp.tool()
async def send_slack_notification(payload: Dict[str, Any]) -> str:
    """Send a formatted notification to the team Slack channel using webhook.

    Args:
        payload: The markdown payload to be posted in the Slack channel.

    NOTE: For CI failures, we use format_ci_failure_alert prompt first!
    NOTE: For CI success, we use format_ci_success_summary prompt first!
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return "Error: SLACK_WEBHOOK_URL env variable not set!"

    try:
        # Sending POST request to Slack webhook
        response = requests.post(webhook_url, json=payload, timeout=10)

        # Check if successful request
        if response.status_code == 200:
            return "✅ Slack Message sent successfully"
        else:
            return f"❌ Failed to send message. Status: {response.status_code}, Response: {response.text}"

    except requests.exceptions.Timeout:
        return "⏰ Request timed out. Check your internet connection and try again."
    except requests.exceptions.ConnectionError:
        return "🔌 Connection error. Check your internet connection and webhook URL."
    except Exception as e:
        return f"❌ Error sending message: {str(e)}"


@mcp.prompt(
    description="creates a slack payload for GitHub Actions failure alert using the json template."
)
async def ci_failure_alert_template() -> str:
    """Create a Slack alert for CI/CD failures using the template."""

    async with aiofiles.open(PROMPTS_DIR / "ci_failure.json", "r") as f:
        failure_json = await json.load(f)

    prompt_header = textwrap.dedent(
        f"""Use this Slack Payload as template for failing GitHub Actions runs..."""
    )

    return f"{prompt_header}\n{failure_json}"


@mcp.prompt(
    description="creates a slack payload for successful GitHub Actions run using the json template."
)
async def ci_success_summary_template() -> str:
    """Create a Slack message for successful deployments using the template."""

    async with aiofiles.open(PROMPTS_DIR / "ci_success.json", "r") as f:
        success_json = await json.load(f)

    prompt_header = (
        """Use this Slack Payload as template for successful GitHub Actions runs..."""
    )
    return f"{prompt_header}\n{success_json}"


@mcp.prompt(description="analyzes the latest CI/CD result from the GitHub Actions.")
async def analyze_ci_results():
    """Analyze recent CI/CD results and provide insights."""

    async with aiofiles.open(PROMPTS_DIR / "analyze_result.md", "r") as f:
        analysis_template = await f.read()

    prompt_header = textwrap.dedent(
        """
    Please analyze the recent CI/CD results from GitHub Actions:
    
    1. First, call get_recent_actions_events() to fetch the latest CI/CD events
    2. Then call get_workflow_status() to check current workflow states
    3. Identify any failures or issues that need attention
    4. Provide actionable next steps based on the results
    
    Format your response as:"""
    )

    return f"{prompt_header}\n\n{analysis_template}"


@mcp.prompt(description="generates deployment summary for the team communication.")
async def create_deployment_summary():
    """Generate a deployment summary for team communication."""

    async with aiofiles.open(PROMPTS_DIR / "generate_summary.md", "r") as f:
        generate_template = await f.read()

    prompt_header = textwrap.dedent(
        """
    Create a deployment summary for team communication:

    1. Check workflow status with get_workflow_status()
    2. Look specifically for deployment-related workflows
    3. Note the deployment outcome, timing, and any issues
    
    Keep it brief but informative for team awareness.
    
    Format as a concise message suitable for Slack:"""
    )

    return f"{prompt_header}\n{generate_template}"


@mcp.prompt(description="generates a detailed PR status report.")
async def generate_pr_status_report():
    """Generate a comprehensive PR status report including CI/CD results."""

    async with aiofiles.open(PROMPTS_DIR / "generate_report.md", "r") as f:
        report_template = await f.read()

    prompt_header = textwrap.dedent(
        """
    Generate a comprehensive PR status report:

    1. Use analyze_file_changes() to understand what changed
    2. Use get_workflow_status() to check CI/CD status
    3. Use suggest_template() to recommend the appropriate PR template
    4. Combine all information into a cohesive report
    
    Create a detailed report with: """
    )

    return f"{prompt_header}\n{report_template}"


@mcp.prompt(description="troubleshoots the failing GitHub Actions workflows.")
async def troubleshoot_workflow_failure():
    """Help troubleshoot a failing GitHub Actions workflow."""
    async with aiofiles.open(PROMPTS_DIR / "troubleshoot.md", "r") as f:
        troubleshoot_template = await f.read()

    prompt_header = textwrap.dedent(
        """
    Help troubleshoot failing GitHub Actions workflows:

    1. Use get_recent_actions_events() to find recent failures
    2. Use get_workflow_status() to see which workflows are failing
    3. Analyze the failure patterns and timing
    4. Provide systematic troubleshooting steps
    
    Structure your response as:"""
    )

    return f"{prompt_header}\n{troubleshoot_template}"


if __name__ == "__main__":
    # Run MCP server and run webhook server separately
    # print("Starting PR Agent Slack MCP server...")
    # mcp.run()

    DEFAULT_TEMPLATES = {
        file.split(".")[0].capitalize().replace("_", " ") : file
        for file in os.listdir(TEMPLATES_DIR) if file.endswith(".md")
    }
    for name, template in DEFAULT_TEMPLATES.items():
        print(name, template)
    #
    # TYPE_MAPPING = {
    #     "Bug fix": ["bug", "fix"],
    #     "Feature": ["feature", "enhancement"],
    #     "Documentation": ["docs", "documentation"],
    #     "Refactor": ["refactor", "cleanup"],
    #     "Test": ["test", "testing"],
    #     "Performance": ["performance", "optimization"],
    #     "Security": ["security"]
    # }
    # change_type = "docs"
    # matched_key = next(iter(templates for templates, aliases in TYPE_MAPPING.items() if change_type.lower() in aliases), None)
    # print(matched_key)
    # print(f"Matched file is: {DEFAULT_TEMPLATES[matched_key]}")
    # print(f"Matched file is: {DEFAULT_TEMPLATES.get(matched_key, "gello.md")}")

    # result  = asyncio.run(ci_failure_alert_template())
    # print(result)
