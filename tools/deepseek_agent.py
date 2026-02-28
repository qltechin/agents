"""DeepSeek agentic code editor.

Replaces Codex CLI with a tool-calling loop using the DeepSeek API
(OpenAI-compatible). DeepSeek-V3 / DeepSeek-Coder handles reading,
writing, and running commands autonomously to resolve GitHub issues.

Usage:
    from tools.deepseek_agent import run_deepseek_agent

    result = await run_deepseek_agent(
        work_dir="/tmp/repo",
        prompt="Fix the nav bar issue...",
        api_key="sk-...",
        logger=self.log,
    )
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path


# Tool definitions for DeepSeek function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the repository root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories at a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to repo root. Defaults to '.'.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (create or overwrite) a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to repo root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a bash command in the repository directory. "
                "Use for flutter analyze, dart format, grep, find, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an expert software engineer working autonomously on a GitHub issue.

Your job:
1. Understand the issue thoroughly
2. Explore the codebase using read_file and list_files
3. Make precise, minimal code changes using write_file
4. Verify your changes with run_bash (e.g. linting, formatting)
5. Stop when the issue is resolved

Rules:
- Only modify files that are necessary to fix the issue
- Follow existing code style and conventions
- Do not add unnecessary comments or documentation
- Do not install packages or modify lock files unless explicitly required
- When done, stop calling tools — do not loop indefinitely
"""


def _execute_tool(name: str, args: dict, work_dir: str, logger=None) -> str:
    """Execute a tool call and return the result as a string."""

    def log(msg: str):
        if logger:
            logger(f"[DeepSeek:{name}] {msg}")

    try:
        if name == "read_file":
            path = Path(work_dir) / args["path"]
            if not path.exists():
                return f"Error: File not found: {args['path']}"
            content = path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 8000:
                content = content[:8000] + f"\n... (truncated, {len(content)} total chars)"
            log(f"Read {args['path']} ({len(content)} chars)")
            return content

        elif name == "list_files":
            target = Path(work_dir) / args.get("path", ".")
            if not target.exists():
                return f"Error: Path not found: {args.get('path', '.')}"
            entries = []
            for entry in sorted(target.iterdir()):
                prefix = "📁 " if entry.is_dir() else "📄 "
                entries.append(f"{prefix}{entry.name}")
            result = "\n".join(entries) if entries else "(empty directory)"
            log(f"Listed {args.get('path', '.')} — {len(entries)} entries")
            return result

        elif name == "write_file":
            path = Path(work_dir) / args["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"], encoding="utf-8")
            log(f"Wrote {args['path']} ({len(args['content'])} chars)")
            return f"OK: wrote {args['path']}"

        elif name == "run_bash":
            cmd = args["command"]
            log(f"$ {cmd}")
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = ""
            if result.stdout:
                output += result.stdout[:3000]
            if result.stderr:
                output += f"\nSTDERR: {result.stderr[:1000]}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            log(f"Exit {result.returncode}")
            return output or "(no output)"

        else:
            return f"Error: Unknown tool: {name}"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 120s"
    except Exception as e:
        return f"Error: {e}"


async def run_deepseek_agent(
    work_dir: str,
    prompt: str,
    api_key: str,
    logger=None,
    model: str = "deepseek-chat",
    max_turns: int = 25,
) -> dict:
    """
    Run an agentic loop using DeepSeek to resolve a coding issue.

    Args:
        work_dir: Path to the cloned repository
        prompt: The full issue prompt (built by PromptBuilder)
        api_key: DeepSeek API key
        logger: Optional logging function (takes a string)
        model: DeepSeek model to use (deepseek-chat or deepseek-coder)
        max_turns: Maximum agentic turns before stopping

    Returns:
        dict with keys: fix_applied (bool), files_changed (list), turns (int)
    """
    from openai import OpenAI

    def log(msg: str):
        if logger:
            logger(f"[DeepSeek] {msg}")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    files_written: list[str] = []
    turn = 0

    log(f"Starting agentic loop (model={model}, max_turns={max_turns})")

    for turn in range(max_turns):
        log(f"Turn {turn + 1}/{max_turns}")

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=4096,
            temperature=0.0,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Append assistant message
        messages.append({"role": "assistant", "content": message.content, "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (message.tool_calls or [])
        ]})

        if message.content:
            log(f"Assistant: {message.content[:200]}")

        # No tool calls → agent is done
        if not message.tool_calls or finish_reason == "stop":
            log("Agent finished (no more tool calls)")
            break

        # Execute each tool call
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result = await asyncio.to_thread(_execute_tool, name, args, work_dir, logger)

            if name == "write_file" and "Error" not in result:
                files_written.append(args.get("path", "unknown"))

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
    else:
        log(f"Reached max turns ({max_turns})")

    fix_applied = len(files_written) > 0
    log(f"Done. Files written: {files_written}")

    return {
        "fix_applied": fix_applied,
        "files_changed": files_written,
        "turns": turn + 1,
    }
