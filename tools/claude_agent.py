"""Claude agentic code editor.

Uses the Anthropic API with tool use to autonomously read, write,
and run commands to resolve GitHub issues.

Usage:
    from tools.claude_agent import run_claude_agent

    result = await run_claude_agent(
        work_dir="/tmp/repo",
        prompt="Fix the nav bar issue...",
        api_key="sk-ant-...",
        logger=self.log,
    )
"""

import asyncio
import json
import subprocess
from pathlib import Path

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the full contents of a file in the repository.",
        "input_schema": {
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
    {
        "name": "list_files",
        "description": "List files and directories at a given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to repo root. Defaults to '.'.",
                }
            },
        },
    },
    {
        "name": "write_file",
        "description": "Write (create or overwrite) a file with the given content.",
        "input_schema": {
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
    {
        "name": "run_bash",
        "description": (
            "Run a bash command in the repository directory. "
            "Use for flutter analyze, dart format, grep, find, etc."
        ),
        "input_schema": {
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
            logger(f"[Claude:{name}] {msg}")

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


async def run_claude_agent(
    work_dir: str,
    prompt: str,
    api_key: str,
    logger=None,
    model: str = "claude-sonnet-4-6",
    max_turns: int = 25,
) -> dict:
    """
    Run an agentic loop using Claude to resolve a coding issue.

    Args:
        work_dir: Path to the cloned repository
        prompt: The full issue prompt (built by PromptBuilder)
        api_key: Anthropic API key
        logger: Optional logging function (takes a string)
        model: Claude model to use
        max_turns: Maximum agentic turns before stopping

    Returns:
        dict with keys: fix_applied (bool), files_changed (list), turns (int)
    """
    from anthropic import Anthropic

    def log(msg: str):
        if logger:
            logger(f"[Claude] {msg}")

    client = Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": prompt}]
    files_written: list[str] = []
    turn = 0

    log(f"Starting agentic loop (model={model}, max_turns={max_turns})")

    for turn in range(max_turns):
        log(f"Turn {turn + 1}/{max_turns}")

        response = await asyncio.to_thread(
            client.messages.create,
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Log text content
        for block in response.content:
            if hasattr(block, "text") and block.text:
                log(f"Assistant: {block.text[:200]}")

        # No tool calls → agent is done
        if response.stop_reason == "end_turn":
            log("Agent finished (end_turn)")
            break

        if response.stop_reason != "tool_use":
            log(f"Agent stopped with reason: {response.stop_reason}")
            break

        # Execute each tool use block
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            name = block.name
            args = block.input if isinstance(block.input, dict) else {}

            result = await asyncio.to_thread(_execute_tool, name, args, work_dir, logger)

            if name == "write_file" and "Error" not in result:
                files_written.append(args.get("path", "unknown"))

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    else:
        log(f"Reached max turns ({max_turns})")

    fix_applied = len(files_written) > 0
    log(f"Done. Files written: {files_written}")

    return {
        "fix_applied": fix_applied,
        "files_changed": files_written,
        "turns": turn + 1,
    }
