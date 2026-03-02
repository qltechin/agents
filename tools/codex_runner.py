"""Codex CLI runner — wraps the Codex CLI as a code agent.

Extracted from builder_agent.py so it's a clean, swappable module
alongside claude_agent.py and deepseek_agent.py.
"""

import asyncio
import json
import os
import subprocess


async def run_codex_agent(
    work_dir: str,
    prompt: str,
    settings,
    logger=None,
) -> dict:
    """
    Run Codex CLI to resolve a coding issue.

    Requires:
      - codex CLI installed (npm install -g @openai/codex)
      - CODEX_AUTH_JSON env var (or ~/.codex/auth.json)

    Returns:
        dict with keys: fix_applied (bool), files_changed (list), turns (int)
    """
    from agents.codex_auth import CodexAuthContext

    def log(msg: str):
        if logger:
            logger(f"[Codex] {msg}")

    # Verify codex is installed
    cli_version = subprocess.run(["codex", "--version"], capture_output=True, text=True)
    if cli_version.returncode == 0:
        log(f"Codex CLI version: {cli_version.stdout.strip()}")
    else:
        log(f"Codex CLI not found: {cli_version.stderr}", )
        raise RuntimeError("Codex CLI is not installed. Run: npm install -g @openai/codex")

    build_id = os.environ.get("CODEBUILD_BUILD_ID", "local")
    changes_made: list[str] = []
    codex_timeout = 900  # 15 minutes

    with CodexAuthContext(build_id=build_id):
        log("Codex auth setup complete")

        codex_cmd = [
            "codex",
            "exec",
            "--yolo",
            "--json",
            "-C", work_dir,
            prompt,
        ]

        log(f"Running: codex exec --yolo --json -C {work_dir} [prompt]")

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                codex_cmd,
                capture_output=True,
                text=True,
                timeout=codex_timeout,
                cwd=work_dir,
            )

            log(f"Exit code: {result.returncode}")

            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        event_type = event.get("type", "")

                        if event_type == "item.completed":
                            item = event.get("item", {})
                            if item.get("type") == "file_change":
                                for change in item.get("changes", []):
                                    file_path = change.get("path")
                                    kind = change.get("kind", "")
                                    if file_path and kind in ("add", "update"):
                                        changes_made.append(file_path)
                                log(f"[FILE_CHANGE] {len(item.get('changes', []))} file(s)")
                            elif item.get("type") == "agent_message":
                                log(f"[MESSAGE] {item.get('text', '')[:200]}")

                    except json.JSONDecodeError:
                        if getattr(settings, "verbose", False):
                            log(f"[OUTPUT] {line[:200]}")

            if result.returncode != 0 and result.stderr:
                log(f"stderr: {result.stderr[:500]}")
                if "auth" in result.stderr.lower() or "login" in result.stderr.lower():
                    return {
                        "fix_applied": False,
                        "error": "Codex authentication failed — refresh token may have expired",
                        "auth_error": True,
                    }

        except subprocess.TimeoutExpired:
            log(f"Timed out after {codex_timeout}s")
            return {"fix_applied": False, "error": f"Codex timed out after {codex_timeout}s"}

    fix_applied = len(changes_made) > 0
    log(f"Done. Files changed: {changes_made}")

    return {
        "fix_applied": fix_applied,
        "files_changed": changes_made,
        "turns": 1,
    }
