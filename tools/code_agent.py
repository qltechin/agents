"""Unified code agent router.

Selects and runs the appropriate code generation agent based on
CODE_AGENT_PROVIDER env var or settings.code_agent_provider.

Provider priority (when set to 'auto'):
  1. claude   — if ANTHROPIC_API_KEY is set
  2. deepseek — if DEEPSEEK_API_KEY is set
  3. codex    — fallback (requires CODEX_AUTH_JSON)

Usage:
    from tools.code_agent import run_code_agent

    result = await run_code_agent(
        work_dir="/tmp/repo",
        prompt="Fix the nav bar issue...",
        settings=settings,
        logger=self.log,
    )
"""

import os


async def run_code_agent(
    work_dir: str,
    prompt: str,
    settings,
    logger=None,
    max_turns: int = 60,
) -> dict:
    """
    Run the configured code agent to resolve a coding issue.

    Provider is resolved in this order:
      1. CODE_AGENT_PROVIDER env var
      2. settings.code_agent_provider
      3. Auto-detect from available API keys

    Returns:
        dict with keys: fix_applied (bool), files_changed (list), turns (int)
    """

    def log(msg: str):
        if logger:
            logger(msg)

    provider = (
        os.environ.get("CODE_AGENT_PROVIDER")
        or getattr(settings, "code_agent_provider", "auto")
    )

    # Auto-detect: pick the first available key
    if provider == "auto":
        if _get_anthropic_key(settings):
            provider = "claude"
        elif _get_deepseek_key(settings):
            provider = "deepseek"
        else:
            provider = "codex"

    log(f"Code agent provider: {provider}")

    if provider == "claude":
        api_key = _get_anthropic_key(settings)
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Cannot use claude provider.")
        model = (
            os.environ.get("CLAUDE_CODE_MODEL")
            or getattr(settings, "claude_code_model", "claude-sonnet-4-6")
        )
        from tools.claude_agent import run_claude_agent
        return await run_claude_agent(
            work_dir=work_dir,
            prompt=prompt,
            api_key=api_key,
            logger=logger,
            model=model,
            max_turns=max_turns,
        )

    elif provider == "deepseek":
        api_key = _get_deepseek_key(settings)
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set. Cannot use deepseek provider.")
        model = (
            os.environ.get("DEEPSEEK_MODEL")
            or getattr(settings, "deepseek_model", "deepseek-chat")
        )
        from tools.deepseek_agent import run_deepseek_agent
        return await run_deepseek_agent(
            work_dir=work_dir,
            prompt=prompt,
            api_key=api_key,
            logger=logger,
            model=model,
            max_turns=max_turns,
        )

    elif provider == "codex":
        from tools.codex_runner import run_codex_agent
        return await run_codex_agent(
            work_dir=work_dir,
            prompt=prompt,
            settings=settings,
            logger=logger,
        )

    else:
        raise ValueError(
            f"Unknown code agent provider: '{provider}'. "
            "Valid options: claude, deepseek, codex, auto"
        )


def _get_anthropic_key(settings) -> str | None:
    return (
        os.environ.get("ANTHROPIC_API_KEY")
        or getattr(settings, "anthropic_api_key", None)
    )


def _get_deepseek_key(settings) -> str | None:
    return (
        os.environ.get("DEEPSEEK_API_KEY")
        or getattr(settings, "deepseek_api_key", None)
    )
