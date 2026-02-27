# AI Agents

Autonomous AI agent system for managing GitHub repositories.

## Agents

- **Builder Agent** — picks up issues labeled `aibuild`, writes code, creates PRs
- **Issue Analyzer** — LLM-based complexity analysis for task decomposition
- **Task Orchestrator** — cyclical execution for complex multi-step issues

## Workflow

1. Add `aibuild` label to any issue in a monitored repo
2. Builder Agent triggers automatically, clones the repo, fixes the issue
3. PR is created and issue label changes to `aicomplete`

## Setup

```bash
pip install -e .
cp .env.example .env
# fill in .env with your API keys
```

## Configuration

All configuration is via environment variables or `.env` file. See `.env.example`.

Key variables:
- `GH_OWNER` — GitHub username or org
- `GH_ACCOUNT_TYPE` — `user` or `organization`
- `ANTHROPIC_API_KEY` — for LLM and Claude Code Action
- `MONITORED_REPOS` — comma-separated list of repos to monitor
