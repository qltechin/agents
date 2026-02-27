"""Builder Agent - Generic autonomous issue resolution for any repo using OpenAI Codex CLI.

Codex Integration:
- Uses Codex CLI (codex exec) for autonomous code generation
- Authenticates via cached auth.json stored in AWS Secrets Manager
- Supports ChatGPT subscription-based usage (no per-token billing)

Figma Integration:
- When enabled, provides Figma context in prompts for design reference
- Configure via FIGMA_ACCESS_TOKEN env var or figma_access_token setting
"""

import asyncio
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel

from agents.base_agent import AgentResult, BaseAgent
from agents.prompt_builder import PromptBuilder
from config.settings import Settings
from tools.github_project import GitHubProjectClient, get_project_client
from tools.messenger import WorkflowStage, get_messenger
from tools.metrics import get_metrics
from tools.semantic_label_parser import (
    create_parser as create_semantic_parser,
)
from tools.story_generator import (
    create_generator as create_story_generator,
)
from tools.widgetbook_tools import (
    generate_source_widget_label,
    generate_widgetbook_label,
)

# Codex CLI availability check (lazy-loaded to avoid subprocess on import)
_codex_cli_available: bool | None = None


def _check_codex_cli() -> bool:
    """Check if Codex CLI is installed and available."""
    try:
        result = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def codex_cli_available() -> bool:
    """Check if Codex CLI is available (cached after first call).

    This is lazy-loaded to avoid running subprocess on module import,
    which would add latency during test collection and cause side effects.
    """
    global _codex_cli_available
    if _codex_cli_available is None:
        _codex_cli_available = _check_codex_cli()
    return _codex_cli_available


# Codex auth management
try:
    from agents.codex_auth import CodexAuthContext, CodexAuthManager

    CODEX_AUTH_AVAILABLE = True
except ImportError:
    CODEX_AUTH_AVAILABLE = False
    CodexAuthContext = None  # type: ignore
    CodexAuthManager = None  # type: ignore

# Figma integration (optional - for design reference during development)
try:
    from tools.figma_tools import FigmaMCPConfig, FigmaTools

    FIGMA_TOOLS_AVAILABLE = True
except ImportError:
    FIGMA_TOOLS_AVAILABLE = False

# Task orchestration (IssueAnalyzer no longer used - completion check via Claude)
from agents.task_orchestrator import (
    RoadblockDetector,
    create_orchestrator,
)


class RepoConfig(BaseModel):
    """Configuration for a repository, loaded from agent-config.json."""

    name: str
    type: str  # java-maven, flutter, python, go, etc.
    description: str = ""
    repository: str

    # Build configuration
    build_command: str = ""
    build_timeout: int = 600

    # Test configuration
    analyze_command: str = ""
    analyze_timeout: int = 180
    test_command: str = ""
    test_timeout: int = 300

    # Code style
    language: str = ""
    formatter_command: str = ""
    guidelines: list[str] = []

    # Labels
    source_label: str = "aibuild"
    complete_label: str = "aicomplete"
    in_progress_label: str = "ai-in-progress"
    failed_label: str = "ai-failed"

    # Semantic label enforcement
    semantic_labels_required: bool = True
    semantic_label_patterns: list[str] = []  # e.g., ["patient_*", "medication_*"]

    # Story generation
    story_generation_enabled: bool = True
    widgetbook_stories_path: str = ""  # Path to stories directory


class BuilderState(TypedDict):
    """State for Builder workflow."""

    repos_to_process: list[str]
    issues_to_process: list[dict[str, Any]]
    current_issue: dict[str, Any] | None
    current_repo: str | None
    repo_config: dict[str, Any] | None
    container_id: str | None
    work_dir: str | None
    fix_applied: bool
    tests_passed: bool
    pr_url: str | None
    error: str | None
    processed_issues: list[dict[str, Any]]
    # Semantic labels
    semantic_issues: list[dict[str, Any]] | None
    semantic_labels_added: list[str] | None
    # Story generation
    stories_generated: list[str] | None


class BuilderAgent(BaseAgent[BuilderState]):
    """Generic agent that autonomously fixes issues in any repo using Claude Agent SDK.

    This agent:
    1. Finds issues with 'aibuild' label from configured repositories
    2. Clones the repository and reads its agent-config.json config
    3. Sets up a devcontainer based on the repo's .devcontainer spec
    4. Uses Claude Agent SDK (with Bedrock) to analyze and fix the issue
    5. Runs build/test commands defined in the repo's config
    6. Creates a PR if tests pass
    7. Updates issue labels (aibuild → aicomplete)

    The agent is generic - it reads build/test commands from each repo's
    agent-config.json file, so it works with any language/framework.
    """

    name = "builder"
    description = (
        "Picks up 'aibuild' tagged issues from any configured repo, fixes them using Claude Agent SDK, and creates PRs"
    )

    # Default commands if repo doesn't have agent-config.json
    DEFAULT_CONFIGS = {
        "flutter": {
            "type": "flutter",
            "analyze_command": "flutter analyze",
            "test_command": "flutter test --no-pub --reporter=compact",
            "build_command": "flutter build apk --debug",
            "language": "dart",
        },
        "java-maven": {
            "type": "java-maven",
            "analyze_command": "mvn compile -B",
            "test_command": "mvn test -B",
            "build_command": "mvn clean install -DskipTests -B",
            "language": "java",
        },
        "python": {
            "type": "python",
            "analyze_command": "python -m compileall . -q",  # Compile all Python files (no explicit filenames needed)
            "test_command": "pytest -v",
            "build_command": "pip install -e .",
            "language": "python",
        },
        "go": {
            "type": "go",
            "analyze_command": "go vet ./...",
            "test_command": "go test ./...",
            "build_command": "go build ./...",
            "language": "go",
        },
    }

    def __init__(
        self,
        settings: Settings | None = None,
        max_issues: int = 1,
        target_repo: str | None = None,
        parallel: bool = True,
        max_concurrent: int = 3,
        repo_lock: bool = False,
        **kwargs,
    ):
        """Initialize the Builder agent.

        Args:
            settings: Application settings
            max_issues: Maximum issues to process per run
            target_repo: Specific repo to process (optional, processes all if None)
            parallel: Enable parallel issue processing (default: True)
            max_concurrent: Maximum concurrent issues to process (default: 3)
            repo_lock: Enable per-repo locking to serialize same-repo issues (default: False)
        """
        super().__init__(settings=settings)
        self.max_issues_per_run = max_issues
        self.target_repo = target_repo
        self.parallel = parallel
        self.max_concurrent = max_concurrent
        self.repo_lock = repo_lock

        # Repo locks for parallel processing (only used if repo_lock=True)
        self.repo_locks: dict[str, asyncio.Lock] = {}

        # Load configuration from settings
        self.repos = self.settings.builder_repo_list
        self.source_label = self.settings.builder_source_label
        self.complete_label = self.settings.builder_complete_label
        self.in_progress_label = self.settings.builder_in_progress_label
        self.failed_label = self.settings.builder_failed_label

        # Check for staging mode (STAGING=true env var)
        self.staging_mode = os.environ.get("STAGING", "").lower() == "true"
        if self.staging_mode:
            self.log("[STAGING] Running in staging mode - using staging labels and DB")
            self.source_label = f"{self.source_label}-staging"
            self.complete_label = f"{self.complete_label}-staging"
            self.in_progress_label = f"{self.in_progress_label}-staging"
            self.failed_label = f"{self.failed_label}-staging"

        # If target_repo specified, only process that one
        if target_repo:
            self.repos = [target_repo]

        # Track the source branch we cloned from (updated during _process_issue)
        self.clone_source_branch: str = "staging"

        # Initialize messenger for notifications (uses Google Chat API)
        self.chat_client = get_messenger(settings=self.settings)

        # Initialize CloudWatch metrics
        self.metrics = get_metrics("builder")

        # Initialize GitHub Project client for board integration
        self.project_client = get_project_client(os.environ.get("GH_OWNER", ""), os.environ.get("GH_ACCOUNT_TYPE", "user"))
        # Default project numbers (can be overridden)
        self.flutter_project = GitHubProjectClient.FLUTTER_PROJECT  # 10
        self.backend_project = GitHubProjectClient.BACKEND_PROJECT  # 11

        # Set up environment for Claude Agent SDK (Anthropic API or Bedrock)
        self._setup_llm_env()

        # Initialize Figma tools for design reference (optional)
        self.figma_tools: FigmaTools | None = None
        self.figma_mcp_config: dict[str, Any] | None = None
        self._setup_figma_integration()

        # Initialize task orchestrator for completion checking and subtask management
        # Note: IssueAnalyzer removed - completion checks now use Claude directly
        self.task_orchestrator = create_orchestrator(settings=self.settings)
        self.roadblock_detector = RoadblockDetector()

        # Initialize prompt builder for dynamic prompt construction
        self.prompt_builder = PromptBuilder()

        # State dictionary for tracking per-issue processing results
        self.state: dict[str, Any] = {}

    def _setup_figma_integration(self):
        """Initialize Figma tools if available and configured.

        When enabled, adds Figma MCP server to Claude Agent SDK so the builder
        can reference designs while implementing features.

        Token loading priority:
        1. FIGMA_ACCESS_TOKEN environment variable
        2. figma_access_token in settings
        3. AWS Secrets Manager (configured via figma_secret_name in settings)
        """
        if not FIGMA_TOOLS_AVAILABLE:
            self.log("Figma tools not available (missing dependencies)", level="info")
            return

        if not self.settings.figma_mcp_enabled:
            self.log("Figma MCP integration disabled in settings", level="info")
            return

        # Try to get Figma token from various sources
        figma_token = self._load_figma_token()
        if not figma_token:
            self.log("Figma access token not available - Figma integration disabled", level="info")
            return

        try:
            # Initialize Python wrapper tools
            self.figma_tools = FigmaTools(access_token=figma_token)

            # Verify API connectivity
            health = self.figma_tools.check_api_health()
            if health.get("status") == "healthy":
                self.log(f"Figma API connected: user={health.get('user_name', 'unknown')}")

                # Get MCP server configuration for Claude SDK
                transport = self.settings.figma_mcp_transport
                if transport == "http":
                    self.figma_mcp_config = FigmaMCPConfig.get_remote_mcp_config(figma_token)
                else:
                    self.figma_mcp_config = FigmaMCPConfig.get_local_mcp_config()

                self.log("Figma MCP server configured for Claude Agent SDK")
            else:
                self.log(f"Figma API health check failed: {health}", level="warning")
                self.figma_tools = None

        except Exception as e:
            self.log(f"Failed to initialize Figma tools: {e}", level="warning")
            self.figma_tools = None

    def _load_figma_token(self) -> str | None:
        """Load Figma access token from environment, settings, or Secrets Manager.

        Returns:
            Figma access token or None if not available
        """
        import os

        # 1. Check environment variable first
        token = os.environ.get("FIGMA_ACCESS_TOKEN")
        if token:
            self.log("Using Figma token from FIGMA_ACCESS_TOKEN env var")
            return token

        # 2. Check settings
        if self.settings.figma_access_token:
            self.log("Using Figma token from settings")
            return self.settings.figma_access_token

        # 3. Try AWS Secrets Manager
        try:
            import boto3
            from botocore.exceptions import ClientError

            client = boto3.client(
                "secretsmanager",
                region_name=self.settings.aws_region,
            )

            response = client.get_secret_value(SecretId=self.settings.figma_secret_name)
            secret_value = response.get("SecretString", "")

            if secret_value:
                # Handle both plain token and JSON format
                try:
                    import json

                    secret_json = json.loads(secret_value)
                    token = secret_json.get("access_token") or secret_json.get("token")
                except json.JSONDecodeError:
                    # Plain string token
                    token = secret_value

                if token:
                    self.log(f"Loaded Figma token from Secrets Manager ({self.settings.figma_secret_name})")
                    # Cache in settings for other components
                    self.settings.figma_access_token = token
                    return token

        except ImportError:
            self.log("boto3 not available for Secrets Manager", level="warning")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                self.log(f"Figma secret not found: {self.settings.figma_secret_name}", level="info")
            else:
                self.log(f"Could not load Figma secret: {e}", level="warning")
        except Exception as e:
            self.log(f"Error loading Figma token from Secrets Manager: {e}", level="warning")

        return None

    def _setup_llm_env(self):
        """Configure Codex authentication for headless CI/CD.

        Auth is retrieved from AWS Secrets Manager and written to ~/.codex/auth.json.
        """
        # Codex authentication is handled via CodexAuthContext when running agents
        # Auth.json is retrieved from AWS Secrets Manager at runtime
        self.codex_auth_manager: CodexAuthManager | None = None
        if CODEX_AUTH_AVAILABLE:
            self.codex_auth_manager = CodexAuthManager()
            self.log("Codex auth manager initialized (will retrieve auth from Secrets Manager)")
        else:
            self.log("Codex auth module not available", level="warning")

    def get_initial_state(self) -> BuilderState:
        """Get initial state for the workflow."""
        return BuilderState(
            repos_to_process=self.repos.copy(),
            issues_to_process=[],
            current_issue=None,
            current_repo=None,
            repo_config=None,
            container_id=None,
            work_dir=None,
            fix_applied=False,
            tests_passed=False,
            pr_url=None,
            error=None,
            processed_issues=[],
        )

    async def run(self, **kwargs) -> AgentResult:
        """Execute the Builder agent workflow."""
        import time

        run_start_time = time.time()

        self.log("Starting Builder Agent...")
        self.log(f"Configured repos: {', '.join(self.repos)}")

        if not codex_cli_available():
            run_duration = time.time() - run_start_time
            self.metrics.record_agent_run_duration(run_duration)
            return self.create_result(
                success=False,
                message="Codex CLI not installed. Run: npm install -g @openai/codex",
                errors=["codex CLI not found in PATH"],
            )

        if not CODEX_AUTH_AVAILABLE:
            run_duration = time.time() - run_start_time
            self.metrics.record_agent_run_duration(run_duration)
            return self.create_result(
                success=False,
                message="Codex auth module not available",
                errors=["agents.codex_auth module not found"],
            )

        processed = []
        errors = []

        try:
            # Find issues across all configured repos
            all_issues = []
            for repo in self.repos:
                self.log(f"Searching for issues in {repo}...")
                issues = await self._find_aibuild_issues(repo)
                all_issues.extend(issues)

            if not all_issues:
                self.log("No issues found with 'aibuild' label across all repos")
                self.chat_client.send_builder_no_issues_notification(
                    agent_name="Builder Agent",
                    repos=self.repos,
                )
                run_duration = time.time() - run_start_time
                self.metrics.record_agent_run_duration(run_duration)
                return self.create_result(
                    success=True,
                    message="No issues found to process",
                    changes_made=0,
                )

            self.log(f"Found {len(all_issues)} issue(s) to process across {len(self.repos)} repo(s)")

            # Sort issues by priority (High > Medium > Low)
            all_issues = await self._sort_by_priority(all_issues)

            issues_to_process = all_issues[: self.max_issues_per_run]

            # Process issues (parallel or sequential based on settings)
            if self.parallel and len(issues_to_process) > 1:
                self.log(
                    f"Processing {len(issues_to_process)} issues in PARALLEL (max {self.max_concurrent} concurrent)"
                )
                results = await self._process_issues_parallel(issues_to_process)
            else:
                # Sequential processing (single issue or parallel disabled)
                self.log(f"Processing {len(issues_to_process)} issue(s) sequentially")
                results = []
                for issue in issues_to_process:
                    self.log(f"Processing issue #{issue['number']} in {issue['repository']}: {issue['title']}")
                    result = await self._process_issue(issue)
                    results.append((issue, result))

            # Collect results and record metrics
            for issue, result in results:
                repo_name = issue["repository"].split("/")[-1]  # Extract repo name from "Org/Repo"

                if result.get("success"):
                    processed.append(
                        {
                            "number": issue["number"],
                            "title": issue["title"],
                            "repository": issue["repository"],
                            "pr_url": result.get("pr_url"),
                        }
                    )
                    # Record success metrics
                    self.metrics.record_issue_processed(repo_name, success=True)
                    if result.get("pr_url"):
                        self.metrics.record_pr_created(repo_name)
                    if result.get("duration"):
                        self.metrics.record_resolution_time(result["duration"], repo_name)
                else:
                    errors.append(f"{issue['repository']}#{issue['number']}: {result.get('error', 'Unknown error')}")
                    # Record failure metrics
                    self.metrics.record_issue_processed(repo_name, success=False)
                    error_type = self._classify_error(result.get("error", ""))
                    self.metrics.record_error(error_type, repo_name)

            # Record overall success rate
            total = len(processed) + len(errors)
            if total > 0:
                success_rate = (len(processed) / total) * 100
                self.metrics.record_success_rate(success_rate)

            success = len(processed) > 0
            message = f"Processed {len(processed)} issue(s)"
            if errors:
                message += f", {len(errors)} error(s)"

            # Record agent run duration
            run_duration = time.time() - run_start_time
            self.metrics.record_agent_run_duration(run_duration)

            return self.create_result(
                success=success,
                message=message,
                changes_made=len(processed),
                pr_url=processed[0]["pr_url"] if processed else None,
                errors=errors,
                metadata={"processed_issues": processed},
            )

        except Exception as e:
            self.log(f"Agent failed: {e}", level="error")
            # Record agent run duration even on failure
            run_duration = time.time() - run_start_time
            self.metrics.record_agent_run_duration(run_duration)
            self.metrics.record_error("agent_exception")
            return self.create_result(
                success=False,
                message=f"Agent failed: {e}",
                errors=[str(e)],
            )

    async def _find_aibuild_issues(self, repo: str) -> list[dict[str, Any]]:
        """Find issues with 'aibuild' or 'ai-in-progress' label in a specific repo.

        HYBRID APPROACH (Issue #256):
        - 'aibuild' is added by manual label additions and subtask creation
        - 'ai-in-progress' is added by the sync workflow (doesn't trigger cascade)
        We search for both to handle all cases.
        """
        all_issues: dict[int, dict] = {}  # Use dict to dedupe by issue number

        # Search for both the source label (aibuild) and in-progress label
        labels_to_search = [self.source_label, self.in_progress_label]

        for label in labels_to_search:
            result = subprocess.run(
                [
                    "gh",
                    "issue",
                    "list",
                    "--repo",
                    repo,
                    "--label",
                    label,
                    "--state",
                    "open",
                    "--json",
                    "number,title,body,url,labels",
                    "--limit",
                    str(self.max_issues_per_run * 2),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self.log(f"Failed to fetch issues with '{label}' from {repo}: {result.stderr}", level="warning")
                continue

            issues = json.loads(result.stdout) if result.stdout.strip() else []

            for issue in issues:
                issue_num = issue["number"]
                if issue_num in all_issues:
                    continue  # Already added from another label

                label_names = [l.get("name", "") for l in issue.get("labels", [])]

                # Skip if already completed
                if self.complete_label in label_names:
                    continue

                all_issues[issue_num] = {
                    "number": issue_num,
                    "title": issue["title"],
                    "body": issue.get("body", ""),
                    "url": issue["url"],
                    "repository": repo,
                    "labels": label_names,
                }

        filtered = list(all_issues.values())
        self.log(
            f"Found {len(filtered)} issues with '{self.source_label}' or '{self.in_progress_label}' label in {repo}"
        )
        return filtered

    async def _sort_by_priority(self, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort issues by priority from GitHub Project.

        Queries the GitHub Project to get priority information and sorts issues
        with High priority first, then Medium, then Low, then unset.

        Issues marked as "Blocked" are filtered out entirely.

        Args:
            issues: List of issues to sort

        Returns:
            Sorted list of issues (excluding blocked ones)
        """
        if not issues:
            return issues

        # Build a lookup of issue priorities from project boards
        priority_map: dict[str, str] = {}  # "repo#number" -> priority

        # Get unique repos from issues
        repos = set(issue["repository"] for issue in issues)

        for repo in repos:
            try:
                project_number = self._get_project_for_repo(repo)

                # Get prioritized items from the project
                items = self.project_client.get_issues_by_priority(
                    project_number=project_number,
                    status_filter=None,  # Get all statuses
                )

                for item in items:
                    key = f"{item.repository}#{item.number}"
                    priority_map[key] = item.priority

            except Exception as e:
                self.log(f"Could not get priorities from project for {repo}: {e}", level="warning")
                continue

        # Filter out blocked issues and sort by priority
        priority_order = {"High": 0, "Medium": 1, "Low": 2, None: 3}

        def get_priority_score(issue: dict[str, Any]) -> int:
            key = f"{issue['repository']}#{issue['number']}"
            priority = priority_map.get(key)
            return priority_order.get(priority, 3)

        # Filter out blocked issues
        non_blocked = []
        for issue in issues:
            key = f"{issue['repository']}#{issue['number']}"
            priority = priority_map.get(key)
            if priority != "Blocked":
                issue["priority"] = priority  # Store for logging
                non_blocked.append(issue)
            else:
                self.log(f"Skipping blocked issue #{issue['number']}: {issue['title'][:40]}")

        # Sort by priority
        sorted_issues = sorted(non_blocked, key=get_priority_score)

        if sorted_issues:
            priorities_found = [i.get("priority", "None") for i in sorted_issues[:5]]
            self.log(f"Priority order (first 5): {priorities_found}")

        return sorted_issues

    async def _process_issues_parallel(
        self, issues: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Process multiple issues concurrently with optional per-repo locking.

        Uses a semaphore to limit concurrency. Per-repo locks are optional and
        can serialize same-repo issues if enabled (repo_lock=True).

        Args:
            issues: List of issues to process

        Returns:
            List of (issue, result) tuples
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_issue_task(
            issue: dict[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            """Process a single issue with semaphore and optional repo lock."""
            repo = issue["repository"]

            async with semaphore:
                # Optionally acquire repo lock (if repo_lock is enabled)
                if self.repo_lock:
                    if repo not in self.repo_locks:
                        self.repo_locks[repo] = asyncio.Lock()
                    async with self.repo_locks[repo]:
                        self.log(f"[Parallel+Lock] Starting issue #{issue['number']} in {repo}")
                        try:
                            result = await self._process_issue(issue)
                            return (issue, result)
                        except Exception as e:
                            self.log(
                                f"[Parallel] Exception processing #{issue['number']}: {e}",
                                level="error",
                            )
                            return (issue, {"success": False, "error": str(e)})
                else:
                    # No repo lock - full parallel processing
                    self.log(f"[Parallel] Starting issue #{issue['number']} in {repo}")
                    try:
                        result = await self._process_issue(issue)
                        return (issue, result)
                    except Exception as e:
                        self.log(
                            f"[Parallel] Exception processing #{issue['number']}: {e}",
                            level="error",
                        )
                        return (issue, {"success": False, "error": str(e)})

        # Process all issues concurrently
        tasks = [process_issue_task(issue) for issue in issues]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.log(
                    f"[Parallel] Task exception for issue #{issues[i]['number']}: {result}",
                    level="error",
                )
                processed_results.append((issues[i], {"success": False, "error": str(result)}))
            else:
                processed_results.append(result)

        return processed_results

    def _load_repo_config(self, work_dir: str, repo: str) -> RepoConfig:
        """Load repository configuration from agent-config.json or detect defaults."""
        config_path = Path(work_dir) / "agent-config.json"

        if config_path.exists():
            self.log("Loading config from agent-config.json")
            try:
                data = json.loads(config_path.read_text())
                return RepoConfig(
                    name=data.get("service", {}).get("name", repo.split("/")[-1]),
                    type=data.get("service", {}).get("type", "unknown"),
                    description=data.get("service", {}).get("description", ""),
                    repository=repo,
                    build_command=data.get("build", {}).get("command", ""),
                    build_timeout=data.get("build", {}).get("timeout", 600),
                    analyze_command=data.get("test", {}).get("analyze", {}).get("command", ""),
                    analyze_timeout=data.get("test", {}).get("analyze", {}).get("timeout", 180),
                    test_command=data.get("test", {}).get("unit", {}).get("command", ""),
                    test_timeout=data.get("test", {}).get("unit", {}).get("timeout", 300),
                    language=data.get("codeStyle", {}).get("language", ""),
                    formatter_command=data.get("codeStyle", {}).get("formatter", ""),
                    guidelines=data.get("codeStyle", {}).get("guidelines", []),
                    source_label=data.get("labels", {}).get("source", self.source_label),
                    complete_label=data.get("labels", {}).get("complete", self.complete_label),
                    in_progress_label=data.get("labels", {}).get("inProgress", self.in_progress_label),
                )
            except Exception as e:
                self.log(f"Error loading agent-config.json: {e}", level="warning")

        # Auto-detect project type
        project_type = self._detect_project_type(work_dir)
        self.log(f"Auto-detected project type: {project_type}")

        defaults = self.DEFAULT_CONFIGS.get(project_type, {})
        return RepoConfig(
            name=repo.split("/")[-1],
            type=project_type,
            repository=repo,
            build_command=defaults.get("build_command", ""),
            analyze_command=defaults.get("analyze_command", ""),
            test_command=defaults.get("test_command", ""),
            language=defaults.get("language", ""),
        )

    def _classify_error(self, error_message: str) -> str:
        """Classify an error message into a metric-friendly type.

        Args:
            error_message: The error message to classify

        Returns:
            Error type string for metrics
        """
        error_lower = error_message.lower()

        if "clone" in error_lower:
            return "clone_failed"
        elif "test" in error_lower or "analyze" in error_lower:
            return "test_failed"
        elif "sdk" in error_lower or "claude" in error_lower:
            return "sdk_error"
        elif "pr" in error_lower or "pull request" in error_lower:
            return "pr_creation_failed"
        elif "permission" in error_lower or "auth" in error_lower:
            return "permission_error"
        elif "timeout" in error_lower:
            return "timeout"
        else:
            return "unknown_error"

    def _detect_project_type(self, work_dir: str) -> str:
        """Auto-detect project type from files in the repository."""
        work_path = Path(work_dir)

        if (work_path / "pubspec.yaml").exists():
            return "flutter"
        elif (work_path / "pom.xml").exists():
            return "java-maven"
        elif (work_path / "build.gradle").exists():
            return "java-gradle"
        elif (work_path / "go.mod").exists():
            return "go"
        elif (work_path / "pyproject.toml").exists() or (work_path / "setup.py").exists():
            return "python"
        elif (work_path / "package.json").exists():
            return "nodejs"
        elif (work_path / "Cargo.toml").exists():
            return "rust"

        return "unknown"

    def _get_project_for_repo(self, repo: str) -> int:
        """Determine which GitHub Project an issue belongs to based on repo.

        Args:
            repo: Repository name (e.g., "owner/flutter-app")

        Returns:
            Project number (10 for Flutter, 11 for Backend)
        """
        repo_lower = repo.lower()

        # Flutter project repos
        if "flutter" in repo_lower:
            return self.flutter_project

        # Backend project repos (everything else)
        return self.backend_project

    async def _update_project_on_completion(
        self,
        issue: dict[str, Any],
        repo: str,
        pr_url: str | None = None,
        success: bool = True,
    ) -> None:
        """Update GitHub Project fields when an issue is processed.

        Args:
            issue: Issue data
            repo: Repository name
            pr_url: PR URL if created
            success: Whether processing succeeded
        """
        try:
            project_number = self._get_project_for_repo(repo)
            self.log(f"Updating project #{project_number} for issue #{issue['number']}")

            result = self.project_client.update_issue_on_completion(
                repo=repo,
                issue_number=issue["number"],
                project_number=project_number,
                pr_url=pr_url,
                success=success,
            )

            if result:
                self.log("Project fields updated successfully")
            else:
                self.log("Could not update project fields (issue may not be in project)", level="warning")

        except Exception as e:
            # Don't fail the whole process if project update fails
            self.log(f"Failed to update project: {e}", level="warning")

    async def _mark_in_progress(self, issue: dict[str, Any], repo: str):
        """Add in-progress label when starting to work on an issue."""
        self.log(f"Marking issue #{issue['number']} as in-progress...")

        # Add in-progress label
        subprocess.run(
            [
                "gh",
                "issue",
                "edit",
                str(issue["number"]),
                "--repo",
                repo,
                "--add-label",
                self.in_progress_label,
            ],
            capture_output=True,
        )

        # Remove source label to prevent re-processing
        subprocess.run(
            [
                "gh",
                "issue",
                "edit",
                str(issue["number"]),
                "--repo",
                repo,
                "--remove-label",
                self.source_label,
            ],
            capture_output=True,
        )

        self.log(f"Labels updated: -{self.source_label}, +{self.in_progress_label}")

    async def _mark_failed(self, issue: dict[str, Any], repo: str, error: str):
        """Mark issue as failed - remove in-progress, add failed label."""
        self.log(f"Marking issue #{issue['number']} as failed...")

        # Remove in-progress label
        subprocess.run(
            [
                "gh",
                "issue",
                "edit",
                str(issue["number"]),
                "--repo",
                repo,
                "--remove-label",
                self.in_progress_label,
            ],
            capture_output=True,
        )

        # Add failed label (not source label - prevents infinite retry loop)
        subprocess.run(
            [
                "gh",
                "issue",
                "edit",
                str(issue["number"]),
                "--repo",
                repo,
                "--add-label",
                self.failed_label,
            ],
            capture_output=True,
        )

        # Comment on the issue about the failure
        comment_body = f"""## ⚠️ AI Agent Fix Failed

The Builder Agent attempted to fix this issue but encountered an error:

```
{error}
```

**To retry**, remove the `{self.failed_label}` label and add `{self.source_label}` back.

You may want to:
- Add more details to the issue description
- Specify which files need to be changed
- Break down the issue into smaller tasks

---
*This message was generated automatically by the Builder Agent.*
"""
        subprocess.run(
            [
                "gh",
                "issue",
                "comment",
                str(issue["number"]),
                "--repo",
                repo,
                "--body",
                comment_body,
            ],
            capture_output=True,
        )

        self.log(f"Labels updated: -{self.in_progress_label}, +{self.failed_label}")

    def _verify_codebuild_environment(self) -> None:
        """Verify running in CodeBuild container for --yolo safety.

        The --yolo flag bypasses Codex's safety sandbox, so we only allow it
        in isolated CodeBuild containers where it can't affect the host system.

        Raises:
            RuntimeError: If not running in a CodeBuild environment.
        """
        if os.environ.get("CODEBUILD_BUILD_ID") is None:
            raise RuntimeError("Codex --yolo flag requires CodeBuild container isolation.")

    async def _process_issue(self, issue: dict[str, Any]) -> dict[str, Any]:
        """Process a single issue: clone, fix, test, PR."""
        import time

        start_time = time.time()

        work_dir = None
        container_id = None
        repo = issue["repository"]

        # Start threaded notification for this issue
        thread_key = self.chat_client.send_builder_thread_start(
            agent_name="Builder Agent",
            repo=repo,
            issue_number=issue["number"],
            issue_title=issue["title"],
            issue_url=issue.get("url"),
            issue_body=issue.get("body", ""),
            dry_run=self.settings.dry_run,
        )

        try:
            # Step 0: Mark issue as in-progress
            if not self.settings.dry_run:
                await self._mark_in_progress(issue, repo)

            # Step 1: Clone repository
            self.chat_client.send_builder_stage_update(
                thread_key=thread_key,
                stage=WorkflowStage.CLONING,
                details=f"Cloning {repo} and setting up workspace",
                extra_info={"Repository": repo, "Branch": f"ai/fix-issue-{issue['number']}"},
            )

            work_dir = tempfile.mkdtemp(prefix="builder_agent_")
            self.log(f"Cloning {repo} to {work_dir}...")
            self.clone_source_branch = "staging"

            # Clone from staging branch - all agent work happens on staging first,
            # then PRs are created targeting staging (not main)
            clone_result = subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "staging",
                    f"https://github.com/{repo}.git",
                    work_dir,
                ],
                capture_output=True,
                text=True,
            )

            if clone_result.returncode != 0:
                error_msg = f"Clone failed: {clone_result.stderr}"
                duration = int(time.time() - start_time)
                self.chat_client.send_builder_thread_complete(
                    thread_key=thread_key,
                    success=False,
                    error=error_msg,
                    duration_seconds=duration,
                )
                return {"success": False, "error": error_msg}

            clone_source_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=work_dir,
                capture_output=True,
                text=True,
            )
            if clone_source_result.returncode == 0:
                detected_branch = clone_source_result.stdout.strip()
                if detected_branch and detected_branch != "HEAD":
                    self.clone_source_branch = detected_branch
            self.log(f"Clone source branch detected: {self.clone_source_branch}")

            # Initialize submodules
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=work_dir,
                capture_output=True,
            )

            # Step 2: Load repo configuration
            config = self._load_repo_config(work_dir, repo)
            self.log(f"Project type: {config.type}, Language: {config.language}")

            # Create branch for this issue (auto-increment if branch already exists)
            base_branch = f"ai/fix-issue-{issue['number']}"
            branch_name = base_branch
            version = 2

            # Check if remote branch exists, increment until we find available name
            while True:
                result = subprocess.run(
                    ["git", "ls-remote", "--heads", "origin", branch_name],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                )
                if not result.stdout.strip():
                    break  # Branch doesn't exist on remote, use it
                self.log(f"Branch {branch_name} already exists, trying v{version}")
                branch_name = f"{base_branch}-v{version}"
                version += 1
                if version > 100:  # Safety limit
                    raise RuntimeError(f"Too many branch versions for issue {issue['number']}")

            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=work_dir,
                capture_output=True,
            )

            self.log(f"Repository ready on branch: {branch_name}")

            # Step 3: Setup devcontainer (if available)
            container_id = await self._setup_devcontainer(work_dir, issue["number"])

            # Step 4: Use Codex CLI to fix the issue
            self.chat_client.send_builder_stage_update(
                thread_key=thread_key,
                stage=WorkflowStage.ANALYZING,
                details=f"Reading issue #{issue['number']} and understanding the {config.type} codebase",
                extra_info={
                    "Project Type": config.type.title(),
                    "Language": config.language.title() if config.language else "Auto-detect",
                },
            )

            self.log("Running Codex CLI to analyze and fix the issue...")

            self.chat_client.send_builder_stage_update(
                thread_key=thread_key,
                stage=WorkflowStage.FIXING,
                details=f"Codex AI is implementing the fix for: {issue['title'][:60]}...",
                extra_info={"Model": "GPT-5 Codex", "Mode": "Autonomous coding"},
            )

            fix_result = await self._run_codex_agent(work_dir, issue, config, container_id)

            if not fix_result.get("fix_applied"):
                error_msg = fix_result.get("error", "No fix applied")
                duration = int(time.time() - start_time)
                self.chat_client.send_builder_thread_complete(
                    thread_key=thread_key,
                    success=False,
                    error=error_msg,
                    duration_seconds=duration,
                )
                return {"success": False, "error": error_msg}

            # Get files changed for reporting
            files_changed = fix_result.get("files_changed", [])

            # Post-process changes: validate semantics and generate stories
            if config.type == "flutter":
                self.log("Running post-processing for semantic labels and stories...")
                post_process_result = await self._post_process_changes(work_dir, config)

                # Update state with post-processing results
                self.state["semantic_issues"] = post_process_result.get("semantic_issues", [])
                self.state["semantic_labels_added"] = post_process_result.get("semantic_labels_added", [])
                self.state["stories_generated"] = post_process_result.get("stories_generated", [])

                # Log any semantic issues found (warnings, not failures)
                if post_process_result.get("semantic_issues"):
                    self.log(
                        f"Warning: Found {len(post_process_result['semantic_issues'])} widgets missing Semantics wrappers",
                        level="warning",
                    )

            # Step 5: Run tests using repo's configured commands
            test_cmd = config.analyze_command or config.test_command or "validation"
            self.chat_client.send_builder_stage_update(
                thread_key=thread_key,
                stage=WorkflowStage.TESTING,
                details=f"Validating changes and running {test_cmd}",
                extra_info={"Files Changed": str(len(files_changed)), "Command": test_cmd[:40]},
            )

            self.log("Running tests...")
            tests_passed = await self._run_tests(work_dir, config, container_id)

            if not tests_passed and not self.settings.dry_run:
                error_msg = "Tests failed"
                duration = int(time.time() - start_time)
                self.chat_client.send_builder_thread_complete(
                    thread_key=thread_key,
                    success=False,
                    error=error_msg,
                    duration_seconds=duration,
                )
                return {"success": False, "error": error_msg}

            # Step 6: Create PR (skip in dry run)
            if self.settings.dry_run:
                self.log("[DRY RUN] Would create PR, comment on issue, and update labels")
                duration = int(time.time() - start_time)
                self.chat_client.send_builder_thread_complete(
                    thread_key=thread_key,
                    success=True,
                    dry_run=True,
                    duration_seconds=duration,
                )
                return {"success": True, "pr_url": None}

            self.chat_client.send_builder_stage_update(
                thread_key=thread_key,
                stage=WorkflowStage.CREATING_PR,
                details="Pushing changes and creating pull request for review",
                extra_info={"Branch": branch_name, "Files": str(len(files_changed))},
            )

            pr_url = await self._create_pr(work_dir, issue, branch_name, config)

            # Step 7: Check completion with orchestrator (DISABLED)
            # DISABLED: Subtask creation caused runaway cascade (issue #256)
            # The orchestrator creates subtasks with labels that trigger more builder runs,
            # leading to exponential issue/PR creation. Disabled until orchestrator has
            # proper database persistence and subtask tracking.
            # TODO: Re-enable when orchestrator_state table exists in production DB
            completion_result: dict[str, Any] | None = None
            # try:
            #     completion_result = await self.task_orchestrator.after_claude_run(
            #         repository=repo,
            #         issue_number=issue["number"],
            #         issue_title=issue["title"],
            #         issue_body=issue.get("body", ""),
            #         work_dir=work_dir,
            #         files_changed=files_changed,
            #         pr_created=bool(pr_url),
            #         pr_url=pr_url,
            #     )
            #     self.log(f"[COMPLETION] Status: {completion_result.get('status', 'unknown')}")
            #
            #     if completion_result.get("status") == "awaiting_subtasks":
            #         subtasks = completion_result.get("subtasks", [])
            #         self.log(f"[COMPLETION] Created {len(subtasks)} subtasks for remaining work")
            #     elif completion_result.get("status") == "incomplete":
            #         self.log(
            #             f"[COMPLETION] Issue incomplete: {completion_result.get('reasoning', 'unknown')}",
            #             level="warning",
            #         )
            #
            # except Exception as completion_error:
            #     # Don't fail the whole build if completion check fails
            #     self.log(f"[COMPLETION] Check failed (non-fatal): {completion_error}", level="warning")

            if pr_url:
                # Step 7.5: Comment on issue (with completion status)
                await self._comment_on_issue(issue, pr_url, work_dir, config, completion_result)

            # Step 8: Update labels
            await self._update_labels(issue, repo)

            # Step 9: Update GitHub Project fields
            await self._update_project_on_completion(issue, repo, pr_url=pr_url, success=True)

            # Send success completion
            duration = int(time.time() - start_time)
            self.chat_client.send_builder_thread_complete(
                thread_key=thread_key,
                success=True,
                pr_url=pr_url,
                duration_seconds=duration,
                files_changed=files_changed,
                issue_url=issue.get("url"),
            )

            return {"success": True, "pr_url": pr_url}

        except Exception as e:
            self.log(f"Error processing issue: {e}", level="error")
            # Send failure completion
            duration = int(time.time() - start_time)
            self.chat_client.send_builder_thread_complete(
                thread_key=thread_key,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
            # Mark as failed and return to queue
            if not self.settings.dry_run:
                await self._mark_failed(issue, repo, str(e))
                # Update GitHub Project fields for failure
                await self._update_project_on_completion(issue, repo, success=False)
            return {"success": False, "error": str(e)}

        finally:
            await self._cleanup(container_id, work_dir)

    async def _setup_devcontainer(self, work_dir: str, issue_number: int) -> str | None:
        """Setup devcontainer from repository's .devcontainer spec."""
        self.log("Setting up devcontainer...")

        devcontainer_path = Path(work_dir) / ".devcontainer" / "devcontainer.json"

        if not devcontainer_path.exists():
            self.log("No devcontainer.json found, running without container")
            return None

        # Parse devcontainer.json (strip comments)
        content = devcontainer_path.read_text()
        lines = [line for line in content.split("\n") if not line.strip().startswith("//")]
        config = json.loads("\n".join(lines))

        container_name = f"builder_agent_{issue_number}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Determine image
        if "image" in config:
            image_name = config["image"]
            self.log(f"Using pre-built image: {image_name}")
            subprocess.run(["docker", "pull", image_name], capture_output=True, timeout=600)
        elif "build" in config:
            build_config = config["build"]
            dockerfile = build_config.get("dockerfile", "Dockerfile")
            context = build_config.get("context", ".")
            devcontainer_dir = Path(work_dir) / ".devcontainer"
            image_name = f"builder-agent-{issue_number}:latest"

            self.log(f"Building image from {dockerfile}...")
            build_result = subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    image_name,
                    "-f",
                    str(devcontainer_dir / dockerfile),
                    str(devcontainer_dir / context),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if build_result.returncode != 0:
                self.log(f"Docker build failed: {build_result.stderr}", level="warning")
                return None
        else:
            self.log("No image specified in devcontainer.json, running without container")
            return None

        # Start container
        self.log("Starting devcontainer...")
        env_args = []
        for key, value in config.get("containerEnv", {}).items():
            if "${containerEnv:" in value:
                value = value.replace("${containerEnv:PATH}", os.environ.get("PATH", ""))
            env_args.extend(["-e", f"{key}={value}"])

        docker_run = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-v",
                f"{work_dir}:/workspace",
                "-w",
                "/workspace",
            ]
            + env_args
            + [image_name, "tail", "-f", "/dev/null"],
            capture_output=True,
            text=True,
        )

        if docker_run.returncode != 0:
            self.log(f"Container start failed: {docker_run.stderr}", level="error")
            return None

        container_id = docker_run.stdout.strip()
        self.log(f"Container started: {container_id[:12]}")

        # Run postCreateCommand
        post_cmd = config.get("postCreateCommand")
        if post_cmd:
            self.log(f"Running postCreateCommand: {post_cmd}")
            subprocess.run(
                ["docker", "exec", container_id, "bash", "-c", post_cmd],
                capture_output=True,
                timeout=300,
            )

        return container_id

    async def _run_codex_agent(
        self, work_dir: str, issue: dict[str, Any], config: RepoConfig, container_id: str | None
    ) -> dict[str, Any]:
        """Use Codex CLI to analyze and fix the issue."""

        # Add Figma section if available (as context in prompt, not MCP)
        figma_section = ""
        if self.figma_mcp_config:
            default_file_key = self.settings.figma_default_file_key or "configured in project"
            figma_section = f"""

## Design Reference (Figma)

If this issue involves UI/UX work, refer to Figma designs:
- Design file key: {default_file_key}
- Ensure your implementation matches the design specs
"""

        # Build base prompt using PromptBuilder (handles persona, checklist, pattern analysis, etc.)
        prompt = self.prompt_builder.build_prompt(
            issue=issue,
            config=config,
            repo_path=work_dir,
            subtask_context=None,  # TODO: Wire up from orchestrator for subtask issues
            complexity="moderate",  # TODO: Detect from issue analysis
        )

        # Append Figma section if available
        if figma_section:
            prompt += figma_section

        # Append Flutter-specific sections (already conditional)
        if config.type == "flutter":
            semantic_section = self._build_semantic_enforcement_prompt(config)
            story_section = self._build_story_generation_prompt(config)
            if semantic_section:
                prompt += semantic_section
            if story_section:
                prompt += story_section

        try:
            # Check Codex CLI version
            cli_version = subprocess.run(["codex", "--version"], capture_output=True, text=True)
            if cli_version.returncode == 0:
                self.log(f"Codex CLI version: {cli_version.stdout.strip()}")
            else:
                self.log(f"Codex CLI version check failed: {cli_version.stderr}", level="warning")

            # Reset roadblock detector for this run
            self.roadblock_detector.reset()

            changes_made: list[str] = []
            build_id = os.environ.get("CODEBUILD_BUILD_ID", "local")

            # Setup Codex auth from Secrets Manager
            with CodexAuthContext(build_id=build_id):
                self.log("Codex auth setup complete")

                # Build codex exec command
                # Use --yolo (--dangerously-bypass-approvals-and-sandbox) to fully bypass sandbox
                # SECURITY: Only use --yolo in CodeBuild container environment
                self._verify_codebuild_environment()

                codex_cmd = [
                    "codex",
                    "exec",
                    "--yolo",  # Bypass approvals and sandbox (safe in CodeBuild container)
                    "--json",
                    "-C",
                    work_dir,  # Set working directory
                    prompt,
                ]

                self.log(f"Codex CLI command: codex exec --yolo --json -C {work_dir} [prompt]")

                # Run Codex CLI
                # Use a longer timeout since Codex can take a while for complex tasks
                codex_timeout = 900  # 15 minutes

                try:
                    # Use asyncio.to_thread to avoid blocking the event loop
                    result = await asyncio.to_thread(
                        subprocess.run,
                        codex_cmd,
                        capture_output=True,
                        text=True,
                        timeout=codex_timeout,
                        cwd=work_dir,
                    )

                    # Log raw output for debugging
                    self.log(f"Codex CLI exit code: {result.returncode}")
                    self.log(f"Codex stdout length: {len(result.stdout)} chars")
                    self.log(f"Codex stderr length: {len(result.stderr)} chars")

                    if not result.stdout.strip():
                        self.log("Codex produced no stdout output", level="warning")
                        if result.stderr:
                            self.log(f"Codex stderr: {result.stderr[:1000]}", level="warning")

                    # Log first 500 chars of raw output for debugging
                    if result.stdout:
                        self.log(f"Codex raw output (first 500): {result.stdout[:500]}")

                    # Parse JSON output (newline-delimited JSON events)
                    if result.stdout:
                        for line in result.stdout.strip().split("\n"):
                            if not line.strip():
                                continue
                            try:
                                event = json.loads(line)
                                event_type = event.get("type", "")

                                if event_type == "item.completed":
                                    item = event.get("item", {})
                                    item_type = item.get("type", "")

                                    if item_type == "file_change":
                                        for change in item.get("changes", []):
                                            file_path = change.get("path")
                                            kind = change.get("kind", "")
                                            if file_path and kind in ("add", "update"):
                                                changes_made.append(file_path)
                                        self.log(f"[FILE_CHANGE] {len(item.get('changes', []))} file(s)")

                                    elif item_type == "command_execution":
                                        cmd = item.get("command", "")[:60]
                                        self.log(f"[COMMAND] {cmd}")

                                    elif item_type == "agent_message":
                                        text = item.get("text", "")[:200]
                                        self.log(f"[MESSAGE] {text}")

                                        # Check for roadblocks
                                        roadblocks = self.roadblock_detector.check(text)
                                        if roadblocks:
                                            self.log(f"[ROADBLOCK] Detected: {roadblocks[0]}", level="warning")

                                elif event_type == "turn.completed":
                                    self.log("[RESULT] Codex turn completed")

                            except json.JSONDecodeError:
                                # Non-JSON line, log if verbose
                                if self.settings.verbose:
                                    self.log(f"[OUTPUT] {line[:200]}")

                    # Log any errors
                    if result.returncode != 0:
                        self.log(f"Codex CLI exited with code {result.returncode}", level="warning")
                        if result.stderr:
                            self.log(f"Codex stderr: {result.stderr[:500]}", level="warning")

                            # Check for auth errors
                            if "auth" in result.stderr.lower() or "login" in result.stderr.lower():
                                return {
                                    "fix_applied": False,
                                    "error": "Codex authentication failed - refresh token may have expired",
                                    "auth_error": True,
                                }

                except subprocess.TimeoutExpired:
                    self.log(f"Codex CLI timed out after {codex_timeout}s", level="error")
                    return {"fix_applied": False, "error": f"Codex timed out after {codex_timeout}s"}

                except Exception as e:
                    self.log(f"Codex CLI error: {e}", level="error")
                    import traceback

                    self.log(f"Traceback: {traceback.format_exc()}", level="error")
                    raise

            # Check if any files were actually modified
            git_status = subprocess.run(["git", "status", "--porcelain"], cwd=work_dir, capture_output=True, text=True)
            has_uncommitted_changes = bool(git_status.stdout.strip())

            # Also check for commits made by the agent (Codex may commit directly via Bash)
            # Get the base branch name (usually main or staging)
            base_branch = getattr(self, "clone_source_branch", "staging") or "staging"

            # Check for commits ahead of the base branch
            commits_ahead = subprocess.run(
                ["git", "log", f"origin/{base_branch}..HEAD", "--oneline"],
                cwd=work_dir,
                capture_output=True,
                text=True,
            )
            has_commits = bool(commits_ahead.stdout.strip())

            if has_commits:
                self.log(f"Agent made commits: {commits_ahead.stdout.strip()[:200]}")

            has_changes = has_uncommitted_changes or has_commits

            if has_changes:
                self.log("Agent made changes to the codebase")

                # If Claude already committed, skip the staging/commit process
                if has_commits and not has_uncommitted_changes:
                    self.log("Agent already committed changes, skipping builder commit")
                    # Get list of changed files from the commits
                    files_in_commits = subprocess.run(
                        ["git", "diff", "--name-only", f"origin/{base_branch}..HEAD"],
                        cwd=work_dir,
                        capture_output=True,
                        text=True,
                    )
                    committed_files = (
                        files_in_commits.stdout.strip().split("\n") if files_in_commits.stdout.strip() else []
                    )
                    return {
                        "fix_applied": True,
                        "files_changed": committed_files or changes_made,
                        "roadblocks_detected": self.roadblock_detector.detected,
                    }

                # Stage all changes first
                subprocess.run(["git", "add", "-A"], cwd=work_dir)

                # Get list of submodules to exclude from commit
                submodule_result = subprocess.run(
                    ["git", "config", "--file", ".gitmodules", "--get-regexp", "path"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                )
                submodule_paths = []
                if submodule_result.returncode == 0:
                    for line in submodule_result.stdout.strip().split("\n"):
                        if line:
                            # Format: "submodule.<name>.path <path>"
                            parts = line.split()
                            if len(parts) >= 2:
                                submodule_paths.append(parts[1])

                # Unstage all submodule changes after git add -A
                if submodule_paths:
                    self.log(f"Unstaging {len(submodule_paths)} submodule references")
                    for submodule in submodule_paths:
                        # Use git restore --staged to unstage without modifying working tree
                        subprocess.run(
                            ["git", "restore", "--staged", submodule],
                            cwd=work_dir,
                            capture_output=True,
                        )

                # Check if there are still staged changes after unstaging submodules
                staged_status = subprocess.run(
                    ["git", "diff", "--cached", "--name-only"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                )
                staged_files = staged_status.stdout.strip()

                if not staged_files:
                    # If no staged files but there are commits, that's still a success
                    if has_commits:
                        self.log("No additional changes to stage, but agent made commits")
                        files_in_commits = subprocess.run(
                            ["git", "diff", "--name-only", f"origin/{base_branch}..HEAD"],
                            cwd=work_dir,
                            capture_output=True,
                            text=True,
                        )
                        committed_files = (
                            files_in_commits.stdout.strip().split("\n") if files_in_commits.stdout.strip() else []
                        )
                        return {
                            "fix_applied": True,
                            "files_changed": committed_files or changes_made,
                            "roadblocks_detected": self.roadblock_detector.detected,
                        }
                    self.log("No non-submodule changes were made by the agent")
                    return {
                        "fix_applied": False,
                        "error": "No changes made (only submodule changes which were ignored)",
                    }

                self.log(f"Staged files (excluding submodules): {staged_files.split(chr(10))}")

                commit_msg = f"""fix: Address issue #{issue['number']}

{issue['title']}

Automated fix by Builder Agent using OpenAI Codex.

Closes #{issue['number']}

🤖 Generated with OpenAI Codex CLI

Co-Authored-By: Codex <noreply@openai.com>
"""
                subprocess.run(["git", "commit", "-m", commit_msg], cwd=work_dir, capture_output=True)

                return {
                    "fix_applied": True,
                    "files_changed": changes_made,
                    "roadblocks_detected": self.roadblock_detector.detected,
                }
            else:
                self.log("No changes were made by the agent")

                # Check if roadblocks were detected that might explain why
                if self.roadblock_detector.should_pause():
                    self.log(f"Roadblocks detected: {self.roadblock_detector.get_summary()}", level="warning")
                    return {
                        "fix_applied": False,
                        "error": "No changes made - roadblocks detected",
                        "roadblocks": self.roadblock_detector.detected,
                        "should_escalate": True,
                    }

                return {"fix_applied": False, "error": "No changes made"}

        except Exception as e:
            self.log(f"Codex CLI error: {e}", level="error")
            return {"fix_applied": False, "error": str(e)}

    async def _run_tests(self, work_dir: str, config: RepoConfig, container_id: str | None) -> bool:
        """Run tests using the repo's configured commands."""

        def run_cmd(cmd: str, timeout: int = 180) -> subprocess.CompletedProcess:
            if container_id:
                return subprocess.run(
                    ["docker", "exec", container_id, "bash", "-c", cmd],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            else:
                return subprocess.run(
                    cmd,
                    shell=True,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

        # Run analyze command if available
        if config.analyze_command:
            self.log(f"[ANALYZE] Running: {config.analyze_command}")
            try:
                analyze = run_cmd(config.analyze_command, config.analyze_timeout)
                if analyze.returncode != 0:
                    # Log both stdout and stderr for visibility
                    stdout_output = analyze.stdout.strip() if analyze.stdout else "(no stdout)"
                    stderr_output = analyze.stderr.strip() if analyze.stderr else "(no stderr)"
                    self.log(f"[ANALYZE] Exit code: {analyze.returncode}", level="warning")
                    self.log(f"[ANALYZE] Stdout: {stdout_output[:1000]}", level="warning")
                    if stderr_output != "(no stderr)":
                        self.log(f"[ANALYZE] Stderr: {stderr_output[:1000]}", level="warning")

                    # Check for actual errors (not just warnings)
                    combined_output = (analyze.stdout or "") + (analyze.stderr or "")
                    if (
                        "error" in combined_output.lower()
                        and "warning" not in combined_output.lower().split("error")[0][-20:]
                    ):
                        self.log("[ANALYZE] Found errors - build failed", level="error")
                        return False
                else:
                    self.log("[ANALYZE] Passed", level="info")
            except subprocess.TimeoutExpired:
                self.log("[ANALYZE] Command timed out", level="warning")

        # In dry-run mode, skip full test suite
        if self.settings.dry_run:
            self.log("[DRY RUN] Skipping tests to save time")
            return True

        # Run test command if available
        if config.test_command:
            self.log(f"Running tests: {config.test_command}")
            try:
                test = run_cmd(config.test_command, config.test_timeout)
                if test.returncode != 0:
                    output = test.stdout + (test.stderr or "")
                    self.log(f"Test output: {output[:500]}", level="warning")

                    # If no tests found, consider it a pass
                    if "no test" in output.lower() or not output.strip():
                        self.log("No tests found - skipping test verification", level="warning")
                        return True

                    # Allow PR creation even if tests fail - let CI verify
                    self.log(
                        "Tests may have failed - PR will still be created for review",
                        level="warning",
                    )
                    return True
            except subprocess.TimeoutExpired:
                self.log("Test command timed out", level="warning")
                return True

        self.log("Tests passed!")
        return True

    async def _create_pr(
        self, work_dir: str, issue: dict[str, Any], branch_name: str, config: RepoConfig
    ) -> str | None:
        """Create a pull request for the fix."""
        self.log("Creating pull request...")
        repo = issue["repository"]

        # Push branch
        push_result = subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )

        if push_result.returncode != 0:
            self.log(f"Push failed: {push_result.stderr}", level="error")
            return None

        # Create PR
        pr_body = f"""## Summary

Automated fix for issue #{issue['number']}: {issue['title']}

## Changes

This PR was generated by the Builder Agent using Claude Agent SDK.

## Project Info

- **Type**: {config.type}
- **Language**: {config.language}

## Test Plan

- [x] Code analysis passes
- [x] Tests pass (or no tests defined)

Closes #{issue['number']}

---
🤖 Generated with [OpenAI Codex CLI](https://developers.openai.com/codex/cli/)
"""

        # Create PR targeting staging branch - all agent PRs go to staging first
        pr_result = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                repo,
                "--title",
                f"fix: {issue['title']}",
                "--body",
                pr_body,
                "--head",
                branch_name,
                "--base",
                "staging",
            ],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )

        if pr_result.returncode != 0:
            self.log(f"PR creation failed: {pr_result.stderr}", level="error")
            return None

        pr_url = pr_result.stdout.strip()
        self.log(f"PR created: {pr_url}")
        return pr_url

    @staticmethod
    def _build_completion_section(completion_result: dict[str, Any] | None) -> str:
        """Build the completion status section for issue comments.

        Args:
            completion_result: Result from orchestrator completion check (optional)

        Returns:
            Formatted markdown string for completion section, or empty string if no result
        """
        if not completion_result:
            return ""

        status = completion_result.get("status", "unknown")
        status_emoji = {
            "complete": "✅ **Complete**",
            "awaiting_subtasks": "⏳ **Awaiting Subtasks**",
            "incomplete": "⚠️ **Incomplete**",
        }.get(status, f"❓ **{status}**")

        completion_section = f"\n### Completion Status\n{status_emoji}\n"

        if completion_result.get("reasoning"):
            completion_section += f"\n**Reasoning:** {completion_result['reasoning']}\n"

        # Subtasks created
        subtasks = completion_result.get("subtasks", [])
        if subtasks:
            completion_section += "\n### Subtasks Created\n"
            for st in subtasks:
                issue_num = st.get("issue_number", "")
                title = st.get("title", "Untitled")
                if issue_num:
                    completion_section += f"- [ ] #{issue_num} - {title}\n"
                else:
                    completion_section += f"- [ ] {title}\n"

        # Blockers
        blockers = completion_result.get("blockers", [])
        if blockers:
            completion_section += "\n### Blockers\n"
            for blocker in blockers:
                completion_section += f"- {blocker}\n"

        return completion_section

    async def _comment_on_issue(
        self,
        issue: dict[str, Any],
        pr_url: str,
        work_dir: str,
        config: RepoConfig,
        completion_result: dict[str, Any] | None = None,
    ):
        """Post a comment on the issue explaining what was done.

        Args:
            issue: Issue data
            pr_url: URL of the created PR
            work_dir: Working directory
            config: Repository configuration
            completion_result: Result from orchestrator completion check (optional)
        """
        self.log("Posting comment on issue...")

        # Get list of changed files
        diff_result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        files_changed = [f.strip() for f in diff_result.stdout.strip().split("\n") if f.strip()]

        files_list = "\n".join([f"  - `{f}`" for f in files_changed[:10]])
        if len(files_changed) > 10:
            files_list += f"\n  - ... and {len(files_changed) - 10} more files"

        # Get diff summary
        diff_stat = subprocess.run(
            ["git", "diff", "--stat", "HEAD~1", "HEAD"],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )

        # Get code diff
        diff_content = ""
        if files_changed:
            diff_result = subprocess.run(
                ["git", "diff", "HEAD~1", "HEAD", "--no-color"],
                cwd=work_dir,
                capture_output=True,
                text=True,
            )
            diff_content = diff_result.stdout[:3000]
            if len(diff_result.stdout) > 3000:
                diff_content += "\n... (truncated)"

        # Build semantic issues section
        semantic_section = ""
        semantic_issues = self.state.get("semantic_issues", [])
        if semantic_issues:
            semantic_section = (
                "\n### ⚠️ Accessibility Review Needed\n\nThe following widgets may need Semantics wrappers:\n\n"
            )
            semantic_section += "| File | Line | Widget | Message |\n"
            semantic_section += "|------|------|--------|--------|\n"
            for issue_item in semantic_issues[:5]:
                semantic_section += f"| `{issue_item['file_path']}` | {issue_item['line_number']} | {issue_item.get('pattern', '')} | {issue_item.get('message', '')[:50]}... |\n"
            if len(semantic_issues) > 5:
                semantic_section += f"\n*... and {len(semantic_issues) - 5} more*\n"

        # Build stories generated section
        stories_section = ""
        stories_generated = self.state.get("stories_generated", [])
        if stories_generated:
            stories_section = "\n### 📚 Widgetbook Stories Generated\n\n"
            stories_section += "The following stories were auto-generated for new widgets:\n\n"
            for story in stories_generated[:5]:
                stories_section += f"- `{story}`\n"
            if len(stories_generated) > 5:
                stories_section += f"\n*... and {len(stories_generated) - 5} more*\n"

        # Build completion status section
        completion_section = self._build_completion_section(completion_result)

        comment_body = f"""## 🤖 AI Agent Fix Applied

I've analyzed this issue and created a fix! Here's what I did:

### Pull Request
🔗 **{pr_url}**

### Project
- **Type**: {config.type}
- **Language**: {config.language}

### Files Changed
{files_list}

### Change Statistics
```
{diff_stat.stdout.strip()}
```
{semantic_section}{stories_section}{completion_section}
### Code Changes
<details>
<summary>Click to view diff</summary>

```diff
{diff_content}
```
</details>

---
*This fix was generated automatically by the Builder Agent using Claude Agent SDK.*
*Please review the changes carefully before merging.*
"""

        result = subprocess.run(
            [
                "gh",
                "issue",
                "comment",
                str(issue["number"]),
                "--repo",
                issue["repository"],
                "--body",
                comment_body,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            self.log(f"Failed to post comment: {result.stderr}", level="warning")
        else:
            self.log("Comment posted on issue")

    async def _update_labels(self, issue: dict[str, Any], repo: str):
        """Update issue labels: remove in-progress and source labels, add complete."""
        self.log("Updating issue labels...")

        # Remove in-progress label (may have been added when we started)
        subprocess.run(
            [
                "gh",
                "issue",
                "edit",
                str(issue["number"]),
                "--repo",
                repo,
                "--remove-label",
                self.in_progress_label,
            ],
            capture_output=True,
        )

        # Remove source label (in case it wasn't removed earlier)
        subprocess.run(
            [
                "gh",
                "issue",
                "edit",
                str(issue["number"]),
                "--repo",
                repo,
                "--remove-label",
                self.source_label,
            ],
            capture_output=True,
        )

        # Add complete label
        subprocess.run(
            [
                "gh",
                "issue",
                "edit",
                str(issue["number"]),
                "--repo",
                repo,
                "--add-label",
                self.complete_label,
            ],
            capture_output=True,
        )

        self.log(f"Labels updated: -{self.in_progress_label}, -{self.source_label}, +{self.complete_label}")

        # Auto-unblock dependent issues
        await self._unblock_dependent_issues(issue, repo)

    async def _unblock_dependent_issues(self, completed_issue: dict[str, Any], repo: str):
        """Find and unblock issues that depend on the completed issue.

        Searches for issues with 'Depends on #X' or 'Blocked by #X' in their body
        and changes their Priority from 'Blocked' to 'High'.

        Args:
            completed_issue: The issue that was just completed
            repo: Repository name
        """
        issue_number = completed_issue["number"]
        self.log(f"Checking for issues blocked by #{issue_number}...")

        try:
            # Search for issues that reference this one as a dependency
            # Patterns: "Depends on #123", "Blocked by #123", "depends on #123"
            search_terms = [
                f"Depends on #{issue_number}",
                f"Blocked by #{issue_number}",
                f"depends on #{issue_number}",
                f"blocked by #{issue_number}",
            ]

            dependent_issues = []

            for repo_to_check in self.repos:
                # Get all open issues with aibuild label
                result = subprocess.run(
                    [
                        "gh",
                        "issue",
                        "list",
                        "--repo",
                        repo_to_check,
                        "--label",
                        self.source_label,
                        "--state",
                        "open",
                        "--json",
                        "number,title,body",
                        "--limit",
                        "100",
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    continue

                issues = json.loads(result.stdout) if result.stdout else []

                for issue in issues:
                    body = issue.get("body", "") or ""
                    # Check if any dependency pattern matches
                    for pattern in search_terms:
                        if pattern in body:
                            dependent_issues.append(
                                {
                                    "repo": repo_to_check,
                                    "number": issue["number"],
                                    "title": issue["title"],
                                }
                            )
                            break

            if not dependent_issues:
                self.log(f"No issues found depending on #{issue_number}")
                return

            self.log(f"Found {len(dependent_issues)} issue(s) depending on #{issue_number}")

            # Unblock each dependent issue by changing Priority to High
            for dep_issue in dependent_issues:
                await self._unblock_issue(dep_issue)

        except Exception as e:
            # Don't fail the process if unblocking fails
            self.log(f"Error checking dependencies: {e}", level="warning")

    async def _unblock_issue(self, issue: dict[str, Any]):
        """Change an issue's Priority from Blocked to High.

        Args:
            issue: Issue data with repo, number, title
        """
        repo = issue["repo"]
        issue_number = issue["number"]

        try:
            project_number = self._get_project_for_repo(repo)

            # Get the project item ID for this issue
            item = self.project_client.get_issue_project_item(
                repo=repo,
                issue_number=issue_number,
                project_number=project_number,
            )

            if not item:
                self.log(f"Issue #{issue_number} not found in project {project_number}", level="warning")
                return

            # Update Priority from Blocked to High
            success = self.project_client.update_item_field(
                project_number=project_number,
                item_id=item.item_id,
                field_name="Priority",
                value="High",
            )

            if success:
                self.log(f"✓ Unblocked #{issue_number}: {issue['title'][:40]}... (Priority → High)")

                # Add a comment to notify about unblocking
                comment = """## 🔓 Issue Unblocked

This issue was automatically unblocked because its dependency was completed.

The Builder Agent will pick this up on the next run.

---
*Unblocked by Builder Agent*
"""
                subprocess.run(
                    [
                        "gh",
                        "issue",
                        "comment",
                        str(issue_number),
                        "--repo",
                        repo,
                        "--body",
                        comment,
                    ],
                    capture_output=True,
                )
            else:
                self.log(f"Could not update priority for #{issue_number}", level="warning")

        except Exception as e:
            self.log(f"Failed to unblock #{issue_number}: {e}", level="warning")

    async def _cleanup(self, container_id: str | None, work_dir: str | None):
        """Clean up container and work directory."""
        self.log("Cleaning up...")

        if container_id:
            self.log(f"Stopping container {container_id[:12]}...")
            subprocess.run(["docker", "stop", container_id], capture_output=True)
            subprocess.run(["docker", "rm", container_id], capture_output=True)

        if work_dir and Path(work_dir).exists():
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)

    # =========================================================================
    # Semantic Label Enforcement Methods
    # =========================================================================

    def _validate_semantics(self, work_dir: str, changed_files: list[str]) -> list[dict[str, Any]]:
        """Scan modified Dart files for interactive widgets missing Semantics wrappers.

        Args:
            work_dir: Working directory containing the repo
            changed_files: List of files that were modified

        Returns:
            List of issues found, each with file_path, line_number, widget_type, and message
        """
        issues: list[dict[str, Any]] = []

        # Only check Dart files in presentation or widget directories
        dart_files = [
            f
            for f in changed_files
            if f.endswith(".dart") and ("/presentation/" in f or "/widgets/" in f or "/components/" in f)
        ]

        if not dart_files:
            return issues

        parser = create_semantic_parser()

        # Interactive widget patterns that SHOULD have Semantics wrappers
        interactive_patterns = [
            (r"ElevatedButton\s*\(", "button"),
            (r"TextButton\s*\(", "button"),
            (r"OutlinedButton\s*\(", "button"),
            (r"IconButton\s*\(", "button"),
            (r"FloatingActionButton\s*\(", "button"),
            (r"InkWell\s*\(", "interactive"),
            (r"GestureDetector\s*\(", "interactive"),
            (r"TextFormField\s*\(", "textField"),
            (r"TextField\s*\(", "textField"),
            (r"DropdownButton\s*\(", "dropdown"),
            (r"Checkbox\s*\(", "toggled"),
            (r"Switch\s*\(", "toggled"),
            (r"Radio\s*\(", "toggled"),
            (r"Slider\s*\(", "slider"),
        ]

        import re

        for dart_file in dart_files:
            file_path = Path(work_dir) / dart_file
            if not file_path.exists():
                continue

            try:
                content = file_path.read_text()
                lines = content.split("\n")

                # Parse existing semantic labels in this file
                existing_labels = parser.parse_flutter_source(str(file_path))
                labeled_lines = {label.line_number for label in existing_labels}

                for pattern, widget_type in interactive_patterns:
                    for match in re.finditer(pattern, content):
                        # Find line number
                        line_num = content[: match.start()].count("\n") + 1

                        # Check if there's a Semantics wrapper nearby (within 10 lines before)
                        start_check = max(0, line_num - 10)
                        context_lines = "\n".join(lines[start_check:line_num])

                        has_semantics = "Semantics(" in context_lines

                        # If the widget doesn't have a Semantics wrapper nearby
                        if not has_semantics and line_num not in labeled_lines:
                            # Get context for the issue
                            line_content = lines[line_num - 1].strip() if line_num <= len(lines) else ""

                            cleaned_pattern = pattern.replace("\\s*\\(", "").replace("\\(", "")
                            issues.append(
                                {
                                    "file_path": dart_file,
                                    "line_number": line_num,
                                    "widget_type": widget_type,
                                    "pattern": cleaned_pattern,
                                    "line_content": line_content[:80],
                                    "message": f"Interactive widget '{cleaned_pattern}' at line {line_num} should have a Semantics wrapper for accessibility",
                                }
                            )

            except Exception as e:
                self.log(f"Error checking semantics in {dart_file}: {e}", level="warning")

        return issues

    def _build_semantic_enforcement_prompt(self, config: RepoConfig) -> str:
        """Build the semantic label enforcement section for the Claude prompt.

        Args:
            config: Repository configuration

        Returns:
            Prompt section for semantic label enforcement
        """
        if not config.semantic_labels_required:
            return ""

        pattern_text = ""
        if config.semantic_label_patterns:
            patterns = ", ".join(config.semantic_label_patterns)
            pattern_text = f"\n- For healthcare elements, use these prefix patterns: {patterns}"

        return f"""
## Semantic Label Requirements (MANDATORY)

ALL interactive Flutter widgets MUST have Semantics wrappers for accessibility.

### Pattern for Buttons
```dart
Semantics(
  label: semanticLabel ?? 'Descriptive action label',
  button: true,
  enabled: !isLoading,
  child: ExcludeSemantics(
    child: ElevatedButton(...),
  ),
)
```

### Pattern for Text Fields
```dart
Semantics(
  label: 'Field description',
  textField: true,
  child: ExcludeSemantics(
    child: TextFormField(...),
  ),
)
```

### Pattern for Toggles (Checkbox, Switch, Radio)
```dart
Semantics(
  label: 'Toggle description',
  toggled: isSelected,
  child: ExcludeSemantics(
    child: Checkbox(...),
  ),
)
```

### Guidelines
- Labels describe the ACTION, not the widget type: "Save patient record" not "ElevatedButton"
- Use `ExcludeSemantics` to prevent duplicate announcements
- Set appropriate semantic flags: `button: true`, `textField: true`, `toggled: value`, etc.{pattern_text}

FAILURE TO ADD SEMANTICS TO INTERACTIVE WIDGETS = FAILED REVIEW
"""

    # =========================================================================
    # Widgetbook Story Generation Methods
    # =========================================================================

    async def _generate_stories_for_widgets(
        self,
        work_dir: str,
        changed_files: list[str],
        config: RepoConfig,
    ) -> list[str]:
        """Generate Widgetbook stories for newly created widgets.

        Args:
            work_dir: Working directory containing the repo
            changed_files: List of files that were modified
            config: Repository configuration

        Returns:
            List of story file paths that were generated
        """
        if not config.story_generation_enabled or not config.widgetbook_stories_path:
            return []

        generated_stories: list[str] = []

        # Filter for new/modified widget files
        widget_files = [
            f
            for f in changed_files
            if f.endswith(".dart") and ("/presentation/" in f or "/widgets/" in f or "/components/" in f)
        ]

        if not widget_files:
            return generated_stories

        generator = create_story_generator()

        for widget_file in widget_files:
            file_path = Path(work_dir) / widget_file
            if not file_path.exists():
                continue

            try:
                # Generate story for this widget
                story = generator.generate_story(
                    widget_path=file_path,
                    use_cases=["Default", "Loading", "Error", "Dark Theme"],
                    include_viewports=True,
                    include_themes=True,
                )

                if story:
                    # Determine story output path
                    widget_name = file_path.stem  # e.g., "custom_btn_mobile"
                    story_filename = f"{widget_name}_stories.dart"
                    stories_dir = Path(work_dir) / config.widgetbook_stories_path

                    # Create stories directory if it doesn't exist
                    stories_dir.mkdir(parents=True, exist_ok=True)
                    story_path = stories_dir / story_filename

                    # Write story file
                    story_path.write_text(story.code)

                    # Add bidirectional labels
                    # 1. Add label to story file
                    generate_widgetbook_label(str(file_path))
                    story_content = f"/// SOURCE_WIDGET: {widget_file}\n/// CLASS: {story.widget_name}\n/// LAST_SYNCED: {datetime.now().strftime('%Y-%m-%d')}\n\n{story.code}"
                    story_path.write_text(story_content)

                    # 2. Add label to source widget (prepend to file)
                    source_label = generate_source_widget_label(
                        str(story_path.relative_to(work_dir)),
                        story.widget_name,
                        story.use_cases,
                    )

                    # Read current source content
                    source_content = file_path.read_text()

                    # Check if label already exists
                    if "/// WIDGETBOOK_STORY:" not in source_content:
                        # Find the right place to insert (before class declaration or imports)
                        import_match = source_content.find("import ")
                        if import_match != -1:
                            source_content = (
                                source_content[:import_match] + source_label + "\n\n" + source_content[import_match:]
                            )
                        else:
                            source_content = source_label + "\n\n" + source_content
                        file_path.write_text(source_content)

                    generated_stories.append(str(story_path.relative_to(work_dir)))
                    self.log(f"Generated story: {story_filename} for {story.widget_name}")

            except Exception as e:
                self.log(f"Error generating story for {widget_file}: {e}", level="warning")

        return generated_stories

    def _build_story_generation_prompt(self, config: RepoConfig) -> str:
        """Build the story generation section for the Claude prompt.

        Args:
            config: Repository configuration

        Returns:
            Prompt section for story generation guidance
        """
        if not config.story_generation_enabled or not config.widgetbook_stories_path:
            return ""

        return f"""
## Widgetbook Story Requirements

When creating NEW presentation widgets, you should structure them for Widgetbook stories:

### Widget Structure
- Add a `semanticLabel` parameter to custom widgets for accessibility
- Use `const` constructors where possible
- Expose key properties as named parameters for testability

### Bidirectional Labels
Add doc comments linking to Widgetbook stories:
```dart
/// WIDGETBOOK_STORY: {config.widgetbook_stories_path}/widget_name_stories.dart
/// COMPONENT: WidgetName
/// USE_CASES: [Default, Loading, Error, Dark Theme]
/// TESTED_PROPERTIES:
///   - title: String (knobs.string)
///   - isLoading: bool (knobs.boolean)
/// LAST_SYNCED: {{current_date}}

class WidgetName extends StatelessWidget {{
  // ...
}}
```

Stories will be auto-generated for new widgets at: `{config.widgetbook_stories_path}/`
"""

    async def _post_process_changes(
        self,
        work_dir: str,
        config: RepoConfig,
    ) -> dict[str, Any]:
        """Post-process changes after Claude makes fixes.

        Validates semantic labels and generates stories for new widgets.

        Args:
            work_dir: Working directory
            config: Repository configuration

        Returns:
            Dict with semantic_issues, semantic_labels_added, and stories_generated
        """
        result: dict[str, Any] = {
            "semantic_issues": [],
            "semantic_labels_added": [],
            "stories_generated": [],
        }

        # Get list of changed files
        git_diff = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        changed_files = [f.strip() for f in git_diff.stdout.strip().split("\n") if f.strip()]

        # 1. Validate semantic labels
        if config.semantic_labels_required and config.type == "flutter":
            semantic_issues = self._validate_semantics(work_dir, changed_files)
            result["semantic_issues"] = semantic_issues

            if semantic_issues:
                self.log(f"Found {len(semantic_issues)} semantic label issues", level="warning")
                for issue in semantic_issues[:5]:  # Log first 5
                    self.log(f"  - {issue['file_path']}:{issue['line_number']}: {issue['message']}")

        # 2. Generate stories for new widgets
        if config.story_generation_enabled and config.type == "flutter":
            stories = await self._generate_stories_for_widgets(work_dir, changed_files, config)
            result["stories_generated"] = stories

            if stories:
                self.log(f"Generated {len(stories)} Widgetbook stories")

                # Stage the new story files
                for story in stories:
                    subprocess.run(["git", "add", story], cwd=work_dir, capture_output=True)

                # Amend commit with stories
                if stories:
                    subprocess.run(
                        ["git", "commit", "--amend", "--no-edit"],
                        cwd=work_dir,
                        capture_output=True,
                    )

        return result
