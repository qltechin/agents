"""Task Orchestrator - Cyclical execution for complex issue decomposition.

This module implements a simplified cyclical task orchestration pattern:
    RUN CLAUDE -> CHECK COMPLETION -> IF NOT COMPLETE: SPAWN SUBTASKS -> WAIT -> INTEGRATE -> LOOP

Key simplification: Instead of pre-analyzing complexity, we run Claude first
and then ask Claude if the issue is complete. This is reactive (based on actual
work done) rather than predictive (guessing complexity upfront).

For issues that can't be completed in one go, it decomposes them into smaller
subtasks, monitors completion, integrates results, and iterates until done.
"""

import json
import re
import subprocess
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from config.settings import Settings, get_settings

# IssueAnalyzer is NOT used - we use Claude directly for completion checks

# Optional persistent storage
try:
    from training.storage.orchestrator_store import OrchestratorState, OrchestratorStore

    STORE_AVAILABLE = True
except ImportError:
    STORE_AVAILABLE = False
    OrchestratorStore = None  # type: ignore
    OrchestratorState = None  # type: ignore


class SubTask(BaseModel):
    """A subtask derived from issue decomposition."""

    title: str = Field(description="Concise title for the subtask issue")
    description: str = Field(description="Detailed description of what needs to be done")
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Specific, testable criteria for completion",
    )
    estimated_files: int = Field(
        default=1,
        description="Estimated number of files to modify",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Titles of subtasks this depends on",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Suggested labels (api, frontend, database, testing, etc.)",
    )


class CompletionCheck(BaseModel):
    """Result of asking Claude if an issue is complete."""

    is_complete: bool = Field(description="Whether the issue is fully complete")
    reasoning: str = Field(description="Explanation of why complete or not")
    remaining_subtasks: list[SubTask] = Field(
        default_factory=list,
        description="If not complete, what subtasks remain",
    )
    files_changed: list[str] = Field(
        default_factory=list,
        description="Files that were changed in this cycle",
    )
    blockers: list[str] = Field(
        default_factory=list,
        description="Any blockers encountered",
    )


class OrchestratorStatus(str, Enum):
    """Status of the orchestrator for a parent task."""

    ANALYZING = "analyzing"
    SPAWNING = "spawning"
    AWAITING = "awaiting"
    INTEGRATING = "integrating"
    COMPLETE = "complete"
    FAILED = "failed"
    ESCALATED = "escalated"


class SubtaskStatus(BaseModel):
    """Status tracking for a spawned subtask."""

    issue_number: int = Field(description="GitHub issue number")
    title: str = Field(description="Subtask title")
    status: str = Field(default="open", description="open, closed, merged")
    pr_number: int | None = Field(default=None, description="Associated PR number if any")
    pr_url: str | None = Field(default=None, description="PR URL")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = Field(default=None)


class IterationState(BaseModel):
    """State for a single iteration cycle."""

    iteration_number: int = Field(description="Which iteration (1-based)")
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = Field(default=None)
    completion_check: CompletionCheck | None = Field(default=None, description="Claude's completion check result")
    subtasks: list[SubtaskStatus] = Field(default_factory=list)
    integration_pr: str | None = Field(default=None, description="Integration PR URL if created")
    notes: str = Field(default="", description="Notes about this iteration")
    files_changed: list[str] = Field(default_factory=list, description="Files modified in this iteration")


class ParentTaskState(BaseModel):
    """Complete state for a parent task being orchestrated."""

    parent_issue_number: int = Field(description="Original issue number")
    repository: str = Field(description="Repository in format owner/repo")
    original_issue_title: str = Field(description="Original issue title")
    original_issue_body: str = Field(description="Original issue body")

    iterations: list[IterationState] = Field(default_factory=list)
    current_iteration: int = Field(default=0, description="Current iteration number (0 = not started)")
    max_iterations: int = Field(default=5, description="Maximum iterations before escalation")

    status: OrchestratorStatus = Field(default=OrchestratorStatus.ANALYZING)
    integration_branch: str = Field(default="", description="Branch for integrating all changes")
    final_pr: str | None = Field(default=None, description="Final PR URL when complete")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Error tracking
    last_error: str | None = Field(default=None)
    error_count: int = Field(default=0)


class RoadblockDetector:
    """Detects when the agent should pause and request human help."""

    ROADBLOCK_PATTERNS = [
        r"I need more (information|context|details|clarification)",
        r"(don't|do not|cannot) have access",
        r"this requires (human|manual) (review|intervention|approval)",
        r"unclear (requirement|specification|what|how)",
        r"(conflicting|contradictory) (requirements|instructions)",
        r"security (concern|risk|implication)",
        r"(architectural|design) decision (needed|required)",
        r"permission (denied|required|needed)",
        r"(api|credentials|token) (missing|unavailable|required)",
    ]

    def __init__(self):
        self.detected: list[str] = []
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.ROADBLOCK_PATTERNS]

    def check(self, text: str) -> list[str]:
        """Check text for roadblock indicators.

        Args:
            text: Text to check (agent output, error messages, etc.)

        Returns:
            List of detected roadblock patterns
        """
        detected = []
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                detected.append(pattern.pattern)

        self.detected.extend(detected)
        return detected

    def should_pause(self, threshold: int = 2) -> bool:
        """Determine if enough roadblocks detected to pause.

        Args:
            threshold: Number of roadblocks before pausing

        Returns:
            True if should pause for human review
        """
        return len(self.detected) >= threshold

    def get_summary(self) -> str:
        """Get summary of detected roadblocks."""
        if not self.detected:
            return "No roadblocks detected"
        return f"Detected {len(self.detected)} roadblock(s): {', '.join(set(self.detected))}"

    def reset(self):
        """Reset detected roadblocks."""
        self.detected = []


class CyclicalTaskOrchestrator:
    """Orchestrates complex issues through iterative decomposition and integration.

    Simplified architecture:
    - NO pre-analysis of complexity
    - Runs Claude first, then asks Claude if complete
    - Reactive decomposition based on actual results
    - Uses Claude Code SDK for completion checks (same as builder)
    """

    # Labels used for orchestration
    LABEL_AWAITING_SUBTASKS = "awaiting-subtasks"
    LABEL_HUMAN_NEEDED = "human-needed"
    LABEL_DECOMPOSED = "decomposed"

    # Completion check prompt template
    COMPLETION_CHECK_PROMPT = """Given the original issue and the work done, determine if this issue is complete.

## Original Issue
**Title**: {issue_title}
**Body**:
{issue_body}

## Changes Made This Cycle
{changes_summary}

## Questions
1. Is this issue FULLY complete based on the requirements? (yes/no)
2. If not, what specific subtasks remain?

Respond in JSON format:
```json
{{
    "is_complete": true/false,
    "reasoning": "Why complete or not complete",
    "remaining_subtasks": [
        {{
            "title": "Subtask title",
            "description": "What needs to be done",
            "acceptance_criteria": ["Criterion 1", "Criterion 2"],
            "estimated_files": 1,
            "labels": ["api", "frontend", etc]
        }}
    ],
    "blockers": ["Any blockers encountered"]
}}
```

Rules:
- Set is_complete=true ONLY if ALL requirements are met
- If code was written but tests fail, is_complete=false
- If partial implementation, list remaining work as subtasks
- Keep subtasks focused (each should be one PR worth of work)
"""

    # Fallback prompt for when first attempt exhausts turns without returning JSON
    # Used with no tools to guarantee a text response
    COMPLETION_CHECK_FALLBACK_PROMPT = """Based on the summary below, determine if this issue is complete.

**Issue**: {issue_title}
**Changes Made**: {changes_summary}

You MUST respond with ONLY this JSON (no other text, no tool calls):
{{"is_complete": true, "reasoning": "brief reason why complete or not", "remaining_subtasks": [], "blockers": []}}

If incomplete, set is_complete to false and briefly explain why in reasoning.
"""

    def __init__(
        self,
        settings: Settings | None = None,
        store: "OrchestratorStore | None" = None,
        use_persistent_storage: bool = True,
    ):
        """Initialize the task orchestrator.

        Args:
            settings: Application settings
            store: Optional OrchestratorStore instance for persistent storage
            use_persistent_storage: Whether to use persistent storage (default True)
        """
        self.settings = settings or get_settings()
        self.roadblock_detector = RoadblockDetector()

        # Persistent storage (if available and enabled)
        self.store: OrchestratorStore | None = None
        if use_persistent_storage and STORE_AVAILABLE:
            try:
                self.store = store or OrchestratorStore()
                self._log("Using persistent storage for orchestrator state")
            except Exception as e:
                self._log(f"Failed to initialize persistent storage: {e}", level="warning")
                self.store = None

        # Fallback in-memory state store
        self.parents: dict[str, ParentTaskState] = {}

    def _get_parent_key(self, repository: str, issue_number: int) -> str:
        """Generate unique key for a parent task."""
        return f"{repository}#{issue_number}"

    def _persist_state(self, state: ParentTaskState) -> None:
        """Persist state to storage (if available).

        Args:
            state: The parent task state to persist
        """
        if not self.store or not STORE_AVAILABLE:
            return

        try:
            # Convert ParentTaskState to OrchestratorState for storage
            store_state = OrchestratorState(
                repository=state.repository,
                parent_issue_number=state.parent_issue_number,
                status=state.status.value,
                current_iteration=state.current_iteration,
                max_iterations=state.max_iterations,
                iterations=[iter.model_dump() for iter in state.iterations],
            )
            self.store.save_state(store_state)
            self._log(f"Persisted state for {state.repository}#{state.parent_issue_number}")
        except Exception as e:
            self._log(f"Failed to persist state: {e}", level="warning")

    def _load_state(self, repository: str, issue_number: int) -> ParentTaskState | None:
        """Load state from storage (if available).

        Args:
            repository: Repository in format 'owner/repo'
            issue_number: The parent issue number

        Returns:
            ParentTaskState if found, None otherwise
        """
        if not self.store or not STORE_AVAILABLE:
            return None

        try:
            store_state = self.store.load_state(repository, issue_number)
            if not store_state:
                return None

            # Convert OrchestratorState back to ParentTaskState
            state = ParentTaskState(
                parent_issue_number=store_state.parent_issue_number,
                repository=store_state.repository,
                original_issue_title="",  # Not stored, will need to fetch if needed
                original_issue_body="",
                status=OrchestratorStatus(store_state.status),
                current_iteration=store_state.current_iteration,
                max_iterations=store_state.max_iterations,
                iterations=[IterationState(**iter_data) for iter_data in store_state.iterations],
            )
            self._log(f"Loaded state for {repository}#{issue_number}")
            return state
        except Exception as e:
            self._log(f"Failed to load state: {e}", level="warning")
            return None

    def _delete_state(self, repository: str, issue_number: int) -> None:
        """Delete state from storage (if available).

        Args:
            repository: Repository in format 'owner/repo'
            issue_number: The parent issue number
        """
        if not self.store or not STORE_AVAILABLE:
            return

        try:
            self.store.delete_state(repository, issue_number)
            self._log(f"Deleted state for {repository}#{issue_number}")
        except Exception as e:
            self._log(f"Failed to delete state: {e}", level="warning")

    def _log(self, message: str, level: str = "info"):
        """Log a message (placeholder for actual logging)."""
        timestamp = datetime.now(UTC).isoformat()
        print(f"[{timestamp}] [orchestrator] [{level.upper()}] {message}")

    def _parse_completion_response(self, response_text: str) -> CompletionCheck | None:
        """Parse a completion check response from Claude.

        Args:
            response_text: The text response from Claude

        Returns:
            CompletionCheck if parsing succeeds, None if parsing fails
        """
        if not response_text.strip():
            self._log("[COMPLETION] No text response from Claude (only tool calls?)", level="warning")
            return None

        try:
            # Extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            # Validate json_str is not empty before parsing
            if not json_str:
                self._log("[COMPLETION] Extracted JSON string is empty", level="warning")
                return None

            data = json.loads(json_str)

            # Convert remaining_subtasks to SubTask objects
            subtasks = []
            for st_data in data.get("remaining_subtasks", []):
                subtasks.append(SubTask(**st_data))

            result = CompletionCheck(
                is_complete=data.get("is_complete", False),
                reasoning=data.get("reasoning", ""),
                remaining_subtasks=subtasks,
                blockers=data.get("blockers", []),
            )

            self._log(
                f"[COMPLETION] Result: is_complete={result.is_complete}, subtasks={len(result.remaining_subtasks)}"
            )
            return result

        except (json.JSONDecodeError, ValueError) as e:
            self._log(f"[COMPLETION] Failed to parse response: {e}", level="warning")
            return None

    async def check_completion_with_claude(
        self,
        issue_title: str,
        issue_body: str,
        changes_summary: str,
        work_dir: str,
    ) -> CompletionCheck:
        """Ask Claude if the issue is complete after running.

        Uses the same Claude Code SDK as the builder, avoiding separate
        Bedrock model configuration issues.

        Args:
            issue_title: Original issue title
            issue_body: Original issue body
            changes_summary: Summary of changes made (files, git diff stats)
            work_dir: Working directory for Claude to examine

        Returns:
            CompletionCheck with is_complete, reasoning, and any remaining subtasks
        """
        try:
            from claude_code_sdk import ClaudeCodeOptions, query
            from claude_code_sdk.types import AssistantMessage
        except ImportError:
            self._log("Claude Code SDK not available, assuming incomplete", level="warning")
            return CompletionCheck(
                is_complete=False,
                reasoning="Claude Code SDK not available for completion check",
                blockers=["SDK not installed"],
            )

        prompt = self.COMPLETION_CHECK_PROMPT.format(
            issue_title=issue_title,
            issue_body=issue_body[:2000],  # Truncate long bodies
            changes_summary=changes_summary,
        )

        options = ClaudeCodeOptions(
            allowed_tools=["Read", "Glob", "Grep", "Bash"],  # Read-only for checking
            permission_mode="bypassPermissions",
            cwd=work_dir,
            max_turns=50,  # Complex assessment task - needs room for file exploration + JSON response
        )

        self._log("[COMPLETION] Asking Claude if issue is complete...")

        response_text = ""
        try:
            async for message in query(prompt=prompt, options=options):
                # Use isinstance() to check message type - AssistantMessage has no 'type' attribute
                if isinstance(message, AssistantMessage):
                    # AssistantMessage has 'content' directly, not 'message.content'
                    for block in message.content:
                        if hasattr(block, "text"):
                            response_text += block.text
        except Exception as e:
            self._log(f"[COMPLETION] Error querying Claude: {e}", level="error")
            return CompletionCheck(
                is_complete=False,
                reasoning=f"Error during completion check: {e}",
                blockers=[str(e)],
            )

        # Try to parse the response
        result = self._parse_completion_response(response_text)
        if result is not None:
            return result

        # Fallback: If first attempt failed (no JSON), retry with no tools to force text response
        self._log("[COMPLETION] First attempt failed to return JSON, retrying with fallback (no tools)...")

        fallback_prompt = self.COMPLETION_CHECK_FALLBACK_PROMPT.format(
            issue_title=issue_title,
            changes_summary=changes_summary,
        )

        fallback_options = ClaudeCodeOptions(
            allowed_tools=[],  # No tools - forces immediate text response
            permission_mode="bypassPermissions",
            cwd=work_dir,
            max_turns=3,  # Minimal turns since no exploration needed
        )

        fallback_response = ""
        try:
            async for message in query(prompt=fallback_prompt, options=fallback_options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if hasattr(block, "text"):
                            fallback_response += block.text
        except Exception as e:
            self._log(f"[COMPLETION] Fallback query failed: {e}", level="error")
            return CompletionCheck(
                is_complete=False,
                reasoning=f"Both completion check attempts failed: {e}",
                blockers=[str(e)],
            )

        # Try to parse fallback response
        result = self._parse_completion_response(fallback_response)
        if result is not None:
            return result

        # Both attempts failed
        self._log("[COMPLETION] Both attempts failed to return valid JSON", level="error")
        return CompletionCheck(
            is_complete=False,
            reasoning="Failed to get valid completion check response after retry",
            blockers=["No valid JSON response from either attempt"],
        )

    async def after_claude_run(
        self,
        repository: str,
        issue_number: int,
        issue_title: str,
        issue_body: str,
        work_dir: str,
        files_changed: list[str],
        pr_created: bool = False,
        pr_url: str | None = None,
    ) -> dict[str, Any]:
        """Called after Claude runs to check if issue is complete.

        This is the simplified flow:
        1. Claude just ran on the issue
        2. We check if the issue is complete
        3. If not, we spawn subtasks for remaining work
        4. Return status for builder to handle

        Args:
            repository: Repository in format owner/repo
            issue_number: Issue number
            issue_title: Issue title
            issue_body: Issue body
            work_dir: Working directory where Claude made changes
            files_changed: List of files that were changed
            pr_created: Whether a PR was already created
            pr_url: URL of the PR if created

        Returns:
            Status dict with completion result and any subtasks spawned
        """
        parent_key = self._get_parent_key(repository, issue_number)
        self._log(f"[AFTER-RUN] Checking completion for {parent_key}")

        # Build changes summary for Claude
        changes_summary = f"Files changed: {len(files_changed)}\n"
        for f in files_changed[:10]:
            changes_summary += f"  - {f}\n"
        if len(files_changed) > 10:
            changes_summary += f"  ... and {len(files_changed) - 10} more\n"

        if pr_created:
            changes_summary += f"\nPR created: {pr_url}\n"

        # Get git diff stats
        try:
            diff_stat = subprocess.run(
                ["git", "diff", "--stat", "HEAD~1", "HEAD"],
                cwd=work_dir,
                capture_output=True,
                text=True,
            )
            if diff_stat.returncode == 0:
                changes_summary += f"\nDiff stats:\n{diff_stat.stdout[:500]}"
        except Exception:
            pass

        # Ask Claude if complete
        completion = await self.check_completion_with_claude(
            issue_title=issue_title,
            issue_body=issue_body,
            changes_summary=changes_summary,
            work_dir=work_dir,
        )

        if completion.is_complete:
            self._log(f"[AFTER-RUN] Issue is COMPLETE: {completion.reasoning}")
            return {
                "status": "complete",
                "reasoning": completion.reasoning,
                "files_changed": files_changed,
            }

        # Not complete - check if we should spawn subtasks
        if completion.remaining_subtasks:
            self._log(f"[AFTER-RUN] Issue NOT complete, {len(completion.remaining_subtasks)} subtasks needed")

            # Initialize orchestration state
            state = await self.start_orchestration(
                repository=repository,
                issue_number=issue_number,
                issue_title=issue_title,
                issue_body=issue_body,
            )

            # Create iteration for tracking
            iteration = IterationState(iteration_number=state.current_iteration + 1)
            iteration.completion_check = completion
            iteration.files_changed = files_changed
            state.iterations.append(iteration)
            state.current_iteration += 1

            # Spawn subtasks
            spawned = await self._create_subtasks(state, completion.remaining_subtasks)
            iteration.subtasks = spawned

            if spawned:
                state.status = OrchestratorStatus.AWAITING
                # Persist state after spawning subtasks
                self._persist_state(state)
                return {
                    "status": "awaiting_subtasks",
                    "subtasks_created": len(spawned),
                    "subtasks": [{"issue_number": s.issue_number, "title": s.title} for s in spawned],
                    "reasoning": completion.reasoning,
                }

        # No subtasks but incomplete - escalate
        if completion.blockers:
            self._log(f"[AFTER-RUN] Blockers found: {completion.blockers}", level="warning")

        return {
            "status": "incomplete",
            "reasoning": completion.reasoning,
            "blockers": completion.blockers,
            "needs_human": True,
        }

    async def start_orchestration(
        self,
        repository: str,
        issue_number: int,
        issue_title: str,
        issue_body: str,
        max_iterations: int = 5,
    ) -> ParentTaskState:
        """Start orchestrating a complex issue.

        Args:
            repository: Repository in format owner/repo
            issue_number: Issue number
            issue_title: Issue title
            issue_body: Issue body/description
            max_iterations: Maximum iterations before escalation

        Returns:
            Initial ParentTaskState
        """
        parent_key = self._get_parent_key(repository, issue_number)

        # Check in-memory cache first
        if parent_key in self.parents:
            self._log(f"Resuming existing orchestration for {parent_key} (from cache)")
            return self.parents[parent_key]

        # Try to load from persistent storage
        persisted_state = self._load_state(repository, issue_number)
        if persisted_state:
            # Update with current issue info (may have changed)
            persisted_state.original_issue_title = issue_title
            persisted_state.original_issue_body = issue_body
            self.parents[parent_key] = persisted_state
            self._log(f"Resuming existing orchestration for {parent_key} (from storage)")
            return persisted_state

        # Create new state
        state = ParentTaskState(
            parent_issue_number=issue_number,
            repository=repository,
            original_issue_title=issue_title,
            original_issue_body=issue_body,
            max_iterations=max_iterations,
            integration_branch=f"ai/orchestrate-issue-{issue_number}",
        )

        self.parents[parent_key] = state
        self._persist_state(state)
        self._log(f"Started orchestration for {parent_key}")

        return state

    async def process_parent(self, parent_key: str, work_dir: str | None = None) -> dict[str, Any]:
        """Process a parent task through one cycle (after subtasks complete).

        Called to re-check completion after subtasks are done.

        Args:
            parent_key: Key identifying the parent task
            work_dir: Working directory for completion check (optional)

        Returns:
            Status dict with current state and next action needed
        """
        # Try to load from storage if not in memory
        if parent_key not in self.parents:
            # Parse repository and issue number from key
            if "#" in parent_key:
                repository, issue_str = parent_key.rsplit("#", 1)
                try:
                    issue_number = int(issue_str)
                    loaded_state = self._load_state(repository, issue_number)
                    if loaded_state:
                        self.parents[parent_key] = loaded_state
                except ValueError:
                    pass

        if parent_key not in self.parents:
            return {"error": f"Unknown parent task: {parent_key}"}

        state = self.parents[parent_key]
        state.updated_at = datetime.now(UTC)

        # Check if max iterations reached
        if state.current_iteration >= state.max_iterations:
            self._log(f"Max iterations ({state.max_iterations}) reached for {parent_key}")
            state.status = OrchestratorStatus.ESCALATED
            self._persist_state(state)
            await self._escalate_to_human(state, "Maximum iterations reached without completion")
            return {
                "status": "escalated",
                "reason": "max_iterations_reached",
                "iterations": state.current_iteration,
            }

        self._log(f"[PROCESS] Iteration {state.current_iteration} for {parent_key}")

        # If we have a work_dir, do completion check
        if work_dir:
            iteration_context = self._build_iteration_context(state)
            changes_summary = f"Previous iterations:\n{iteration_context}"

            completion = await self.check_completion_with_claude(
                issue_title=state.original_issue_title,
                issue_body=state.original_issue_body,
                changes_summary=changes_summary,
                work_dir=work_dir,
            )

            if completion.is_complete:
                state.status = OrchestratorStatus.COMPLETE
                self._persist_state(state)
                await self._finalize_parent(state)
                # Clean up state after completion
                self._delete_state(state.repository, state.parent_issue_number)
                return {
                    "status": "complete",
                    "iterations": state.current_iteration,
                    "final_pr": state.final_pr,
                }

            # Not complete - spawn more subtasks if suggested
            if completion.remaining_subtasks:
                state.current_iteration += 1
                iteration = IterationState(iteration_number=state.current_iteration)
                iteration.completion_check = completion
                state.iterations.append(iteration)

                state.status = OrchestratorStatus.SPAWNING
                spawned_subtasks = await self._create_subtasks(state, completion.remaining_subtasks)
                iteration.subtasks = spawned_subtasks
                state.status = OrchestratorStatus.AWAITING
                self._persist_state(state)

                return {
                    "status": "awaiting",
                    "iteration": state.current_iteration,
                    "subtasks_created": len(spawned_subtasks),
                    "subtasks": [{"issue_number": s.issue_number, "title": s.title} for s in spawned_subtasks],
                }

            # No subtasks but incomplete - escalate
            if completion.blockers:
                state.status = OrchestratorStatus.ESCALATED
                self._persist_state(state)
                await self._escalate_to_human(state, f"Blockers: {', '.join(completion.blockers)}")
                return {
                    "status": "escalated",
                    "reason": "blockers_detected",
                    "blockers": completion.blockers,
                }

        # Return current state
        return {
            "status": state.status.value,
            "iteration": state.current_iteration,
        }

    async def check_and_continue(self, parent_key: str) -> dict[str, Any]:
        """Check subtask status and continue if ready.

        Called periodically to check if awaiting subtasks are complete.
        If complete, integrates them and loops back to analyze.

        Args:
            parent_key: Key identifying the parent task

        Returns:
            Status dict with next action
        """
        if parent_key not in self.parents:
            return {"error": f"Unknown parent task: {parent_key}"}

        state = self.parents[parent_key]

        if state.status != OrchestratorStatus.AWAITING:
            return {"status": state.status.value, "message": "Not in awaiting state"}

        # Check current iteration's subtasks
        if not state.iterations:
            return {"error": "No iterations found"}

        current_iteration = state.iterations[-1]
        all_complete = await self._check_subtasks_complete(state, current_iteration)

        if not all_complete:
            incomplete = [s for s in current_iteration.subtasks if s.status == "open"]
            return {
                "status": "awaiting",
                "incomplete_subtasks": len(incomplete),
                "subtasks": [{"number": s.issue_number, "title": s.title} for s in incomplete],
            }

        # All subtasks complete - integrate
        state.status = OrchestratorStatus.INTEGRATING
        current_iteration.completed_at = datetime.now(UTC)

        integration_result = await self._integrate_iteration(state, current_iteration)

        if integration_result.get("error"):
            state.last_error = integration_result["error"]
            state.error_count += 1
            return integration_result

        # Loop back to analyze for next iteration
        state.status = OrchestratorStatus.ANALYZING
        return await self.process_parent(parent_key)

    def _build_iteration_context(self, state: ParentTaskState) -> str:
        """Build context string from previous iterations.

        Args:
            state: Parent task state

        Returns:
            Formatted context string for LLM
        """
        if not state.iterations or len(state.iterations) <= 1:
            return ""

        context_parts = []
        for iteration in state.iterations[:-1]:  # Exclude current iteration
            context_parts.append(f"""
### Iteration {iteration.iteration_number}
**Files Changed**: {len(iteration.files_changed)}
**Subtasks Created**: {len(iteration.subtasks)}
**Subtasks Completed**: {len([s for s in iteration.subtasks if s.status == 'closed' or s.pr_number])}
""")

            if iteration.completion_check:
                context_parts.append(f"**Reasoning**: {iteration.completion_check.reasoning[:200]}...")

            if iteration.subtasks:
                context_parts.append("**Subtask Details**:")
                for subtask in iteration.subtasks:
                    status = "completed" if subtask.pr_number else subtask.status
                    context_parts.append(f"- {subtask.title}: {status}")

            if iteration.integration_pr:
                context_parts.append(f"**Integration PR**: {iteration.integration_pr}")

        return "\n".join(context_parts)

    async def _create_subtasks(
        self,
        state: ParentTaskState,
        subtasks: list[SubTask],
    ) -> list[SubtaskStatus]:
        """Create GitHub issues for subtasks.

        Args:
            state: Parent task state
            subtasks: List of subtasks from analysis

        Returns:
            List of created SubtaskStatus
        """
        created: list[SubtaskStatus] = []

        for subtask in subtasks:
            # Build issue body
            body = f"""## Parent Issue
This is a subtask of #{state.parent_issue_number}: {state.original_issue_title}

## Description
{subtask.description}

## Acceptance Criteria
"""
            for criterion in subtask.acceptance_criteria:
                body += f"- [ ] {criterion}\n"

            if subtask.dependencies:
                body += "\n## Dependencies\n"
                for dep in subtask.dependencies:
                    body += f"- {dep}\n"

            body += f"\n---\n_Auto-generated by Task Orchestrator (Iteration {state.current_iteration})_"

            # Create the issue via gh CLI
            try:
                # DO NOT add 'aibuild' label to subtasks!
                # Adding 'aibuild' causes a cascade: builder picks up subtask -> creates more subtasks
                # -> builder picks those up -> infinite loop (see issue #256)
                # Subtasks are tracked by the orchestrator and processed when parent completes.
                result = subprocess.run(
                    [
                        "gh",
                        "issue",
                        "create",
                        "--repo",
                        state.repository,
                        "--title",
                        subtask.title,
                        "--body",
                        body,
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    # Parse issue URL to get number
                    issue_url = result.stdout.strip()
                    issue_number = int(issue_url.split("/")[-1])

                    created.append(
                        SubtaskStatus(
                            issue_number=issue_number,
                            title=subtask.title,
                        )
                    )
                    self._log(f"Created subtask #{issue_number}: {subtask.title}")
                else:
                    self._log(f"Failed to create subtask: {result.stderr}", level="error")

            except Exception as e:
                self._log(f"Error creating subtask: {e}", level="error")

        # Update parent issue with decomposed label
        if created:
            try:
                subprocess.run(
                    [
                        "gh",
                        "issue",
                        "edit",
                        str(state.parent_issue_number),
                        "--repo",
                        state.repository,
                        "--add-label",
                        f"{self.LABEL_DECOMPOSED},{self.LABEL_AWAITING_SUBTASKS}",
                        "--remove-label",
                        "aibuild",
                    ],
                    capture_output=True,
                )

                # Add comment listing subtasks
                comment_body = f"""## Task Decomposition (Iteration {state.current_iteration})

This issue has been decomposed into {len(created)} subtasks:

"""
                for subtask in created:
                    comment_body += f"- [ ] #{subtask.issue_number}: {subtask.title}\n"

                comment_body += "\n_Progress will be tracked automatically. Once all subtasks are complete, they will be integrated._"

                subprocess.run(
                    [
                        "gh",
                        "issue",
                        "comment",
                        str(state.parent_issue_number),
                        "--repo",
                        state.repository,
                        "--body",
                        comment_body,
                    ],
                    capture_output=True,
                )

            except Exception as e:
                self._log(f"Error updating parent issue: {e}", level="warning")

        return created

    async def _check_subtasks_complete(
        self,
        state: ParentTaskState,
        iteration: IterationState,
    ) -> bool:
        """Check if all subtasks in an iteration are complete.

        Args:
            state: Parent task state
            iteration: Current iteration

        Returns:
            True if all subtasks are complete
        """
        all_complete = True

        for subtask in iteration.subtasks:
            try:
                result = subprocess.run(
                    [
                        "gh",
                        "issue",
                        "view",
                        str(subtask.issue_number),
                        "--repo",
                        state.repository,
                        "--json",
                        "state,closedAt",
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    issue_data = json.loads(result.stdout)
                    if issue_data.get("state") == "CLOSED":
                        subtask.status = "closed"
                        if issue_data.get("closedAt"):
                            subtask.completed_at = datetime.fromisoformat(issue_data["closedAt"].replace("Z", "+00:00"))

                        # Check for associated PR
                        pr_result = subprocess.run(
                            [
                                "gh",
                                "pr",
                                "list",
                                "--repo",
                                state.repository,
                                "--search",
                                f"#{subtask.issue_number}",
                                "--json",
                                "number,url,state",
                            ],
                            capture_output=True,
                            text=True,
                        )
                        if pr_result.returncode == 0:
                            prs = json.loads(pr_result.stdout)
                            for pr in prs:
                                if pr.get("state") == "MERGED":
                                    subtask.pr_number = pr["number"]
                                    subtask.pr_url = pr["url"]
                                    break
                    else:
                        all_complete = False

            except Exception as e:
                self._log(f"Error checking subtask #{subtask.issue_number}: {e}", level="warning")
                all_complete = False

        return all_complete

    async def _integrate_iteration(
        self,
        state: ParentTaskState,
        iteration: IterationState,
    ) -> dict[str, Any]:
        """Integrate completed subtasks from an iteration.

        Args:
            state: Parent task state
            iteration: Completed iteration

        Returns:
            Integration result dict
        """
        self._log(
            f"Integrating iteration {iteration.iteration_number} for {state.repository}#{state.parent_issue_number}"
        )

        # Update parent issue comment with completion status
        try:
            comment_body = f"""## Iteration {iteration.iteration_number} Complete

All subtasks have been completed:

"""
            for subtask in iteration.subtasks:
                status = "merged" if subtask.pr_number else "closed"
                pr_link = f" ([PR #{subtask.pr_number}]({subtask.pr_url}))" if subtask.pr_url else ""
                comment_body += f"- [x] #{subtask.issue_number}: {subtask.title} - {status}{pr_link}\n"

            comment_body += "\n_Re-analyzing to determine if more work is needed..._"

            subprocess.run(
                [
                    "gh",
                    "issue",
                    "comment",
                    str(state.parent_issue_number),
                    "--repo",
                    state.repository,
                    "--body",
                    comment_body,
                ],
                capture_output=True,
            )

            # Remove awaiting label
            subprocess.run(
                [
                    "gh",
                    "issue",
                    "edit",
                    str(state.parent_issue_number),
                    "--repo",
                    state.repository,
                    "--remove-label",
                    self.LABEL_AWAITING_SUBTASKS,
                ],
                capture_output=True,
            )

        except Exception as e:
            self._log(f"Error updating parent issue: {e}", level="warning")

        return {"success": True, "integrated_subtasks": len(iteration.subtasks)}

    async def _finalize_parent(self, state: ParentTaskState) -> dict[str, Any]:
        """Finalize a completed parent task.

        Args:
            state: Parent task state

        Returns:
            Finalization result
        """
        self._log(f"Finalizing {state.repository}#{state.parent_issue_number}")

        try:
            # Add completion comment
            total_subtasks = sum(len(i.subtasks) for i in state.iterations)
            comment_body = f"""## Task Complete

All work has been completed after {state.current_iteration} iteration(s).

**Summary**:
- Total subtasks created: {total_subtasks}
- Iterations required: {state.current_iteration}

The issue can now be closed.
"""

            subprocess.run(
                [
                    "gh",
                    "issue",
                    "comment",
                    str(state.parent_issue_number),
                    "--repo",
                    state.repository,
                    "--body",
                    comment_body,
                ],
                capture_output=True,
            )

            # Update labels
            subprocess.run(
                [
                    "gh",
                    "issue",
                    "edit",
                    str(state.parent_issue_number),
                    "--repo",
                    state.repository,
                    "--add-label",
                    "aicomplete",
                    "--remove-label",
                    f"{self.LABEL_DECOMPOSED},{self.LABEL_AWAITING_SUBTASKS}",
                ],
                capture_output=True,
            )

        except Exception as e:
            self._log(f"Error finalizing: {e}", level="warning")

        return {"success": True}

    async def _escalate_to_human(self, state: ParentTaskState, reason: str) -> None:
        """Escalate a task to human review.

        Args:
            state: Parent task state
            reason: Reason for escalation
        """
        self._log(f"Escalating {state.repository}#{state.parent_issue_number}: {reason}", level="warning")

        try:
            comment_body = f"""## Human Review Required

The AI agent has paused work on this issue and requests human intervention.

**Reason**: {reason}

**Context**:
- Iterations attempted: {state.current_iteration}
- Last status: {state.status.value}

Please review and either:
1. Clarify requirements and remove the `{self.LABEL_HUMAN_NEEDED}` label to resume
2. Take over the implementation manually

_To resume AI work, remove the `{self.LABEL_HUMAN_NEEDED}` label and add `aibuild`._
"""

            subprocess.run(
                [
                    "gh",
                    "issue",
                    "comment",
                    str(state.parent_issue_number),
                    "--repo",
                    state.repository,
                    "--body",
                    comment_body,
                ],
                capture_output=True,
            )

            subprocess.run(
                [
                    "gh",
                    "issue",
                    "edit",
                    str(state.parent_issue_number),
                    "--repo",
                    state.repository,
                    "--add-label",
                    self.LABEL_HUMAN_NEEDED,
                    "--remove-label",
                    f"aibuild,{self.LABEL_AWAITING_SUBTASKS}",
                ],
                capture_output=True,
            )

        except Exception as e:
            self._log(f"Error escalating: {e}", level="error")


def create_orchestrator(settings: Settings | None = None) -> CyclicalTaskOrchestrator:
    """Factory function to create a CyclicalTaskOrchestrator.

    Args:
        settings: Optional settings override

    Returns:
        Configured orchestrator instance
    """
    return CyclicalTaskOrchestrator(settings=settings)
