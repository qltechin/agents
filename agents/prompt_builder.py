"""
Prompt Builder Module for Builder Agent.

Handles dynamic prompt construction with:
- Repo style detection (CLAUDE.md + auto-analyze fallback)
- YAML-based persona and checklist injection
- Dynamic pattern analysis (read READMEs, analyze existing tests/code)
- Subtask context awareness
- Complexity-based instructions

Issue: #257
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Path to prompts directory
PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class RepoStyle:
    """Detected or configured style patterns for a repository."""

    # Source of style info
    source: str = "default"  # "claude_md", "auto_analyzed", "default"

    # Test patterns
    test_use_fixtures: bool = False
    test_data_from_json: bool = False
    test_logging_with_markers: bool = False
    test_class_based: bool = True

    # Code patterns
    docstring_format: str = "google"  # google, numpy, sphinx
    error_handling_style: str = "exceptions"  # exceptions, result_types, error_codes
    naming_convention: str = "snake_case"  # snake_case, camelCase

    # Raw style guide text (from CLAUDE.md or auto-analysis)
    raw_style_guide: str = ""

    def to_prompt_section(self) -> str:
        """Convert style to prompt section text."""
        if self.raw_style_guide:
            return f"""## Repository Style Guide
{self.raw_style_guide}

Follow these patterns when writing code and tests."""

        # Fallback to structured style
        return f"""## Repository Style Guide
- Test style: {"Class-based" if self.test_class_based else "Function-based"}
- Use fixtures: {"Yes" if self.test_use_fixtures else "No"}
- Docstring format: {self.docstring_format}
- Naming convention: {self.naming_convention}

Follow these patterns when writing code and tests."""


@dataclass
class SubtaskContext:
    """Context for subtask issues (child of a parent issue)."""

    is_subtask: bool = False
    parent_issue_number: int | None = None
    parent_issue_title: str = ""
    iteration_number: int = 1
    total_subtasks: int = 0
    sibling_subtasks: list[dict] = field(default_factory=list)
    completed_work_summary: str = ""

    def to_prompt_section(self) -> str:
        """Convert subtask context to prompt section text."""
        if not self.is_subtask:
            return ""

        siblings_text = ""
        for subtask in self.sibling_subtasks:
            status_marker = "Complete" if subtask.get("completed") else "Pending"
            if subtask.get("current"):
                status_marker = "In Progress - YOU ARE HERE"
            siblings_text += f"- #{subtask['number']}: [{status_marker}] {subtask['title']}\n"

        return f"""## Subtask Context
This is subtask {self.iteration_number} of {self.total_subtasks} for parent issue #{self.parent_issue_number}.

**Parent Issue:** {self.parent_issue_title}

**Subtasks:**
{siblings_text}
**Completed Work Summary:**
{self.completed_work_summary if self.completed_work_summary else "No previous work completed yet."}

Focus only on your specific subtask. Do not duplicate work from other subtasks."""


class RepoStyleAnalyzer:
    """
    Analyzes repository to detect coding style and conventions.

    Uses hybrid approach:
    1. Check for CLAUDE.md - use if exists (preferred)
    2. If missing/incomplete, auto-analyze via Claude
    3. Cache results for future runs
    """

    # Files to look for style documentation
    STYLE_DOC_FILES = ["CLAUDE.md", ".claude/CLAUDE.md", "docs/CLAUDE.md"]

    def __init__(self, cache_dir: str | None = None):
        """Initialize analyzer with optional cache directory."""
        self.cache_dir = cache_dir or "/tmp/repo-style-cache"
        self._style_cache: dict[str, RepoStyle] = {}

    def get_style(self, repo_path: str, repo_name: str = "") -> RepoStyle:
        """
        Get style for a repository.

        Args:
            repo_path: Local path to cloned repository
            repo_name: Repository name for caching (e.g., "owner/repo-name")

        Returns:
            RepoStyle with detected or default patterns
        """
        # Check cache first
        cache_key = repo_name or repo_path
        if cache_key in self._style_cache:
            logger.debug(f"[STYLE] Using cached style for {cache_key}")
            return self._style_cache[cache_key]

        # Try to find CLAUDE.md
        style = self._try_load_claude_md(repo_path)
        if style:
            logger.info(f"[STYLE] Loaded style from CLAUDE.md for {cache_key}")
            self._style_cache[cache_key] = style
            return style

        # Fallback to default style
        logger.info(f"[STYLE] No CLAUDE.md found, using default style for {cache_key}")
        style = RepoStyle(source="default")
        self._style_cache[cache_key] = style
        return style

    def _try_load_claude_md(self, repo_path: str) -> RepoStyle | None:
        """Try to load style from CLAUDE.md file."""
        for doc_file in self.STYLE_DOC_FILES:
            doc_path = Path(repo_path) / doc_file
            if doc_path.exists():
                try:
                    content = doc_path.read_text()
                    return self._parse_claude_md(content)
                except Exception as e:
                    logger.warning(f"[STYLE] Failed to parse {doc_path}: {e}")
        return None

    def _parse_claude_md(self, content: str) -> RepoStyle:
        """Parse CLAUDE.md content into RepoStyle."""
        # Extract relevant sections for the style guide
        # Keep the raw content for now - Claude will understand it
        return RepoStyle(
            source="claude_md",
            raw_style_guide=content,
        )

    async def auto_analyze(self, repo_path: str, claude_client: Any) -> RepoStyle:
        """
        Auto-analyze repository style using Claude.

        This is called when CLAUDE.md is missing or incomplete.

        Args:
            repo_path: Local path to cloned repository
            claude_client: Claude SDK client for analysis

        Returns:
            RepoStyle with auto-detected patterns
        """
        # Read sample files for analysis
        test_samples = self._read_sample_files(repo_path, "tests", limit=3)
        source_samples = self._read_sample_files(repo_path, "src", limit=3)

        if not test_samples and not source_samples:
            # Try alternative paths
            source_samples = self._read_sample_files(repo_path, "lib", limit=3)
            if not source_samples:
                source_samples = self._read_sample_files(repo_path, "agents", limit=3)

        if not test_samples and not source_samples:
            logger.warning("[STYLE] No sample files found for auto-analysis")
            return RepoStyle(source="default")

        analysis_prompt = f"""Analyze these code samples and extract the style patterns.

## Test Files
{test_samples if test_samples else "No test files found"}

## Source Files
{source_samples if source_samples else "No source files found"}

Extract and summarize:
1. Test structure (fixtures usage, class-based vs functions, assertion style)
2. Import patterns and organization
3. Logging style (markers, format)
4. Docstring format (Google, NumPy, etc.)
5. Error handling patterns
6. Naming conventions

Return as concise guidelines (5-10 bullet points) for writing new code in this repo.
Keep it brief and actionable."""

        try:
            # Use Claude to analyze
            response = await claude_client.send_message(analysis_prompt)
            style_guide = response.content if hasattr(response, "content") else str(response)

            return RepoStyle(
                source="auto_analyzed",
                raw_style_guide=style_guide,
            )
        except Exception as e:
            logger.error(f"[STYLE] Auto-analysis failed: {e}")
            return RepoStyle(source="default")

    def _read_sample_files(self, repo_path: str, subdir: str, limit: int = 3) -> str:
        """Read sample files from a directory."""
        dir_path = Path(repo_path) / subdir
        if not dir_path.exists():
            return ""

        samples = []
        count = 0

        for file_path in dir_path.rglob("*.py"):
            if count >= limit:
                break
            if "__pycache__" in str(file_path):
                continue
            try:
                content = file_path.read_text()
                # Limit content size
                if len(content) > 2000:
                    content = content[:2000] + "\n... (truncated)"
                samples.append(f"### {file_path.name}\n```python\n{content}\n```")
                count += 1
            except Exception:
                continue

        # Also check for Dart/Java files
        for ext in ["*.dart", "*.java"]:
            if count >= limit:
                break
            for file_path in dir_path.rglob(ext):
                if count >= limit:
                    break
                try:
                    content = file_path.read_text()
                    if len(content) > 2000:
                        content = content[:2000] + "\n... (truncated)"
                    lang = "dart" if ext == "*.dart" else "java"
                    samples.append(f"### {file_path.name}\n```{lang}\n{content}\n```")
                    count += 1
                except Exception:
                    continue

        return "\n\n".join(samples)


class PromptBuilder:
    """
    Builds dynamic prompts for the Builder Agent.

    Combines:
    - Base task prompt
    - Persona principles (from builder.yaml)
    - Quality checklist (from builder-quality.yaml)
    - Dynamic pattern analysis instructions
    - Repo style (from CLAUDE.md or auto-analysis)
    - Subtask context (if applicable)
    - Complexity-based instructions
    """

    def __init__(self):
        """Initialize prompt builder and load YAML configs."""
        self.persona = self._load_yaml("personas/builder.yaml")
        self.checklist = self._load_yaml("checklists/builder-quality.yaml")
        self.style_analyzer = RepoStyleAnalyzer()

    def _load_yaml(self, relative_path: str) -> dict:
        """Load a YAML file from the prompts directory."""
        file_path = PROMPTS_DIR / relative_path
        if not file_path.exists():
            logger.warning(f"[PROMPT] YAML file not found: {file_path}")
            return {}
        try:
            with open(file_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"[PROMPT] Failed to load {file_path}: {e}")
            return {}

    def build_prompt(
        self,
        issue: dict,
        config: Any,
        repo_path: str,
        subtask_context: SubtaskContext | None = None,
        complexity: str = "moderate",
    ) -> str:
        """
        Build the complete prompt for Claude.

        Args:
            issue: GitHub issue dict with number, title, body
            config: RepoConfig with type, language, repository, etc.
            repo_path: Local path to cloned repository
            subtask_context: Optional subtask context if this is a child issue
            complexity: Complexity level (simple, moderate, complex, epic)

        Returns:
            Complete prompt string for Claude
        """
        sections = []

        # 1. Role and principles from persona
        sections.append(self._build_role_section())

        # 2. Issue details
        sections.append(self._build_issue_section(issue, config))

        # 3. Repo style (from CLAUDE.md or auto-analysis)
        repo_style = self.style_analyzer.get_style(repo_path, config.repository)
        if repo_style.raw_style_guide or repo_style.source != "default":
            sections.append(repo_style.to_prompt_section())

        # 4. Pattern analysis instructions (replaces language-specific guidelines)
        sections.append(self._build_pattern_analysis_section())

        # 5. Subtask context (if applicable)
        if subtask_context and subtask_context.is_subtask:
            sections.append(subtask_context.to_prompt_section())

        # 6. Complexity-based instructions
        sections.append(self._build_complexity_section(complexity))

        # 7. Task instructions
        sections.append(self._build_task_section(config))

        # 8. Quality checklist summary
        sections.append(self._build_checklist_section())

        return "\n\n".join(sections)

    def _build_role_section(self) -> str:
        """Build the role/persona section."""
        principles = self.persona.get("principles", [])
        principles_text = "\n".join(f"- {p}" for p in principles)

        healthcare_context = self.persona.get("healthcare_context", "")

        return f"""## Your Role
You are an expert software developer fixing a GitHub issue.

### Principles
{principles_text}

### Healthcare Context
{healthcare_context}"""

    def _build_issue_section(self, issue: dict, config: Any) -> str:
        """Build the issue details section."""
        return f"""## Issue #{issue['number']}: {issue['title']}

{issue.get('body', 'No description provided.')}

## Project Information
- **Name**: {config.name}
- **Type**: {config.type}
- **Language**: {getattr(config, 'language', 'unknown')}
- **Repository**: {config.repository}"""

    def _build_pattern_analysis_section(self) -> str:
        """Build instructions for analyzing and following existing repo patterns."""
        return """## Before Writing Code or Tests

**IMPORTANT: Follow existing patterns in this repository.**

1. **Read README files** in any directories you're working in
2. **Analyze 2-3 existing test files** to understand the testing patterns:
   - Look at imports, fixtures, and test structure
   - Note logging patterns (markers, format)
   - Check if tests use JSON data files or inline data
   - See if tests are class-based or function-based
3. **Analyze 2-3 existing source files** in the same area:
   - Note docstring format and style
   - Observe error handling patterns
   - Check naming conventions
4. **Model your code after existing patterns** in this repo
5. **Only deviate from existing patterns** if you have a clearly better approach
   - If you do deviate, explain why in your PR description

This ensures consistency with the existing codebase."""

    def _build_complexity_section(self, complexity: str) -> str:
        """Build complexity-specific instructions."""
        complexity_handling = self.persona.get("complexity_handling", {})
        complexity_info = complexity_handling.get(complexity, {})

        if not complexity_info:
            return f"## Complexity: {complexity.title()}"

        approach = complexity_info.get("approach", "")
        description = complexity_info.get("description", "")

        return f"""## Complexity: {complexity.title()}
{description}

### Recommended Approach
{approach}"""

    def _build_task_section(self, config: Any) -> str:
        """Build the task instructions section."""
        guidelines = getattr(config, "guidelines", [])
        guidelines_text = "\n".join(f"- {g}" for g in guidelines) if guidelines else ""

        return f"""## Your Task
1. First, explore the codebase to understand the issue
2. Find the relevant files that need to be modified
3. Make the necessary code changes to fix the issue
4. Keep changes minimal and focused on the issue
5. Follow the repository's existing patterns and conventions

## Important Guidelines
{guidelines_text}
- Do NOT modify test files unless the issue specifically mentions tests
- Do NOT add new dependencies unless absolutely necessary
- Make sure your changes don't break existing functionality

Start by exploring the codebase structure, then find and fix the issue."""

    def _build_checklist_section(self) -> str:
        """Build the quality checklist summary section."""
        sections = self.checklist.get("sections", {})

        critical_items = []
        for section_data in sections.values():
            items = section_data.get("items", [])
            for item in items:
                if item.get("severity") == "critical":
                    critical_items.append(item.get("text", ""))

        if not critical_items:
            return ""

        items_text = "\n".join(f"- [ ] {item}" for item in critical_items[:8])

        return f"""## Before You Finish (Critical Checks)
{items_text}

Ensure all critical items pass before committing."""
