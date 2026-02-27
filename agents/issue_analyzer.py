"""Issue Analyzer - LLM-based complexity analysis for intelligent task orchestration.

This module analyzes GitHub issues to determine complexity, suggest decomposition
into subtasks, and detect whether an issue requires human intervention.
"""

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from config.settings import Settings, get_settings


class IssueComplexity(str, Enum):
    """Complexity levels for issue classification."""

    SIMPLE = "simple"  # Single file, minor change, clear fix
    MODERATE = "moderate"  # Multiple files, clear approach, 1-2 hours
    COMPLEX = "complex"  # Multiple components, requires design decisions
    EPIC = "epic"  # Major feature, needs decomposition into subtasks


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


class IssueAnalysis(BaseModel):
    """Result of LLM-based issue analysis."""

    complexity: IssueComplexity = Field(
        description="Determined complexity level",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis (0-1)",
    )
    is_complete: bool = Field(
        default=False,
        description="Whether the issue appears to be complete (for re-analysis after iterations)",
    )
    reasoning: str = Field(
        description="Explanation of why this complexity was assigned",
    )
    estimated_files: int = Field(
        description="Estimated number of files that will need modification",
    )
    requires_new_files: bool = Field(
        description="Whether new files need to be created",
    )
    suggested_approach: str = Field(
        description="Recommended approach to implementing this issue",
    )
    subtasks: list[SubTask] = Field(
        default_factory=list,
        description="Suggested subtasks for complex/epic issues",
    )
    should_escalate: bool = Field(
        default=False,
        description="Whether this should be escalated to human review",
    )
    escalation_reason: str | None = Field(
        default=None,
        description="Reason for escalation if should_escalate is True",
    )
    roadblocks: list[str] = Field(
        default_factory=list,
        description="Identified blockers or unclear requirements",
    )


class IssueAnalyzer:
    """Analyzes GitHub issues using LLM to determine complexity and suggest decomposition."""

    def __init__(self, settings: Settings | None = None):
        """Initialize the issue analyzer."""
        self.settings = settings or get_settings()
        self._llm = None

    def _get_llm(self):
        """Lazily initialize and return the LLM client."""
        if self._llm is not None:
            return self._llm

        from tools.llm_providers import get_llm_client

        self._llm = get_llm_client(
            self.settings,
            temperature=0.1,
            max_tokens=4096,
        )
        return self._llm

    def _build_analysis_prompt(
        self,
        issue: dict[str, Any],
        iteration_context: str = "",
        repo_structure: str = "",
    ) -> str:
        """Build the prompt for issue analysis."""
        title = issue.get("title", "")
        body = issue.get("body", "")
        labels = issue.get("labels", [])
        if isinstance(labels, list) and labels and isinstance(labels[0], dict):
            labels = [label.get("name", "") for label in labels]

        prompt = f"""You are an expert software architect analyzing a GitHub issue to determine its complexity and implementation approach.

## Issue Details
**Title**: {title}
**Labels**: {', '.join(labels) if labels else 'None'}

**Description**:
{body}

"""

        if repo_structure:
            prompt += f"""## Repository Structure
{repo_structure}

"""

        if iteration_context:
            prompt += f"""## Previous Iteration Context
{iteration_context}

Based on the work already completed, analyze what remains to be done.

"""

        prompt += """## Your Task

Analyze this issue and provide a JSON response with the following structure:

```json
{
    "complexity": "simple|moderate|complex|epic",
    "confidence": 0.0-1.0,
    "is_complete": false,
    "reasoning": "Why you assigned this complexity level",
    "estimated_files": number,
    "requires_new_files": true|false,
    "suggested_approach": "Step-by-step approach to implement this",
    "subtasks": [
        {
            "title": "Subtask title",
            "description": "What needs to be done",
            "acceptance_criteria": ["Criterion 1", "Criterion 2"],
            "estimated_files": 1,
            "dependencies": ["Other subtask titles this depends on"],
            "labels": ["api", "frontend", etc]
        }
    ],
    "should_escalate": false,
    "escalation_reason": null,
    "roadblocks": ["List of blockers or unclear requirements"]
}
```

## Complexity Guidelines

- **simple**: Single file change, clear fix, < 30 min work. Examples: typo fix, small bug fix, config change.
- **moderate**: 2-5 files, clear approach, 1-2 hours. Examples: add endpoint, update component, fix integration.
- **complex**: 5+ files, requires design decisions, half day to full day. Examples: new feature, refactor module.
- **epic**: Major feature, needs decomposition, multiple days. Examples: new service, major UI overhaul.

## Rules

1. For **simple** and **moderate** issues, subtasks should be empty - the issue can be done in one PR.
2. For **complex** issues, create 2-4 focused subtasks that can each be done in one PR.
3. For **epic** issues, create 4-8 subtasks organized by logical layers (models, services, API, UI).
4. Set `should_escalate: true` if:
   - Requirements are ambiguous or conflicting
   - Needs architectural decision that wasn't specified
   - Requires access/permissions not available
   - Security implications need review
5. Set `is_complete: true` only during re-analysis if all work appears done.
6. Be specific in subtask descriptions - include file paths where possible.
7. Confidence should be lower (<0.6) when issue description is vague.

Respond ONLY with the JSON object, no additional text."""

        return prompt

    async def analyze(
        self,
        issue: dict[str, Any],
        iteration_context: str = "",
        repo_structure: str = "",
    ) -> IssueAnalysis:
        """Analyze an issue to determine complexity and decomposition."""
        prompt = self._build_analysis_prompt(issue, iteration_context, repo_structure)

        llm = self._get_llm()
        response = await llm.ainvoke(prompt)

        response_text = response.content
        if isinstance(response_text, list):
            response_text = response_text[0].get("text", "") if response_text else ""

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        try:
            analysis_data = json.loads(response_text)
            return IssueAnalysis(**analysis_data)
        except (json.JSONDecodeError, ValueError) as e:
            return IssueAnalysis(
                complexity=IssueComplexity.MODERATE,
                confidence=0.3,
                reasoning=f"Failed to parse LLM response: {e}. Defaulting to moderate complexity.",
                estimated_files=3,
                requires_new_files=False,
                suggested_approach="Manual review recommended due to analysis failure.",
                should_escalate=True,
                escalation_reason=f"LLM analysis failed: {e}",
                roadblocks=[str(e)],
            )

    def analyze_sync(
        self,
        issue: dict[str, Any],
        iteration_context: str = "",
        repo_structure: str = "",
    ) -> IssueAnalysis:
        """Synchronous wrapper for analyze()."""
        import asyncio

        return asyncio.run(self.analyze(issue, iteration_context, repo_structure))


def create_analyzer(settings: Settings | None = None) -> IssueAnalyzer:
    """Factory function to create an IssueAnalyzer."""
    return IssueAnalyzer(settings=settings)
