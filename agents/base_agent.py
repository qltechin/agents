"""Base agent class for all AI Agents."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from config.settings import Settings, get_settings


class AgentResult(BaseModel):
    """Result from agent execution."""

    success: bool = Field(description="Whether the agent succeeded")
    message: str = Field(description="Human-readable result message")
    changes_made: int = Field(default=0, description="Number of changes made")
    files_updated: list[str] = Field(default_factory=list, description="List of updated files")
    pr_url: str | None = Field(default=None, description="URL of created PR")
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Start timestamp")
    completed_at: datetime | None = Field(default=None, description="Completion timestamp")

    @property
    def duration_seconds(self) -> float | None:
        """Calculate execution duration in seconds."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


StateT = TypeVar("StateT")


class BaseAgent(ABC, Generic[StateT]):
    """Abstract base class for all agents."""

    name: str = "base_agent"
    description: str = "Base agent"

    def __init__(self, settings: Settings | None = None):
        """Initialize the agent."""
        self.settings = settings or get_settings()
        self._result: AgentResult | None = None

    @abstractmethod
    async def run(self, **kwargs) -> AgentResult:
        """Execute the agent's main workflow."""
        pass

    @abstractmethod
    def get_initial_state(self) -> StateT:
        """Get the initial state for the agent workflow."""
        pass

    def log(self, message: str, level: str = "info") -> None:
        """Log a message."""
        timestamp = datetime.utcnow().isoformat()
        prefix = f"[{timestamp}] [{self.name}] [{level.upper()}]"
        print(f"{prefix} {message}")

    def create_result(
        self,
        success: bool,
        message: str,
        changes_made: int = 0,
        files_updated: list[str] | None = None,
        pr_url: str | None = None,
        errors: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Create an agent result."""
        return AgentResult(
            success=success,
            message=message,
            changes_made=changes_made,
            files_updated=files_updated or [],
            pr_url=pr_url,
            errors=errors or [],
            metadata=metadata or {},
            completed_at=datetime.utcnow(),
        )
