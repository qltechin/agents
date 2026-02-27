"""Pydantic settings for AI Agents."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # GitHub Organization / User Configuration
    gh_owner: str = Field(default="", description="GitHub owner (org or personal account)")
    gh_account_type: Literal["user", "organization"] = Field(
        default="user", description="Account type: 'user' for personal, 'organization' for org"
    )
    gh_project_number: int = Field(default=1, description="GitHub Project number for the main project board")
    gh_agents_repo_name: str = Field(default="agents", description="Name of the agents repository")

    # LLM Configuration
    llm_provider: Literal["anthropic", "bedrock", "cerebras"] = Field(
        default="anthropic", description="LLM provider to use (anthropic, bedrock, cerebras)"
    )
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    bedrock_model_id: str = Field(
        default="global.anthropic.claude-opus-4-6-20251001-v1:0",
        description="Bedrock inference profile ID (Claude Opus 4.6)",
    )
    bedrock_region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock API calls",
    )

    # AWS Configuration (optional - only needed for Bedrock)
    aws_region: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: str | None = Field(default=None, description="AWS access key")
    aws_secret_access_key: str | None = Field(default=None, description="AWS secret key")

    # Cerebras z.ai Configuration (GLM 4.7)
    cerebras_api_key: str | None = Field(default=None, description="Cerebras z.ai API key")
    cerebras_base_url: str = Field(
        default="https://api.cerebras.ai/v1",
        description="Cerebras API base URL",
    )
    cerebras_model: str = Field(
        default="glm-4.7",
        description="Cerebras model ID (default: glm-4.7)",
    )

    # Langfuse Configuration
    langfuse_public_key: str | None = Field(default=None, description="Langfuse public key")
    langfuse_secret_key: str | None = Field(default=None, description="Langfuse secret key")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", description="Langfuse host URL")

    # GitHub Configuration
    github_token: str | None = Field(default=None, description="GitHub personal access token")
    github_repo: str = Field(default="", description="Target GitHub repository (owner/repo)")

    # Builder Agent Configuration (generic, works with any repo)
    builder_repos: str = Field(
        default="",
        description="Comma-separated list of repos for Builder Agent to monitor (e.g., owner/repo1,owner/repo2)",
    )
    builder_source_label: str = Field(default="aibuild", description="Label for issues to be picked up by agent")
    builder_complete_label: str = Field(
        default="aicomplete", description="Label to apply when agent completes an issue"
    )
    builder_in_progress_label: str = Field(
        default="ai-in-progress", description="Label to apply while agent is working"
    )
    builder_failed_label: str = Field(
        default="ai-failed", description="Label to apply when agent fails to fix an issue"
    )

    # Tester Agent Configuration
    tester_source_label: str = Field(default="needs-tests", description="Label for PRs to be picked up by Tester Agent")
    tester_complete_label: str = Field(
        default="tests-reviewed", description="Label to apply when Tester Agent completes review"
    )
    tester_in_progress_label: str = Field(
        default="test-review-in-progress",
        description="Label to apply while Tester Agent is working",
    )
    tester_coverage_threshold: int = Field(default=80, description="Default coverage threshold percentage")

    # Designer Agent Configuration
    designer_source_label: str = Field(
        default="needs-design-review", description="Label for PRs to be picked up by Designer Agent"
    )
    designer_complete_label: str = Field(
        default="design-reviewed", description="Label to apply when Designer Agent completes review"
    )
    designer_in_progress_label: str = Field(
        default="design-review-in-progress",
        description="Label to apply while Designer Agent is working",
    )
    designer_accessibility_level: str = Field(default="AA", description="Default WCAG accessibility level (A, AA, AAA)")

    # Figma Integration (for Designer Agent)
    figma_access_token: str | None = Field(default=None, description="Figma personal access token")
    figma_team_id: str | None = Field(default=None, description="Figma team ID for library access")
    figma_default_file_key: str | None = Field(default=None, description="Default Figma file key for design system")
    figma_mcp_enabled: bool = Field(default=True, description="Enable Figma MCP server integration")
    figma_mcp_transport: str = Field(
        default="http", description="MCP transport type: 'http' (remote) or 'sse' (local desktop app)"
    )

    # Deployer Agent Configuration
    deployer_staging_label: str = Field(default="deploy-staging", description="Label to trigger staging deployment")
    deployer_production_label: str = Field(
        default="deploy-production", description="Label to trigger production deployment"
    )
    deployer_complete_label: str = Field(default="deployed", description="Label to apply when deployment completes")
    deployer_failed_label: str = Field(default="deploy-failed", description="Label to apply when deployment fails")
    deployer_awaiting_approval_label: str = Field(
        default="awaiting-deploy-approval",
        description="Label for production deployments awaiting approval",
    )
    deployer_production_requires_approval: bool = Field(
        default=True, description="Always require human approval for production deployments"
    )
    deployer_staging_auto_deploy: bool = Field(
        default=True, description="Automatically deploy to staging without approval"
    )
    deployer_rollback_on_failure: bool = Field(default=True, description="Automatically rollback failed deployments")
    deployer_smoke_test_timeout: int = Field(default=60, description="Timeout in seconds for smoke tests")
    deployer_health_check_retries: int = Field(default=5, description="Number of health check retries after deployment")

    # Google Chat Integration
    google_chat_space_ai_builders: str | None = Field(
        default=None, description="Google Chat space ID for ai-builders (format: spaces/AAAA...)"
    )
    google_chat_space_infrastructure: str | None = Field(
        default=None, description="Google Chat space ID for infrastructure (format: spaces/AAAA...)"
    )
    google_chat_space_business: str | None = Field(
        default=None, description="Google Chat space ID for business (format: spaces/AAAA...)"
    )
    google_chat_space_supervisor: str | None = Field(
        default=None,
        description="Google Chat space ID for supervisor agent (format: spaces/AAAA...)",
    )
    google_chat_enabled: bool = Field(default=True, description="Enable Google Chat notifications")

    # Monorepo Configuration (optional)
    monorepo_path: Path | None = Field(default=None, description="Local path to monorepo")

    # Agent Configuration
    dry_run: bool = Field(default=False, description="Run without making changes")
    verbose: bool = Field(default=False, description="Verbose output")

    # Maintenance Agent Configuration
    maintenance_stale_metrics_threshold_hours: int = Field(
        default=24, description="Hours without data before a metric is considered stale"
    )
    maintenance_stale_docs_threshold_days: int = Field(
        default=7, description="Days without update before documentation is considered stale"
    )
    maintenance_stale_issue_threshold_hours: int = Field(
        default=24, description="Hours in AI Queue before issue is considered stale"
    )
    maintenance_agent_success_threshold: float = Field(
        default=80.0, description="Minimum success rate (%) before agent is flagged as degraded"
    )
    maintenance_deprecated_repos: str = Field(
        default="",
        description="Comma-separated list of deprecated repos to flag for cleanup",
    )

    @property
    def maintenance_deprecated_repo_list(self) -> list[str]:
        """Return list of deprecated repos."""
        return [r.strip() for r in self.maintenance_deprecated_repos.split(",") if r.strip()]

    @property
    def builder_repo_list(self) -> list[str]:
        """Return list of repos for Builder Agent."""
        return [r.strip() for r in self.builder_repos.split(",") if r.strip()]

    @field_validator("monorepo_path", mode="before")
    @classmethod
    def validate_monorepo_path(cls, v: str | None) -> Path | None:
        """Convert string path to Path object."""
        if v is None:
            return None
        return Path(v).expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
