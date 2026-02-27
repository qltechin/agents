"""Metrics stub - no-op implementation.

CloudWatch metrics are not configured in this deployment (no AWS).
All methods are no-ops that silently succeed.
"""


class NoOpMetrics:
    """No-op metrics when CloudWatch is not configured."""

    def record_agent_run_duration(self, *args, **kwargs) -> None:
        pass

    def record_issue_processed(self, *args, **kwargs) -> None:
        pass

    def record_pr_created(self, *args, **kwargs) -> None:
        pass

    def record_resolution_time(self, *args, **kwargs) -> None:
        pass

    def record_error(self, *args, **kwargs) -> None:
        pass

    def record_success_rate(self, *args, **kwargs) -> None:
        pass


def get_metrics(name: str) -> NoOpMetrics:
    """Return a no-op metrics collector."""
    return NoOpMetrics()
