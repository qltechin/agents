"""Messenger stub - no-op implementation.

Google Chat notifications are not configured in this deployment.
All methods are no-ops that silently succeed.
"""

from enum import Enum


class WorkflowStage(str, Enum):
    CLONING = "cloning"
    ANALYZING = "analyzing"
    FIXING = "fixing"
    TESTING = "testing"
    CREATING_PR = "creating_pr"


class NoOpMessenger:
    """No-op messenger when Google Chat is not configured."""

    def send_builder_thread_start(self, **kwargs) -> str | None:
        return None

    def send_builder_stage_update(self, thread_key=None, stage=None, **kwargs) -> None:
        pass

    def send_builder_thread_complete(self, thread_key=None, **kwargs) -> None:
        pass

    def send_builder_no_issues_notification(self, **kwargs) -> None:
        pass

    def send_text(self, *args, **kwargs) -> None:
        pass

    def send_card(self, *args, **kwargs) -> None:
        pass


def get_messenger(settings=None):
    """Return a no-op messenger."""
    return NoOpMessenger()
