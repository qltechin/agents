"""Semantic label parser stub - no-op implementation.

Flutter/Widgetbook/Maestro semantic label parsing is not active
in this deployment. All methods return empty results.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SemanticLabel:
    label: str = ""
    widget_type: str = ""
    file_path: str = ""
    line_number: int = -1


class NoOpParser:
    """No-op parser when Flutter tooling is not configured."""

    def parse_flutter_source(self, file_path: str) -> list[SemanticLabel]:
        return []

    def parse_widgetbook_story(self, file_path: str) -> list[SemanticLabel]:
        return []

    def parse_maestro_test(self, file_path: str) -> list[SemanticLabel]:
        return []


def create_parser() -> NoOpParser:
    """Return a no-op semantic label parser."""
    return NoOpParser()
