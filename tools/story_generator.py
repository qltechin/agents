"""Story generator stub - no-op implementation.

Widgetbook story generation is not active in this deployment.
All methods return empty results.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GeneratedStory:
    code: str = ""
    widget_name: str = ""
    use_cases: list[str] = field(default_factory=list)


class NoOpGenerator:
    """No-op story generator when Widgetbook is not configured."""

    def generate_story(self, widget_path=None, use_cases=None, **kwargs) -> GeneratedStory | None:
        return None


def create_generator() -> NoOpGenerator:
    """Return a no-op story generator."""
    return NoOpGenerator()
