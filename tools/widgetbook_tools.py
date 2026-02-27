"""Widgetbook tools stub - no-op implementation.

Widgetbook label generation is not active in this deployment.
All functions return empty strings.
"""


def generate_widgetbook_label(file_path: str) -> str:
    """Generate a bidirectional widgetbook label for a source widget file.

    No-op stub - returns empty string when Widgetbook is not configured.
    """
    return ""


def generate_source_widget_label(
    story_path: str,
    class_name: str,
    use_cases: list[str],
) -> str:
    """Generate a bidirectional source label for a Widgetbook story file.

    No-op stub - returns empty string when Widgetbook is not configured.
    """
    return ""
