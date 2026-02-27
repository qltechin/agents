"""
Codex authentication management for CI/CD environments.

Handles retrieval and persistence of Codex auth tokens.
Auth JSON can be provided via environment variable or file path.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"

# Environment variable names for auth JSON
CODEX_AUTH_JSON_ENV = "CODEX_AUTH_JSON"  # Full JSON content as env var
CODEX_AUTH_FILE_ENV = "CODEX_AUTH_FILE"  # Path to auth file


class CodexAuthManager:
    """Manages Codex authentication for headless CI/CD environments."""

    def __init__(self):
        """Initialize the auth manager."""
        self._original_auth: dict[str, Any] | None = None

    def setup_auth(self) -> bool:
        """
        Set up Codex auth from environment or file.

        Auth source priority:
        1. CODEX_AUTH_JSON environment variable (full JSON string)
        2. CODEX_AUTH_FILE environment variable (path to auth file)
        3. ~/.codex/auth.json (already exists)

        Returns:
            True if auth was set up successfully, False otherwise
        """
        try:
            CODEX_AUTH_PATH.parent.mkdir(parents=True, exist_ok=True)

            # 1. Check CODEX_AUTH_JSON env var
            auth_json_str = os.environ.get(CODEX_AUTH_JSON_ENV)
            if auth_json_str:
                self._original_auth = json.loads(auth_json_str)
                fd = os.open(CODEX_AUTH_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as f:
                    json.dump(self._original_auth, f, indent=2)
                logger.info(f"Codex auth.json written from {CODEX_AUTH_JSON_ENV} env var")
                return True

            # 2. Check CODEX_AUTH_FILE env var
            auth_file_path = os.environ.get(CODEX_AUTH_FILE_ENV)
            if auth_file_path:
                source_path = Path(auth_file_path)
                if source_path.exists():
                    with open(source_path) as f:
                        self._original_auth = json.load(f)
                    fd = os.open(CODEX_AUTH_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, "w") as f:
                        json.dump(self._original_auth, f, indent=2)
                    logger.info(f"Codex auth.json written from {auth_file_path}")
                    return True
                else:
                    logger.error(f"Auth file not found: {auth_file_path}")
                    return False

            # 3. Check if auth.json already exists
            if CODEX_AUTH_PATH.exists():
                with open(CODEX_AUTH_PATH) as f:
                    self._original_auth = json.load(f)
                logger.info("Using existing Codex auth.json")
                return True

            logger.error(
                "No Codex auth found. Set CODEX_AUTH_JSON or CODEX_AUTH_FILE environment variable, "
                "or ensure ~/.codex/auth.json exists."
            )
            return False

        except Exception as e:
            logger.error(f"Unexpected error setting up Codex auth: {e}")
            return False

    def cleanup_auth(self) -> None:
        """Remove local auth.json file."""
        try:
            if CODEX_AUTH_PATH.exists():
                CODEX_AUTH_PATH.unlink()
                logger.debug("Removed local auth.json")
        except Exception as e:
            logger.warning(f"Failed to cleanup auth.json: {e}")


class CodexAuthContext:
    """Context manager for Codex auth setup and teardown."""

    def __init__(self, cleanup_on_exit: bool = True, build_id: str | None = None):
        """
        Initialize the context manager.

        Args:
            cleanup_on_exit: Whether to remove auth.json after execution
            build_id: Optional build identifier (e.g., CodeBuild build ID) for logging
        """
        self.manager = CodexAuthManager()
        self.cleanup_on_exit = cleanup_on_exit
        self.build_id = build_id

    def __enter__(self) -> "CodexAuthContext":
        """Set up Codex auth."""
        if not self.manager.setup_auth():
            raise RuntimeError("Failed to setup Codex auth")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup Codex auth."""
        if self.cleanup_on_exit:
            self.manager.cleanup_auth()
        return None  # Don't suppress exceptions
