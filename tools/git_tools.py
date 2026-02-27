"""Git tools for lazy submodule cloning and repository operations."""

import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from config.settings import get_settings


class RepoHost(Enum):
    """Repository hosting platforms."""

    GITHUB = "github"
    UNKNOWN = "unknown"


@dataclass
class SubmoduleInfo:
    """Information about a git submodule."""

    name: str
    path: str
    url: str
    host: RepoHost
    is_cloned: bool = False
    branch: str | None = None


@dataclass
class LazySubmoduleManager:
    """Manages lazy cloning of git submodules.

    This class provides on-demand submodule cloning, which is essential
    when working with monorepos that have many submodules.
    """

    repo_path: Path
    _submodules: dict[str, SubmoduleInfo] = field(default_factory=dict)
    _initialized: bool = False

    def __post_init__(self):
        """Convert string path to Path if needed."""
        if isinstance(self.repo_path, str):
            self.repo_path = Path(self.repo_path)

    def _run_git(
        self,
        args: list[str],
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = ["git"] + args
        result = subprocess.run(
            cmd,
            cwd=cwd or self.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"Git command failed: {' '.join(cmd)}\n{result.stderr}")
        return result

    def _determine_host(self, url: str) -> RepoHost:
        """Determine the hosting platform from a git URL."""
        if "github.com" in url:
            return RepoHost.GITHUB
        return RepoHost.UNKNOWN

    def _parse_gitmodules(self) -> dict[str, SubmoduleInfo]:
        """Parse .gitmodules file to get submodule information."""
        gitmodules_path = self.repo_path / ".gitmodules"
        if not gitmodules_path.exists():
            return {}

        content = gitmodules_path.read_text()
        submodules = {}

        current_name = None
        current_data = {}

        for line in content.splitlines():
            line = line.strip()

            match = re.match(r'\[submodule\s+"([^"]+)"\]', line)
            if match:
                if current_name and current_data.get("url"):
                    url = current_data["url"]
                    submodules[current_name] = SubmoduleInfo(
                        name=current_name,
                        path=current_data.get("path", current_name),
                        url=url,
                        host=self._determine_host(url),
                        branch=current_data.get("branch"),
                    )
                current_name = match.group(1)
                current_data = {}

            elif "=" in line and current_name:
                key, value = line.split("=", 1)
                current_data[key.strip()] = value.strip()

        if current_name and current_data.get("url"):
            url = current_data["url"]
            submodules[current_name] = SubmoduleInfo(
                name=current_name,
                path=current_data.get("path", current_name),
                url=url,
                host=self._determine_host(url),
                branch=current_data.get("branch"),
            )

        return submodules

    def initialize(self) -> None:
        """Initialize submodule tracking without cloning."""
        if self._initialized:
            return

        self._submodules = self._parse_gitmodules()
        self._run_git(["submodule", "init"], check=False)

        for name, info in self._submodules.items():
            submodule_path = self.repo_path / info.path
            git_dir = submodule_path / ".git"
            info.is_cloned = submodule_path.exists() and (git_dir.exists() or (submodule_path / ".git").is_file())

        self._initialized = True

    def get_submodules(self) -> dict[str, SubmoduleInfo]:
        """Get all submodule information."""
        if not self._initialized:
            self.initialize()
        return self._submodules.copy()

    def get_submodule(self, name_or_path: str) -> SubmoduleInfo | None:
        """Get a specific submodule by name or path."""
        if not self._initialized:
            self.initialize()

        if name_or_path in self._submodules:
            return self._submodules[name_or_path]

        for info in self._submodules.values():
            if info.path == name_or_path:
                return info

        return None

    def is_cloned(self, name_or_path: str) -> bool:
        """Check if a submodule is already cloned."""
        info = self.get_submodule(name_or_path)
        if not info:
            return False

        submodule_path = self.repo_path / info.path
        info.is_cloned = submodule_path.exists() and any(
            (submodule_path / marker).exists()
            for marker in [".git", "CLAUDE.md", "pom.xml", "package.json", "setup.py"]
        )
        return info.is_cloned

    def clone_submodule(
        self,
        name_or_path: str,
        recursive: bool = False,
    ) -> bool:
        """Clone a specific submodule on demand."""
        info = self.get_submodule(name_or_path)
        if not info:
            print(f"Submodule not found: {name_or_path}")
            return False

        if self.is_cloned(name_or_path):
            print(f"Submodule already cloned: {info.path}")
            return True

        print(f"Cloning submodule: {info.path} ({info.host.value})...")

        args = ["submodule", "update", "--init"]
        if recursive:
            args.append("--recursive")
        args.append("--")
        args.append(info.path)

        try:
            result = self._run_git(args, check=False)
            if result.returncode == 0:
                info.is_cloned = True
                print(f"Successfully cloned: {info.path}")
                return True
            else:
                print(f"Failed to clone {info.path}: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error cloning {info.path}: {e}")
            return False

    def clone_by_host(
        self,
        host: RepoHost,
        recursive: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Clone all submodules from a specific host."""
        if not self._initialized:
            self.initialize()

        successful = []
        failed = []

        for info in self._submodules.values():
            if info.host == host and not info.is_cloned:
                if self.clone_submodule(info.path, recursive=recursive):
                    successful.append(info.path)
                else:
                    failed.append(info.path)

        return successful, failed

    def clone_github_submodules(self, recursive: bool = False) -> tuple[list[str], list[str]]:
        """Clone all GitHub-hosted submodules."""
        return self.clone_by_host(RepoHost.GITHUB, recursive=recursive)

    def ensure_submodule(
        self,
        name_or_path: str,
        recursive: bool = False,
    ) -> Path:
        """Ensure a submodule is cloned and return its path."""
        info = self.get_submodule(name_or_path)
        if not info:
            raise RuntimeError(f"Unknown submodule: {name_or_path}")

        if not self.is_cloned(name_or_path):
            if not self.clone_submodule(name_or_path, recursive=recursive):
                raise RuntimeError(f"Failed to clone submodule: {name_or_path}")

        return self.repo_path / info.path

    def get_cloned_submodules(self) -> list[SubmoduleInfo]:
        """Get list of currently cloned submodules."""
        if not self._initialized:
            self.initialize()
        return [info for info in self._submodules.values() if self.is_cloned(info.path)]

    def get_uncloned_submodules(self) -> list[SubmoduleInfo]:
        """Get list of submodules that are not yet cloned."""
        if not self._initialized:
            self.initialize()
        return [info for info in self._submodules.values() if not self.is_cloned(info.path)]

    def status_summary(self) -> dict:
        """Get a summary of submodule status."""
        if not self._initialized:
            self.initialize()

        github_subs = [s for s in self._submodules.values() if s.host == RepoHost.GITHUB]
        other_subs = [s for s in self._submodules.values() if s.host == RepoHost.UNKNOWN]

        cloned = self.get_cloned_submodules()

        return {
            "total": len(self._submodules),
            "cloned": len(cloned),
            "uncloned": len(self._submodules) - len(cloned),
            "by_host": {
                "github": {
                    "total": len(github_subs),
                    "cloned": len([s for s in github_subs if s.is_cloned]),
                },
                "other": {
                    "total": len(other_subs),
                    "cloned": len([s for s in other_subs if s.is_cloned]),
                },
            },
        }


def setup_git_credentials() -> None:
    """Configure git credentials for GitHub.

    This should be called at the start of agent execution to set up
    authentication for GitHub.
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        subprocess.run(
            [
                "git",
                "config",
                "--global",
                'url."https://x-access-token:' + github_token + '@github.com/".insteadOf',
                "https://github.com/",
            ],
            check=False,
        )
        print("Configured GitHub authentication")


def get_submodule_manager(repo_path: Path | None = None) -> LazySubmoduleManager:
    """Get a LazySubmoduleManager for the specified or default repo path."""
    if repo_path is None:
        settings = get_settings()
        repo_path = settings.monorepo_path

    if repo_path is None:
        raise ValueError("No repository path specified and MONOREPO_PATH not configured")

    return LazySubmoduleManager(repo_path=repo_path)
