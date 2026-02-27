"""Tools module for AI Agents."""

from tools.git_tools import (
    LazySubmoduleManager,
    RepoHost,
    SubmoduleInfo,
    get_submodule_manager,
    setup_git_credentials,
)
from tools.github_tools import GitHubTools, get_github_client

__all__ = [
    "GitHubTools",
    "get_github_client",
    "LazySubmoduleManager",
    "RepoHost",
    "SubmoduleInfo",
    "get_submodule_manager",
    "setup_git_credentials",
]
