"""GitHub tools for repository operations."""

from datetime import datetime
from pathlib import Path
from typing import Any

from github import Github, GithubException, InputGitTreeElement
from pydantic import BaseModel

from config.settings import get_settings


def get_github_client() -> Github:
    """Get authenticated GitHub client."""
    settings = get_settings()
    if not settings.github_token:
        raise ValueError("GITHUB_TOKEN not configured")
    return Github(settings.github_token)


class PRInfo(BaseModel):
    """Pull request information."""

    number: int
    title: str
    url: str
    state: str
    created_at: datetime
    branch: str


class FileChange(BaseModel):
    """File change for a commit/PR."""

    path: str
    content: str
    message: str


class GitHubTools:
    """Tools for GitHub operations."""

    def __init__(self, client: Github | None = None):
        """Initialize GitHub tools."""
        self.client = client or get_github_client()
        self.settings = get_settings()
        self._repo = None

    @property
    def repo(self):
        """Get repository object (lazy loaded)."""
        if self._repo is None:
            self._repo = self.client.get_repo(self.settings.github_repo)
        return self._repo

    def get_file_content(self, path: str, ref: str = "main") -> str | None:
        """Get content of a file from the repository."""
        try:
            content = self.repo.get_contents(path, ref=ref)
            if isinstance(content, list):
                return None  # It's a directory
            return content.decoded_content.decode("utf-8")
        except GithubException as e:
            if e.status == 404:
                return None
            raise

    def create_or_update_file(
        self,
        path: str,
        content: str,
        message: str,
        branch: str = "main",
    ) -> dict[str, Any]:
        """Create or update a file in the repository."""
        try:
            # Check if file exists
            existing = self.repo.get_contents(path, ref=branch)
            # Update existing file
            result = self.repo.update_file(
                path=path,
                message=message,
                content=content,
                sha=existing.sha,
                branch=branch,
            )
            return {"action": "updated", "commit": result["commit"].sha}
        except GithubException as e:
            if e.status == 404:
                # Create new file
                result = self.repo.create_file(
                    path=path,
                    message=message,
                    content=content,
                    branch=branch,
                )
                return {"action": "created", "commit": result["commit"].sha}
            raise

    def create_branch(self, branch_name: str, from_branch: str = "main") -> bool:
        """Create a new branch from an existing branch."""
        try:
            source = self.repo.get_branch(from_branch)
            self.repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=source.commit.sha)
            return True
        except GithubException as e:
            if e.status == 422:  # Branch already exists
                return True
            raise

    def create_pull_request(
        self,
        title: str,
        body: str,
        head: str,
        base: str = "main",
        labels: list[str] | None = None,
    ) -> PRInfo:
        """Create a pull request."""
        pr = self.repo.create_pull(
            title=title,
            body=body,
            head=head,
            base=base,
        )

        if labels:
            pr.add_to_labels(*labels)

        return PRInfo(
            number=pr.number,
            title=pr.title,
            url=pr.html_url,
            state=pr.state,
            created_at=pr.created_at,
            branch=head,
        )

    def get_open_prs(self, head_prefix: str = "automated/") -> list[PRInfo]:
        """Get open PRs with a specific head branch prefix."""
        prs = self.repo.get_pulls(state="open")
        result = []

        for pr in prs:
            if pr.head.ref.startswith(head_prefix):
                result.append(
                    PRInfo(
                        number=pr.number,
                        title=pr.title,
                        url=pr.html_url,
                        state=pr.state,
                        created_at=pr.created_at,
                        branch=pr.head.ref,
                    )
                )

        return result

    def commit_multiple_files(
        self,
        changes: list[FileChange],
        branch: str,
        commit_message: str,
    ) -> str:
        """Commit multiple file changes in a single commit."""
        # Get the current commit SHA for the branch
        ref = self.repo.get_git_ref(f"heads/{branch}")
        base_tree = self.repo.get_git_tree(ref.object.sha)

        # Create tree elements for each file using InputGitTreeElement
        tree_elements = []
        for change in changes:
            blob = self.repo.create_git_blob(change.content, "utf-8")
            tree_elements.append(
                InputGitTreeElement(
                    path=change.path,
                    mode="100644",
                    type="blob",
                    sha=blob.sha,
                )
            )

        # Create new tree
        new_tree = self.repo.create_git_tree(tree_elements, base_tree)

        # Create commit
        parent = self.repo.get_git_commit(ref.object.sha)
        commit = self.repo.create_git_commit(commit_message, new_tree, [parent])

        # Update branch reference
        ref.edit(commit.sha)

        return commit.sha

    def create_automated_pr(
        self,
        title: str,
        changes: list[FileChange],
        pr_body: str,
        branch_name: str = "automated/sync",
    ) -> PRInfo:
        """Create an automated PR with multiple file changes."""
        # Generate unique branch name with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        branch = f"{branch_name}-{timestamp}"

        # Create branch
        self.create_branch(branch)

        # Commit changes
        commit_message = f"docs: {title}\n\nAutomated update by ai-agents"
        self.commit_multiple_files(changes, branch, commit_message)

        # Create PR
        return self.create_pull_request(
            title=f"[Automated] {title}",
            body=pr_body,
            head=branch,
            labels=["automated", "documentation"],
        )


class LocalGitTools:
    """Tools for local git operations."""

    def __init__(self, repo_path: Path):
        """Initialize local git tools."""
        self.repo_path = repo_path

    def read_file(self, relative_path: str) -> str | None:
        """Read a file from the local repository."""
        file_path = self.repo_path / relative_path
        if not file_path.exists():
            return None
        return file_path.read_text()

    def write_file(self, relative_path: str, content: str) -> None:
        """Write a file to the local repository."""
        file_path = self.repo_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    def list_files(self, pattern: str = "**/*") -> list[str]:
        """List files matching a pattern."""
        return [str(p.relative_to(self.repo_path)) for p in self.repo_path.glob(pattern) if p.is_file()]

    def find_claude_md_files(self) -> list[str]:
        """Find all CLAUDE.md files in the repository."""
        return self.list_files("**/CLAUDE.md")
