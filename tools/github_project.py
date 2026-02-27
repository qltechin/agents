"""GitHub Project integration for AI agents."""

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC
from typing import Any


@dataclass
class ProjectItem:
    """Represents an item in a GitHub Project."""

    item_id: str
    content_id: str  # Issue or PR node ID
    number: int
    title: str
    repository: str
    status: str | None = None
    priority: str | None = None
    assignee: str | None = None
    pr_link: str | None = None
    last_agent_run: str | None = None


class GitHubProjectClient:
    """Client for interacting with GitHub Projects (v2).

    Uses the GitHub CLI (gh) for GraphQL queries since it handles
    authentication automatically via GITHUB_TOKEN.

    Supports both personal user accounts and organizations via account_type parameter.
    """

    def __init__(self, owner: str, account_type: str = "user"):
        """Initialize the GitHub Project client.

        Args:
            owner: GitHub owner (personal username or org name)
            account_type: 'user' for personal accounts, 'organization' for orgs
        """
        self.owner = owner
        self.account_type = account_type
        self._project_ids: dict[int, str] = {}  # Cache project node IDs
        self._field_ids: dict[str, dict[str, str]] = {}  # Cache field IDs per project

    def _run_graphql(self, query: str, variables: dict | None = None) -> dict[str, Any]:
        """Execute a GraphQL query using gh CLI.

        Args:
            query: GraphQL query string
            variables: Optional variables for the query

        Returns:
            Parsed JSON response

        Raises:
            RuntimeError: If the query fails
        """
        cmd = ["gh", "api", "graphql", "-f", f"query={query}"]

        if variables:
            for key, value in variables.items():
                if isinstance(value, bool):
                    cmd.extend(["-F", f"{key}={str(value).lower()}"])
                elif isinstance(value, int):
                    cmd.extend(["-F", f"{key}={value}"])
                else:
                    cmd.extend(["-f", f"{key}={value}"])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"GraphQL query failed: {result.stderr}")

        return json.loads(result.stdout)

    def get_project_id(self, project_number: int) -> str:
        """Get the node ID for a project number.

        Args:
            project_number: The project number (e.g., 1, 2)

        Returns:
            Project node ID
        """
        if project_number in self._project_ids:
            return self._project_ids[project_number]

        if self.account_type == "organization":
            query = """
            query($owner: String!, $number: Int!) {
                organization(login: $owner) {
                    projectV2(number: $number) {
                        id
                        title
                    }
                }
            }
            """
            result = self._run_graphql(query, {"owner": self.owner, "number": project_number})
            project_id = result["data"]["organization"]["projectV2"]["id"]
        else:
            query = """
            query($owner: String!, $number: Int!) {
                user(login: $owner) {
                    projectV2(number: $number) {
                        id
                        title
                    }
                }
            }
            """
            result = self._run_graphql(query, {"owner": self.owner, "number": project_number})
            project_id = result["data"]["user"]["projectV2"]["id"]

        self._project_ids[project_number] = project_id
        return project_id

    def get_field_id(self, project_number: int, field_name: str) -> str | None:
        """Get the node ID for a project field.

        Args:
            project_number: The project number
            field_name: Name of the field (e.g., "Status", "Priority", "PR Link")

        Returns:
            Field node ID or None if not found
        """
        cache_key = f"{project_number}"
        if cache_key not in self._field_ids:
            self._field_ids[cache_key] = {}
            self._load_project_fields(project_number)

        return self._field_ids[cache_key].get(field_name)

    def _load_project_fields(self, project_number: int) -> None:
        """Load all field IDs for a project into cache."""
        project_id = self.get_project_id(project_number)

        query = """
        query($projectId: ID!) {
            node(id: $projectId) {
                ... on ProjectV2 {
                    fields(first: 50) {
                        nodes {
                            ... on ProjectV2Field {
                                id
                                name
                            }
                            ... on ProjectV2SingleSelectField {
                                id
                                name
                                options {
                                    id
                                    name
                                }
                            }
                            ... on ProjectV2IterationField {
                                id
                                name
                            }
                        }
                    }
                }
            }
        }
        """

        result = self._run_graphql(query, {"projectId": project_id})
        fields = result["data"]["node"]["fields"]["nodes"]

        cache_key = f"{project_number}"
        for field in fields:
            if field and "name" in field:
                self._field_ids[cache_key][field["name"]] = field["id"]
                if "options" in field:
                    for option in field["options"]:
                        option_key = f"{field['name']}:{option['name']}"
                        self._field_ids[cache_key][option_key] = option["id"]

    def get_issue_project_item(self, repo: str, issue_number: int, project_number: int) -> ProjectItem | None:
        """Get project item info for an issue.

        Args:
            repo: Repository name (e.g., "owner/repo")
            issue_number: Issue number
            project_number: Project number to check

        Returns:
            ProjectItem if found in project, None otherwise
        """
        owner, name = repo.split("/")

        query = """
        query($owner: String!, $name: String!, $number: Int!) {
            repository(owner: $owner, name: $name) {
                issue(number: $number) {
                    id
                    title
                    projectItems(first: 10) {
                        nodes {
                            id
                            project {
                                number
                            }
                            fieldValues(first: 20) {
                                nodes {
                                    ... on ProjectV2ItemFieldTextValue {
                                        text
                                        field { ... on ProjectV2Field { name } }
                                    }
                                    ... on ProjectV2ItemFieldSingleSelectValue {
                                        name
                                        field { ... on ProjectV2SingleSelectField { name } }
                                    }
                                    ... on ProjectV2ItemFieldDateValue {
                                        date
                                        field { ... on ProjectV2Field { name } }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        result = self._run_graphql(query, {"owner": owner, "name": name, "number": issue_number})

        issue = result["data"]["repository"]["issue"]
        if not issue:
            return None

        for item in issue["projectItems"]["nodes"]:
            if item["project"]["number"] == project_number:
                status = None
                priority = None
                pr_link = None
                last_agent_run = None

                for fv in item["fieldValues"]["nodes"]:
                    if not fv or "field" not in fv:
                        continue
                    field_name = fv["field"].get("name", "")
                    if field_name == "Status":
                        status = fv.get("name")
                    elif field_name == "Priority":
                        priority = fv.get("name")
                    elif field_name == "PR Link":
                        pr_link = fv.get("text")
                    elif field_name == "Last Agent Run":
                        last_agent_run = fv.get("date") or fv.get("text")

                return ProjectItem(
                    item_id=item["id"],
                    content_id=issue["id"],
                    number=issue_number,
                    title=issue["title"],
                    repository=repo,
                    status=status,
                    priority=priority,
                    pr_link=pr_link,
                    last_agent_run=last_agent_run,
                )

        return None

    def update_item_field(
        self,
        project_number: int,
        item_id: str,
        field_name: str,
        value: str,
    ) -> bool:
        """Update a field value on a project item.

        Args:
            project_number: Project number
            item_id: Project item node ID
            field_name: Name of the field to update
            value: New value (for single-select, use the option name)

        Returns:
            True if successful
        """
        project_id = self.get_project_id(project_number)
        field_id = self.get_field_id(project_number, field_name)

        if not field_id:
            print(f"Field '{field_name}' not found in project {project_number}")
            return False

        option_key = f"{field_name}:{value}"
        option_id = self.get_field_id(project_number, option_key)

        if option_id:
            mutation = """
            mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
                updateProjectV2ItemFieldValue(input: {
                    projectId: $projectId,
                    itemId: $itemId,
                    fieldId: $fieldId,
                    value: { singleSelectOptionId: $optionId }
                }) {
                    projectV2Item { id }
                }
            }
            """
            variables = {
                "projectId": project_id,
                "itemId": item_id,
                "fieldId": field_id,
                "optionId": option_id,
            }
        else:
            mutation = """
            mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $value: String!) {
                updateProjectV2ItemFieldValue(input: {
                    projectId: $projectId,
                    itemId: $itemId,
                    fieldId: $fieldId,
                    value: { text: $value }
                }) {
                    projectV2Item { id }
                }
            }
            """
            variables = {
                "projectId": project_id,
                "itemId": item_id,
                "fieldId": field_id,
                "value": value,
            }

        try:
            self._run_graphql(mutation, variables)
            return True
        except RuntimeError as e:
            print(f"Failed to update field: {e}")
            return False

    def update_issue_on_completion(
        self,
        repo: str,
        issue_number: int,
        project_number: int,
        pr_url: str | None = None,
        success: bool = True,
    ) -> bool:
        """Update project item when agent completes work on an issue.

        Args:
            repo: Repository name
            issue_number: Issue number
            project_number: Project number
            pr_url: PR URL if one was created
            success: Whether the agent succeeded

        Returns:
            True if updates were successful
        """
        from datetime import datetime

        item = self.get_issue_project_item(repo, issue_number, project_number)
        if not item:
            print(f"Issue #{issue_number} not found in project {project_number}")
            return False

        updates_made = 0

        if success and pr_url:
            if self.update_item_field(project_number, item.item_id, "Status", "In Review"):
                updates_made += 1
        elif not success:
            if self.update_item_field(project_number, item.item_id, "Status", "AI Failed"):
                updates_made += 1

        if pr_url:
            if self.update_item_field(project_number, item.item_id, "PR Link", pr_url):
                updates_made += 1

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        if self.update_item_field(project_number, item.item_id, "Last Agent Run", timestamp):
            updates_made += 1

        return updates_made > 0

    def get_issues_by_priority(
        self,
        project_number: int,
        status_filter: str | None = "AI Queue",
    ) -> list[ProjectItem]:
        """Get issues from a project sorted by priority.

        Args:
            project_number: Project number
            status_filter: Only return items with this status (default: "AI Queue")

        Returns:
            List of ProjectItems sorted by priority (High > Medium > Low)
        """
        project_id = self.get_project_id(project_number)

        query = """
        query($projectId: ID!, $cursor: String) {
            node(id: $projectId) {
                ... on ProjectV2 {
                    items(first: 100, after: $cursor) {
                        pageInfo {
                            hasNextPage
                            endCursor
                        }
                        nodes {
                            id
                            content {
                                ... on Issue {
                                    id
                                    number
                                    title
                                    state
                                    repository {
                                        nameWithOwner
                                    }
                                    labels(first: 10) {
                                        nodes { name }
                                    }
                                }
                            }
                            fieldValues(first: 20) {
                                nodes {
                                    ... on ProjectV2ItemFieldSingleSelectValue {
                                        name
                                        field { ... on ProjectV2SingleSelectField { name } }
                                    }
                                    ... on ProjectV2ItemFieldTextValue {
                                        text
                                        field { ... on ProjectV2Field { name } }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        all_items = []
        cursor = None

        while True:
            variables = {"projectId": project_id}
            if cursor:
                variables["cursor"] = cursor

            result = self._run_graphql(query, variables)
            items_data = result["data"]["node"]["items"]

            for item in items_data["nodes"]:
                if not item.get("content"):
                    continue

                content = item["content"]
                if content.get("state") != "OPEN":
                    continue

                status = None
                priority = None
                pr_link = None

                for fv in item["fieldValues"]["nodes"]:
                    if not fv or "field" not in fv:
                        continue
                    field_name = fv["field"].get("name", "")
                    if field_name == "Status":
                        status = fv.get("name")
                    elif field_name == "Priority":
                        priority = fv.get("name")
                    elif field_name == "PR Link":
                        pr_link = fv.get("text")

                if status_filter and status != status_filter:
                    continue

                if priority == "Blocked":
                    continue

                all_items.append(
                    ProjectItem(
                        item_id=item["id"],
                        content_id=content["id"],
                        number=content["number"],
                        title=content["title"],
                        repository=content["repository"]["nameWithOwner"],
                        status=status,
                        priority=priority,
                        pr_link=pr_link,
                    )
                )

            if not items_data["pageInfo"]["hasNextPage"]:
                break
            cursor = items_data["pageInfo"]["endCursor"]

        priority_order = {"High": 0, "Medium": 1, "Low": 2, None: 3}
        all_items.sort(key=lambda x: priority_order.get(x.priority, 3))

        return all_items

    def get_stale_ai_queue_issues(
        self,
        repos: list[str],
        hours_threshold: int = 24,
    ) -> list[dict[str, Any]]:
        """Get issues with aibuild label that have been waiting too long.

        Args:
            repos: List of repository names to check (e.g., ["owner/repo"])
            hours_threshold: Hours before issue is considered stale

        Returns:
            List of dicts with issue details (repository, number, title, hours_in_queue)
        """
        from datetime import datetime, timedelta

        stale_issues = []
        now = datetime.now(UTC)

        for repo in repos:
            try:
                cmd = [
                    "gh",
                    "issue",
                    "list",
                    "--repo",
                    repo,
                    "--label",
                    "aibuild",
                    "--state",
                    "open",
                    "--json",
                    "number,title,createdAt,labels,url",
                    "--limit",
                    "50",
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    continue

                issues = json.loads(result.stdout)

                for issue in issues:
                    created_str = issue.get("createdAt", "")
                    if not created_str:
                        continue

                    try:
                        created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                        hours_in_queue = (now - created).total_seconds() / 3600

                        if hours_in_queue > hours_threshold:
                            stale_issues.append(
                                {
                                    "repository": repo,
                                    "number": issue["number"],
                                    "title": issue.get("title", ""),
                                    "url": issue.get("url", ""),
                                    "hours_in_queue": hours_in_queue,
                                    "created_at": created_str,
                                }
                            )
                    except (ValueError, TypeError):
                        continue

            except Exception as e:
                print(f"Error checking {repo}: {e}")
                continue

        stale_issues.sort(key=lambda x: x.get("hours_in_queue", 0), reverse=True)

        return stale_issues


def get_project_client(owner: str, account_type: str = "user") -> GitHubProjectClient:
    """Factory function to get a project client.

    Args:
        owner: GitHub owner (username or org name)
        account_type: 'user' for personal accounts, 'organization' for orgs

    Returns:
        GitHubProjectClient instance
    """
    return GitHubProjectClient(owner=owner, account_type=account_type)
