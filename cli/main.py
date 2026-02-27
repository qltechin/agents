"""CLI application for AI Agents."""

import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from config.settings import get_settings

# Note: training module imports are lazy to avoid requiring psycopg2 for non-training commands
# The train_app is registered via a callback that imports on-demand

app = typer.Typer(
    name="ai-agents",
    help="AI agents for software project management",
    no_args_is_help=True,
)
console = Console()

# Create a lazy-loaded training subcommand
# This avoids importing psycopg2 until the train command is actually used
train_app_lazy = typer.Typer(
    name="train",
    help="Train and optimize AI agents with DSPy + GEPA",
    no_args_is_help=True,
)


@train_app_lazy.callback(invoke_without_command=True)
def train_callback(ctx: typer.Context):
    """Lazy-load training commands when train subcommand is invoked."""
    if ctx.invoked_subcommand is None:
        # If no subcommand, show help
        console.print(ctx.get_help())
        raise typer.Exit(0)


def _handle_training_import_error(e: ImportError):
    """Handle ImportError for training dependencies."""
    console.print("[red]Training dependencies not installed.[/red]")
    console.print("[yellow]Install with: pip install -e '.[training]'[/yellow]")
    console.print(f"[dim]Error: {e}[/dim]")
    raise typer.Exit(1)


@train_app_lazy.command(name="agent")
def train_agent_cmd(
    agent: str = typer.Argument(..., help="Agent to train (builder, planning, maintenance, etc.)"),
    budget: str = typer.Option("medium", "--budget", "-b", help="Training budget: light, medium, heavy"),
    synthetic_ratio: float | None = typer.Option(
        None, "--synthetic-ratio", "-s", help="Ratio of synthetic data (0.0-1.0)"
    ),
    target_size: int = typer.Option(200, "--target-size", "-t", help="Target dataset size"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Preview without training"),
    deploy: bool = typer.Option(False, "--deploy", help="Deploy to prompt store after training"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Train an agent with DSPy + GEPA optimization."""
    try:
        from training.cli.train_commands import AgentType, Budget
        from training.cli.train_commands import train as train_fn

        # Convert string args to enums
        agent_type = AgentType(agent)
        budget_enum = Budget(budget)
        # Call the training function directly
        train_fn(
            agent=agent_type,
            budget=budget_enum,
            synthetic_ratio=synthetic_ratio,
            target_size=target_size,
            min_examples=50,
            save_path=None,
            deploy=deploy,
            dry_run=dry_run,
            verbose=verbose,
        )
    except ImportError as e:
        _handle_training_import_error(e)
    except ValueError as e:
        console.print(f"[red]Invalid argument: {e}[/red]")
        console.print(
            "[yellow]Valid agents: builder, planning, maintenance, infrastructure, documentation, cost_monitor[/yellow]"
        )
        console.print("[yellow]Valid budgets: light, medium, heavy[/yellow]")
        raise typer.Exit(1)


@train_app_lazy.command(name="evaluate")
def train_evaluate_cmd(
    agent: str = typer.Argument(..., help="Agent to evaluate (e.g., builder)"),
    version: str | None = typer.Option(None, "--version", "-v", help="Specific version (default: active)"),
    dataset: str = typer.Option("historical", "--dataset", "-d", help="Dataset: historical, synthetic, both"),
    limit: int = typer.Option(100, "--limit", "-l", help="Max examples to evaluate"),
):
    """Evaluate an agent's current performance."""
    try:
        from training.cli.train_commands import AgentType
        from training.cli.train_commands import evaluate as evaluate_cmd

        agent_type = AgentType(agent)
        evaluate_cmd.callback(agent=agent_type, version=version, dataset=dataset, limit=limit)
    except ImportError as e:
        _handle_training_import_error(e)
    except ValueError as e:
        console.print(f"[red]Invalid agent: {e}[/red]")
        raise typer.Exit(1)


@train_app_lazy.command(name="compare")
def train_compare_cmd(
    agent: str = typer.Argument(..., help="Agent to compare versions for"),
    baseline: str = typer.Option(..., "--baseline", help="Baseline version"),
    candidate: str = typer.Option(..., "--candidate", help="Candidate version to compare"),
    size: int = typer.Option(50, "--size", help="Evaluation dataset size"),
):
    """Compare two versions of an agent's prompts."""
    try:
        from training.cli.train_commands import AgentType
        from training.cli.train_commands import compare as compare_cmd

        agent_type = AgentType(agent)
        compare_cmd.callback(agent=agent_type, baseline=baseline, candidate=candidate, dataset_size=size)
    except ImportError as e:
        _handle_training_import_error(e)
    except ValueError as e:
        console.print(f"[red]Invalid agent: {e}[/red]")
        raise typer.Exit(1)


@train_app_lazy.command(name="deploy")
def train_deploy_cmd(
    agent: str = typer.Argument(..., help="Agent to deploy"),
    version: str = typer.Option(..., "--version", "-v", help="Version to deploy"),
    notes: str = typer.Option("", "--notes", "-n", help="Deployment notes"),
):
    """Deploy a specific prompt version as active."""
    try:
        from training.cli.train_commands import AgentType
        from training.cli.train_commands import deploy as deploy_cmd

        agent_type = AgentType(agent)
        deploy_cmd.callback(agent=agent_type, version=version, notes=notes)
    except ImportError as e:
        _handle_training_import_error(e)
    except ValueError as e:
        console.print(f"[red]Invalid agent: {e}[/red]")
        raise typer.Exit(1)


@train_app_lazy.command(name="list-versions")
def train_list_versions_cmd(
    agent: str = typer.Argument(..., help="Agent to list versions for"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max versions to show"),
):
    """List all prompt versions for an agent."""
    try:
        from training.cli.train_commands import AgentType
        from training.cli.train_commands import list_versions as list_versions_cmd

        agent_type = AgentType(agent)
        list_versions_cmd.callback(agent=agent_type, limit=limit)
    except ImportError as e:
        _handle_training_import_error(e)
    except ValueError as e:
        console.print(f"[red]Invalid agent: {e}[/red]")
        raise typer.Exit(1)


@train_app_lazy.command(name="rollback")
def train_rollback_cmd(
    agent: str = typer.Argument(..., help="Agent to rollback"),
    to: str = typer.Option(..., "--to", help="Version to rollback to"),
):
    """Rollback to a previous prompt version."""
    try:
        from training.cli.train_commands import AgentType
        from training.cli.train_commands import rollback as rollback_cmd

        agent_type = AgentType(agent)
        rollback_cmd.callback(agent=agent_type, version=to)
    except ImportError as e:
        _handle_training_import_error(e)
    except ValueError as e:
        console.print(f"[red]Invalid agent: {e}[/red]")
        raise typer.Exit(1)


# Add the lazy training subcommand
app.add_typer(train_app_lazy, name="train")


def run_async(coro):
    """Run an async coroutine."""
    return asyncio.run(coro)


@app.command()
def sync_inventory(
    monorepo_path: Path | None = typer.Option(
        None,
        "--monorepo-path",
        "-m",
        help="Path to monorepo",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Run without making changes",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """Sync ECS service inventory with documentation.

    This command queries AWS ECS for the current state of all services
    and updates the documentation in the monorepo to match.
    """
    from agents.infrastructure_sync_agent import InfrastructureSyncAgent

    settings = get_settings()

    # Override settings from CLI
    if monorepo_path:
        settings.monorepo_path = monorepo_path
    settings.dry_run = dry_run
    settings.verbose = verbose

    console.print("[bold blue]Infrastructure Sync Agent[/bold blue]")
    console.print(f"Clusters: {', '.join(settings.ecs_clusters)}")
    console.print(f"Dry run: {dry_run}")
    console.print()

    agent = InfrastructureSyncAgent(settings=settings)
    result = run_async(agent.run())

    if result.success:
        console.print(f"[green]{result.message}[/green]")

        if result.files_updated:
            console.print("\nFiles updated:")
            for f in result.files_updated:
                console.print(f"  - {f}")

        if result.pr_url:
            console.print(f"\n[bold]PR created:[/bold] {result.pr_url}")
    else:
        console.print(f"[red]Error: {result.message}[/red]")
        for error in result.errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)


@app.command()
def sync_documentation(
    monorepo_path: Path | None = typer.Option(
        None,
        "--monorepo-path",
        "-m",
        help="Path to monorepo",
    ),
    check_api_changes: bool = typer.Option(
        True,
        "--check-api-changes/--no-check-api-changes",
        help="Check for API changes in submodules",
    ),
    create_missing_claude_md: bool = typer.Option(
        False,
        "--create-missing-claude-md/--no-create-missing-claude-md",
        help="Create CLAUDE.md files for repos that don't have them",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Run without making changes",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """Sync documentation with code changes.

    This command monitors code changes in submodules and updates
    CLAUDE.md files when APIs change.

    Use --create-missing-claude-md to automatically create CLAUDE.md files
    for repos in the builder_repos list that don't have them.
    """
    from agents.documentation_agent import DocumentationAgent

    settings = get_settings()

    # Override settings from CLI
    if monorepo_path:
        settings.monorepo_path = monorepo_path
    settings.dry_run = dry_run
    settings.verbose = verbose

    console.print("[bold blue]Documentation Sync Agent[/bold blue]")
    console.print(f"Monorepo: {settings.monorepo_path or 'Not set'}")
    console.print(f"Check API changes: {check_api_changes}")
    console.print(f"Create missing CLAUDE.md: {create_missing_claude_md}")
    console.print(f"Dry run: {dry_run}")
    console.print()

    if not settings.monorepo_path:
        console.print("[red]Error: --monorepo-path is required[/red]")
        raise typer.Exit(1)

    agent = DocumentationAgent(
        settings=settings,
        create_missing_claude_md=create_missing_claude_md,
    )
    result = run_async(agent.run())

    if result.success:
        console.print(f"[green]{result.message}[/green]")

        if result.files_updated:
            console.print("\nFiles updated:")
            for f in result.files_updated:
                console.print(f"  - {f}")

        if result.pr_url:
            console.print(f"\n[bold]PR created:[/bold] {result.pr_url}")

        if result.metadata.get("api_changes"):
            console.print(f"\nAPI files analyzed: {len(result.metadata['api_changes'])}")

        if result.metadata.get("claude_md_created"):
            console.print("\nCLAUDE.md creation results:")
            for item in result.metadata["claude_md_created"]:
                status = (
                    "[green]Created[/green]"
                    if item["success"]
                    else f"[red]Failed: {item.get('error', 'Unknown')}[/red]"
                )
                console.print(f"  - {item['repo']}: {status}")
                if item.get("pr_url"):
                    console.print(f"    PR: {item['pr_url']}")
    else:
        console.print(f"[red]Error: {result.message}[/red]")
        for error in result.errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)


@app.command()
def analyze_costs(
    days: int = typer.Option(
        7,
        "--days",
        "-d",
        help="Number of days to analyze",
    ),
    alert_threshold: float = typer.Option(
        100.0,
        "--alert-threshold",
        "-t",
        help="Daily cost threshold for alerts (USD)",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, or summary",
    ),
):
    """Analyze AWS costs and detect anomalies.

    This command fetches cost data from AWS Cost Explorer and
    identifies any unusual spending patterns.
    """
    from tools.aws_tools import CostExplorerTools

    console.print("[bold blue]Cost Monitor Agent[/bold blue]")
    console.print(f"Analyzing costs for the last {days} days...")
    console.print()

    try:
        cost_tools = CostExplorerTools()
        summary = cost_tools.get_cost_by_service(days=days)
        anomalies = cost_tools.detect_anomalies(days=days, threshold_percentage=20.0)

        # Display summary
        console.print(f"[bold]Total cost:[/bold] ${summary.total:.2f}")
        console.print(f"[bold]Daily average:[/bold] ${summary.daily_average:.2f}")
        console.print(f"[bold]Period:[/bold] {summary.period_start} to {summary.period_end}")
        console.print()

        # Display cost breakdown
        if output_format == "table":
            table = Table(title="Cost by Service")
            table.add_column("Service", style="cyan")
            table.add_column("Cost (USD)", justify="right", style="green")

            for cost in summary.by_service[:15]:  # Top 15
                if cost.amount > 0.01:  # Skip negligible costs
                    table.add_row(cost.service, f"${cost.amount:.2f}")

            console.print(table)

        # Display anomalies
        if anomalies:
            console.print("\n[bold red]Anomalies Detected:[/bold red]")
            for anomaly in anomalies:
                console.print(
                    f"  - {anomaly['type']}: {anomaly['change_percentage']:.1f}% increase "
                    f"(baseline: ${anomaly['baseline_daily']:.2f}/day, "
                    f"recent: ${anomaly['recent_daily']:.2f}/day)"
                )
        else:
            console.print("\n[green]No anomalies detected[/green]")

        # Check threshold
        if summary.daily_average > alert_threshold:
            console.print(
                f"\n[bold yellow]Warning:[/bold yellow] Daily average "
                f"(${summary.daily_average:.2f}) exceeds threshold (${alert_threshold:.2f})"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status():
    """Show current agent status and configuration."""
    settings = get_settings()

    console.print("[bold blue]AI Agents Status[/bold blue]")
    console.print()

    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("AWS Region", settings.aws_region)
    table.add_row("Production Cluster", settings.ecs_production_cluster)
    table.add_row("Staging Cluster", settings.ecs_staging_cluster)
    table.add_row("LLM Provider", settings.llm_provider)
    table.add_row("GitHub Repo", settings.github_repo)
    table.add_row(
        "Monorepo Path",
        str(settings.monorepo_path) if settings.monorepo_path else "Not set",
    )
    table.add_row("Langfuse Host", settings.langfuse_host)

    console.print(table)

    # Check connectivity
    console.print("\n[bold]Connectivity Status:[/bold]")

    # AWS
    try:
        from tools.aws_tools import ECSTools

        ecs = ECSTools()
        services = ecs.list_services(settings.ecs_production_cluster)
        console.print(f"  AWS ECS: [green]Connected[/green] ({len(services)} services found)")
    except Exception as e:
        console.print(f"  AWS ECS: [red]Error[/red] ({e})")

    # GitHub
    if settings.github_token:
        try:
            from tools.github_tools import GitHubTools

            gh = GitHubTools()
            _ = gh.repo.full_name
            console.print(f"  GitHub: [green]Connected[/green] ({settings.github_repo})")
        except Exception as e:
            console.print(f"  GitHub: [red]Error[/red] ({e})")
    else:
        console.print("  GitHub: [yellow]Not configured[/yellow]")


@app.command()
def builder(
    max_issues: int = typer.Option(
        3,
        "--max-issues",
        "-n",
        help="Maximum number of issues to process per run",
    ),
    parallel: bool = typer.Option(
        True,
        "--parallel/--no-parallel",
        help="Enable parallel issue processing (default: enabled)",
    ),
    max_concurrent: int = typer.Option(
        3,
        "--max-concurrent",
        "-c",
        help="Maximum concurrent issues to process in parallel mode",
    ),
    repo_lock: bool = typer.Option(
        False,
        "--repo-lock/--no-repo-lock",
        help="Serialize same-repo issues (default: disabled for max velocity)",
    ),
    target_repo: str | None = typer.Option(
        None,
        "--target-repo",
        "-r",
        help="Specific repo to process (default: all configured repos)",
    ),
    issue_number: int | None = typer.Option(
        None,
        "--issue-number",
        "-i",
        help="Specific issue number to process (requires --target-repo)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run without creating PRs or updating labels",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """Run Generic Builder Agent to process 'aibuild' tagged issues.

    This command finds issues with the 'aibuild' label across configured repositories,
    uses Claude Agent SDK to analyze and fix the issue, runs tests based on the
    repo's configuration (agent-config.json), creates a PR, and updates labels.

    The agent is generic - it auto-detects project type and reads build/test commands
    from each repo's agent-config.json file, so it works with any language/framework.

    By default, all issues process fully in parallel (up to max_concurrent at a time).
    Use --repo-lock to serialize same-repo issues if you encounter merge conflicts.
    """
    from agents.builder_agent import BuilderAgent

    settings = get_settings()
    settings.dry_run = dry_run
    settings.verbose = verbose

    console.print("[bold blue]Builder Agent[/bold blue]")
    console.print(f"Max issues: {max_issues}")
    console.print(f"Parallel: {parallel} (max concurrent: {max_concurrent}, repo-lock: {repo_lock})")
    if target_repo:
        console.print(f"Target repo: {target_repo}")
    else:
        console.print(f"Repos: {settings.builder_repos}")
    if issue_number:
        console.print(f"Issue number: #{issue_number}")
    console.print(f"Dry run: {dry_run}")
    console.print()

    agent = BuilderAgent(
        settings=settings,
        max_issues=max_issues,
        target_repo=target_repo,
        issue_number=issue_number,
        parallel=parallel,
        max_concurrent=max_concurrent,
        repo_lock=repo_lock,
    )
    result = run_async(agent.run())

    if result.success:
        console.print(f"[green]{result.message}[/green]")

        if result.pr_url:
            console.print(f"\n[bold]PR created:[/bold] {result.pr_url}")

        if result.metadata.get("processed_issues"):
            console.print("\nProcessed issues:")
            for issue in result.metadata["processed_issues"]:
                status = "[green]PR Created[/green]" if issue.get("pr_url") else "[yellow]No PR[/yellow]"
                console.print(f"  - {issue['repository']}#{issue['number']}: {issue['title']} - {status}")
    else:
        console.print(f"[red]Error: {result.message}[/red]")
        for error in result.errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)


@app.command()
def planner(
    issue_number: int | None = typer.Argument(
        None,
        help="Specific issue number to process (if not provided, enters watch mode)",
    ),
    watch: bool = typer.Option(
        False,
        "--watch",
        "-w",
        help="Watch mode: continuously monitor for ai-plan issues",
    ),
    interval: int = typer.Option(
        60,
        "--interval",
        "-i",
        help="Polling interval in seconds for watch mode",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """Run Planning Agent to create PRDs from 'ai-plan' labeled issues.

    This command monitors GitHub issues with the 'ai-plan' label, creates a PRD
    (Product Requirements Document) as a comment on the issue, and generates
    implementation issues with the 'aibuild' label for Builder Agents to process.

    Usage:
      ai-agents planner 64           # Process issue #64
      ai-agents planner --watch      # Watch mode (continuous)
      ai-agents planner -w -i 30     # Watch mode, 30s interval

    Label workflow:
      ai-plan → (planning agent) → ai-planned + implementation issues (aibuild)
    """
    from agents.planning_agent import PlanningAgent

    console.print("[bold blue]Planning Agent[/bold blue]")

    agent = PlanningAgent()

    if issue_number:
        # Process specific issue
        console.print(f"Processing issue #{issue_number}")
        console.print()
        result = run_async(agent.process_issue(issue_number))
        if result:
            console.print(f"[green]Successfully processed issue #{issue_number}[/green]")
        else:
            console.print(f"[red]Failed to process issue #{issue_number}[/red]")
            raise typer.Exit(1)
    elif watch:
        # Watch mode
        console.print(f"Watch mode enabled (interval: {interval}s)")
        console.print(f"Input label: {agent.INPUT_LABEL}")
        console.print(f"Output label: {agent.OUTPUT_LABEL}")
        console.print("Press Ctrl+C to stop")
        console.print()
        try:
            run_async(agent.watch_issues(interval=interval))
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped by user[/yellow]")
    else:
        console.print("[yellow]Usage:[/yellow]")
        console.print("  ai-agents planner <issue_number>  # Process specific issue")
        console.print("  ai-agents planner --watch         # Watch for ai-plan issues")
        raise typer.Exit(1)


@app.command()
def flutter_builder(
    max_issues: int = typer.Option(
        1,
        "--max-issues",
        "-n",
        help="Maximum number of issues to process per run",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run without creating PRs or updating labels",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """Run Flutter Builder Agent to process 'aibuild' tagged issues.

    This command finds issues with the 'aibuild' label on the GitHub project board,
    spins up a devcontainer, uses Claude Code to fix the issue, runs tests,
    creates a PR, and updates labels.

    NOTE: This agent is deprecated. Use 'ai-agents builder' instead, which
    handles all repositories including Flutter.
    """
    console.print("[yellow]NOTE: flutter-builder is deprecated. Use 'ai-agents builder' instead.[/yellow]")
    console.print(
        "[red]This command has been removed. Please use 'ai-agents builder' which handles all repos including Flutter.[/red]"
    )
    raise typer.Exit(1)

    # Legacy code kept for reference - FlutterBuilderAgent import removed
    from agents.flutter_builder_agent import FlutterBuilderAgent

    console.print()

    settings = get_settings()
    settings.dry_run = dry_run
    settings.verbose = verbose

    console.print("[bold blue]Flutter Builder Agent[/bold blue]")
    console.print(f"Max issues: {max_issues}")
    console.print(f"Dry run: {dry_run}")
    console.print()

    agent = FlutterBuilderAgent(settings=settings, max_issues=max_issues)
    result = run_async(agent.run())

    if result.success:
        console.print(f"[green]{result.message}[/green]")

        if result.pr_url:
            console.print(f"\n[bold]PR created:[/bold] {result.pr_url}")

        if result.metadata.get("processed_issues"):
            console.print("\nProcessed issues:")
            for issue in result.metadata["processed_issues"]:
                status = "[green]PR Created[/green]" if issue.get("pr_url") else "[yellow]No PR[/yellow]"
                console.print(f"  - #{issue['number']}: {issue['title']} - {status}")
    else:
        console.print(f"[red]Error: {result.message}[/red]")
        for error in result.errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)


@app.command()
def maintenance(
    full_report: bool = typer.Option(
        True,
        "--full-report/--quick-check",
        help="Run full report or quick health check only",
    ),
    send_notification: bool = typer.Option(
        True,
        "--notify/--no-notify",
        help="Send results to Google Chat",
    ),
    check_metrics: bool = typer.Option(
        True,
        "--check-metrics/--no-check-metrics",
        help="Check CloudWatch metrics health",
    ),
    check_docs: bool = typer.Option(
        True,
        "--check-docs/--no-check-docs",
        help="Check documentation health",
    ),
    check_monorepo: bool = typer.Option(
        True,
        "--check-monorepo/--no-check-monorepo",
        help="Check monorepo and submodule health",
    ),
    check_dashboard: bool = typer.Option(
        True,
        "--check-dashboard/--no-check-dashboard",
        help="Check CloudWatch dashboard and GitHub Project health",
    ),
    check_cleanup: bool = typer.Option(
        True,
        "--check-cleanup/--no-check-cleanup",
        help="Check for deprecated configuration",
    ),
    auto_fix_docs: bool = typer.Option(
        False,
        "--auto-fix-docs/--no-auto-fix-docs",
        help="Trigger documentation sync workflow when stale docs detected",
    ),
    auto_update_submodules: bool = typer.Option(
        False,
        "--auto-update-submodules/--no-auto-update-submodules",
        help="Update stale submodules to latest commits and create PR",
    ),
    monorepo_path: Path | None = typer.Option(
        None,
        "--monorepo-path",
        "-m",
        help="Path to monorepo",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Run checks but don't send notifications",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """Run maintenance health checks and generate daily report.

    This command performs comprehensive health checks across:
    - CloudWatch metrics (data freshness, agent success rates)
    - Documentation (CLAUDE.md, SERVICE-MANIFEST.yaml staleness)
    - Monorepo (submodule status, orphaned directories)
    - Dashboard & Project boards (data flow, stale issues)
    - Google Chat (webhook connectivity)
    - Configuration cleanup (deprecated repos)

    Results are sent to Google Chat infrastructure space by default.

    Use --auto-update-submodules to automatically update stale submodules
    and create a PR in the monorepo.
    """
    from agents.maintenance_agent import MaintenanceAgent

    settings = get_settings()

    # Override settings from CLI
    if monorepo_path:
        settings.monorepo_path = monorepo_path
    settings.dry_run = dry_run
    settings.verbose = verbose

    console.print("[bold blue]Maintenance Agent[/bold blue]")
    console.print(f"Full report: {full_report}")
    console.print(f"Send notification: {send_notification and not dry_run}")
    console.print(f"Auto-fix docs: {auto_fix_docs}")
    console.print(f"Auto-update submodules: {auto_update_submodules}")
    console.print()

    agent = MaintenanceAgent(
        settings=settings,
        checks={
            "metrics": check_metrics,
            "documentation": check_docs,
            "monorepo": check_monorepo,
            "dashboard": check_dashboard,
            "google_chat": True,  # Always check Google Chat
            "cleanup": check_cleanup,
        },
        send_notification=send_notification and not dry_run,
        auto_fix_docs=auto_fix_docs,
        auto_update_submodules=auto_update_submodules,
    )

    result = run_async(agent.run())

    if result.success:
        console.print(f"[green]{result.message}[/green]")

        # Display health summary
        if result.metadata.get("health_summary"):
            console.print()
            table = Table(title="Health Check Summary")
            table.add_column("Check", style="cyan")
            table.add_column("Status")

            for check, healthy in result.metadata["health_summary"].items():
                status = "[green]Healthy[/green]" if healthy else "[red]Issues[/red]"
                table.add_row(check.replace("_", " ").title(), status)

            console.print(table)

        if result.metadata.get("detected_issues"):
            console.print("\n[yellow]Issues Detected:[/yellow]")
            for issue in result.metadata["detected_issues"]:
                console.print(f"  - {issue}")

        if result.metadata.get("recommendations"):
            console.print("\n[blue]Recommendations:[/blue]")
            for rec in result.metadata["recommendations"]:
                console.print(f"  - {rec}")

        if result.metadata.get("report_sent"):
            console.print("\n[green]Report sent to Google Chat[/green]")

        if result.metadata.get("doc_sync_triggered"):
            console.print("[green]Documentation sync workflow triggered[/green]")

        if result.metadata.get("submodules_updated"):
            console.print("[green]Stale submodules updated[/green]")
            if result.metadata.get("submodule_update_pr_url"):
                console.print(f"  PR: {result.metadata['submodule_update_pr_url']}")
    else:
        console.print(f"[red]Error: {result.message}[/red]")
        for error in result.errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)


@app.command()
def supervisor(
    message: str = typer.Option(
        ...,
        "--message",
        "-m",
        help="Natural language message from user",
    ),
    user: str = typer.Option(
        ...,
        "--user",
        "-u",
        help="User who sent the message",
    ),
    space: str = typer.Option(
        ...,
        "--space",
        "-s",
        help="Google Chat space name",
    ),
    thread: str | None = typer.Option(
        None,
        "--thread",
        "-t",
        help="Thread name for replies",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Run without executing actions",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """Run Supervisor Agent to process natural language commands.

    The Supervisor Agent receives natural language messages from Google Chat
    and uses Claude Code SDK to understand user intent and execute appropriate
    actions. It can orchestrate other agents, manage GitHub issues, and query
    system status.

    Examples:
        ai-agents supervisor -m "fix issue 42 in flutter" -u "Sumit" -s "spaces/ABC"
        ai-agents supervisor -m "what's the queue status?" -u "Dev" -s "spaces/ABC"
        ai-agents supervisor -m "sync the documentation" -u "Admin" -s "spaces/ABC"
    """
    from agents.supervisor_agent import SupervisorAgent

    settings = get_settings()
    settings.dry_run = dry_run
    settings.verbose = verbose

    console.print("[bold blue]Supervisor Agent[/bold blue]")
    console.print(f"User: {user}")
    console.print(f"Message: {message[:80]}{'...' if len(message) > 80 else ''}")
    console.print(f"Space: {space}")
    if thread:
        console.print(f"Thread: {thread}")
    console.print(f"Dry run: {dry_run}")
    console.print()

    agent = SupervisorAgent(settings=settings)
    result = run_async(
        agent.run(
            message=message,
            user=user,
            space_name=space,
            thread_name=thread,
        )
    )

    if result.success:
        console.print(f"[green]{result.message}[/green]")
        if result.metadata.get("response"):
            console.print("\n[bold]Response preview:[/bold]")
            console.print(result.metadata["response"][:500])
    else:
        console.print(f"[red]Error: {result.message}[/red]")
        for error in result.errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)


@app.command()
def tester(
    target_repo: str | None = typer.Option(
        None,
        "--target-repo",
        "-r",
        help="Specific repo to process (default: all configured repos)",
    ),
    target_pr: int | None = typer.Option(
        None,
        "--pr",
        "-p",
        help="Specific PR number to review",
    ),
    max_prs: int = typer.Option(
        1,
        "--max-prs",
        "-n",
        help="Maximum number of PRs to review per run",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run without posting comments or updating labels",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """Run Tester Agent (TEA) to analyze test coverage in PRs.

    This command finds PRs with the 'needs-tests' label (or a specific PR),
    uses Claude Agent SDK to analyze test coverage gaps, identifies risk areas
    based on the repo's agent-config.json config, and posts review comments with
    specific test suggestions.

    The agent reads per-repo configuration from agent-config.json for:
    - Risk areas (critical/high/medium paths)
    - Coverage threshold
    - Test patterns and commands

    Usage:
      ai-agents tester                        # Review PRs from all repos
      ai-agents tester -r owner/app -p 123     # Review specific PR
      ai-agents tester --dry-run              # Preview without posting

    Label workflow:
      needs-tests → (tester agent) → tests-reviewed
    """
    from agents.tester_agent import TesterAgent

    settings = get_settings()
    settings.dry_run = dry_run
    settings.verbose = verbose

    console.print("[bold blue]Tester Agent (TEA)[/bold blue]")
    console.print(f"Max PRs: {max_prs}")
    if target_repo and target_pr:
        console.print(f"Target: {target_repo}#{target_pr}")
    elif target_repo:
        console.print(f"Target repo: {target_repo}")
    else:
        console.print(f"Repos: {settings.builder_repos}")
    console.print(f"Dry run: {dry_run}")
    console.print()

    agent = TesterAgent(
        settings=settings,
        target_repo=target_repo,
        target_pr=target_pr,
        max_prs=max_prs,
    )
    result = run_async(agent.run())

    if result.success:
        console.print(f"[green]{result.message}[/green]")

        if result.metadata.get("reviewed_prs"):
            console.print("\nReviewed PRs:")
            for pr in result.metadata["reviewed_prs"]:
                status = "[green]Comment Posted[/green]" if pr.get("comment_url") else "[yellow]No Comment[/yellow]"
                console.print(f"  - {pr['repository']}#{pr['number']}: {pr['title']} - {status}")
    else:
        console.print(f"[red]Error: {result.message}[/red]")
        for error in result.errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)


@app.command("test-run")
def test_run(
    trigger: str = typer.Argument(
        ...,
        help="Test trigger: pr, merge-staging, or merge-production",
    ),
    target_repo: str = typer.Option(
        ...,
        "--target-repo",
        "-r",
        help="Repository in owner/repo format",
    ),
    ref: str | None = typer.Option(
        None,
        "--ref",
        help="Git ref (branch, tag, commit SHA) to test",
    ),
    pr_number: int | None = typer.Option(
        None,
        "--pr",
        "-p",
        help="PR number (for posting results)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run without executing tests (preview mode)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """Execute tests based on agent-config.json configuration.

    This command reads the repo's agent-config.json to determine which tests to run
    based on the trigger, and executes them via the configured runner (codebuild,
    self-hosted, device-farm, or local).

    Each repo defines its own test strategy in agent-config.json:

    \b
    {
      "tester": {
        "unit": {
          "trigger": "pr",
          "command": "flutter test --coverage",
          "runner": "codebuild",
          "timeout": 600
        },
        "integration": {
          "trigger": "merge-staging",
          "command": "./run_integration_tests.sh",
          "runner": "device-farm",
          "timeout": 1800
        }
      }
    }

    Usage:
      ai-agents test-run pr -r owner/app --pr 123
      ai-agents test-run merge-staging -r owner/app --ref staging
      ai-agents test-run merge-production -r owner/app --ref main

    Triggers:
      pr              - Unit tests on PR (fast, every PR)
      merge-staging   - Integration tests when merged to staging
      merge-production - E2E tests when merged to production
    """
    from agents.tester_agent import TesterAgent

    settings = get_settings()
    settings.dry_run = dry_run
    settings.verbose = verbose

    console.print("[bold blue]Test Execution (TEA)[/bold blue]")
    console.print(f"Trigger: {trigger}")
    console.print(f"Repository: {target_repo}")
    if ref:
        console.print(f"Ref: {ref}")
    if pr_number:
        console.print(f"PR: #{pr_number}")
    console.print(f"Dry run: {dry_run}")
    console.print()

    agent = TesterAgent(
        settings=settings,
        target_repo=target_repo,
    )
    result = run_async(
        agent.run_tests(
            trigger=trigger,
            repo=target_repo,
            ref=ref,
            pr_number=pr_number,
        )
    )

    if result.success:
        console.print(f"[green]{result.message}[/green]")

        if result.metadata.get("results"):
            console.print("\nTest Results:")
            for test in result.metadata["results"]:
                status = "[green]✓[/green]" if test["success"] else "[red]✗[/red]"
                console.print(f"  {status} {test['name']} ({test['runner']}) - {test['duration']}s")
    else:
        console.print(f"[red]Error: {result.message}[/red]")
        for error in result.errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)


@app.command()
def designer(
    target_repo: str | None = typer.Option(
        None,
        "--target-repo",
        "-r",
        help="Specific repo to process (default: all configured repos)",
    ),
    target_pr: int | None = typer.Option(
        None,
        "--pr",
        "-p",
        help="Specific PR number to review",
    ),
    max_prs: int = typer.Option(
        1,
        "--max-prs",
        "-n",
        help="Maximum number of PRs to review per run",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run without posting comments or updating labels",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """Run Designer Agent (UXA) to review UI/UX and accessibility in PRs.

    This command finds PRs with the 'needs-design-review' label (or a specific PR),
    uses Claude Agent SDK to analyze UI/UX changes, checks accessibility compliance
    (WCAG 2.1 AA), and posts review comments with specific recommendations.

    The agent reads per-repo configuration from agent-config.json for:
    - UI paths to focus on
    - Design system (material, cupertino, custom)
    - Accessibility level (A, AA, AAA)

    Usage:
      ai-agents designer                        # Review PRs from all repos
      ai-agents designer -r owner/app -p 123     # Review specific PR
      ai-agents designer --dry-run              # Preview without posting

    Label workflow:
      needs-design-review → (designer agent) → design-reviewed
    """
    from agents.designer_agent import DesignerAgent

    settings = get_settings()
    settings.dry_run = dry_run
    settings.verbose = verbose

    console.print("[bold blue]Designer Agent (UXA)[/bold blue]")
    console.print(f"Max PRs: {max_prs}")
    if target_repo and target_pr:
        console.print(f"Target: {target_repo}#{target_pr}")
    elif target_repo:
        console.print(f"Target repo: {target_repo}")
    else:
        console.print(f"Repos: {settings.builder_repos}")
    console.print(f"Dry run: {dry_run}")
    console.print()

    agent = DesignerAgent(
        settings=settings,
        target_repo=target_repo,
        target_pr=target_pr,
        max_prs=max_prs,
    )
    result = run_async(agent.run())

    if result.success:
        console.print(f"[green]{result.message}[/green]")

        if result.metadata.get("reviewed_prs"):
            console.print("\nReviewed PRs:")
            for pr in result.metadata["reviewed_prs"]:
                status = "[green]Comment Posted[/green]" if pr.get("comment_url") else "[yellow]No Comment[/yellow]"
                issues = f"({pr.get('issues_found', 0)} issues)" if pr.get("issues_found") else ""
                console.print(f"  - {pr['repository']}#{pr['number']}: {pr['title']} - {status} {issues}")
    else:
        console.print(f"[red]Error: {result.message}[/red]")
        for error in result.errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)


@app.command(name="design-generator")
def design_generator(
    target_repo: str | None = typer.Option(
        None,
        "--target-repo",
        "-r",
        help="Specific repo to process (default: all configured repos)",
    ),
    issue: int | None = typer.Option(
        None,
        "--issue",
        "-i",
        help="Specific issue number to process",
    ),
    max_issues: int = typer.Option(
        5,
        "--max-issues",
        "-n",
        help="Maximum number of issues to process per run",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run without creating PRs or updating labels",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """Run Design Generator Agent (Figma → Stitch → Codex pipeline).

    This command finds issues with the 'needs-design' label, fetches Figma
    design context, generates UI screens via Google Stitch, and creates
    Flutter code via Codex. Results are submitted as PRs.

    Usage:
      ai-agents design-generator                              # Process all repos
      ai-agents design-generator -r owner/app -i 1886 # Specific issue
      ai-agents design-generator --dry-run                    # Preview without PRs

    Label workflow:
      needs-design → design-in-progress → design-generated
    """
    from agents.design_generator_agent import DesignGeneratorAgent

    settings = get_settings()
    settings.dry_run = dry_run
    settings.verbose = verbose

    console.print("[bold blue]Design Generator Agent[/bold blue]")
    console.print(f"Max issues: {max_issues}")
    if target_repo and issue:
        console.print(f"Target: {target_repo}#{issue}")
    elif target_repo:
        console.print(f"Target repo: {target_repo}")
    else:
        console.print(f"Repos: {settings.builder_repos}")
    console.print(f"Dry run: {dry_run}")
    console.print()

    agent = DesignGeneratorAgent(
        settings=settings,
        target_repo=target_repo,
        target_issue=issue,
        max_issues=max_issues,
    )
    result = run_async(agent.run())

    if result.success:
        console.print(f"[green]{result.message}[/green]")

        if result.metadata.get("processed_issues"):
            console.print("\nProcessed issues:")
            for processed_issue in result.metadata["processed_issues"]:
                status = "[green]PR Created[/green]" if processed_issue.get("pr_url") else "[yellow]No PR[/yellow]"
                console.print(
                    f"  - {processed_issue['repository']}#{processed_issue['number']}: "
                    f"{processed_issue['title']} - {status}"
                )
    else:
        console.print(f"[red]Error: {result.message}[/red]")
        for error in result.errors:
            console.print(f"  - {error}")
        raise typer.Exit(1)


@app.command()
def deployer(
    environment: str = typer.Argument(
        ...,
        help="Deployment environment: staging or production",
    ),
    target_repo: str = typer.Argument(
        ...,
        help="Target repository (e.g., owner/service-name)",
    ),
    pr_number: int | None = typer.Option(
        None,
        "--pr",
        "-p",
        help="PR number (required for production deployments)",
    ),
    skip_approval: bool = typer.Option(
        False,
        "--skip-approval",
        help="Skip human approval for production (NOT RECOMMENDED)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run pre-deploy checks only, don't actually deploy",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """Run Deployer Agent (DRA) to orchestrate safe deployments.

    The Deployer Agent handles safe deployment to staging and production
    environments with approval workflows, pre-deploy checks, and rollback
    capabilities.

    IMPORTANT: Production deployments ALWAYS require human approval unless
    --skip-approval is explicitly passed (which is strongly discouraged).

    The agent reads per-repo configuration from agent-config.json for:
    - ECS cluster and service names
    - Health endpoints for smoke tests
    - Approval requirements per environment
    - Rollback configuration

    Usage:
      ai-agents deployer staging owner/app              # Deploy to staging
      ai-agents deployer production owner/app --pr 123  # Request production deploy
      ai-agents deployer staging owner/app --dry-run    # Pre-deploy checks only

    Label workflow:
      deploy-staging / deploy-production → (deployer agent) → deployed / deploy-failed
    """
    from agents.deployer_agent import DeployerAgent

    settings = get_settings()
    settings.dry_run = dry_run
    settings.verbose = verbose

    # Validate environment
    if environment not in ("staging", "production"):
        console.print(f"[red]Error: Invalid environment '{environment}'. Must be 'staging' or 'production'[/red]")
        raise typer.Exit(1)

    # Production requires PR number unless dry run
    if environment == "production" and not pr_number and not dry_run:
        console.print("[red]Error: Production deployments require --pr <number>[/red]")
        console.print("[yellow]This ensures we can track which changes are being deployed.[/yellow]")
        raise typer.Exit(1)

    console.print("[bold blue]Deployer Agent (DRA)[/bold blue]")
    console.print(f"Environment: {environment}")
    console.print(f"Target repo: {target_repo}")
    if pr_number:
        console.print(f"PR: #{pr_number}")
    console.print(f"Skip approval: {skip_approval}")
    console.print(f"Dry run: {dry_run}")
    console.print()

    # Warning for skip-approval
    if skip_approval and environment == "production":
        console.print("[bold red]WARNING: Skipping production approval is dangerous![/bold red]")
        console.print("[yellow]This should only be used in emergencies.[/yellow]")
        console.print()

    agent = DeployerAgent(
        settings=settings,
        target_repo=target_repo,
    )
    result = run_async(
        agent.run(
            environment=environment,
            pr_number=pr_number,
            skip_approval=skip_approval,
        )
    )

    if result.success:
        console.print(f"[green]{result.message}[/green]")

        if result.metadata.get("deployment_status"):
            status = result.metadata["deployment_status"]
            console.print(f"\n[bold]Deployment Status:[/bold] {status}")

        if result.metadata.get("service_url"):
            console.print(f"[bold]Service URL:[/bold] {result.metadata['service_url']}")

        if result.metadata.get("smoke_test_results"):
            console.print("\nSmoke Tests:")
            for test in result.metadata["smoke_test_results"]:
                status = "[green]Passed[/green]" if test.get("passed") else "[red]Failed[/red]"
                console.print(f"  - {test['name']}: {status}")

        if result.metadata.get("awaiting_approval"):
            console.print("\n[yellow]Deployment is awaiting human approval.[/yellow]")
            if result.metadata.get("approval_url"):
                console.print(f"Approve at: {result.metadata['approval_url']}")
    else:
        console.print(f"[red]Error: {result.message}[/red]")
        for error in result.errors:
            console.print(f"  - {error}")

        if result.metadata.get("rollback_performed"):
            console.print("\n[yellow]Automatic rollback was performed.[/yellow]")

        raise typer.Exit(1)


@app.command(name="orchestrator-check")
def orchestrator_check(
    repository: str = typer.Option(
        ...,
        "--repository",
        "-r",
        help="Repository where the subtask was closed (e.g., owner/ai-agents)",
    ),
    subtask_issue: int = typer.Option(
        ...,
        "--subtask-issue",
        "-s",
        help="Issue number of the closed subtask",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    """Check if parent issue can be completed after subtask closes.

    This command is triggered automatically by the webhook when a subtask
    (issue with iteration-* label) is closed. It:
    1. Looks up the parent issue from the orchestrator state store
    2. Checks if all subtasks for that parent are complete
    3. If all complete, triggers the builder to finalize the parent issue
    4. If not all complete, logs progress and exits

    Usage:
      ai-agents orchestrator-check --repository owner/repo --subtask-issue 123
    """
    console.print("[bold blue]Orchestrator Check[/bold blue]")
    console.print(f"Repository: {repository}")
    console.print(f"Subtask Issue: #{subtask_issue}")
    console.print()

    try:
        from training.storage.orchestrator_store import OrchestratorStore
    except ImportError as e:
        console.print("[red]Orchestrator store not available.[/red]")
        console.print(f"[dim]Error: {e}[/dim]")
        raise typer.Exit(1)

    try:
        store = OrchestratorStore()
    except Exception as e:
        console.print(f"[red]Failed to connect to orchestrator store: {e}[/red]")
        raise typer.Exit(1)

    # Find the parent issue for this subtask
    parent_state = store.find_parent_for_subtask(repository, subtask_issue)

    if not parent_state:
        console.print(f"[yellow]No parent found for subtask #{subtask_issue}[/yellow]")
        console.print("This may be a standalone issue or the parent state was cleaned up.")
        raise typer.Exit(0)

    console.print(f"[green]Found parent issue: #{parent_state.parent_issue_number}[/green]")
    console.print(f"Status: {parent_state.status}")
    console.print(f"Current iteration: {parent_state.current_iteration}")

    # Check if all subtasks for this parent are complete
    # We need to query GitHub to check issue states
    import subprocess

    all_complete = True
    incomplete_subtasks = []

    for iteration in parent_state.iterations:
        subtasks = iteration.get("subtasks", [])
        for subtask in subtasks:
            subtask_num = subtask.get("issue_number")
            if not subtask_num:
                continue

            # Check if subtask issue is closed
            result = subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/{repository}/issues/{subtask_num}",
                    "--jq",
                    ".state",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                state = result.stdout.strip()
                if verbose:
                    console.print(f"  Subtask #{subtask_num}: {state}")
                if state != "closed":
                    all_complete = False
                    incomplete_subtasks.append(subtask_num)
            else:
                console.print(f"[yellow]Failed to check subtask #{subtask_num}[/yellow]")

    if not all_complete:
        console.print("\n[yellow]Not all subtasks complete.[/yellow]")
        console.print(f"Incomplete: {incomplete_subtasks}")
        console.print("Waiting for remaining subtasks to close.")
        raise typer.Exit(0)

    console.print("\n[green]All subtasks complete![/green]")
    console.print("Triggering builder to finalize parent issue...")

    # Trigger the builder workflow for the parent issue
    result = subprocess.run(
        [
            "gh",
            "workflow",
            "run",
            "on-demand-agent.yml",
            "--repo",
            os.environ.get("GH_OWNER_REPO", ""),
            "-f",
            "agent=builder",
            "-f",
            f"target_repo={repository}",
            "-f",
            f"issue_number={parent_state.parent_issue_number}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print(f"[green]Builder triggered for parent #{parent_state.parent_issue_number}[/green]")
    else:
        console.print(f"[red]Failed to trigger builder: {result.stderr}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    console.print("[bold]ai-agents[/bold] v0.1.0")
    console.print("AI agents for software project management")


if __name__ == "__main__":
    app()
