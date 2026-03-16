import subprocess
from pathlib import Path

import click


def check_git_status(project_dir: Path, skip: bool = False) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_dir,
        )
        if result.returncode != 0:
            click.echo("Warning: not a git repository, skipping git checks")
            return "unknown"
        commit_sha = result.stdout.strip()
    except FileNotFoundError:
        click.echo("Warning: git not found, skipping git checks")
        return "unknown"

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=project_dir,
    )
    dirty = bool(result.stdout.strip())

    if dirty and not skip:
        click.echo("Warning: you have uncommitted changes.")
        if not click.confirm("Continue?", default=False):
            raise SystemExit(1)

    return commit_sha
