import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import git


class SnapshotError(Exception):
    pass


@dataclass
class SnapshotResult:
    commit_sha: str
    branch_name: str
    head_sha: str


def create_snapshot(
    project_dir: Path, experiment_name: str, command: str | None = None
) -> SnapshotResult:
    try:
        repo = git.Repo(project_dir)
    except git.InvalidGitRepositoryError:
        raise SnapshotError(f"{project_dir} is not a git repository")

    head_sha = repo.head.commit.hexsha

    fd, tmp_index_path = tempfile.mkstemp(suffix=".idx")
    os.close(fd)
    os.unlink(tmp_index_path)
    try:
        env = {**os.environ, "GIT_INDEX_FILE": tmp_index_path}

        repo.git.add("-A", env=env)

        tree_sha = repo.git.write_tree(env=env)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"wm/{experiment_name}/{timestamp}"
        message = f"wm snapshot: {experiment_name} at {timestamp}"
        if command:
            message += f"\n\nCommand: {command}"

        commit_sha = repo.git.commit_tree(
            tree_sha, "-p", head_sha, "-m", message
        )

        repo.git.update_ref(f"refs/heads/{branch_name}", commit_sha)
    finally:
        if os.path.exists(tmp_index_path):
            os.unlink(tmp_index_path)

    if not repo.remotes:
        raise SnapshotError("no remote configured — wm requires a git remote")

    repo.remotes.origin.push(f"{branch_name}:{branch_name}")

    return SnapshotResult(
        commit_sha=commit_sha,
        branch_name=branch_name,
        head_sha=head_sha,
    )
