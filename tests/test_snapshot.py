import re
import subprocess

import git
import pytest

from wm.snapshot import SnapshotError, create_snapshot


def test_clean_repo(tmp_git_project):
    result = create_snapshot(tmp_git_project, "train")
    assert len(result.commit_sha) == 40
    assert result.branch_name.startswith("wm/train/")
    assert result.head_sha == git.Repo(tmp_git_project).head.commit.hexsha
    assert result.pushed is False


def test_dirty_and_untracked(tmp_git_project):
    (tmp_git_project / "pyproject.toml").write_text("[project]\nname='dirty'\n")
    (tmp_git_project / "new_file.txt").write_text("hello")

    status_before = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=tmp_git_project,
    ).stdout

    result = create_snapshot(tmp_git_project, "train")

    status_after = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=tmp_git_project,
    ).stdout
    assert status_before == status_after


def test_commit_message_includes_command(tmp_git_project):
    command = "uv run birdclef run baseline --lr 0.001 --detach"
    result = create_snapshot(tmp_git_project, "baseline", command=command)

    repo = git.Repo(tmp_git_project)
    commit = repo.commit(result.commit_sha)
    assert "wm snapshot: baseline at" in commit.message
    assert f"Command: {command}" in commit.message


def test_commit_message_without_command(tmp_git_project):
    result = create_snapshot(tmp_git_project, "train")

    repo = git.Repo(tmp_git_project)
    commit = repo.commit(result.commit_sha)
    assert "wm snapshot: train at" in commit.message
    assert "Command:" not in commit.message


def test_staged_changes_preserved(tmp_git_project):
    (tmp_git_project / "staged.txt").write_text("staged content")
    subprocess.run(["git", "add", "staged.txt"], cwd=tmp_git_project)

    status_before = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=tmp_git_project,
    ).stdout

    create_snapshot(tmp_git_project, "train")

    status_after = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=tmp_git_project,
    ).stdout
    assert status_before == status_after
    assert "A  staged.txt" in status_after


def test_branch_naming(tmp_git_project):
    result = create_snapshot(tmp_git_project, "my_exp")
    assert re.match(r"^wm/my_exp/\d{8}-\d{6}$", result.branch_name)


def test_no_remote(tmp_git_project):
    result = create_snapshot(tmp_git_project, "train")
    assert result.pushed is False


def test_not_a_git_repo(tmp_path):
    with pytest.raises(SnapshotError):
        create_snapshot(tmp_path, "train")


def test_working_tree_fully_preserved(tmp_git_project):
    (tmp_git_project / "pyproject.toml").write_text("[project]\nname='modified'\n")
    (tmp_git_project / "staged.txt").write_text("staged")
    subprocess.run(["git", "add", "staged.txt"], cwd=tmp_git_project)
    (tmp_git_project / "untracked.txt").write_text("untracked")

    status_before = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=tmp_git_project,
    ).stdout

    create_snapshot(tmp_git_project, "train")

    status_after = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=tmp_git_project,
    ).stdout
    assert status_before == status_after
