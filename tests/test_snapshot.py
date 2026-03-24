import re
import subprocess

import git
import pytest

from wm.snapshot import SnapshotError, create_snapshot


def test_clean_repo(tmp_git_project_with_remote):
    result = create_snapshot(tmp_git_project_with_remote, "train")
    assert len(result.commit_sha) == 40
    assert result.branch_name.startswith("wm/train/")
    assert result.head_sha == git.Repo(tmp_git_project_with_remote).head.commit.hexsha


def test_snapshot_captures_uncommitted_changes(tmp_git_project_with_remote):
    """Snapshot commit must contain dirty + untracked files at their modified content."""
    (tmp_git_project_with_remote / "pyproject.toml").write_text("[project]\nname='dirty'\n")
    (tmp_git_project_with_remote / "new_file.txt").write_text("hello from untracked")

    result = create_snapshot(tmp_git_project_with_remote, "train")

    repo = git.Repo(tmp_git_project_with_remote)
    snapshot_commit = repo.commit(result.commit_sha)
    tree = snapshot_commit.tree

    assert tree["pyproject.toml"].data_stream.read().decode() == "[project]\nname='dirty'\n"
    assert tree["new_file.txt"].data_stream.read().decode() == "hello from untracked"


def test_dirty_and_untracked_working_tree_preserved(tmp_git_project_with_remote):
    (tmp_git_project_with_remote / "pyproject.toml").write_text("[project]\nname='dirty'\n")
    (tmp_git_project_with_remote / "new_file.txt").write_text("hello")

    status_before = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=tmp_git_project_with_remote,
    ).stdout

    create_snapshot(tmp_git_project_with_remote, "train")

    status_after = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=tmp_git_project_with_remote,
    ).stdout
    assert status_before == status_after


def test_commit_message_includes_command(tmp_git_project_with_remote):
    command = "uv run birdclef run baseline --lr 0.001 --detach"
    result = create_snapshot(tmp_git_project_with_remote, "baseline", command=command)

    repo = git.Repo(tmp_git_project_with_remote)
    commit = repo.commit(result.commit_sha)
    assert "wm snapshot: baseline at" in commit.message
    assert f"Command: {command}" in commit.message


def test_commit_message_without_command(tmp_git_project_with_remote):
    result = create_snapshot(tmp_git_project_with_remote, "train")

    repo = git.Repo(tmp_git_project_with_remote)
    commit = repo.commit(result.commit_sha)
    assert "wm snapshot: train at" in commit.message
    assert "Command:" not in commit.message


def test_staged_changes_preserved(tmp_git_project_with_remote):
    (tmp_git_project_with_remote / "staged.txt").write_text("staged content")
    subprocess.run(["git", "add", "staged.txt"], cwd=tmp_git_project_with_remote)

    status_before = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=tmp_git_project_with_remote,
    ).stdout

    create_snapshot(tmp_git_project_with_remote, "train")

    status_after = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=tmp_git_project_with_remote,
    ).stdout
    assert status_before == status_after
    assert "A  staged.txt" in status_after


def test_branch_naming(tmp_git_project_with_remote):
    result = create_snapshot(tmp_git_project_with_remote, "my_exp")
    assert re.match(r"^wm/my_exp/\d{8}-\d{6}$", result.branch_name)


def test_no_remote(tmp_git_project):
    with pytest.raises(SnapshotError, match="no remote"):
        create_snapshot(tmp_git_project, "train")


def test_not_a_git_repo(tmp_path):
    with pytest.raises(SnapshotError):
        create_snapshot(tmp_path, "train")


def test_snapshot_skipped_on_snapshot_branch(tmp_git_project_with_remote):
    """create_snapshot returns None when already on a snapshot branch."""
    import subprocess

    subprocess.run(
        ["git", "checkout", "-b", "wm/train/20260324-120000"],
        cwd=tmp_git_project_with_remote,
        capture_output=True,
    )

    result = create_snapshot(tmp_git_project_with_remote, "train")
    assert result is None


def test_snapshot_force_on_snapshot_branch(tmp_git_project_with_remote):
    """force=True allows snapshotting even when already on a snapshot branch."""
    import subprocess

    subprocess.run(
        ["git", "checkout", "-b", "wm/train/20260324-120000"],
        cwd=tmp_git_project_with_remote,
        capture_output=True,
    )

    result = create_snapshot(tmp_git_project_with_remote, "train", force=True)
    assert result.branch_name.startswith("wm/train/")


def test_working_tree_fully_preserved(tmp_git_project_with_remote):
    (tmp_git_project_with_remote / "pyproject.toml").write_text("[project]\nname='modified'\n")
    (tmp_git_project_with_remote / "staged.txt").write_text("staged")
    subprocess.run(["git", "add", "staged.txt"], cwd=tmp_git_project_with_remote)
    (tmp_git_project_with_remote / "untracked.txt").write_text("untracked")

    status_before = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=tmp_git_project_with_remote,
    ).stdout

    create_snapshot(tmp_git_project_with_remote, "train")

    status_after = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=tmp_git_project_with_remote,
    ).stdout
    assert status_before == status_after
