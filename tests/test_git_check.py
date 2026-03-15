import subprocess

import pytest

from wm.git_check import check_git_status


def test_clean_repo(tmp_git_project):
    sha = check_git_status(tmp_git_project, force=False)
    assert len(sha) == 40  # full SHA


def test_dirty_repo_with_force(tmp_git_project):
    (tmp_git_project / "dirty.txt").write_text("dirty")
    sha = check_git_status(tmp_git_project, force=True)
    assert len(sha) == 40


def test_dirty_repo_aborts(tmp_git_project, monkeypatch):
    (tmp_git_project / "dirty.txt").write_text("dirty")
    monkeypatch.setattr("click.confirm", lambda *a, **kw: False)
    with pytest.raises(SystemExit):
        check_git_status(tmp_git_project, force=False)


def test_dirty_repo_continues(tmp_git_project, monkeypatch):
    (tmp_git_project / "dirty.txt").write_text("dirty")
    monkeypatch.setattr("click.confirm", lambda *a, **kw: True)
    sha = check_git_status(tmp_git_project, force=False)
    assert len(sha) == 40


def test_not_a_git_repo(tmp_path):
    sha = check_git_status(tmp_path, force=False)
    assert sha == "unknown"
