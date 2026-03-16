from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from wm.config import ProjectConfig
from wm.container import build_container, _BUNDLED_DOCKERFILE


def _make_project(**kwargs):
    defaults = dict(
        name="test",
        gpu=None,
        timeout=3600,
        wandb_secret="wandb-secret",
        dockerfile=None,
        volume=None,
        data_mount="/data",
    )
    defaults.update(kwargs)
    return ProjectConfig(**defaults)


def _write_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text(".venv\n__pycache__\n")


def _write_git_dir(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")


@patch("wm.container.modal")
def test_default_build(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    project = _make_project()
    mock_image = MagicMock()
    mock_modal.Image.from_dockerfile.return_value = mock_image

    result = build_container(project, tmp_path)

    mock_modal.Image.from_dockerfile.assert_called_once()
    args, kwargs = mock_modal.Image.from_dockerfile.call_args
    assert args[0] == str(_BUNDLED_DOCKERFILE)
    assert kwargs["context_dir"] == str(tmp_path)
    assert result.gpu is None
    assert result.timeout == 3600


@patch("wm.container.modal")
def test_custom_dockerfile_build(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    (tmp_path / "containers").mkdir()
    (tmp_path / "containers" / "custom.Dockerfile").write_text("FROM python:3.12")

    project = _make_project(dockerfile="containers/custom.Dockerfile")
    mock_modal.Image.from_dockerfile.return_value = MagicMock()

    build_container(project, tmp_path)

    args, kwargs = mock_modal.Image.from_dockerfile.call_args
    assert args[0] == str(tmp_path / "containers" / "custom.Dockerfile")


@patch("wm.container.modal")
def test_gitignore_used_when_present(mock_modal, tmp_path):
    (tmp_path / ".gitignore").write_text(".venv\n__pycache__\n")

    project = _make_project()
    mock_modal.Image.from_dockerfile.return_value = MagicMock()

    build_container(project, tmp_path)

    # Patterns from .gitignore plus .git should be passed to FilePatternMatcher
    args = mock_modal.FilePatternMatcher.call_args[0]
    assert ".venv" in args
    assert "__pycache__" in args
    assert ".git" in args


@patch("wm.container.modal")
@patch("wm.container.click")
def test_no_gitignore_warns(mock_click, mock_modal, tmp_path):
    project = _make_project()
    mock_modal.Image.from_dockerfile.return_value = MagicMock()

    build_container(project, tmp_path)

    mock_modal.FilePatternMatcher.from_file.assert_not_called()
    mock_click.echo.assert_called_once()
    assert ".gitignore" in mock_click.echo.call_args[0][0]


@patch("wm.container.modal")
def test_git_dir_added_as_separate_layer(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    _write_git_dir(tmp_path)
    project = _make_project()
    mock_image = MagicMock()
    mock_modal.Image.from_dockerfile.return_value = mock_image
    mock_image.add_local_dir.return_value = mock_image

    result = build_container(project, tmp_path)

    mock_image.add_local_dir.assert_called_once_with(
        str(tmp_path / ".git"), "/repo/.git", copy=True
    )


@patch("wm.container.modal")
def test_no_git_dir_skips_add(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    project = _make_project()
    mock_image = MagicMock()
    mock_modal.Image.from_dockerfile.return_value = mock_image

    build_container(project, tmp_path)

    mock_image.add_local_dir.assert_not_called()


@patch("wm.container.modal")
def test_volume_and_mount_passed_through(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    project = _make_project(volume="my-vol", data_mount="/mnt/data")
    mock_modal.Image.from_dockerfile.return_value = MagicMock()

    result = build_container(project, tmp_path)

    assert result.volume_name == "my-vol"
    assert result.data_mount == "/mnt/data"


def test_bundled_dockerfile_exists():
    assert _BUNDLED_DOCKERFILE.exists()
