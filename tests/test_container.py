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


def _setup_mock_image(mock_modal):
    mock_image = MagicMock()
    mock_modal.Image.from_dockerfile.return_value = mock_image
    mock_image.add_local_dir.return_value = mock_image
    mock_image.run_commands.return_value = mock_image
    return mock_image


@patch("wm.container.modal")
def test_default_build(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    project = _make_project()
    mock_image = _setup_mock_image(mock_modal)

    result = build_container(project, tmp_path)

    # Dockerfile built with deps-only context
    args, kwargs = mock_modal.Image.from_dockerfile.call_args
    assert args[0] == str(_BUNDLED_DOCKERFILE)
    assert kwargs["context_dir"] == str(tmp_path)

    # Code added via add_local_dir, then uv sync
    mock_image.add_local_dir.assert_called()
    mock_image.run_commands.assert_called_once_with("uv sync --frozen --no-dev")

    assert result.gpu is None
    assert result.timeout == 3600


@patch("wm.container.modal")
def test_custom_dockerfile_build(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    (tmp_path / "containers").mkdir()
    (tmp_path / "containers" / "custom.Dockerfile").write_text("FROM python:3.12")

    project = _make_project(dockerfile="containers/custom.Dockerfile")
    _setup_mock_image(mock_modal)

    build_container(project, tmp_path)

    args, kwargs = mock_modal.Image.from_dockerfile.call_args
    assert args[0] == str(tmp_path / "containers" / "custom.Dockerfile")


@patch("wm.container.modal")
def test_code_ignore_uses_gitignore_patterns(mock_modal, tmp_path):
    (tmp_path / ".gitignore").write_text(".venv\n__pycache__\n")
    project = _make_project()
    _setup_mock_image(mock_modal)

    build_container(project, tmp_path)

    # The first add_local_dir call is for code (second is .git if present)
    code_call = mock_modal.FilePatternMatcher.call_args_list[-1]
    patterns = code_call[0]
    assert ".venv" in patterns
    assert "__pycache__" in patterns
    assert ".git" in patterns


@patch("wm.container.modal")
@patch("wm.container.click")
def test_no_gitignore_warns(mock_click, mock_modal, tmp_path):
    project = _make_project()
    _setup_mock_image(mock_modal)

    build_container(project, tmp_path)

    mock_click.echo.assert_called_once()
    assert ".gitignore" in mock_click.echo.call_args[0][0]


@patch("wm.container.modal")
def test_git_dir_added_as_separate_layer(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    _write_git_dir(tmp_path)
    project = _make_project()
    mock_image = _setup_mock_image(mock_modal)

    build_container(project, tmp_path)

    # Last add_local_dir should be .git
    last_add_call = mock_image.add_local_dir.call_args_list[-1]
    assert last_add_call[0][0] == str(tmp_path / ".git")
    assert last_add_call[0][1] == "/repo/.git"


@patch("wm.container.modal")
def test_no_git_dir_skips_git_layer(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    project = _make_project()
    mock_image = _setup_mock_image(mock_modal)

    build_container(project, tmp_path)

    # Only one add_local_dir call (code), no .git
    assert mock_image.add_local_dir.call_count == 1
    assert ".git" not in mock_image.add_local_dir.call_args[0][0]


@patch("wm.container.modal")
def test_volume_and_mount_passed_through(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    project = _make_project(volume="my-vol", data_mount="/mnt/data")
    _setup_mock_image(mock_modal)

    result = build_container(project, tmp_path)

    assert result.volume_name == "my-vol"
    assert result.data_mount == "/mnt/data"


def test_bundled_dockerfile_exists():
    assert _BUNDLED_DOCKERFILE.exists()
