from pathlib import Path
from unittest.mock import MagicMock, patch

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


@patch("wm.container.modal")
def test_default_build(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    project = _make_project()
    mock_image = MagicMock()
    mock_modal.Image.from_dockerfile.return_value = mock_image
    mock_modal.FilePatternMatcher.from_file.return_value = "gitignore_matcher"

    result = build_container(project, tmp_path)

    mock_modal.Image.from_dockerfile.assert_called_once_with(
        str(_BUNDLED_DOCKERFILE),
        context_dir=str(tmp_path),
        ignore="gitignore_matcher",
    )
    assert result.gpu is None
    assert result.timeout == 3600


@patch("wm.container.modal")
def test_custom_dockerfile_build(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    (tmp_path / "containers").mkdir()
    (tmp_path / "containers" / "custom.Dockerfile").write_text("FROM python:3.12")

    project = _make_project(dockerfile="containers/custom.Dockerfile")
    mock_image = MagicMock()
    mock_modal.Image.from_dockerfile.return_value = mock_image
    mock_modal.FilePatternMatcher.from_file.return_value = "gitignore_matcher"

    build_container(project, tmp_path)

    mock_modal.Image.from_dockerfile.assert_called_once_with(
        str(tmp_path / "containers" / "custom.Dockerfile"),
        context_dir=str(tmp_path),
        ignore="gitignore_matcher",
    )


@patch("wm.container.modal")
def test_gitignore_used_when_present(mock_modal, tmp_path):
    (tmp_path / ".gitignore").write_text(".venv\n__pycache__\n")

    project = _make_project()
    mock_image = MagicMock()
    mock_modal.Image.from_dockerfile.return_value = mock_image
    mock_modal.FilePatternMatcher.from_file.return_value = "gitignore_matcher"

    build_container(project, tmp_path)

    mock_modal.FilePatternMatcher.from_file.assert_called_once_with(
        str(tmp_path / ".gitignore")
    )
    call_kwargs = mock_modal.Image.from_dockerfile.call_args[1]
    assert call_kwargs["ignore"] == "gitignore_matcher"


@patch("wm.container.modal")
def test_auto_dockerignore_when_no_gitignore(mock_modal, tmp_path):
    project = _make_project()
    mock_image = MagicMock()
    mock_modal.Image.from_dockerfile.return_value = mock_image

    build_container(project, tmp_path)

    # Should use FilePatternMatcher with auto patterns, not from_file
    mock_modal.FilePatternMatcher.from_file.assert_not_called()
    mock_modal.FilePatternMatcher.assert_called_once()


@patch("wm.container.modal")
def test_volume_and_mount_passed_through(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    project = _make_project(volume="my-vol", data_mount="/mnt/data")
    mock_modal.Image.from_dockerfile.return_value = MagicMock()
    mock_modal.FilePatternMatcher.from_file.return_value = "matcher"

    result = build_container(project, tmp_path)

    assert result.volume_name == "my-vol"
    assert result.data_mount == "/mnt/data"


def test_bundled_dockerfile_exists():
    assert _BUNDLED_DOCKERFILE.exists()
