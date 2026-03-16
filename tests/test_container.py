from unittest.mock import MagicMock, patch

import pytest

from wm.config import ProjectConfig
from wm.container import build_container, resolve_container_spec


def _make_project(**kwargs):
    defaults = dict(
        name="test",
        gpu=None,
        timeout=3600,
        wandb_secret="wandb-secret",
        dependencies=["torch>=2.10.0"],
        apt_packages=None,
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
    mock_modal.Image.debian_slim.return_value = mock_image
    mock_image.apt_install.return_value = mock_image
    mock_image.pip_install.return_value = mock_image
    mock_image.add_local_dir.return_value = mock_image
    mock_image.workdir.return_value = mock_image

    result = build_container(project, None, tmp_path)

    mock_modal.Image.debian_slim.assert_called_once()
    mock_image.apt_install.assert_called_once()
    # git should always be in apt
    apt_args = mock_image.apt_install.call_args[0]
    assert "git" in apt_args
    # wandb and torch should be in pip
    pip_args = mock_image.pip_install.call_args[0]
    assert "torch>=2.10.0" in pip_args
    assert "wandb" in pip_args
    assert any("wm @ git+https://github.com/josh-gree/wm.git" in d for d in pip_args)

    # ignore kwarg should be passed
    add_local_dir_kwargs = mock_image.add_local_dir.call_args[1]
    assert "ignore" in add_local_dir_kwargs

    assert result.gpu is None
    assert result.timeout == 3600


@patch("wm.container.modal")
def test_dockerfile_build(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    (tmp_path / "containers").mkdir()
    (tmp_path / "containers" / "custom.Dockerfile").write_text("FROM python:3.12")

    project = _make_project(dockerfile="containers/custom.Dockerfile")
    mock_image = MagicMock()
    mock_modal.Image.from_dockerfile.return_value = mock_image
    mock_image.add_local_dir.return_value = mock_image
    mock_image.workdir.return_value = mock_image

    build_container(project, None, tmp_path)

    mock_modal.Image.from_dockerfile.assert_called_once_with(
        str(tmp_path / "containers" / "custom.Dockerfile")
    )


@patch("wm.container.modal")
def test_experiment_overrides(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    project = _make_project(gpu=None, timeout=3600)
    mock_image = MagicMock()
    mock_modal.Image.debian_slim.return_value = mock_image
    mock_image.apt_install.return_value = mock_image
    mock_image.pip_install.return_value = mock_image
    mock_image.add_local_dir.return_value = mock_image
    mock_image.workdir.return_value = mock_image

    container = {
        "extra_dependencies": ["flash-attn"],
        "gpu": "A100",
        "timeout": 7200,
    }

    result = build_container(project, container, tmp_path)

    pip_args = mock_image.pip_install.call_args[0]
    assert "flash-attn" in pip_args
    assert result.gpu == "A100"
    assert result.timeout == 7200


@patch("wm.container.modal")
def test_wandb_always_included(mock_modal, tmp_path):
    _write_gitignore(tmp_path)
    project = _make_project(dependencies=["wandb>=0.19.0", "torch"])
    mock_image = MagicMock()
    mock_modal.Image.debian_slim.return_value = mock_image
    mock_image.apt_install.return_value = mock_image
    mock_image.pip_install.return_value = mock_image
    mock_image.add_local_dir.return_value = mock_image
    mock_image.workdir.return_value = mock_image

    build_container(project, None, tmp_path)

    pip_args = mock_image.pip_install.call_args[0]
    # wandb already present, should not be duplicated
    wandb_count = sum(
        1
        for d in pip_args
        if d == "wandb" or d.startswith("wandb>") or d.startswith("wandb=")
    )
    assert wandb_count == 1


def test_wandb_not_confused_by_similar_name():
    """wandb-utils should not be treated as 'wandb'."""
    project = _make_project(dependencies=["wandb-utils", "torch"])
    spec = resolve_container_spec(project, None)
    assert "wandb" in spec.dependencies


def test_resolve_container_spec_defaults():
    project = _make_project()
    spec = resolve_container_spec(project, None)
    assert spec.gpu is None
    assert spec.timeout == 3600
    assert spec.volume_name is None
    assert spec.data_mount == "/data"
    assert spec.dockerfile is None
    assert "torch>=2.10.0" in spec.dependencies
    assert "wandb" in spec.dependencies
    assert any("wm @ git+https://github.com/josh-gree/wm.git" in d for d in spec.dependencies)
    assert "git" in spec.apt_packages


def test_resolve_container_spec_experiment_overrides():
    project = _make_project()
    ec = {"gpu": "A100", "timeout": 7200, "extra_dependencies": ["flash-attn"]}
    spec = resolve_container_spec(project, ec)
    assert spec.gpu == "A100"
    assert spec.timeout == 7200
    assert "flash-attn" in spec.dependencies


def test_resolve_container_spec_dockerfile():
    project = _make_project(dockerfile="containers/base.Dockerfile")
    spec = resolve_container_spec(project, None)
    assert spec.dockerfile == "containers/base.Dockerfile"
    assert spec.dependencies == []
    assert spec.apt_packages == []


@patch("wm.container.modal")
def test_gitignore_used_when_present(mock_modal, tmp_path):
    (tmp_path / ".gitignore").write_text(".venv\n__pycache__\n")

    project = _make_project()
    mock_image = MagicMock()
    mock_modal.Image.debian_slim.return_value = mock_image
    mock_image.apt_install.return_value = mock_image
    mock_image.pip_install.return_value = mock_image
    mock_image.add_local_dir.return_value = mock_image
    mock_image.workdir.return_value = mock_image
    mock_modal.FilePatternMatcher.from_file.return_value = "gitignore_matcher"

    build_container(project, None, tmp_path)

    mock_modal.FilePatternMatcher.from_file.assert_called_once_with(
        str(tmp_path / ".gitignore")
    )
    add_kwargs = mock_image.add_local_dir.call_args[1]
    assert add_kwargs["ignore"] == "gitignore_matcher"


@patch("wm.container.modal")
def test_missing_gitignore_raises(mock_modal, tmp_path):
    project = _make_project()
    mock_image = MagicMock()
    mock_modal.Image.debian_slim.return_value = mock_image
    mock_image.apt_install.return_value = mock_image
    mock_image.pip_install.return_value = mock_image

    with pytest.raises(FileNotFoundError, match=".gitignore"):
        build_container(project, None, tmp_path)
