import textwrap
from unittest.mock import patch

import pytest

from wm.config import ProjectConfig, load_project_config


def test_valid_config(tmp_project):
    config = load_project_config(tmp_project)
    assert config.name == "test-project"
    assert config.gpu is None
    assert config.timeout == 3600
    assert config.dependencies == ["torch>=2.10.0"]


def test_missing_name(tmp_path):
    (tmp_path / "project.yaml").write_text("gpu: A100\n")
    with pytest.raises(ValueError, match="must contain a 'name' field"):
        load_project_config(tmp_path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_project_config(tmp_path)


def test_dockerfile_and_deps_conflict(tmp_path):
    (tmp_path / "project.yaml").write_text(
        textwrap.dedent("""\
        name: test
        dockerfile: containers/base.Dockerfile
        dependencies:
          - torch
        """)
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_project_config(tmp_path)


def test_defaults(tmp_path):
    (tmp_path / "project.yaml").write_text("name: minimal\n")
    config = load_project_config(tmp_path)
    assert config.name == "minimal"
    assert config.volume is None
    assert config.data_mount == "/data"
    assert config.wandb_secret == "wandb-secret"
    assert config.dependencies is None


def test_all_fields(tmp_path):
    (tmp_path / "project.yaml").write_text(
        textwrap.dedent("""\
        name: full
        volume: my-vol
        data_mount: /mnt/data
        gpu: A100
        timeout: 7200
        wandb_secret: my-secret
        dependencies:
          - torch
          - numpy
        apt_packages:
          - libgl1
        """)
    )
    config = load_project_config(tmp_path)
    assert config.name == "full"
    assert config.volume == "my-vol"
    assert config.data_mount == "/mnt/data"
    assert config.gpu == "A100"
    assert config.timeout == 7200
    assert config.apt_packages == ["libgl1"]


def test_unknown_keys_warns(tmp_path):
    (tmp_path / "project.yaml").write_text(
        textwrap.dedent("""\
        name: test
        bogus: 42
        dependencis:
          - torch
        """)
    )
    with patch("wm.config.click") as mock_click:
        config = load_project_config(tmp_path)
    assert config.name == "test"
    mock_click.echo.assert_called_once()
    warning_msg = mock_click.echo.call_args[0][0]
    assert "unknown keys" in warning_msg
    assert "bogus" in warning_msg
    assert "dependencis" in warning_msg
