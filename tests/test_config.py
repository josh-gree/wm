import textwrap
from unittest.mock import patch

import pytest

from wm.config import load_project_config


def test_valid_config(tmp_project):
    config = load_project_config(tmp_project)
    assert config.name == "test-project"
    assert config.gpu is None
    assert config.timeout == 3600


def test_missing_project_name(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    with pytest.raises(ValueError, match="must contain .* 'name' field"):
        load_project_config(tmp_path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_project_config(tmp_path)


def test_defaults(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""\
        [project]
        name = "minimal"
        """)
    )
    config = load_project_config(tmp_path)
    assert config.name == "minimal"
    assert config.volume is None
    assert config.data_mount == "/data"
    assert config.wandb_secret == "wandb-secret"
    assert config.gpu is None
    assert config.dockerfile is None


def test_missing_wm_section_uses_defaults(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""\
        [project]
        name = "no-wm-section"
        """)
    )
    config = load_project_config(tmp_path)
    assert config.name == "no-wm-section"
    assert config.timeout == 3600
    assert config.gpu is None


def test_all_fields(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""\
        [project]
        name = "full"

        [tool.wm]
        volume = "my-vol"
        data_mount = "/mnt/data"
        gpu = "A100"
        timeout = 7200
        wandb_secret = "my-secret"
        dockerfile = "custom.Dockerfile"
        """)
    )
    config = load_project_config(tmp_path)
    assert config.name == "full"
    assert config.volume == "my-vol"
    assert config.data_mount == "/mnt/data"
    assert config.gpu == "A100"
    assert config.timeout == 7200
    assert config.wandb_secret == "my-secret"
    assert config.dockerfile == "custom.Dockerfile"


def test_unknown_keys_warns(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""\
        [project]
        name = "test"

        [tool.wm]
        bogus = 42
        dependencis = ["torch"]
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
