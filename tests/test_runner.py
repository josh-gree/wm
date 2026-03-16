from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from wm import Experiment
from wm.config import ProjectConfig


class TestExp(Experiment):
    name = "test_exp"

    class Config(BaseModel):
        lr: float = 1e-3

    @staticmethod
    def run(config, wandb_run):
        pass


def _make_project(**kwargs):
    defaults = dict(
        name="test-project",
        gpu=None,
        timeout=3600,
        wandb_secret="wandb-secret",
        dependencies=["torch"],
        volume=None,
        data_mount="/data",
        apt_packages=None,
        dockerfile=None,
    )
    defaults.update(kwargs)
    return ProjectConfig(**defaults)


@patch("wm.runner.modal")
@patch("wm.runner.build_container")
def test_dispatch_constructs_app(mock_build, mock_modal, tmp_path):
    """Test that dispatch creates a Modal app and function."""
    mock_resolved = MagicMock()
    mock_resolved.volume_name = None
    mock_resolved.data_mount = "/data"
    mock_resolved.gpu = None
    mock_resolved.timeout = 3600
    mock_build.return_value = mock_resolved

    project = _make_project()
    config = TestExp.Config(lr=0.001)

    mock_app = MagicMock()
    mock_modal.App.return_value = mock_app

    from wm.runner import dispatch

    dispatch(project, TestExp, config, tmp_path, commit_sha="abc123")

    mock_modal.App.assert_called_once_with("test-project")
    mock_app.run.assert_called_once()
    mock_modal.Secret.from_name.assert_called_once_with("wandb-secret")
