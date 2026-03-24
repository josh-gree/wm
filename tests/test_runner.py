from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from pydantic import BaseModel

from wm import Experiment
from wm.config import ProjectConfig


class TestExp(Experiment):
    name = "test_exp"

    class Config(BaseModel):
        lr: float = 1e-3

    @staticmethod
    def run(config, wandb_run, run_dir):
        pass


def _make_project(**kwargs):
    defaults = dict(
        name="test-project",
        gpu=None,
        timeout=3600,
        wandb_secret="wandb-secret",
        volume=None,
        data_mount="/data",
        dockerfile=None,
        ephemeral_disk=None,
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
    mock_build.return_value = mock_resolved

    project = _make_project()
    config = TestExp.Config(lr=0.001)

    mock_app = MagicMock()
    mock_modal.App.return_value = mock_app

    from wm.runner import dispatch

    dispatch(
        project,
        TestExp,
        config,
        tmp_path,
        gpu=None,
        timeout=3600,
        ephemeral_disk=None,
        commit_sha="abc123",
    )

    mock_build.assert_called_once_with(project, tmp_path, snapshot_branch=None)
    mock_modal.App.assert_called_once_with("test-project")
    mock_app.run.assert_called_once()
    mock_modal.Secret.from_name.assert_called_once_with("wandb-secret")


@patch("wm.runner.modal")
@patch("wm.runner.build_container")
def test_dispatch_mounts_storage_volume(mock_build, mock_modal, tmp_path):
    """Test that dispatch always mounts the storage volume."""
    mock_resolved = MagicMock()
    mock_resolved.volume_name = None
    mock_resolved.data_mount = "/data"
    mock_build.return_value = mock_resolved

    project = _make_project()
    config = TestExp.Config(lr=0.001)

    mock_app = MagicMock()
    mock_modal.App.return_value = mock_app

    captured_volumes = {}

    def capture_function(**kwargs):
        captured_volumes.update(kwargs.get("volumes", {}))
        def decorator(f):
            mock_f = MagicMock()
            mock_f.remote = MagicMock()
            return mock_f
        return decorator

    mock_app.function.side_effect = capture_function

    from wm.runner import dispatch

    dispatch(project, TestExp, config, tmp_path)

    mock_modal.Volume.from_name.assert_any_call("test-project-storage", create_if_missing=True)
    assert "/storage" in captured_volumes


@patch("wm.runner.modal")
@patch("wm.runner.build_container")
def test_dispatch_storage_volume_mounted_without_data_volume(mock_build, mock_modal, tmp_path):
    """Test that storage volume is mounted even when there is no data volume."""
    mock_resolved = MagicMock()
    mock_resolved.volume_name = None
    mock_resolved.data_mount = "/data"
    mock_build.return_value = mock_resolved

    project = _make_project(volume=None)
    config = TestExp.Config(lr=0.001)

    mock_app = MagicMock()
    mock_modal.App.return_value = mock_app

    captured_volumes = {}

    def capture_function(**kwargs):
        captured_volumes.update(kwargs.get("volumes", {}))
        def decorator(f):
            mock_f = MagicMock()
            mock_f.remote = MagicMock()
            return mock_f
        return decorator

    mock_app.function.side_effect = capture_function

    from wm.runner import dispatch

    dispatch(project, TestExp, config, tmp_path)

    assert "/storage" in captured_volumes
    assert "/data" not in captured_volumes


def test_run_experiment_creates_run_dir():
    """Test that _run_experiment creates the correct run_dir."""
    from wm.runner import _run_experiment

    mock_wandb_run = MagicMock()
    mock_wandb_run.id = "abc123"
    mock_storage_vol = MagicMock()

    with (
        patch("wandb.init", return_value=mock_wandb_run),
        patch("wandb.finish"),
        patch("modal.Volume.from_name", return_value=mock_storage_vol),
        patch.object(Path, "mkdir") as mock_mkdir,
    ):
        _run_experiment(TestExp, {"lr": 0.001}, "test-project", None, "test-project-storage")

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_run_experiment_passes_run_dir_to_run():
    """Test that run_dir is passed to experiment_cls.run."""
    from wm.runner import _run_experiment as real_run_experiment

    mock_wandb_run = MagicMock()
    mock_wandb_run.id = "run456"

    mock_storage_vol = MagicMock()
    run_calls = []

    class CapturingExp(Experiment):
        name = "cap_exp"

        class Config(BaseModel):
            lr: float = 1e-3

        @staticmethod
        def run(config, wandb_run, run_dir):
            run_calls.append((config, wandb_run, run_dir))

    with (
        patch("wandb.init", return_value=mock_wandb_run),
        patch("wandb.finish"),
        patch("modal.Volume.from_name", return_value=mock_storage_vol),
        patch.object(Path, "mkdir"),
    ):
        real_run_experiment(
            CapturingExp,
            {"lr": 0.001},
            "test-project",
            None,
            "test-project-storage",
        )

    assert len(run_calls) == 1
    _, _, run_dir = run_calls[0]
    assert run_dir == Path("/storage") / "cap_exp" / "run456"


def test_run_experiment_commits_on_success():
    """Test that storage_vol.commit() is called after a successful run."""
    from wm.runner import _run_experiment

    mock_wandb_run = MagicMock()
    mock_wandb_run.id = "run789"
    mock_storage_vol = MagicMock()

    with (
        patch("wandb.init", return_value=mock_wandb_run),
        patch("wandb.finish"),
        patch("modal.Volume.from_name", return_value=mock_storage_vol),
        patch.object(Path, "mkdir"),
    ):
        _run_experiment(TestExp, {"lr": 0.001}, "test-project", None, "test-project-storage")

    mock_storage_vol.commit.assert_called_once()


def test_run_experiment_commits_on_error():
    """Test that storage_vol.commit() is called even when experiment raises."""
    from wm.runner import _run_experiment

    mock_wandb_run = MagicMock()
    mock_wandb_run.id = "runerr"
    mock_storage_vol = MagicMock()

    class ErrorExp(Experiment):
        name = "error_exp"

        class Config(BaseModel):
            lr: float = 1e-3

        @staticmethod
        def run(config, wandb_run, run_dir):
            raise RuntimeError("boom")

    with (
        patch("wandb.init", return_value=mock_wandb_run),
        patch("wandb.finish"),
        patch("wandb.log"),
        patch("modal.Volume.from_name", return_value=mock_storage_vol),
        patch.object(Path, "mkdir"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            _run_experiment(ErrorExp, {"lr": 0.001}, "test-project", None, "test-project-storage")

    mock_storage_vol.commit.assert_called_once()
