import textwrap

import pytest

from pydantic import BaseModel
from wm import App, Experiment


class MyExp(Experiment):
    name = "my_exp"

    class Config(BaseModel):
        lr: float = 1e-3
        epochs: int = 10
        batch_size: int = 64

    @staticmethod
    def run(config, wandb_run):
        pass


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project directory with pyproject.toml."""
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""\
        [project]
        name = "test-project"

        [tool.wm]
        timeout = 3600
        wandb_secret = "wandb-secret"
        """)
    )
    (tmp_path / ".gitignore").write_text(".venv\n__pycache__\n")
    return tmp_path


@pytest.fixture
def tmp_app(tmp_project):
    """Create an App with MyExp registered."""
    from wm.config import load_project_config

    config = load_project_config(tmp_project)
    app = App(config)
    app.register(MyExp)
    return app


@pytest.fixture
def tmp_git_project(tmp_project):
    """A tmp_project that is also a git repo."""
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_project, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_project,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_project,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=tmp_project, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_project,
        capture_output=True,
    )
    return tmp_project
