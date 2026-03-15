import os
import textwrap

import pytest


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project directory with project.yaml and an experiment."""
    (tmp_path / "project.yaml").write_text(
        textwrap.dedent("""\
        name: test-project
        gpu: null
        timeout: 3600
        wandb_secret: wandb-secret
        dependencies:
          - torch>=2.10.0
        """)
    )

    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    (experiments_dir / "__init__.py").write_text("")
    (experiments_dir / "my_exp.py").write_text(
        textwrap.dedent("""\
        from dataclasses import dataclass

        @dataclass
        class HyperParams:
            lr: float = 1e-3
            epochs: int = 10
            batch_size: int = 64

        def run(config, wandb_run):
            pass
        """)
    )

    return tmp_path


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
