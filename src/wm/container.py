from dataclasses import dataclass
from pathlib import Path

import click
import modal

from wm.config import ProjectConfig

_BUNDLED_DOCKERFILE = Path(__file__).parent / "Dockerfile"


@dataclass
class ResolvedContainer:
    image: modal.Image
    gpu: str | None
    timeout: int
    volume_name: str | None
    data_mount: str


def build_container(
    project: ProjectConfig,
    project_dir: Path,
) -> ResolvedContainer:
    if project.dockerfile:
        dockerfile_path = project_dir / project.dockerfile
    else:
        dockerfile_path = _BUNDLED_DOCKERFILE

    # Always exclude .git from the Dockerfile context so dependency layers cache.
    # We add .git back as a separate layer below for wandb git integration.
    gitignore = project_dir / ".gitignore"
    if gitignore.exists():
        patterns = [p for p in gitignore.read_text().splitlines() if p.strip() and not p.startswith("#")]
    else:
        click.echo(
            "Warning: no .gitignore found. "
            "Consider adding one to avoid copying .venv/, __pycache__/, etc. into the container image.",
            err=True,
        )
        patterns = []
    patterns.append(".git")

    image = modal.Image.from_dockerfile(
        str(dockerfile_path),
        context_dir=str(project_dir),
        ignore=modal.FilePatternMatcher(*patterns),
    )

    git_dir = project_dir / ".git"
    if git_dir.exists():
        image = image.add_local_dir(str(git_dir), "/repo/.git", copy=True)

    return ResolvedContainer(
        image=image,
        gpu=project.gpu,
        timeout=project.timeout,
        volume_name=project.volume,
        data_mount=project.data_mount,
    )
