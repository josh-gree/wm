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

    gitignore = project_dir / ".gitignore"
    if gitignore.exists():
        ignore = modal.FilePatternMatcher.from_file(str(gitignore))
    else:
        click.echo(
            "Warning: no .gitignore found. "
            "Consider adding one to avoid copying .venv/, __pycache__/, etc. into the container image.",
            err=True,
        )
        ignore = None

    kwargs = dict(context_dir=str(project_dir))
    if ignore is not None:
        kwargs["ignore"] = ignore

    image = modal.Image.from_dockerfile(str(dockerfile_path), **kwargs)

    return ResolvedContainer(
        image=image,
        gpu=project.gpu,
        timeout=project.timeout,
        volume_name=project.volume,
        data_mount=project.data_mount,
    )
