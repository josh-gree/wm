import re
from dataclasses import dataclass, field
from pathlib import Path

import modal

from wm.config import ProjectConfig


def _package_name(dep: str) -> str:
    """Extract the package name from a dependency string like 'wandb>=0.19.0'."""
    return re.split(r"[><=!\[;@\s]", dep, maxsplit=1)[0].strip()


@dataclass
class ContainerSpec:
    """Resolved container specification (no Modal objects)."""

    gpu: str | None
    timeout: int
    volume_name: str | None
    data_mount: str
    dockerfile: str | None
    dependencies: list[str] = field(default_factory=list)
    apt_packages: list[str] = field(default_factory=list)


def resolve_container_spec(
    project: ProjectConfig,
    experiment_container: dict | None,
) -> ContainerSpec:
    """Merge project defaults with experiment overrides into a ContainerSpec."""
    ec = experiment_container or {}

    gpu = ec.get("gpu", project.gpu)
    timeout = ec.get("timeout", project.timeout)
    volume_name = ec.get("volume", project.volume)
    data_mount = ec.get("data_mount", project.data_mount)
    dockerfile = ec.get("dockerfile", project.dockerfile)

    deps: list[str] = []
    apt: list[str] = []

    if not dockerfile:
        deps = list(project.dependencies or [])
        apt = list(project.apt_packages or [])

        deps.extend(ec.get("extra_dependencies", []))
        apt.extend(ec.get("apt_packages", []))

        if "git" not in apt:
            apt.append("git")

        if not any(_package_name(d) == "wandb" for d in deps):
            deps.append("wandb")

        if not any(_package_name(d) == "wm" for d in deps):
            deps.append("wm @ git+https://github.com/josh-gree/wm.git")

    return ContainerSpec(
        gpu=gpu,
        timeout=timeout,
        volume_name=volume_name,
        data_mount=data_mount,
        dockerfile=dockerfile,
        dependencies=deps,
        apt_packages=apt,
    )


@dataclass
class ResolvedContainer:
    image: modal.Image
    gpu: str | None
    timeout: int
    volume_name: str | None
    data_mount: str


def build_container(
    project: ProjectConfig,
    experiment_container: dict | None,
    project_dir: Path,
) -> ResolvedContainer:
    spec = resolve_container_spec(project, experiment_container)

    if spec.dockerfile:
        dockerfile_path = project_dir / spec.dockerfile
        image = modal.Image.from_dockerfile(str(dockerfile_path))
    else:
        image = modal.Image.debian_slim()
        if spec.apt_packages:
            image = image.apt_install(*spec.apt_packages)
        if spec.dependencies:
            image = image.pip_install(*spec.dependencies)

    # Always bake project code into the image
    gitignore = project_dir / ".gitignore"
    if not gitignore.exists():
        raise FileNotFoundError(
            f"No .gitignore found in {project_dir}. "
            "A .gitignore is required to avoid copying .venv/, __pycache__/, etc. into the container image. "
            "Run 'wm init' to scaffold one, or create it manually."
        )
    ignore = modal.FilePatternMatcher.from_file(str(gitignore))

    image = image.add_local_dir(
        str(project_dir),
        "/repo",
        copy=True,
        ignore=ignore,
    ).workdir("/repo")

    return ResolvedContainer(
        image=image,
        gpu=spec.gpu,
        timeout=spec.timeout,
        volume_name=spec.volume_name,
        data_mount=spec.data_mount,
    )
