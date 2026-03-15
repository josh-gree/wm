from dataclasses import dataclass
from pathlib import Path

import modal

from wm.config import ProjectConfig


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
    ec = experiment_container or {}

    # Resolve effective settings (experiment overrides project defaults)
    gpu = ec.get("gpu", project.gpu)
    timeout = ec.get("timeout", project.timeout)
    volume_name = ec.get("volume", project.volume)
    data_mount = ec.get("data_mount", project.data_mount)
    dockerfile = ec.get("dockerfile", project.dockerfile)

    if dockerfile:
        dockerfile_path = project_dir / dockerfile
        image = modal.Image.from_dockerfile(str(dockerfile_path))
    else:
        deps = list(project.dependencies or [])
        apt = list(project.apt_packages or [])

        # Add experiment extra dependencies
        extra_deps = ec.get("extra_dependencies", [])
        deps.extend(extra_deps)

        extra_apt = ec.get("apt_packages", [])
        apt.extend(extra_apt)

        # Always include git and wandb
        if "git" not in apt:
            apt.append("git")

        # Always include wandb in pip deps
        wandb_present = any(d.startswith("wandb") for d in deps)
        if not wandb_present:
            deps.append("wandb")

        image = modal.Image.debian_slim()
        if apt:
            image = image.apt_install(*apt)
        if deps:
            image = image.pip_install(*deps)

    # Always bake project code into the image
    image = image.add_local_dir(
        str(project_dir),
        "/repo",
        copy=True,
    ).workdir("/repo")

    return ResolvedContainer(
        image=image,
        gpu=gpu,
        timeout=timeout,
        volume_name=volume_name,
        data_mount=data_mount,
    )
