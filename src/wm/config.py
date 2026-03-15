from dataclasses import dataclass, field
from pathlib import Path

import click
import yaml


@dataclass
class ProjectConfig:
    name: str
    volume: str | None = None
    data_mount: str = "/data"
    gpu: str | None = None
    timeout: int = 3600
    wandb_secret: str = "wandb-secret"
    dependencies: list[str] | None = None
    apt_packages: list[str] | None = None
    dockerfile: str | None = None


def load_project_config(project_dir: Path) -> ProjectConfig:
    config_path = project_dir / "project.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No project.yaml found in {project_dir}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("project.yaml must be a YAML mapping")

    if "name" not in raw:
        raise ValueError("project.yaml must contain a 'name' field")

    if raw.get("dockerfile") and (raw.get("dependencies") or raw.get("apt_packages")):
        raise ValueError(
            "project.yaml: 'dockerfile' is mutually exclusive with "
            "'dependencies' and 'apt_packages'"
        )

    known_fields = {f.name for f in ProjectConfig.__dataclass_fields__.values()}

    unknown_keys = set(raw.keys()) - known_fields
    if unknown_keys:
        click.echo(f"Warning: unknown keys in project.yaml: {', '.join(sorted(unknown_keys))}", err=True)

    filtered = {k: v for k, v in raw.items() if k in known_fields}

    return ProjectConfig(**filtered)
