import tomllib
from dataclasses import dataclass
from pathlib import Path

import click


@dataclass
class ProjectConfig:
    name: str
    gpu: str | None = None
    timeout: int = 3600
    wandb_secret: str = "wandb-secret"
    volume: str | None = None
    data_mount: str = "/data"
    dockerfile: str | None = None
    ephemeral_disk: int | None = None


def load_project_config(project_dir: Path) -> ProjectConfig:
    config_path = project_dir / "pyproject.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"No pyproject.toml found in {project_dir}")

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    project_table = raw.get("project")
    if not isinstance(project_table, dict) or "name" not in project_table:
        raise ValueError("pyproject.toml must contain [project] with a 'name' field")

    name = project_table["name"]

    wm_table = raw.get("tool", {}).get("wm", {})

    known_fields = {f.name for f in ProjectConfig.__dataclass_fields__.values()} - {
        "name"
    }
    unknown_keys = set(wm_table.keys()) - known_fields
    if unknown_keys:
        click.echo(
            f"Warning: unknown keys in [tool.wm]: {', '.join(sorted(unknown_keys))}",
            err=True,
        )

    filtered = {k: v for k, v in wm_table.items() if k in known_fields}

    return ProjectConfig(name=name, **filtered)
