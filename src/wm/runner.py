from pathlib import Path

import click
import modal

from wm.config import ProjectConfig
from wm.container import build_container
from wm.discovery import ExperimentInfo


def dispatch(
    project: ProjectConfig,
    experiment: ExperimentInfo,
    config: dict,
    project_dir: Path,
    commit_sha: str | None = None,
):
    click.echo(f"Building container for {experiment.name}...")
    resolved = build_container(project, experiment.container, project_dir)

    app = modal.App(project.name)

    volume_mount = {}
    if resolved.volume_name:
        volume_mount[resolved.data_mount] = modal.Volume.from_name(
            resolved.volume_name
        )

    @app.function(
        image=resolved.image,
        volumes=volume_mount,
        secrets=[modal.Secret.from_name(project.wandb_secret)],
        gpu=resolved.gpu,
        timeout=resolved.timeout,
        serialized=True,
    )
    def execute(experiment_name: str, config: dict, project_name: str, commit_sha: str | None):
        import importlib
        import traceback

        import wandb

        tags = []
        if commit_sha and commit_sha != "unknown":
            tags.append(f"git:{commit_sha}")

        run = wandb.init(
            project=project_name,
            group=experiment_name,
            config=config,
            save_code=True,
            tags=tags or None,
        )

        try:
            mod = importlib.import_module(f"experiments.{experiment_name}")
            mod.run(config, run)
            wandb.finish(exit_code=0)
        except Exception:
            wandb.log({"error": traceback.format_exc()})
            wandb.finish(exit_code=1)
            raise

    click.echo(f"Dispatching {experiment.name} to Modal...")
    with app.run():
        execute.remote(experiment.name, config, project.name, commit_sha)
    click.echo("Done.")
