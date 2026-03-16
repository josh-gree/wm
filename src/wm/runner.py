from pathlib import Path

import click
import modal

from wm.config import ProjectConfig
from wm.container import build_container
from wm.experiment import Experiment


def dispatch(
    project: ProjectConfig,
    exp_cls: type[Experiment],
    config,
    project_dir: Path,
    gpu: str | None = None,
    timeout: int = 3600,
    commit_sha: str | None = None,
):
    click.echo(f"Building container for {exp_cls.name}...")
    resolved = build_container(project, project_dir)

    app = modal.App(project.name)

    volume_mount = {}
    if resolved.volume_name:
        volume_mount[resolved.data_mount] = modal.Volume.from_name(resolved.volume_name)

    config_dict = config.model_dump()

    @app.function(
        image=resolved.image,
        volumes=volume_mount,
        secrets=[modal.Secret.from_name(project.wandb_secret)],
        gpu=gpu,
        timeout=timeout,
        serialized=True,
    )
    def execute(
        experiment_cls: type[Experiment],
        serialized_config: dict,
        project_name: str,
        commit_sha: str | None,
    ):
        import traceback

        import wandb

        tags = []
        if commit_sha and commit_sha != "unknown":
            tags.append(f"git:{commit_sha}")

        config_instance = experiment_cls.Config.model_validate(serialized_config)

        run = wandb.init(
            project=project_name,
            group=experiment_cls.name,
            config=serialized_config,
            save_code=True,
            tags=tags or None,
        )

        try:
            experiment_cls.run(config_instance, run)
            wandb.finish(exit_code=0)
        except Exception:
            wandb.log({"error": traceback.format_exc()})
            wandb.finish(exit_code=1)
            raise

    click.echo(f"Dispatching {exp_cls.name} to Modal...")
    with modal.enable_output():
        with app.run():
            execute.remote(exp_cls, config_dict, project.name, commit_sha)
    click.echo("Done.")
