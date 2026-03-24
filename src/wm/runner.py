from pathlib import Path

import click
import modal

from wm.config import ProjectConfig
from wm.container import build_container
from wm.experiment import Experiment


def _run_experiment(
    experiment_cls: type[Experiment],
    serialized_config: dict,
    project_name: str,
    commit_sha: str | None,
    storage_volume_name: str,
    snapshot_branch: str | None = None,
):
    import traceback

    import wandb
    from pathlib import Path
    import modal

    tags = []
    if commit_sha and commit_sha != "unknown":
        tags.append(f"git:{commit_sha}")
    if snapshot_branch:
        tags.append(f"branch:{snapshot_branch}")

    config_instance = experiment_cls.Config.model_validate(serialized_config)

    run = wandb.init(
        project=project_name,
        group=experiment_cls.name,
        config=serialized_config,
        save_code=True,
        tags=tags or None,
    )

    run_dir = Path("/storage") / experiment_cls.name / run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    storage_vol = modal.Volume.from_name(storage_volume_name)

    try:
        experiment_cls.run(config_instance, run, run_dir)
        wandb.finish(exit_code=0)
    except Exception:
        wandb.log({"error": traceback.format_exc()})
        wandb.finish(exit_code=1)
        raise
    finally:
        storage_vol.commit()


def dispatch(
    project: ProjectConfig,
    exp_cls: type[Experiment],
    config,
    project_dir: Path,
    gpu: str | None = None,
    timeout: int = 3600,
    ephemeral_disk: int | None = None,
    commit_sha: str | None = None,
    snapshot_branch: str | None = None,
    detach: bool = False,
):
    click.echo(f"Building container for {exp_cls.name}...")
    resolved = build_container(project, project_dir, snapshot_branch=snapshot_branch)

    app = modal.App(project.name)

    volume_mount = {}
    if resolved.volume_name:
        volume_mount[resolved.data_mount] = modal.Volume.from_name(resolved.volume_name)

    storage_volume_name = f"{project.name}-storage"
    storage_vol = modal.Volume.from_name(storage_volume_name, create_if_missing=True)
    volume_mount["/storage"] = storage_vol

    config_dict = config.model_dump()

    @app.function(
        image=resolved.image,
        volumes=volume_mount,
        secrets=[modal.Secret.from_name(project.wandb_secret)],
        gpu=gpu,
        timeout=timeout,
        ephemeral_disk=ephemeral_disk,
        serialized=True,
    )
    def execute(
        experiment_cls: type[Experiment],
        serialized_config: dict,
        project_name: str,
        commit_sha: str | None,
        storage_volume_name: str,
        snapshot_branch: str | None,
    ):
        _run_experiment(experiment_cls, serialized_config, project_name, commit_sha, storage_volume_name, snapshot_branch)

    click.echo(f"Dispatching {exp_cls.name} to Modal...")
    with modal.enable_output():
        if detach:
            with app.run(detach=True):
                call = execute.spawn(exp_cls, config_dict, project.name, commit_sha, storage_volume_name, snapshot_branch)
                click.echo(f"Detached. Function call ID: {call.object_id}")
        else:
            with app.run():
                execute.remote(exp_cls, config_dict, project.name, commit_sha, storage_volume_name, snapshot_branch)
            click.echo("Done.")
