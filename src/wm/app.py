from pathlib import Path

import functools

import click
from pydantic import Field
from pydantic_settings import BaseSettings

from wm.config import load_project_config
from wm.container import resolve_container_spec
from wm.experiment import Experiment
from wm.git_check import check_git_status
from wm.runner import dispatch


def _parse_config(config_cls, cli_args: list[str]):
    """Parse CLI args into a Config instance plus wm flags using pydantic-settings."""
    fields_dict = {}
    annotations = {}
    for name, field in config_cls.model_fields.items():
        annotations[name] = field.annotation
        fields_dict[name] = field.default

    # Add wm flags so they appear in --help alongside config options
    annotations["force"] = bool
    fields_dict["force"] = Field(False, description="Skip confirmation prompt")
    annotations["skip_git_check"] = bool
    fields_dict["skip_git_check"] = Field(False, description="Skip dirty git tree check")

    settings_cls = type(
        "CliConfig",
        (BaseSettings,),
        {
            **fields_dict,
            "__annotations__": annotations,
            "model_config": {
                "cli_parse_args": True,
                "cli_kebab_case": True,
                "cli_implicit_flags": True,
            },
        },
    )
    parsed = settings_cls(_cli_parse_args=cli_args)
    dump = parsed.model_dump()
    force = dump.pop("force")
    skip_git_check = dump.pop("skip_git_check")
    config = config_cls.model_validate(dump)
    return config, force, skip_git_check


class App:
    def __init__(self, project_config):
        self._project = project_config
        self._experiments: dict[str, type[Experiment]] = {}

    @classmethod
    def from_yaml(cls, path: str) -> "App":
        yaml_path = Path(path)
        if not yaml_path.is_absolute():
            yaml_path = Path.cwd() / yaml_path
        config = load_project_config(yaml_path.parent, yaml_path.name)
        return cls(config)

    def register(self, experiment_cls: type[Experiment]):
        if not isinstance(experiment_cls, type) or not issubclass(
            experiment_cls, Experiment
        ):
            raise TypeError(f"{experiment_cls} is not an Experiment subclass")
        self._experiments[experiment_cls.name] = experiment_cls

    @functools.cached_property
    def cli(self) -> click.Group:
        project = self._project
        experiments = self._experiments

        @click.group()
        def group():
            """wm — Modal + W&B experiment framework."""

        @group.command("list")
        def list_experiments():
            """List available experiments."""
            if not experiments:
                click.echo("No experiments registered.")
                return

            click.echo(f"Project: {project.name}\n")
            for name, exp_cls in sorted(experiments.items()):
                click.echo(f"  {name}")
                for field_name, field in exp_cls.Config.model_fields.items():
                    annotation = exp_cls.Config.__annotations__[field_name]
                    type_name = (
                        annotation.__name__
                        if isinstance(annotation, type)
                        else str(annotation)
                    )
                    click.echo(f"    --{field_name} ({type_name}): {field.default}")
                click.echo()

        @group.group()
        def run():
            """Run an experiment on Modal."""

        for exp_name, exp_cls in experiments.items():
            _register_run_subcommand(run, exp_name, exp_cls, project)

        return group


def _register_run_subcommand(run_group, exp_name, exp_cls, project):
    @click.command(
        exp_name,
        help=f"Run the {exp_name} experiment.",
        # Disable Click's --help so pydantic-settings handles it,
        # giving users a single --help that includes both config fields and wm flags.
        context_settings={
            "allow_extra_args": True,
            "ignore_unknown_options": True,
            "help_option_names": [],
        },
    )
    @click.pass_context
    def cmd(ctx):
        config, force, skip_git_check = _parse_config(exp_cls.Config, ctx.args)

        project_dir = Path.cwd()
        commit_sha = check_git_status(project_dir, skip_git_check)

        click.echo(f"Experiment: {exp_name}")
        click.echo(f"Config: {config.model_dump()}")
        click.echo(f"Git SHA: {commit_sha}")
        click.echo("Container:")
        click.echo(describe_container(project, exp_cls))

        if not force:
            click.confirm("Continue?", abort=True)

        dispatch(project, exp_cls, config, project_dir, commit_sha=commit_sha)

    run_group.add_command(cmd)


def describe_container(project, exp_cls: type[Experiment]):
    spec = resolve_container_spec(project, exp_cls.container_dict())
    lines = []
    lines.append(f"  GPU: {spec.gpu or 'none'}")
    lines.append(f"  Timeout: {spec.timeout}s")
    if spec.dockerfile:
        lines.append(f"  Dockerfile: {spec.dockerfile}")
    else:
        lines.append(f"  Dependencies: {spec.dependencies}")
    return "\n".join(lines)
