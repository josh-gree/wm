from dataclasses import fields
from pathlib import Path

import click

from wm.config import load_project_config
from wm.discovery import discover_experiments
from wm.git_check import check_git_status
from wm.runner import dispatch
from wm.validation import resolve_config


def parse_extra_args(args: list[str]) -> dict[str, str]:
    overrides = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                overrides[key] = args[i + 1]
                i += 2
            else:
                overrides[key] = "true"
                i += 1
        else:
            i += 1
    return overrides


def describe_container(project, experiment_container):
    ec = experiment_container or {}
    lines = []
    gpu = ec.get("gpu", project.gpu)
    timeout = ec.get("timeout", project.timeout)
    dockerfile = ec.get("dockerfile", project.dockerfile)
    lines.append(f"  GPU: {gpu or 'none'}")
    lines.append(f"  Timeout: {timeout}s")
    if dockerfile:
        lines.append(f"  Dockerfile: {dockerfile}")
    else:
        deps = list(project.dependencies or [])
        deps.extend(ec.get("extra_dependencies", []))
        lines.append(f"  Dependencies: {deps}")
    return "\n".join(lines)


@click.group()
def cli():
    """wm — Modal + W&B experiment framework."""


@cli.command("list")
def list_experiments():
    """List available experiments."""
    project_dir = Path.cwd()
    project = load_project_config(project_dir)
    experiments = discover_experiments(project_dir)

    if not experiments:
        click.echo("No experiments found in experiments/")
        return

    click.echo(f"Project: {project.name}\n")
    for name, info in sorted(experiments.items()):
        params = fields(info.hyper_params)
        defaults = info.hyper_params()
        click.echo(f"  {name}")
        for p in params:
            val = getattr(defaults, p.name)
            click.echo(f"    --{p.name} ({p.type.__name__}): {val}")
        click.echo()


@cli.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@click.argument("experiment_name")
@click.option("--dry-run", is_flag=True, help="Validate config without launching.")
@click.option("--force", is_flag=True, help="Skip dirty git tree prompt.")
@click.pass_context
def run(ctx, experiment_name, dry_run, force):
    """Run an experiment on Modal."""
    project_dir = Path.cwd()
    project = load_project_config(project_dir)
    experiments = discover_experiments(project_dir)

    if experiment_name not in experiments:
        click.echo(f"Unknown experiment: {experiment_name}")
        click.echo(f"Available: {', '.join(sorted(experiments.keys()))}")
        raise SystemExit(1)

    exp = experiments[experiment_name]

    overrides = parse_extra_args(ctx.args)
    config = resolve_config(exp.hyper_params, overrides)

    check_git_status(project_dir, force)

    if dry_run:
        click.echo(f"Experiment: {experiment_name}")
        click.echo(f"Config: {config}")
        click.echo(f"Container:")
        click.echo(describe_container(project, exp.container))
        return

    dispatch(project, exp, config, project_dir)


@cli.command()
@click.argument("project_name")
def init(project_name):
    """Scaffold a new wm project."""
    from importlib.resources import files as resource_files

    target = Path.cwd() / project_name
    if target.exists():
        click.echo(f"Error: {target} already exists")
        raise SystemExit(1)

    target.mkdir(parents=True)
    (target / "experiments").mkdir()
    (target / "shared").mkdir()
    (target / "containers").mkdir()
    (target / "data").mkdir()

    templates = resource_files("wm.templates")

    # project.yaml
    content = templates.joinpath("project.yaml").read_text()
    (target / "project.yaml").write_text(content.replace("{name}", project_name))

    # pyproject.toml
    content = templates.joinpath("pyproject.toml.template").read_text()
    (target / "pyproject.toml").write_text(content.replace("{name}", project_name))

    # example experiment
    content = templates.joinpath("example_experiment.py").read_text()
    (target / "experiments" / "example.py").write_text(content)

    # .gitignore
    content = templates.joinpath("gitignore").read_text()
    (target / ".gitignore").write_text(content)

    # __init__.py files
    (target / "experiments" / "__init__.py").write_text("")
    (target / "shared" / "__init__.py").write_text("")

    click.echo(f"Created project: {project_name}/")
    click.echo("  experiments/example.py  — example experiment")
    click.echo("  project.yaml            — project config")
    click.echo("  pyproject.toml          — Python project file")
    click.echo()
    click.echo("Next steps:")
    click.echo(f"  cd {project_name}")
    click.echo("  uv sync")
    click.echo("  wm list")
