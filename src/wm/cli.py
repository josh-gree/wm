from pathlib import Path

import click


@click.group()
def cli():
    """wm — project scaffolding tools."""


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

    content = templates.joinpath("project.yaml").read_text()
    (target / "project.yaml").write_text(content.replace("{name}", project_name))

    content = templates.joinpath("pyproject.toml.template").read_text()
    (target / "pyproject.toml").write_text(content.replace("{name}", project_name))

    content = templates.joinpath("example_experiment.py").read_text()
    (target / "experiments" / "example.py").write_text(content)

    content = templates.joinpath("app.py.template").read_text()
    (target / "app.py").write_text(content.replace("{name}", project_name))

    content = templates.joinpath("gitignore").read_text()
    (target / ".gitignore").write_text(content)

    (target / "experiments" / "__init__.py").write_text("")
    (target / "shared" / "__init__.py").write_text("")

    click.echo(f"Created project: {project_name}/")
    click.echo("  app.py                  — CLI entry point")
    click.echo("  experiments/example.py  — example experiment")
    click.echo("  project.yaml            — project config")
    click.echo("  pyproject.toml          — Python project file")
    click.echo()
    click.echo("Next steps:")
    click.echo(f"  cd {project_name}")
    click.echo("  uv sync")
    click.echo(f"  {project_name} list")
