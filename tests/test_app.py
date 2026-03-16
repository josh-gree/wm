import pytest
from click.testing import CliRunner

from wm import App


def test_register(tmp_app):
    assert "my_exp" in tmp_app._experiments


def test_register_non_experiment(tmp_app):
    with pytest.raises(TypeError):
        tmp_app.register(str)


def test_list(tmp_app):
    runner = CliRunner()
    result = runner.invoke(tmp_app.cli, ["list"])
    assert result.exit_code == 0
    assert "my_exp" in result.output
    assert "lr" in result.output
    assert "epochs" in result.output


def test_list_no_experiments(tmp_project):
    from wm.config import load_project_config

    config = load_project_config(tmp_project)
    app = App(config)
    runner = CliRunner()
    result = runner.invoke(app.cli, ["list"])
    assert result.exit_code == 0
    assert "No experiments registered" in result.output


def test_run_dry_run(tmp_git_project, tmp_app, monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_git_project.parent):
        monkeypatch.chdir(tmp_git_project)
        result = runner.invoke(tmp_app.cli, ["run", "my_exp", "--dry-run", "--force"])
        assert result.exit_code == 0
        assert "Experiment: my_exp" in result.output
        assert "Config:" in result.output
        assert "Git SHA:" in result.output


def test_run_dry_run_with_overrides(tmp_git_project, tmp_app, monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_git_project.parent):
        monkeypatch.chdir(tmp_git_project)
        result = runner.invoke(
            tmp_app.cli,
            ["run", "my_exp", "--dry-run", "--force", "--lr", "0.01"],
        )
        assert result.exit_code == 0
        assert "0.01" in result.output


def test_run_unknown_experiment(tmp_git_project, tmp_app, monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_git_project.parent):
        monkeypatch.chdir(tmp_git_project)
        result = runner.invoke(
            tmp_app.cli, ["run", "nonexistent", "--dry-run", "--force"]
        )
        assert result.exit_code != 0
        assert "No such command" in result.output


def test_run_help_shows_options(tmp_app):
    runner = CliRunner()
    result = runner.invoke(tmp_app.cli, ["run", "my_exp", "--help"])
    assert result.exit_code == 0
    assert "--lr" in result.output
    assert "--epochs" in result.output
    assert "--batch-size" in result.output


def test_cli_callable_as_entry_point(tmp_app):
    """app.cli must work as a script entry point: calling app.cli directly should run the CLI."""
    runner = CliRunner()
    # Entry points call app.cli — the result must be invocable by Click's runner
    result = runner.invoke(tmp_app.cli, ["list"])
    assert result.exit_code == 0
    assert "my_exp" in result.output


