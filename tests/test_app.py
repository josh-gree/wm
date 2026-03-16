import pytest
from click.testing import CliRunner
from unittest.mock import patch

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


def test_run_shows_summary_and_confirms(tmp_git_project, tmp_app, monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_git_project.parent):
        monkeypatch.chdir(tmp_git_project)
        result = runner.invoke(tmp_app.cli, ["run", "my_exp"], input="n\n")
        assert "Experiment: my_exp" in result.output
        assert "Config:" in result.output
        assert "Git SHA:" in result.output
        assert "Continue?" in result.output
        assert result.exit_code != 0


def test_run_confirm_yes_dispatches(tmp_git_project, tmp_app, monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_git_project.parent):
        monkeypatch.chdir(tmp_git_project)
        with patch("wm.app.dispatch") as mock_dispatch:
            result = runner.invoke(tmp_app.cli, ["run", "my_exp"], input="y\n")
            assert result.exit_code == 0
            mock_dispatch.assert_called_once()


def test_run_with_overrides(tmp_git_project, tmp_app, monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_git_project.parent):
        monkeypatch.chdir(tmp_git_project)
        result = runner.invoke(
            tmp_app.cli,
            ["run", "my_exp", "--lr", "0.01"],
            input="n\n",
        )
        assert result.exit_code != 0
        assert "0.01" in result.output


def test_run_unknown_experiment(tmp_git_project, tmp_app, monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_git_project.parent):
        monkeypatch.chdir(tmp_git_project)
        result = runner.invoke(tmp_app.cli, ["run", "nonexistent"])
        assert result.exit_code != 0
        assert "No such command" in result.output


def test_run_help_shows_options(tmp_app):
    runner = CliRunner()
    result = runner.invoke(tmp_app.cli, ["run", "my_exp", "--help"])
    assert result.exit_code == 0
    assert "--lr" in result.output
    assert "--epochs" in result.output
    assert "--batch-size" in result.output


def test_run_force_skips_confirm(tmp_git_project, tmp_app, monkeypatch):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_git_project.parent):
        monkeypatch.chdir(tmp_git_project)
        with patch("wm.app.dispatch") as mock_dispatch:
            result = runner.invoke(
                tmp_app.cli, ["run", "my_exp", "--force"]
            )
            assert result.exit_code == 0
            assert "Continue?" not in result.output
            mock_dispatch.assert_called_once()


def test_cli_callable_as_entry_point(tmp_app):
    """app.cli must work as a script entry point: calling app.cli directly should run the CLI."""
    runner = CliRunner()
    result = runner.invoke(tmp_app.cli, ["list"])
    assert result.exit_code == 0
    assert "my_exp" in result.output
