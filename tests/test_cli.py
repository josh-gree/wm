import os
import textwrap

from click.testing import CliRunner

from wm.cli import cli


def test_list_experiments(tmp_project):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_project.parent):
        os.chdir(tmp_project)
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "my_exp" in result.output
        assert "lr" in result.output
        assert "epochs" in result.output


def test_list_no_experiments(tmp_path):
    (tmp_path / "project.yaml").write_text("name: empty\n")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path.parent):
        os.chdir(tmp_path)
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No experiments found" in result.output


def test_run_dry_run(tmp_git_project):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_git_project.parent):
        os.chdir(tmp_git_project)
        result = runner.invoke(cli, ["run", "my_exp", "--dry-run", "--force"])
        assert result.exit_code == 0
        assert "Experiment: my_exp" in result.output
        assert "Config:" in result.output


def test_run_dry_run_with_overrides(tmp_git_project):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_git_project.parent):
        os.chdir(tmp_git_project)
        result = runner.invoke(
            cli, ["run", "my_exp", "--dry-run", "--force", "--lr", "0.01"]
        )
        assert result.exit_code == 0
        assert "0.01" in result.output


def test_run_unknown_experiment(tmp_git_project):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_git_project.parent):
        os.chdir(tmp_git_project)
        result = runner.invoke(cli, ["run", "nonexistent", "--dry-run", "--force"])
        assert result.exit_code == 1
        assert "Unknown experiment" in result.output


def test_init(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path.parent):
        os.chdir(tmp_path)
        result = runner.invoke(cli, ["init", "my-project"])
        assert result.exit_code == 0
        assert "Created project: my-project/" in result.output

        project_dir = tmp_path / "my-project"
        assert (project_dir / "project.yaml").exists()
        assert (project_dir / "pyproject.toml").exists()
        assert (project_dir / "experiments" / "example.py").exists()
        assert (project_dir / ".gitignore").exists()
        assert (project_dir / "shared").is_dir()
        assert (project_dir / "containers").is_dir()
        assert (project_dir / "data").is_dir()

        # Check name substitution
        content = (project_dir / "project.yaml").read_text()
        assert "my-project" in content


def test_init_already_exists(tmp_path):
    (tmp_path / "existing").mkdir()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path.parent):
        os.chdir(tmp_path)
        result = runner.invoke(cli, ["init", "existing"])
        assert result.exit_code == 1
        assert "already exists" in result.output
