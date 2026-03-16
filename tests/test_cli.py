from click.testing import CliRunner

from wm.cli import cli


def test_init(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli, ["init", "my-project"])
    assert result.exit_code == 0
    assert "Created project: my-project/" in result.output

    project_dir = tmp_path / "my-project"
    assert (project_dir / "project.yaml").exists()
    assert (project_dir / "pyproject.toml").exists()
    assert (project_dir / "experiments" / "example.py").exists()
    assert (project_dir / "app.py").exists()
    assert (project_dir / ".gitignore").exists()
    assert (project_dir / "shared").is_dir()
    assert (project_dir / "containers").is_dir()
    assert (project_dir / "data").is_dir()

    content = (project_dir / "project.yaml").read_text()
    assert "my-project" in content


def test_init_already_exists(tmp_path, monkeypatch):
    (tmp_path / "existing").mkdir()
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli, ["init", "existing"])
    assert result.exit_code == 1
    assert "already exists" in result.output
