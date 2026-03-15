import textwrap
from unittest.mock import patch

from wm.discovery import discover_experiments


def test_valid_experiment(tmp_project):
    exps = discover_experiments(tmp_project)
    assert "my_exp" in exps
    info = exps["my_exp"]
    assert info.name == "my_exp"
    assert hasattr(info.hyper_params, "__dataclass_fields__")
    assert callable(info.module.run)
    assert info.container is None


def test_missing_hyperparams(tmp_project):
    (tmp_project / "experiments" / "no_hp.py").write_text(
        textwrap.dedent("""\
        def run(config, wandb_run):
            pass
        """)
    )
    exps = discover_experiments(tmp_project)
    assert "no_hp" not in exps
    assert "my_exp" in exps


def test_missing_run(tmp_project):
    (tmp_project / "experiments" / "no_run.py").write_text(
        textwrap.dedent("""\
        from dataclasses import dataclass

        @dataclass
        class HyperParams:
            lr: float = 1e-3
        """)
    )
    exps = discover_experiments(tmp_project)
    assert "no_run" not in exps


def test_non_python_files_ignored(tmp_project):
    (tmp_project / "experiments" / "readme.txt").write_text("not python")
    exps = discover_experiments(tmp_project)
    assert "readme" not in exps


def test_dunder_files_ignored(tmp_project):
    exps = discover_experiments(tmp_project)
    assert "__init__" not in exps


def test_with_container(tmp_project):
    (tmp_project / "experiments" / "custom.py").write_text(
        textwrap.dedent("""\
        from dataclasses import dataclass

        @dataclass
        class HyperParams:
            lr: float = 1e-3

        CONTAINER = {
            "extra_dependencies": ["flash-attn"],
            "gpu": "A100",
        }

        def run(config, wandb_run):
            pass
        """)
    )
    exps = discover_experiments(tmp_project)
    assert "custom" in exps
    assert exps["custom"].container == {
        "extra_dependencies": ["flash-attn"],
        "gpu": "A100",
    }


def test_no_experiments_dir(tmp_path):
    exps = discover_experiments(tmp_path)
    assert exps == {}


def test_syntax_error_warns(tmp_project):
    (tmp_project / "experiments" / "bad.py").write_text("def oops(\n")
    with patch("wm.discovery.click") as mock_click:
        exps = discover_experiments(tmp_project)
    assert "bad" not in exps
    mock_click.echo.assert_called()
    warning_msg = mock_click.echo.call_args_list[-1][0][0]
    assert "failed to import" in warning_msg
    assert "experiments.bad" in warning_msg
