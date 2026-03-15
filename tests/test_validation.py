from dataclasses import dataclass

import pytest

from wm.validation import resolve_config


@dataclass
class HP:
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 64
    use_dropout: bool = True
    name: str = "default"


def test_all_defaults():
    config = resolve_config(HP, {})
    assert config == {
        "lr": 1e-3,
        "epochs": 10,
        "batch_size": 64,
        "use_dropout": True,
        "name": "default",
    }


def test_override_float():
    config = resolve_config(HP, {"lr": "0.01"})
    assert config["lr"] == 0.01


def test_override_int():
    config = resolve_config(HP, {"epochs": "20"})
    assert config["epochs"] == 20
    assert isinstance(config["epochs"], int)


def test_override_bool_true():
    config = resolve_config(HP, {"use_dropout": "true"})
    assert config["use_dropout"] is True


def test_override_bool_false():
    config = resolve_config(HP, {"use_dropout": "false"})
    assert config["use_dropout"] is False


def test_override_string():
    config = resolve_config(HP, {"name": "experiment_1"})
    assert config["name"] == "experiment_1"


def test_unknown_key():
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        resolve_config(HP, {"nonexistent": "value"})


def test_multiple_overrides():
    config = resolve_config(HP, {"lr": "0.01", "epochs": "5", "use_dropout": "no"})
    assert config["lr"] == 0.01
    assert config["epochs"] == 5
    assert config["use_dropout"] is False
