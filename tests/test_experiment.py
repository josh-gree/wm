import pytest
from pydantic import BaseModel
from wm import Experiment


def test_valid_experiment():
    class Good(Experiment):
        name = "good"

        class Config(BaseModel):
            lr: float = 1e-3

        @staticmethod
        def run(config, wandb_run):
            pass

    assert Good.name == "good"


def test_missing_name():
    with pytest.raises(TypeError, match="must define a 'name'"):

        class Bad(Experiment):
            class Config(BaseModel):
                lr: float = 1e-3

            @staticmethod
            def run(config, wandb_run):
                pass


def test_missing_config():
    with pytest.raises(TypeError, match="must define a 'Config'"):

        class Bad(Experiment):
            name = "bad"

            @staticmethod
            def run(config, wandb_run):
                pass


def test_missing_run():
    with pytest.raises(TypeError, match="must define a 'run'"):

        class Bad(Experiment):
            name = "bad"

            class Config(BaseModel):
                lr: float = 1e-3


def test_config_not_basemodel():
    with pytest.raises(TypeError, match="must define a 'Config'"):

        class Bad(Experiment):
            name = "bad"

            class Config:
                lr: float = 1e-3

            @staticmethod
            def run(config, wandb_run):
                pass
