from pydantic import BaseModel
from wm import Experiment


class Example(Experiment):
    name = "example"

    class Config(BaseModel):
        lr: float = 1e-3
        epochs: int = 10
        batch_size: int = 64

    @staticmethod
    def run(config: "Example.Config", wandb_run) -> None:
        """Example experiment. Replace with your training code."""
        for epoch in range(config.epochs):
            loss = 1.0 / (epoch + 1)
            wandb_run.log({"epoch": epoch, "loss": loss})
