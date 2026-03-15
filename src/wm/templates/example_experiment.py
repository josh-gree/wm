from dataclasses import dataclass


@dataclass
class HyperParams:
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 64


def run(config: dict, wandb_run) -> None:
    """Example experiment. Replace with your training code."""
    for epoch in range(config["epochs"]):
        loss = 1.0 / (epoch + 1)
        wandb_run.log({"epoch": epoch, "loss": loss})
