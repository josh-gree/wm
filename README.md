# wm

A lightweight framework for running ML experiments on [Modal](https://modal.com) with [Weights & Biases](https://wandb.ai) tracking.

## Installation

```bash
uv add wm
```

## Setting up a project

There are no structural constraints on how you organise your code. You need three things:

### 1. `project.yaml`

Defines your project's default container and runtime settings:

```yaml
name: my-project
gpu: null
timeout: 3600
wandb_secret: wandb-secret

dependencies:
  - torch>=2.10.0
  - torchvision>=0.25.0
# apt_packages:
#   - libgl1

# Or use a custom Dockerfile instead:
# dockerfile: containers/base.Dockerfile

# Optional Modal volume for data:
# volume: my-data-volume
# data_mount: /data
```

### 2. Experiments

Define experiments anywhere in your project. Each experiment is a class that subclasses `wm.Experiment`:

```python
from pydantic import BaseModel
from wm import Experiment


class MnistCnn(Experiment):
    name = "mnist-cnn"

    class Config(BaseModel):
        lr: float = 1e-3
        epochs: int = 10
        batch_size: int = 64

    @staticmethod
    def run(config: "MnistCnn.Config", wandb_run) -> None:
        for epoch in range(config.epochs):
            loss = 1.0 / (epoch + 1)
            wandb_run.log({"epoch": epoch, "loss": loss})
```

Experiments can override container settings per-experiment via class attributes:

```python
class HeavyExperiment(Experiment):
    name = "heavy"
    gpu = "A100"
    timeout = 7200
    extra_dependencies = ["transformers"]
    # ...
```

### 3. App entry point

Create a module (e.g. `app.py`) that wires everything together:

```python
from wm import App
from my_experiments.cnn import MnistCnn
from my_experiments.mlp import MnistMlp

app = App.from_yaml("project.yaml")
app.register(MnistCnn)
app.register(MnistMlp)

if __name__ == "__main__":
    app.cli()
```

Then hook it into your `pyproject.toml`:

```toml
[project.scripts]
my-project = "app:app.cli"
```

After `uv sync`, you get a CLI:

```bash
my-project list              # list registered experiments
my-project run mnist-cnn     # run an experiment on Modal
my-project run mnist-cnn --lr 0.01 --epochs 20
```

## Project structure

Organise your code however you like. Here's one example:

```
my-project/
  app.py
  project.yaml
  pyproject.toml
  my_experiments/
    cnn.py
    mlp.py
  shared/
    data.py
```
