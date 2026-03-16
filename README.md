# wm

A lightweight framework for running ML experiments on [Modal](https://modal.com) with [Weights & Biases](https://wandb.ai) tracking.

## Installation

```bash
uv add wm
```

## Setting up a project

There are no structural constraints on how you organise your code. You need three things:

### 1. `pyproject.toml` with `[tool.wm]`

Your `pyproject.toml` is the single source of truth. The project name comes from `[project].name`, and runtime settings go in `[tool.wm]`:

```toml
[project]
name = "my-project"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.10.0",
    "torchvision>=0.25.0",
    "wandb>=0.19.0",
    "wm",
]

[tool.wm]
gpu = "A100"
timeout = 3600
wandb_secret = "wandb-secret"
volume = "my-data-volume"
data_mount = "/data"
# dockerfile = "custom.Dockerfile"  # optional escape hatch
```

All `[tool.wm]` fields are optional with sensible defaults:

| Field | Default | Description |
|-------|---------|-------------|
| `gpu` | `null` | Modal GPU type (e.g. `"A100"`, `"T4"`) |
| `timeout` | `3600` | Function timeout in seconds |
| `wandb_secret` | `"wandb-secret"` | Modal secret name for W&B API key |
| `volume` | `null` | Modal volume name for data |
| `data_mount` | `"/data"` | Mount path for the volume |
| `dockerfile` | `null` | Path to a custom Dockerfile |

### 2. Container build

By default, wm uses a bundled Dockerfile that:
- Starts from `debian:bookworm-slim`
- Installs `uv` and `git`
- Runs `uv sync` to install your project and all dependencies from `pyproject.toml`

This means your `pyproject.toml` dependencies are automatically available in the container. You need a `uv.lock` file — run `uv lock` to generate one.

Python version is controlled by `.python-version` or `requires-python` in your `pyproject.toml` (uv manages it).

#### Custom Dockerfile

Set `dockerfile` in `[tool.wm]` to use your own Dockerfile. It should follow a similar pattern:

```dockerfile
FROM debian:bookworm-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
WORKDIR /repo
COPY pyproject.toml uv.lock* .python-version* ./
RUN uv sync --frozen --no-install-project --no-dev
COPY . .
RUN uv sync --frozen --no-dev
ENV PATH="/repo/.venv/bin:$PATH"
```

### 3. Experiments

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

Experiments can override `gpu` and `timeout` per-experiment via class attributes:

```python
class HeavyExperiment(Experiment):
    name = "heavy"
    gpu = "A100"
    timeout = 7200
    # ...
```

### 4. App entry point

Create a module (e.g. `app.py`) that wires everything together:

```python
from wm import App
from my_experiments.cnn import MnistCnn
from my_experiments.mlp import MnistMlp

app = App.from_pyproject()
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
  pyproject.toml
  uv.lock
  src/
    my_project/
      app.py
      experiments/
        cnn.py
        mlp.py
      shared/
        data.py
```
