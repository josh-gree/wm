import importlib
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType

import click


@dataclass
class ExperimentInfo:
    name: str
    module: ModuleType
    hyper_params: type
    container: dict | None


def discover_experiments(project_dir: Path) -> dict[str, ExperimentInfo]:
    experiments_dir = project_dir / "experiments"
    if not experiments_dir.is_dir():
        return {}

    results = {}
    project_str = str(project_dir)
    added_to_path = project_str not in sys.path

    if added_to_path:
        sys.path.insert(0, project_str)

    try:
        # Clear cached experiments package so it's re-imported from the current path
        sys.modules.pop("experiments", None)

        for py_file in sorted(experiments_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            module_name = f"experiments.{py_file.stem}"

            # Remove from cache to allow re-import
            sys.modules.pop(module_name, None)

            try:
                mod = importlib.import_module(module_name)
            except Exception as exc:
                click.echo(f"Warning: failed to import {module_name}: {exc}", err=True)
                continue

            hyper_params = getattr(mod, "HyperParams", None)
            run_fn = getattr(mod, "run", None)

            if hyper_params is None or run_fn is None:
                continue

            if not hasattr(hyper_params, "__dataclass_fields__"):
                continue

            container = getattr(mod, "CONTAINER", None)

            results[py_file.stem] = ExperimentInfo(
                name=py_file.stem,
                module=mod,
                hyper_params=hyper_params,
                container=container,
            )
    finally:
        if added_to_path:
            sys.path.remove(project_str)

    return results
