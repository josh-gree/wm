# Changelog

## Unreleased

## [0.1.0] — 2026-03-20

### Added

- **Per-project storage volume** — every project automatically gets a `{project.name}-storage` Modal volume mounted at `/storage`. Created on first use, no manual setup required.
- **Per-run directory** — each run gets an isolated subdirectory at `/storage/{experiment_name}/{wandb_run_id}/`, passed to `experiment.run()` as the new `run_dir: Path` argument.
- **Volume auto-commit** — storage volume is committed after every run (success or failure) so partial artifacts are always persisted.
- **`ephemeral_disk` support** — `Experiment`, `ProjectConfig`, and the runner now support the `ephemeral_disk` field for larger scratch space on Modal.
- **Layered image build** — dependency installation and code copying are split into separate image layers for better caching. Code changes no longer bust the dep layer.
- **`prepare-data` pattern** — projects can add a `prepare-data` CLI command (see `wm-mnist` for an example) to populate the data volume before running experiments.

### Changed

- `Experiment.run()` signature updated from `run(config, wandb_run)` to `run(config, wandb_run, run_dir)`.
- Project config moved from `project.yaml` to `[tool.wm]` in `pyproject.toml` — single source of truth.
- Bundled Dockerfile now uses `uv` for fast, reproducible installs.

### Removed

- `init` command removed — project setup is now just `pyproject.toml` + `uv lock`.
- `project.yaml` support removed in favour of `pyproject.toml`.

## 2026-03-22

### Added

- **`--detach` flag** — run experiments in detached mode. Dispatches the function to Modal and exits immediately. The experiment continues running on Modal in the background.

## 2026-03-24

### Added

- **Automatic git snapshots** — every experiment run creates a snapshot branch (`wm/{experiment}/{timestamp}`) capturing the full working tree, including uncommitted changes. The branch is pushed to origin before dispatch, and the branch name is tagged on the W&B run.

### Changed

- A git repository with a configured remote is now required. Runs abort if either is missing.

### Removed

- `--skip-git-check` flag removed — replaced by the snapshot mechanism.
