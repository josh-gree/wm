FROM debian:bookworm-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
WORKDIR /repo
COPY pyproject.toml uv.lock* .python-version* ./
RUN uv sync --frozen --no-install-project --no-dev
ENV PATH="/repo/.venv/bin:$PATH"
