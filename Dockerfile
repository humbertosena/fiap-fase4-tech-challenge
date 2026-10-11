# syntax=docker/dockerfile:1.7
# Imagem da CLI credit-sentinel: build com uv e runtime enxuto com usuário não-root.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.12-slim AS runtime
RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir /home/appuser appuser \
    && mkdir -p /workspace \
    && chown appuser:appuser /workspace
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    CS_PROJECT_ROOT=/workspace
# O repositório é montado em /workspace: docker run --rm -v "$PWD:/workspace" credit-sentinel <comando>
WORKDIR /workspace
USER appuser
ENTRYPOINT ["credit-sentinel"]
CMD ["--help"]
