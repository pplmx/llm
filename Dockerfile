# Base image for building the virtual environment
FROM python:3.14-bookworm AS builder

ENV PATH="/root/.local/bin:$PATH" \
    UV_INDEX_URL="https://mirrors.cernet.edu.cn/pypi/web/simple" \
    PIP_INDEX_URL="https://mirrors.cernet.edu.cn/pypi/web/simple"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Production runtime deps only (exclude dev groups; extras aren't installed by default)
RUN uv sync --frozen --no-group test --no-group docs

# Separate stage for validation (build and test)
FROM builder AS validator

WORKDIR /app
COPY . .

# Install test deps (default-groups) for make test
RUN uv sync --frozen

RUN make build && make test

# Final image for running the application
FROM python:3.14-slim-bookworm

LABEL author="Mystic"

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the virtual environment and application code
COPY --from=builder /app/.venv /app/.venv
COPY src ./src

EXPOSE 8000

# Use uvicorn for production serving (package is `llm`, not `src.llm`).
#
# The bind address is a SINGLE source of truth: the lifespan guard in
# ``llm.serving.api`` reads ``ServingConfig.host`` (env ``LLM_SERVING_HOST``),
# so uvicorn must bind the SAME value the guard validates. Hardcoding
# ``--host 0.0.0.0`` here while ``LLM_SERVING_HOST`` defaulted to 127.0.0.1
# meant ``docker run -p 8000:8000 <image>`` served anonymous inference on
# 0.0.0.0 — the guard saw a loopback config and never fired (RIL ISS-164).
#
# Defaulting to 0.0.0.0 now makes the OUT-OF-THE-BOX container fail CLOSED:
# launching it without ``-e LLM_SERVING_API_KEY=...`` refuses to start with
# the guard's clear error instead of silently exposing /generate anonymously.
ENV LLM_SERVING_HOST="0.0.0.0"
CMD uvicorn llm.serving.api:app --host "$LLM_SERVING_HOST" --port 8000
