# Command runner for the llm project (replaces Makefile; `just` http://just.systems).
# Mirror of the historical Makefile targets; all test recipes run `uv run --frozen`
# so a test run never silently rewrites uv.lock (e.g. a stray re-lock bumping
# torch mid-run) — run `uv lock` explicitly to change dependencies.

set shell := ["bash", "-c"]

APP_NAME := "llm"

# Init the venv + install pre-commit/prek hooks
init: sync
    uvx prek install

# Sync the project with the venv (includes default test group)
sync:
    uv sync

# Sync all dependency groups (test, docs, dev) + runtime extras
dev:
    uv sync --all-groups --all-extras

# Build wheel
build:
    uv build

# Test all
test:
    uv run --frozen pytest

# Test with doctests (/// examples verified as tests)
test-doc:
    uv run --frozen pytest --doctest-modules src/llm/

# Test fast: the CPU-clean daily gate.
# Excludes heavy, e2e AND every GPU-requiring marker, so it runs unchanged on
# CPU-only hosts (CI / laptops). GPU tests belong to `test` (full) / `test-gpu`.
test-fast:
    uv run --frozen pytest -m "not heavy and not e2e and not gpu and not multi_gpu and not full_cluster and not need_gpu"

# Parallel test — one worker per GPU (min 1), GPU/heavy/e2e excluded (CPU-clean).
# Each worker pins to a distinct GPU via PYTEST_XDIST_WORKER env.
test-parallel:
    #!/usr/bin/env bash
    set -euo pipefail
    N="$(uv run --frozen python -c 'import torch; print(max(min(torch.cuda.device_count(), 32), 1))' 2>/dev/null)"
    echo ">>> $N workers across $N GPU(s)"
    uv run --frozen pytest -n "$N" -m "not heavy and not e2e and not gpu and not multi_gpu and not full_cluster and not need_gpu"

# Test quick only
test-quick:
    uv run --frozen pytest -m quick

# GPU-only: for GPU-capable hosts / self-hosted GPU runners. Tests also
# auto-skip individually when too few usable GPUs (tests/conftest.py).
test-gpu:
    uv run --frozen pytest -m "gpu or multi_gpu or need_gpu or full_cluster"

# Heavy resource tests only (slow, memory/CPU-intensive; separate job).
test-heavy:
    uv run --frozen pytest -m heavy

# Test with coverage and allure reports (for CI).
# ``--cov-fail-under`` enforces the bar from CI (see pyproject [tool.coverage.report]).
test-cov:
    uv run --frozen pytest --cov=llm --cov-fail-under=77 --cov-report=term-missing --cov-report=html --cov-report=lcov --cov-report=xml --alluredir=allure-results

# Core CPU suite + coverage — the per-push CI gate.
# Excludes GPU/heavy/e2e (CPU-clean); the coverage bar is calibrated to THIS
# suite (lower than the full-suite `test-cov` bar, which adds heavy/e2e paths).
test-cov-fast:
    uv run --frozen pytest -m "not heavy and not e2e and not gpu and not multi_gpu and not full_cluster and not need_gpu" --cov=llm --cov-fail-under=77 --cov-report=term-missing --cov-report=xml

# Test e2e only
test-e2e:
    uv run --frozen pytest -m e2e

# Allure report
allure:
    allure serve allure-results

# Ruff (lint + format)
ruff: lint fmt

# Lint only
lint:
    uv run ruff check . --fix

# Format only
fmt:
    uv run ruff format .

# ty type check
ty:
    uv run ty check src/llm/

# Full static lint gate (mirrors the CI lint job: lock freshness, ruff, ty, examples)
check:
    uv lock --check
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check src/llm/
    # examples must compile (mirrors the CI lint job)
    for f in examples/*.py; do uv run --frozen python -m py_compile "$f" || exit 1; done

# Build the docs site locally (mirrors the docs CI job)
docs:
    uv run --frozen --group docs mkdocs build --strict

# Build image
image:
    docker image build -t {{APP_NAME}} .

# Start a compose service
compose-up:
    docker compose -f ./compose.yml -p {{APP_NAME}} up -d

# Shutdown a compose service
compose-down:
    docker compose -f ./compose.yml down

# Clean build artifacts (docker teardown is opt-in, see clean-docker)
clean:
    rm -rf build dist *.egg-info htmlcov .coverage coverage.xml coverage.lcov

# Tear down the compose stack + project image (destructive; opt-in)
clean-docker:
    docker compose -f ./compose.yml -p {{APP_NAME}} down -v
    docker image rm -f {{APP_NAME}}
