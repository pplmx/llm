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

# Sync every dependency group + all extras.
#
# `--all-extras` includes `perf`, i.e. flash-attn. PyPI ships flash-attn as an
# sdist only, and compiling it against CUDA takes many minutes (and fails on a
# CPU-only host), so `pyproject.toml` declares the Astral GPU index, which carries
# a pre-built wheel matched to the local torch build (torch 2.14.1+cu130 ->
# cu130). The download is still ~230 MiB, so for day-to-day work prefer
# `just sync`, or add just the extras you need:
# `uv sync --extra eval --extra logging --extra dvc` (what CI installs).
dev:
    uv sync --all-groups --all-extras

# NOTE `uv sync` is exclusive: this prunes every extra it does not name, so running
# it alone drops `eval` (and `lm_eval`), failing six evaluation tests.
# Sync only the optional `perf` extra (flash-attn) from the Astral wheels.
sync-perf:
    uv sync --extra perf

# NOTE use this rather than `sync-perf` before the test gates -- sync is exclusive.
# Sync the dev group and every runtime extra the suite needs, flash-attn included.
sync-full:
    uv sync --group dev --extra perf --extra eval --extra logging --extra dvc --extra compat --extra streaming

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
# CPU-clean fast gate (no heavy / e2e / GPU markers).
test-fast:
    uv run --frozen pytest -m "not heavy and not e2e and not gpu and not multi_gpu and not full_cluster and not need_gpu"

# Parallel test — one worker per GPU (min 1), GPU/heavy/e2e excluded (CPU-clean).
# Each worker pins to a distinct GPU via PYTEST_XDIST_WORKER env.
# Parallel CPU-clean run, one xdist worker per GPU.
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
# GPU-marked tests only (auto-skip when too few usable GPUs).
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
# Substantial multi-rank/multimodal/training-loop tests are `heavy`/`e2e`
# marked, so the CPU-clean gate is a fast path-usable smoke test (macOS CI
# timed out at ~90 min on the torch.multiprocessing.spawn distributed tests —
# those are now excluded here and run in test/heavy/e2e/full).
# CPU-clean suite + coverage — the per-push CI gate.
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
    uv run --frozen --group dev ruff check . --fix

# Format only
fmt:
    uv run --frozen --group dev ruff format .

# ty type check
ty:
    uv run --frozen --group dev ty check src/llm/

# Full static lint gate (mirrors the CI lint job: lock freshness, ruff, ty, examples)
check:
    # --no-config: uv.lock records pypi.org URLs (what CI resolves against). A
    # user-level uv.toml that sets a regional mirror as the default index would
    # otherwise make this reject a lockfile that is correct for everyone else.
    uv lock --check --no-config
    uv run --frozen --group dev ruff format --check .
    uv run --frozen --group dev ruff check .
    uv run --frozen --group dev ty check src/llm/
    # examples must compile (mirrors the CI lint job)
    for f in examples/*.py; do uv run --frozen python -m py_compile "$f" || exit 1; done

# Build the docs site locally (mirrors the docs CI job)
docs:
    uv run --frozen --group docs mkdocs build --strict

# CI has no GPU runner, so the GPU suite only ever runs on a host like this one.
# That makes it the one gate no automation will catch for you -- run this before
# pushing. Deliberately NOT a git hook: it takes ~8 minutes on 8 GPUs, and a hook
# that slow gets bypassed with `--no-verify`, which is worse than no hook.
#
# Refuses to report success without a GPU: `test-gpu` on a GPU-less host skips
# every test and exits 0, so it would print a green line for work it never did.
# Run every gate a push should satisfy, GPU suite included (needs a GPU host).
preflight:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! uv run --frozen python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        echo "preflight needs a GPU host: torch.cuda.is_available() is False." >&2
        echo "The GPU suite would skip everything and report success; run the" >&2
        echo "CPU-only gates on a CPU host instead: just check && just docs && just test-fast" >&2
        exit 1
    fi
    # Do NOT pin CUDA_VISIBLE_DEVICES: the GPU tests call
    # tests/support/devices.all_gpu_devices(), which filters by free VRAM and
    # raises MIN_FREE_VRAM_BYTES so they pick idle cards themselves. Forcing all
    # eight visible makes a busy card fail a test that passes on a free one --
    # measured: test_tp_checkpoint_roundtrip_four_gpu_deep_tp timed out at 480 s on
    # a card with 78 GiB used by another job, and passed in 10 s once it chose.
    uv run --frozen python -c 'import torch; print(f">>> {torch.cuda.device_count()} GPU(s) visible; free-VRAM filtering is per-test")'
    echo ">>> 1/6 check (lock, ruff, ty, examples)"
    just check
    echo ">>> 2/6 docs"
    just docs
    echo ">>> 3/6 test-fast"
    just test-fast
    echo ">>> 4/6 test-gpu"
    just test-gpu
    echo ">>> 5/6 test-heavy"
    just test-heavy
    echo ">>> 6/6 test-e2e"
    just test-e2e
    echo ">>> preflight passed"

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
