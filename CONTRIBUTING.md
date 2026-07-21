# Contributing Guide

We warmly welcome your contributions to the `llm` project! Your contributions will help us improve the project and make it more useful for everyone.

## How to Contribute

We encourage contributions through the following ways:

- **Report Bugs**: If you find any issues or errors, please submit them in [GitHub Issues](https://github.com/pplmx/llm/issues).
- **Suggest Features**: If you have new feature ideas, please also propose them in [GitHub Issues](https://github.com/pplmx/llm/issues).
- **Submit Code**: If you wish to contribute code directly, please follow this process:
    1. Fork this repository.
    2. Create your feature branch (`git checkout -b feature/YourFeature`).
    3. Make changes and commit (`git commit -m 'feat: Add some feature'`).
    4. Push to your branch (`git push origin feature/YourFeature`).
    5. Submit a Pull Request.

## Development Guide

Before you start writing code, please make sure to read our [Development Guide](docs/development/README.md). It contains detailed information on setting up the development environment, running tests, code style, and the contribution process.

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) to understand the expected behavior within our community.

## Testing Policy

- **Test coverage gate**: `pyproject.toml` sets
  `tool.coverage.report.fail_under = 77` and `make test-cov` passes
  `--cov-fail-under=77` to enforce it on CI. This is the coverage the
  CI can actually measure when running the full suite on a GPU-less
  runner (CUDA-required tests fail and don't exercise their code
  paths). CPU-only test runs (excluding `tests/core/attn/test_mla.py`
  and the other GPU-only modules) report ~82.6%; the gap is the
  CUDA-only paths in `paged_attention`, `training/core/engine`,
  `training/distributed`, etc. Any PR that drops coverage below this
  bar will fail the lint job. New code should come with tests that keep
  coverage at or above the current floor.
- **Ruff**: `make ruff` runs `ruff format` and `ruff check`. CI runs both.
- **Type checks**: `make ty` runs `ty check`. CI runs it.
- **Markers**: when adding tests, mark them with the appropriate marker
  (`quick`, `slow`, `heavy`, `e2e`, `gpu`, `multi_gpu`) so the local test
  loop (`make test-fast`) stays fast for daily development.

Thank you for your contributions!
