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

- **Test coverage gate**: `pyproject.toml` sets `tool.coverage.report.fail_under = 62`,
  which is the current measured coverage for the CPU-friendly test subset
  (the full suite, including GPU paths, cannot run in this CI). Any PR that
  drops coverage below this bar will fail. New code should come with tests
  that keep coverage at or above the current floor.
- **Ruff**: `make ruff` runs `ruff format` and `ruff check`. CI runs both.
- **Type checks**: `make ty` runs `ty check`. CI runs it.
- **Markers**: when adding tests, mark them with the appropriate marker
  (`quick`, `slow`, `heavy`, `e2e`, `gpu`, `multi_gpu`) so the local test
  loop (`make test-fast`) stays fast for daily development.

Thank you for your contributions!
