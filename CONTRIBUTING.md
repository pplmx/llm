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

## Security Findings Policy

CI runs two security gates on every PR (and weekly via cron for
dependency-vulnerability scanning):

1. **pip-audit** (`uv run pip-audit --strict`): scans the locked dependency
   set (`uv.lock`) for known CVEs. Any finding fails the build. The tool
   and its version are pinned in `pyproject.toml` under
   `[dependency-groups] security`. Run locally with
   `uv sync --group security && uv run pip-audit --strict`.

2. **bandit** (`uv run bandit -r src/llm/ --severity-level high`): static
   security analysis of the source tree. Findings of severity **HIGH** or
   **CRITICAL** fail the build; **LOW** and **MEDIUM** are reported as
   warnings. Run locally with `uv run bandit -r src/llm/ --severity-level high`.

### Handling findings

| Severity | Action |
|----------|--------|
| CRITICAL | **Block** — fix before merge. If a false positive, add a `# noqa: Sxxx` comment or `bandit: skip` directive with a justification. |
| HIGH | **Block** — fix before merge. Same false-positive process. |
| MEDIUM | **Warn** — fix if easy; otherwise document in the PR description. |
| LOW | **Info** — optional fix; good first issues. |

### When adding `# noqa` / `bandit: skip`

- Always include a brief justification: `# noqa: S603 — internal CLI, no user input`.
- Prefer the narrowest suppression: `# bandit: skip B603` before the specific `subprocess.run` call rather than disabling the rule repo-wide.
- If suppressing in CI config, link to the audit ticket: `docs/audits/2026-07-12-tickets/09-pip-audit-bandit-ci.md`.

### Dependency vulnerabilities (pip-audit)

- Fix: `uv lock` to pull the patched version, then `uv sync`. If no patch
  is available upstream, document the mitigation in the `pyproject.toml`
  comment for that dependency (e.g., the `pillow>=12.3.0` and
  `sacrebleu>=2.6.0` entries).
- The `uv.lock` is the source of truth — `uv lock --check` also runs in CI.

Thank you for your contributions!
