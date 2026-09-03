# `llm.cli` — CLI Tools

Command-line utilities registered under `pyproject.toml` `[project.scripts]`
(`llm-train` lives in `llm.training.train`; the `llm-serve` entry point
lives in `llm.serving.api`). Each module below is a thin Typer app.

## Checkpoint migration

::: llm.cli.migrate_ckpt

## Quantization

::: llm.cli.quantize

## Pruning

::: llm.cli.prune

## Low-rank decomposition

::: llm.cli.decompose
