"""Tests for the unified ``llm`` CLI root that dispatches to subcommands.

The framework's commands (``train`` / ``serve`` / ``quantize`` / ``prune`` /
``decompose`` / ``migrate-ckpt``) live under a single ``llm`` binary rather
than six flat ``llm-*`` programs, so ``llm <sub>`` must route to each
command's parser. The per-command option tables themselves are covered by the
existing ``python -m llm.cli.* --help`` tests; here we only assert the root
dispatches each name to its parser.
"""

from __future__ import annotations

from typer.testing import CliRunner

from llm.cli.main import app


def test_llm_root_help_lists_all_subcommands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("train", "serve", "quantize", "prune", "decompose", "migrate-ckpt"):
        assert name in result.output


def test_llm_routes_each_subcommand_to_its_options():
    # each subcommand's --help must reach that parser's own option tokens
    cases = {
        "train": "--task",
        "serve": "--host",
        "prune": "--model",
        "decompose": "--model",
    }
    for sub, option in cases.items():
        result = CliRunner().invoke(app, [sub, "--help"])
        assert result.exit_code == 0, f"llm {sub} --help failed: {result.output}"
        assert option in result.output, f"llm {sub} --help missing {option}"


def test_llm_quantize_routes_to_gptq_subcommand():
    result = CliRunner().invoke(app, ["quantize", "gptq", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output


def test_llm_migrate_ckpt_routes():
    # positional-arg command: --help must render its parser's help
    result = CliRunner().invoke(app, ["migrate-ckpt", "--help"])
    assert result.exit_code == 0
    assert "legacy v0.0.5" in result.output
