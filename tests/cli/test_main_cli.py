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
from tests.support.ansi import help_render_diag, strip_ansi


def _registered_subcommands() -> set[str]:
    """Subcommand names actually wired into the root app."""
    names = {cmd.name for cmd in app.registered_commands if cmd.name}
    names |= {group.name for group in app.registered_groups if group.name}
    return names


def test_llm_root_help_lists_all_subcommands():
    """``--help`` must mention EVERY subcommand the app registers.

    The names used to be a hard-coded six-tuple. That passes for the six that
    exist today and silently stops covering the seventh: a new subcommand whose
    ``help`` text never reached the root would leave this test green, which is
    the opposite of what a test named "lists all subcommands" should do.

    Derived from the app, so adding a subcommand extends the check
    automatically — and removing one from the help now fails instead of
    quietly shrinking the covered set.
    """
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0

    expected = _registered_subcommands()
    assert expected, "the root app registers no subcommands; the CLI wiring moved"

    missing = sorted(name for name in expected if name not in result.output)
    assert not missing, f"subcommand(s) missing from the root --help output: {missing}"


def test_registered_subcommands_match_their_help_routes():
    """Every registered subcommand must be invokable (its own ``--help`` works).

    ``registered_commands`` is declarative; a name can be present yet bind to a
    broken callback. Listing it is therefore only half the property the root
    help claims — "you can run these".
    """
    for name in sorted(_registered_subcommands()):
        result = CliRunner().invoke(app, [name, "--help"])
        assert result.exit_code == 0, f"`llm {name} --help` failed:\n{result.output[-800:]}"


def test_llm_version_prints_and_exits_zero():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip().startswith("llm ")

    short = CliRunner().invoke(app, ["-V"])
    assert short.exit_code == 0
    assert short.output.strip() == result.output.strip()


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
        # Strip ANSI: rich may split a styled option token into separate color
        # runs on a CI tty, breaking the literal raw substring (see ansi.py).
        clean = strip_ansi(result.output)
        assert result.exit_code == 0, f"llm {sub} --help failed: {help_render_diag(result.output)}"
        assert option in clean, f"llm {sub} --help missing {option}: {help_render_diag(result.output)}"


def test_llm_quantize_routes_to_gptq_subcommand():
    result = CliRunner().invoke(app, ["quantize", "gptq", "--help"])
    assert result.exit_code == 0
    assert "--model" in strip_ansi(result.output), help_render_diag(result.output)


def test_llm_migrate_ckpt_routes():
    # positional-arg command: --help must render its parser's help
    result = CliRunner().invoke(app, ["migrate-ckpt", "--help"])
    assert result.exit_code == 0
    assert "legacy v0.0.5" in result.output
