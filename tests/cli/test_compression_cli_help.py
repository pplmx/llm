"""The four compression CLIs must describe `--model` the same way, and truthfully.

`prune`, `decompose`, `quantize gptq` and `quantize fp8` all load checkpoints
through the same shared helper (``llm.cli._model_loading.load_trained_model``),
yet their ``--help`` text disagreed three ways:

* prune / decompose said "Path to model blob (.pt torch.save of a DecoderModel)"
* quantize said "Path to model checkpoint (.pt with DecoderModel state_dict)"

Both descriptions predate the v2 split layout, so neither mentioned the form
the framework actually writes by default — a user following `llm train` output
saw a help string that did not describe their checkpoint. Tests here pin the
text to the loader's real contract, so the next reader cannot drift again.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

# Every command that exposes the same `--model` input.
_COMMANDS: dict[str, tuple[str, list[str]]] = {
    "prune": ("llm.cli.prune", []),
    "decompose": ("llm.cli.decompose", []),
    "quantize gptq": ("llm.cli.quantize", ["gptq"]),
    "quantize fp8": ("llm.cli.quantize", ["fp8"]),
}


def _render_help_unwrapped(invoke) -> str:
    """Render ``--help`` wide enough that option names are not truncated.

    Rich wraps help text to the terminal width and **truncates** an option name
    that does not fit (``--model`` renders as ``--mod…`` at 40 columns), which
    made ``text.index("--model")`` raise ``ValueError: substring not found``.

    The width is not simply ``COLUMNS``: typer sets
    ``rich_utils.FORCE_TERMINAL = True`` whenever ``GITHUB_ACTIONS`` is in the
    environment (so CI renders colours), and a forced terminal with no real tty
    makes rich query the terminal size, fall back to a narrow default, and
    truncate. Pinning COLUMNS therefore did nothing on CI — only setting
    ``FORCE_TERMINAL = False`` and a wide ``MAX_WIDTH`` restores the full names.
    Both are patched because a real terminal would otherwise still take the
    ambient width. These tests compare the help TEXT, not rich's line wrapping.
    """
    import typer.rich_utils as rich_utils

    original_force_terminal = rich_utils.FORCE_TERMINAL
    original_max_width = rich_utils.MAX_WIDTH
    rich_utils.FORCE_TERMINAL = False
    rich_utils.MAX_WIDTH = 200
    try:
        result = invoke()
    finally:
        rich_utils.FORCE_TERMINAL = original_force_terminal
        rich_utils.MAX_WIDTH = original_max_width
    assert result.exit_code == 0, result.output
    return result.output


def _help_text(module: str, argv: list[str]) -> str:
    import importlib

    app = importlib.import_module(module).app
    return _render_help_unwrapped(lambda: CliRunner().invoke(app, [*argv, "--help"]))


@pytest.mark.parametrize("name", sorted(_COMMANDS))
def test_model_help_mentions_the_split_layout(name: str) -> None:
    """The v2 split stem is the default on-disk form and must be advertised.

    A user who just ran `llm train` has ``checkpoints/epoch_5.safetensors`` and
    friends; the help has to say that stem is acceptable.
    """
    module, argv = _COMMANDS[name]
    text = _help_text(module, argv)
    assert "v2 split stem" in text, f"{name}: --model help does not mention the v2 split stem"


@pytest.mark.parametrize("name", sorted(_COMMANDS))
def test_model_help_mentions_the_legacy_blob(name: str) -> None:
    """The legacy single-file ``.pt`` blob must stay documented (still supported)."""
    module, argv = _COMMANDS[name]
    text = _help_text(module, argv)
    assert "legacy" in text.lower(), f"{name}: --model help does not mention the legacy blob"


def test_model_help_text_is_identical_across_commands() -> None:
    """One shared loader, one description — otherwise they drift independently.

    The three divergent wordings this test replaces are exactly what shared
    loading was meant to end; keeping four paraphrases means a future layout
    change has four places to miss.
    """

    def model_line(name: str) -> str:
        module, argv = _COMMANDS[name]
        text = _help_text(module, argv)
        start = text.index("--model")
        # The help body wraps; join the remaining panel lines into one blob.
        return " ".join(text[start:].split())

    descriptions = {}
    for name in _COMMANDS:
        joined = model_line(name)
        # Cut at the next option so only the --model description is compared.
        body = joined.split("--output")[0].strip()
        # The panels wrap at different widths per command (quantize has a
        # SUBCOMMAND column), so compare the WORDS, not the line breaks.
        descriptions[name] = " ".join(body.split("\u2502")).replace("\u2502", "")
        descriptions[name] = " ".join(descriptions[name].split())

    first_name = next(iter(descriptions))
    first = descriptions[first_name]
    for name, desc in descriptions.items():
        assert desc == first, f"{name} describes --model differently from {first_name}:\n  {desc}\n  {first}"
