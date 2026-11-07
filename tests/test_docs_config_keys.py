"""Docs must not name config keys the schema does not have.

This repo has already paid for this class of drift twice:

* `training.learning_rate` appeared in **eight** shipped YAML configs (the real
  key is `lr`), so the intended learning rates were silently replaced by the
  1e-3 default — a 50x error for ``sft_alpaca.yaml``.
* The same typo then lived on in the tutorials and in an API doc example that
  was copy-paste executable and raised ``AttributeError``.

Fixing the YAML files did not fix the prose. This test scans every fenced
``yaml`` block in ``docs/`` and every ``config.<section>.<field>`` Python
access, and requires the names to exist in the real schema — so a rename has to
be followed through the documentation.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from llm.serving.config import ServingConfig
from llm.training.core.config import Config

_DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
# Guard against the silent-vacuous failure this file was written to avoid: a
# wrong parents[] index pointed at a non-existent directory, so every scan
# iterated ZERO documents and passed unconditionally.
assert _DOCS.is_dir(), f"docs directory not found at {_DOCS}"


def _training_schema() -> dict[str, set[str]]:
    schema = {"<top>": set(Config.model_fields)}
    for section, field in Config.model_fields.items():
        if hasattr(field.annotation, "model_fields"):
            schema[section] = set(field.annotation.model_fields)
    return schema


def _all_docs() -> list[pathlib.Path]:
    """Markdown files whose content is enforced.

    ``docs/superpowers/`` holds historical design records for modules that were
    proposed and either renamed or never shipped; they are not instructions a
    reader follows, so holding them to today's API would only produce noise.
    """
    return sorted(p for p in _DOCS.rglob("*.md") if "superpowers" not in p.parts)


def _yaml_blocks(text: str) -> list[str]:
    return re.findall(r"```ya?ml\n(.*?)```", text, re.S)


def test_no_doc_yaml_block_uses_an_unknown_training_key() -> None:
    """Every key inside a documented YAML block must exist in the schema.

    A doc block is what users copy, so a stale key here reproduces the exact
    silent-default bug the runtime guard now prevents for real config files.
    """
    schema = _training_schema()
    serving_fields = set(ServingConfig.model_fields)
    # Sections that belong to a different schema (Kubernetes, Prometheus, DPO's
    # descriptive pseudo-YAML, ...) are not training config; a block is only
    # checked when its top-level sections are recognised.
    problems: list[str] = []
    for path in _all_docs():
        for block in _yaml_blocks(path.read_text(encoding="utf-8", errors="ignore")):
            section: str | None = None
            for line in block.split("\n"):
                top = re.match(r"^([a-z_][a-z0-9_]*):\s*$", line)
                if top:
                    section = top.group(1)
                    continue
                key = re.match(r"^ {2}([a-z_][a-z0-9_]*):", line)
                if not key or section is None:
                    continue
                pool = schema.get(section) or serving_fields
                if pool and key.group(1) not in pool:
                    problems.append(f"{path.relative_to(_DOCS.parent)}: {section}.{key.group(1)}")
    assert not problems, "documented YAML keys missing from the schema:\n  " + "\n  ".join(problems)


def test_no_doc_attribute_access_uses_an_unknown_config_field() -> None:
    """``config.<section>.<field>`` in a doc snippet must be a real attribute."""
    schema = _training_schema()
    problems: list[str] = []
    for path in _all_docs():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"\bconfig\.([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)", text):
            section, field_name = match.group(1), match.group(2)
            pool = schema.get(section) or set(ServingConfig.model_fields)
            if pool and field_name not in pool:
                problems.append(f"{path.relative_to(_DOCS.parent)}: config.{section}.{field_name}")
    assert not problems, "documented config attribute accesses that do not exist:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("stale", ["learning_rate"])
def test_the_known_rename_stays_gone(stale: str) -> None:
    """``learning_rate`` was renamed to ``lr``; no doc may reintroduce it.

    The generic scans above cover this, but an explicit case documents WHY the
    scans exist and gives a readable failure if the name comes back.
    """
    offenders = [str(p.relative_to(_DOCS.parent)) for p in _all_docs() if stale in p.read_text(errors="ignore")]
    assert not offenders, f"docs still mention the removed field {stale!r}: {offenders}"


# --- Documented Python imports must resolve ---


def _python_blocks() -> list[tuple[pathlib.Path, str]]:
    blocks: list[tuple[pathlib.Path, str]] = []
    for path in _all_docs():
        text = path.read_text(encoding="utf-8", errors="ignore")
        blocks.extend((path, match.group(1)) for match in re.finditer(r"```python\n(.*?)```", text, re.S))
    return blocks


def test_documented_llm_imports_resolve() -> None:
    """Every ``from llm.<mod> import <name>`` in a doc block must exist.

    Docs are executable instructions; an import of a symbol the package no
    longer exports sends the reader straight to ``ImportError``. This caught
    ``from llm.tokenization import load_tokenizer`` (the factory is
    ``TokenizerFactory`` now) and ``DecoderModel.from_pretrained`` in the
    evaluation guide.
    """
    import importlib

    problems: list[str] = []
    for path, block in _python_blocks():
        for match in re.finditer(r"^from (llm[\w.]*) import ([\w, ]+)$", block, re.M):
            module_name, names = match.group(1), match.group(2)
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                problems.append(f"{path.relative_to(_DOCS.parent)}: module {module_name} does not exist")
                continue
            problems.extend(
                f"{path.relative_to(_DOCS.parent)}: {module_name}.{name} is not exported"
                for name in (n.strip() for n in names.split(","))
                if name and not hasattr(module, name)
            )
    assert not problems, "documented imports that do not resolve:\n  " + "\n  ".join(problems)


# --- Documented CLI invocations must match the real commands and flags ---

#: ``(llm argv prefix)`` -> nothing; the flag set is read from ``--help`` once.
_CLI_TIMEOUT = 120


def _cli_flags(prefix: tuple[str, ...]) -> set[str]:
    """Flags accepted by ``llm <prefix>``, read from its real ``--help``."""
    import subprocess

    result = subprocess.run(  # noqa: S603 — argv is built from internal constants, not user input
        ["uv", "run", "--frozen", "llm", *prefix, "--help"],  # noqa: S607 — `uv` comes from PATH by design
        capture_output=True,
        text=True,
        timeout=_CLI_TIMEOUT,
        check=False,
    )
    assert result.returncode == 0, f"`llm {' '.join(prefix)} --help` failed:\n{result.stderr[-800:]}"
    return set(re.findall(r"(--[a-z][a-z0-9-]+)", result.stdout))


#: ``command -> subcommands`` for the commands that take one. ``quantize`` is
#: the only two-level command; a doc line without the subcommand would reference
#: flags that live one level down and never reach them.
_SUBCOMMANDS = {"quantize": ["gptq", "fp8"]}


def _documented_invocations() -> list[tuple[pathlib.Path, str]]:
    """Every ``llm <command> ...`` occurrence in a live doc, with its line."""
    found: list[tuple[pathlib.Path, str]] = []
    for path in _all_docs():
        for line in path.read_text(encoding="utf-8", errors="ignore").split("\n"):
            if not re.search(r"\bllm\s+(train|serve|quantize|prune|decompose|migrate-ckpt)\b", line):
                continue
            # Prose that EXPLAINS a flag does not exist ("`llm train` has no
            # --scheduler-type") names a flag without invoking it. Only lines
            # that look like an invocation are checked.
            if re.search(r"没有|不存在|不支持|has no|does not have|is not a", line):
                continue
            found.append((path, line))
    return found


def test_documented_cli_flags_exist() -> None:
    """A documented flag must be accepted by the command it is written for.

    Docs are instructions: an invented or renamed flag sends the reader to a
    Typer "No such option" error. Reading the flag sets from ``--help`` rather
    than from source keeps the test honest about what the CLI really accepts,
    including aliases Typer synthesises.
    """
    real: dict[tuple[str, ...], set[str]] = {}
    problems: list[str] = []
    for path, line in _documented_invocations():
        command = re.search(r"\bllm\s+(train|serve|quantize|prune|decompose|migrate-ckpt)", line).group(1)
        prefix: tuple[str, ...] = (command,)
        if command in _SUBCOMMANDS:
            # Pick the subcommand this line names, if any; otherwise the flags
            # are compared against the union of all its subcommands.
            named = [s for s in _SUBCOMMANDS[command] if re.search(rf"\bllm\s+{command}\s+{s}\b", line)]
            if named:
                prefix = (command, named[0])
        if prefix not in real:
            real[prefix] = _cli_flags(prefix)
        if command in _SUBCOMMANDS and len(prefix) == 1:
            allowed = set().union(*(_cli_flags((command, s)) for s in _SUBCOMMANDS[command]))
        else:
            allowed = real[prefix]
        for flag in re.findall(r"(--[a-z][a-z0-9-]+)", line):
            if flag == "--help":
                continue
            if flag not in allowed:
                problems.append(
                    f"{path.relative_to(_DOCS.parent)}: `llm {' '.join(prefix)}` has no {flag}\n    {line.strip()}"
                )
    assert not problems, "documented CLI flags that do not exist:\n  " + "\n  ".join(sorted(set(problems)))
