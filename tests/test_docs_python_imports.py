"""Every ``from llm...`` import in the docs must name something that exists.

``docs/development/guide-extending.md``'s "add a task" recipe was once
copy-paste executable and raised ``AttributeError``; the config-key sibling of
this test (`test_docs_config_keys.py`) exists because the same drift class hit
eight shipped YAML files. The Python examples are the part a reader actually
pastes into a REPL, so a rename that is not followed through the docs produces
an ``ImportError`` in the reader's first minute.

This test parses every fenced ``python`` block, keeps the ``ImportFrom`` nodes
whose module starts with ``llm``, imports the module and requires each imported
name to be an attribute of it.

Two deliberate exclusions keep the check honest rather than noisy:

* Blocks under ``docs/superpowers/`` are historical design records for modules
  that were proposed and renamed or never shipped — the config-key test already
  excludes them for the same reason.
* A block that imports a module the prose tells the reader to CREATE (the
  "add a task" recipe does exactly this) is skipped. Those modules legitimately
  do not exist yet; requiring them would forbid the tutorial.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import re

_DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
# Guard against the silent-vacuous failure this kind of scan invites: a wrong
# parents[] index once pointed at a non-existent directory and every assertion
# passed over zero documents.
assert _DOCS.is_dir(), f"docs directory not found at {_DOCS}"

_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def _enforced_docs() -> list[pathlib.Path]:
    return sorted(p for p in _DOCS.rglob("*.md") if "superpowers" not in p.parts)


def _llm_imports(text: str):
    """Yield ``(lineno, module, names)`` for every ``from llm...`` import."""
    for block in _PYTHON_BLOCK.findall(text):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            # Not every fenced block is standalone-parseable (some are
            # fragments with placeholders). Those are out of scope.
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("llm"):
                yield node.lineno, node.module, [alias.name for alias in node.names]


def test_every_documented_llm_import_resolves() -> None:
    """A renamed or deleted symbol must not linger in the examples."""
    problems: list[str] = []
    checked = 0
    teaching = 0
    for path in _enforced_docs():
        rel = path.relative_to(_DOCS.parent)
        for lineno, module, names in _llm_imports(path.read_text(encoding="utf-8")):
            try:
                imported = importlib.import_module(module)
            except ModuleNotFoundError:
                # A module the prose tells the reader to create.
                teaching += 1
                continue
            except Exception as exc:  # noqa: BLE001 — any import failure is a doc bug
                problems.append(f"{rel}:{lineno}: importing {module} raised {exc!r}")
                continue
            for name in names:
                checked += 1
                if not hasattr(imported, name):
                    problems.append(f"{rel}:{lineno}: {module} has no attribute {name!r}")

    # Anti-vacuity floor: today this covers ~176 symbols. If the parser or the
    # regex breaks, the count collapses and this fails instead of going green.
    assert checked >= 100, (
        f"only {checked} documented imports were checked (expected 100+); the scan is "
        "not seeing the docs, so a pass here would be meaningless"
    )
    assert not problems, "documented imports that do not resolve:\n  " + "\n  ".join(problems)


def test_the_scan_actually_finds_documented_imports() -> None:
    """Positive control: the extractor must see a known-good example.

    ``docs/guides/inference.md`` and friends import real ``llm`` symbols; if this
    string-level extraction silently stopped matching fenced blocks, the test
    above would pass over an empty set.
    """
    sample = "```python\nfrom llm.training.core.config import Config\n```\n"
    found = list(_llm_imports(sample))
    # lineno is block-relative (the fenced snippet is parsed on its own), which
    # is what the failure message needs: it points inside the example.
    assert found == [(1, "llm.training.core.config", ["Config"])], found


def test_a_renamed_symbol_would_be_caught() -> None:
    """Positive control for the failure path, not just the happy path."""
    import llm.training.core.config as real_config

    assert hasattr(real_config, "Config")
    assert not hasattr(real_config, "Configr"), "pick a name that really is absent"

    problems = []
    for _lineno, module, names in _llm_imports("```python\nfrom llm.training.core.config import Configr\n```\n"):
        imported = importlib.import_module(module)
        problems.extend(name for name in names if not hasattr(imported, name))
    assert problems == ["Configr"], problems
