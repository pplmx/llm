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
    return sorted(_DOCS.rglob("*.md"))


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
