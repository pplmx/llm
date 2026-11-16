"""The extension-point table must name where each built-in really comes from.

`docs/reference/architecture.md` summarises how extensions plug in, and that
table is the page a contributor reads before adding one. It used to present a
single "setuptools entry points in `pyproject.toml`" column listing every
example — so a reader looking for `lora`, or any of the eight PEFT methods,
found nothing in `pyproject.toml`: all eight are registered **in code**, and
`llm.peft_methods` is declared as an empty group on purpose so third parties
can add methods. The same applied to `dedup_local` / `dedup_hf` (code) versus
`local` / `hf` (pyproject), and to `onnx` (code) versus `torchscript` / `gguf`
(pyproject).

A table that misplaces half its rows is worse than no table: it sends the
reader to grep a file that cannot contain the answer. The table now separates
the two columns, and this test keeps each column honest against the real
sources of truth — `pyproject.toml` for the entry-point column, and the
registries after import for the in-code column.
"""

from __future__ import annotations

import pathlib
import re
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ARCHITECTURE = REPO_ROOT / "docs" / "reference" / "architecture.md"

# The in-code column explains a handful of built-ins with a parenthetical note;
# strip that before comparing names.
_TRAILING_NOTE = re.compile(r"\s*\([^)]*\)")


def _doc_rows() -> list[tuple[str, str, str]]:
    """Return ``(group, code_column, pyproject_column)`` for each table row."""
    text = ARCHITECTURE.read_text(encoding="utf-8")
    rows = []
    for line in text.split("\n"):
        match = re.match(
            r"^\|\s*`(llm\.[a-z_]+)`\s*\|[^|]*\|([^|]*)\|([^|]*)\|\s*$",
            line,
        )
        if not match:
            continue
        rows.append((match.group(1), match.group(2).strip(), match.group(3).strip()))
    return rows


def _names(cell: str) -> set[str]:
    """Backticked identifiers in a cell, as a set."""
    return set(re.findall(r"`([a-z_0-9]+)`", cell))


def _pyproject_groups() -> dict[str, set[str]]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {group: set(entries) for group, entries in data["project"]["entry-points"].items()}


def test_table_has_the_expected_shape() -> None:
    """An anti-vacuity floor: the parse found real rows with real content."""
    rows = _doc_rows()
    assert len(rows) == 6, f"expected 6 extension-point rows, parsed {len(rows)}: {[r[0] for r in rows]}"
    groups = {row[0] for row in rows}
    assert "llm.peft_methods" in groups
    assert "llm.data_sources" in groups


@pytest.mark.parametrize(("group", "code_cell", "pyproject_cell"), _doc_rows() or [("", "", "")])
def test_pyproject_column_is_really_in_pyproject(group: str, code_cell: str, pyproject_cell: str) -> None:
    """Every name in the ``pyproject.toml`` column must be declared there."""
    if not group:
        pytest.skip("no rows parsed (test_table_has_the_expected_shape reports that)")
    del code_cell  # the other column is checked by its own test
    if "third-party" in pyproject_cell:
        pytest.skip(f"{group} documents third-party hooks rather than built-in names")
    declared = _pyproject_groups().get(group, set())
    claimed = _names(pyproject_cell)
    missing = claimed - declared
    assert not missing, (
        f"{group}: the docs' pyproject column claims {sorted(missing)}, but pyproject.toml declares {sorted(declared)}"
    )


@pytest.mark.parametrize(("group", "code_cell", "pyproject_cell"), _doc_rows() or [("", "", "")])
def test_code_column_is_not_in_pyproject(group: str, code_cell: str, pyproject_cell: str) -> None:
    """A name in the ``registered in code`` column must NOT be an entry point.

    This is the half that was wrong: the table used to put these names in the
    pyproject column, sending readers to a file that never mentioned them.
    """
    if not group:
        pytest.skip("no rows parsed (test_table_has_the_expected_shape reports that)")
    del pyproject_cell  # the other column is checked by its own test
    declared = _pyproject_groups().get(group, set())
    in_code = _names(_TRAILING_NOTE.sub("", code_cell))
    misplaced = in_code & declared
    assert not misplaced, (
        f"{group}: {sorted(misplaced)} is listed under 'registered in code' but pyproject.toml really does declare it"
    )


def test_peft_methods_are_registered_in_code_not_as_entry_points() -> None:
    """The concrete case that made the old table misleading.

    All eight built-in PEFT methods register in code; the
    ``llm.peft_methods`` entry-point group is declared EMPTY on purpose, so a
    contributor's grep of ``pyproject.toml`` for ``lora`` finds nothing. Pin
    both halves so a future move in either direction is visible here.
    """
    from llm.core.registry import PEFT_REGISTRY, ensure_peft_methods_registered

    ensure_peft_methods_registered()
    names = set(PEFT_REGISTRY.names()) if hasattr(PEFT_REGISTRY, "names") else set(PEFT_REGISTRY)
    expected = {"lora", "qlora", "adalora", "prefix_tuning", "ia3", "bitfit", "adapter", "pfeiffer_adapter"}
    assert expected <= names, f"missing built-in PEFT methods: {sorted(expected - names)}"

    declared = _pyproject_groups().get("llm.peft_methods", set())
    overlap = expected & declared
    assert not overlap, (
        f"{sorted(overlap)} is both a built-in (registered in code) and a pyproject entry point; "
        "the docs table says the group is empty for third parties"
    )
