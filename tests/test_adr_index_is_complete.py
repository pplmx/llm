"""Every ADR file must be listed in ``docs/adr/README.md``'s index.

The index had drifted silently: ADR-013 (multimodal encoder contract) shipped
without an entry, and the list stopped at 012. Nothing caught it, because no
test looked at the directory — ``mkdocs --strict`` builds the pages fine whether
or not the index mentions them, and a reader browsing the index simply never
learns the newer decisions exist.

This is the same class as ``test_pytest_markers.py`` (a declared marker nothing
uses) and ``test_entry_point_doc_consistency.py`` (a table naming the wrong
source of truth): the artefact and its advertisement disagree, and only the
artefact works.
"""

from __future__ import annotations

import pathlib
import re

ADR_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs" / "adr"
INDEX = ADR_DIR / "README.md"

# ``template.md`` is a skeleton, not a decision; README is the index itself.
_NOT_ADRS = {"README.md", "template.md"}

_INDEX_LINK = re.compile(r"^- \[ADR-\d{3}: [^\]]+\]\(([\w.-]+\.md)\)$", re.M)


def _adr_files() -> set[str]:
    files = {p.name for p in ADR_DIR.glob("*.md")} - _NOT_ADRS
    # Anti-vacuity: a wrong ADR_DIR would otherwise make every assertion below
    # compare empty sets and pass.
    assert len(files) >= 10, f"expected the ADR set, found {sorted(files)} in {ADR_DIR}"
    return files


def _indexed_files() -> set[str]:
    return set(_INDEX_LINK.findall(INDEX.read_text(encoding="utf-8")))


def test_every_adr_is_listed_in_the_index() -> None:
    missing = _adr_files() - _indexed_files()
    assert not missing, (
        "these ADRs exist but are absent from docs/adr/README.md's index, so a reader "
        f"browsing the index never learns they exist: {sorted(missing)}"
    )


def test_the_index_links_no_nonexistent_adr() -> None:
    dangling = _indexed_files() - _adr_files()
    assert not dangling, f"the index links ADRs that do not exist: {sorted(dangling)}"


def test_the_parser_actually_sees_the_index() -> None:
    """Positive control: a broken regex must fail, not pass over nothing."""
    indexed = _indexed_files()
    assert len(indexed) >= 10, f"the index parser found only {len(indexed)} entries; the regex broke"
    assert "001-use-gqa.md" in indexed, "the oldest ADR should always be indexed"
