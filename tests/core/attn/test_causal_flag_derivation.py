"""Both attention backends must DERIVE ``has_past``, never assume it.

``has_past`` decides the attention call's ``causal`` flag. Hardcoding it to
``True`` on the ``batch_indices`` branch made every first-pass prefill
non-causal — position 0 could attend to the last prompt token, silently
(measured 0.14 max divergence from the causal result, no error, no warning).

``mha.py`` was fixed first and is covered behaviourally in ``test_mha.py``.
``flash_attn.py`` had the identical line, but the backend is an optional
CUDA-only install, so the analogous behavioural test would be skipped on most
machines — which is exactly how the bug survived. These tests check the
DERIVATION itself, which is backend-independent pure logic, plus a source-level
guard so the unconditional form cannot come back to either file.
"""

from __future__ import annotations

import pathlib

import pytest
import torch
from torch import Tensor

_ATTN_DIR = pathlib.Path(__file__).resolve().parents[3] / "src" / "llm" / "core" / "attn"


def _derive_has_past(start_pos: int | Tensor) -> bool:
    """The expression both backends now use, exercised in isolation.

    Kept as a copy rather than imported so the test does not depend on reaching
    a private branch of a module whose backend may be unavailable.
    """
    return bool((start_pos.min() > 0).item()) if isinstance(start_pos, Tensor) else int(start_pos) > 0


@pytest.mark.parametrize(
    ("start_pos", "expected", "case"),
    [
        (0, False, "first-pass prefill (int)"),
        (3, True, "decode after 3 cached tokens (int)"),
        (torch.tensor([[0, 1, 2]]), False, "first-pass prefill ([B, S] grid)"),
        (torch.tensor([[3]]), True, "single-token decode ([B, S] grid)"),
        (torch.tensor([[0], [5]]), False, "mixed batch, one row at position 0"),
    ],
)
def test_has_past_is_derived_from_the_start_position(start_pos, expected, case):
    """A prefill (nothing cached) must report no past; a decode must report past."""
    assert _derive_has_past(start_pos) is expected, f"wrong has_past for {case}"


def test_a_prefill_start_is_zero_for_every_shape():
    """The one case that matters: position 0 means "no history", whatever the dtype."""
    assert not _derive_has_past(0)
    assert not _derive_has_past(torch.tensor(0))
    assert not _derive_has_past(torch.tensor([[0, 0]]))
    assert not _derive_has_past(torch.zeros(2, 2, dtype=torch.long))


@pytest.mark.parametrize("backend", ["mha.py", "flash_attn.py"])
def test_neither_backend_asserts_has_past_unconditionally(backend):
    """Source guard: the unconditional assignment must not reappear.

    A behavioural test for flash_attn would only run on a CUDA host with the
    optional package, so it would not have caught the original bug on the
    machines where it was written. This checks the source instead, and applies
    to both backends identically.
    """
    source = (_ATTN_DIR / backend).read_text(encoding="utf-8")

    assert "has_past = True" not in source, (
        f"{backend} asserts has_past unconditionally again; derive it from start_pos "
        "instead, or a first-pass prefill becomes non-causal"
    )
    # And the derivation must be present, not merely absent-by-deletion.
    assert "start_pos.min() > 0" in source, f"{backend} no longer derives has_past from start_pos"
