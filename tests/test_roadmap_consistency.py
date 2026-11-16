"""ROADMAP checkboxes must not describe work that is already in the tree.

The roadmap is a planning document, so most of it cannot be verified
mechanically — "减少峰值显存占用" has no predicate a test can evaluate. But a
narrow, high-value slice CAN be checked: when an unchecked item names a
concrete module or class that exists and is registered, the roadmap is
describing the present as the future, and every reader is misled about what
the project actually does.

Two live examples were found this way:

* ``10.1 Flash Attention 集成`` still said "暂缓 … 优先级降低" with three
  unchecked boxes, while ``core/attn/flash_attn.py`` shipped a
  GQA/MQA-capable, KV-cache-capable, TP-tested ``FlashAttention`` registered
  as ``attn_impl="flash_attn"``. The CHANGELOG had already recorded it as
  "previously listed under Planned, now shipped" — the roadmap section was
  simply never updated, and nothing noticed.
* ``12.3 设计通用的多模态数据接口`` was unchecked although
  ``MultimodalDataModule`` and ``MODALITY_ENCODER_REGISTRY`` exist and
  ADR-013 accepted that contract.

The check below is deliberately narrow: it only asserts that a small, explicit
set of capabilities that ARE implemented is not still listed as planned. It is
not a general roadmap linter — a keyword scan over the whole file would
constantly fire on genuinely future work that merely mentions a module name.
"""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROADMAP = REPO_ROOT / "ROADMAP.md"


def _unchecked_lines() -> list[tuple[int, str]]:
    """Every unchecked checkbox line in ROADMAP.md, with its line number."""
    return [
        (lineno, line)
        for lineno, line in enumerate(ROADMAP.read_text(encoding="utf-8").split("\n"), 1)
        if line.lstrip().startswith("- [ ]")
    ]


def test_flash_attention_is_not_listed_as_unimplemented():
    """The Flash Attention backend ships, so 10.1 must not claim otherwise.

    Guards the specific regression found this round: two of the three boxes
    were unchecked (and a note called the work "暂缓") for a backend that is
    registered, GQA-capable and covered by a TP parity test.
    """
    from llm.core.registry import ATTENTION_REGISTRY

    names = list(ATTENTION_REGISTRY.names()) if hasattr(ATTENTION_REGISTRY, "names") else list(ATTENTION_REGISTRY)
    assert "flash_attn" in names, "the flash_attn backend disappeared from the registry"

    offenders = [
        f"ROADMAP.md:{lineno}: {line.strip()}"
        for lineno, line in _unchecked_lines()
        if "集成 Flash Attention" in line or "在 MHA 和 GQA 中启用 Flash Attention" in line
    ]
    assert not offenders, (
        "flash_attn is implemented and registered (attn_impl='flash_attn'), "
        "but the roadmap still lists it as planned:\n  " + "\n  ".join(offenders)
    )


def test_multimodal_data_interface_is_not_listed_as_unimplemented():
    """``MultimodalDataModule`` + the encoder registry exist; ADR-013 accepted it."""
    from llm.multimodal.data import MultimodalDataModule
    from llm.multimodal.encoders import MODALITY_ENCODER_REGISTRY

    assert MultimodalDataModule is not None
    assert MODALITY_ENCODER_REGISTRY is not None

    offenders = [
        f"ROADMAP.md:{lineno}: {line.strip()}"
        for lineno, line in _unchecked_lines()
        if "设计通用的多模态数据接口" in line
    ]
    assert not offenders, (
        "MultimodalDataModule and MODALITY_ENCODER_REGISTRY ship (ADR-013 accepted the "
        "contract), but the roadmap still lists the interface as planned:\n  " + "\n  ".join(offenders)
    )


def test_roadmap_still_distinguishes_done_from_remaining():
    """A sanity floor: the checks above must not pass by the file going empty.

    If a future edit deleted the roadmap or stripped its checkboxes, the two
    tests above would vacuously succeed; this pins that both states still exist
    in meaningful quantity.
    """
    text = ROADMAP.read_text(encoding="utf-8")
    checked = text.count("- [x]")
    unchecked = text.count("- [ ]")
    assert checked > 100, f"only {checked} completed items — the roadmap looks truncated"
    assert unchecked > 5, f"only {unchecked} remaining items — every checkbox may have been flipped"
