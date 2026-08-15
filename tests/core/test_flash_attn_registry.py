"""Tests for the Flash Attention 2 registry entry (audit T3 #4 / Finding AO).

These tests pin the **soft-dependency contract** of
:class:`llm.core.attn.FlashAttention`:

1. The module imports cleanly even when ``flash-attn`` is not
   installed (CI runs on CPU-only hosts by default).
2. ``ATTENTION_REGISTRY`` always has a ``flash_attn`` entry, with
   the correct KV-cache capability flag.
3. Constructing :class:`FlashAttention` without ``flash-attn``
   installed raises :class:`ImportError` with a remediation hint.
4. :class:`FlashAttention` is exported from :mod:`llm.core.attn`.

Heavy GPU-only behavior (kernel correctness vs MHA, real speedup) is
out of scope here — that lives in ``tests/core/attn/test_mha.py`` for
the reference path, and would need a CUDA host with ``flash-attn``
installed.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

import llm.core.attn
import llm.core.attn.flash_attn  # noqa: F401
from llm.core.attn import FLASH_ATTN_AVAILABLE, FlashAttention
from llm.core.registry import (
    ATTENTION_KV_CACHE_CAPABILITY,
    ATTENTION_REGISTRY,
    attention_supports_kv_cache,
)

# --- Registry wiring --------------------------------------------------------


def test_flash_attn_is_registered():
    """The ``flash_attn`` entry is always present, even on CPU-only hosts."""
    assert "flash_attn" in ATTENTION_REGISTRY.names()


def test_flash_attn_registry_entry_resolves_to_class():
    """The registry resolves to the :class:`FlashAttention` class object."""
    cls = ATTENTION_REGISTRY.get("flash_attn")
    assert cls is FlashAttention


def test_flash_attn_kv_cache_capability_matches_mha():
    """``flash_attn`` advertises KV-cache support like ``mha``.

    The continuous batching engine keys off this map to decide whether
    the impl can drive autoregressive decode.
    """
    assert ATTENTION_KV_CACHE_CAPABILITY["flash_attn"] is True
    assert attention_supports_kv_cache("flash_attn") is True


def test_flash_attn_exported_from_attn_package():
    """``from llm.core.attn import FlashAttention`` works."""
    module = importlib.import_module("llm.core.attn")
    assert module.FlashAttention is FlashAttention
    assert module.FLASH_ATTN_AVAILABLE is FLASH_ATTN_AVAILABLE


# --- Soft-dependency contract ----------------------------------------------


def test_module_imports_without_flash_attn_installed():
    """The module imports even when ``flash_attn`` is missing.

    CPU-only CI must be able to ``import llm.core.attn`` without
    installing the CUDA-only ``flash-attn`` package.
    """
    # If we got here, the imports at the top of the file succeeded.
    # FLASH_ATTN_AVAILABLE is the explicit signal.
    assert FLASH_ATTN_AVAILABLE is False or FLASH_ATTN_AVAILABLE is True  # always bool


@pytest.mark.skipif(FLASH_ATTN_AVAILABLE, reason="flash-attn is installed — gate on the no-install branch")
def test_instantiation_without_flash_attn_raises_import_error():
    """Without ``flash-attn`` installed, ``FlashAttention()`` raises.

    The error message must mention the install command so users can
    self-remediate without reading the source.
    """
    with pytest.raises(ImportError, match="flash-attn"):
        FlashAttention(hidden_size=64, num_heads=4)


# --- Heavy behavior guarded by the optional dep ----------------------------


@pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="flash-attn is optional; install via `llm[perf]`")
def test_flash_attn_instantiates_when_dependency_present():
    """Sanity-check that the class wires up its projections when available.

    Run only when ``flash-attn`` is installed in the environment. The
    heavy correctness checks (vs MHA on GPU) are out of scope here.
    """
    attn = FlashAttention(hidden_size=64, num_heads=4, bias=False)
    # Combined QKV projection = (num_heads + 2*num_kv_heads) * head_dim.
    # Default num_kv_heads == num_heads, head_dim = hidden_size / num_heads.
    assert attn.qkv_dim == (4 + 2 * 4) * (64 // 4)  # 192
    assert attn.out_proj.in_features == 64


def test_flash_attn_constructible_through_transformer_block():
    """RIL ISS-137: ``attn_impl='flash_attn'`` must build through the model
    stack (TransformerBlock threads ``window_size=`` and — when requested —
    the RoPE kwargs into every attention backend).

    Before the fix, ``FlashAttention.__init__`` declared none of
    ``window_size`` / ``max_seq_len`` / ``use_rope`` / ``rope_theta``, so a
    block passed ``window_size=None`` unconditionally and EVERY
    ``attn_impl='flash_attn'`` model raised
    ``TypeError: got an unexpected keyword argument 'window_size'`` at build
    time (configs check out fine — it only broke on construction).
    """
    from llm.core.transformer_block import TransformerBlock

    # Patch the availability gate so construction trips on kwarg binding, not
    # the (absent) flash-attn package. The kernel is never invoked here.
    with (
        patch("llm.core.attn.flash_attn.FLASH_ATTN_AVAILABLE", True),
        patch.object(FlashAttention, "forward", lambda *a, **k: None),
    ):
        block = TransformerBlock(
            hidden_size=64,
            num_heads=4,
            attn_impl="flash_attn",
            mlp_impl="mlp",
            is_causal=True,
        )
    assert isinstance(block.self_attn, FlashAttention)

    # Same build when the model requests RoPE (block threads the Rope kwargs).
    with (
        patch("llm.core.attn.flash_attn.FLASH_ATTN_AVAILABLE", True),
        patch.object(FlashAttention, "forward", lambda *a, **k: None),
    ):
        block = TransformerBlock(
            hidden_size=64,
            num_heads=4,
            attn_impl="flash_attn",
            mlp_impl="mlp",
            is_causal=True,
            use_rope=True,
            max_seq_len=32,
            rope_theta=10000.0,
        )
    assert isinstance(block.self_attn, FlashAttention)
    assert block.self_attn.use_rope is True
