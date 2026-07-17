"""Attention implementations, registered via ATTENTION_REGISTRY.

Each submodule in this package registers its implementation as an
``attn_impl`` value. Importing them here guarantees the side-effect
registration runs at engine-bootstrap time so users can reference
``attn_impl="mha"``, ``attn_impl="mla"``, or ``attn_impl="flash_attn"``
in their :class:`ModelConfig`.

The :class:`FlashAttention` class is the opt-in Flash Attention 2
backend; see ``core/attn/flash_attn.py`` for the soft-dependency
contract (the ``flash-attn`` package is optional).
"""

from .base import PrefixCapableAttention
from .flash_attn import FLASH_ATTN_AVAILABLE, FlashAttention
from .mha import MultiHeadAttention
from .mla import MultiLatentAttention
from .sdpa import sdpa

__all__ = [
    "FLASH_ATTN_AVAILABLE",
    "FlashAttention",
    "MultiHeadAttention",
    "MultiLatentAttention",
    "PrefixCapableAttention",
    "sdpa",
]
