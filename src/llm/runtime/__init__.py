"""Runtime plugin kernel: registries and shared factories."""

from llm.runtime.bootstrap import ensure_builtins_registered
from llm.runtime.checkpoint import CheckpointContributor, collect_extra_state, load_extra_state
from llm.runtime.model_factory import MODEL_REGISTRY, ModelFactory, build_decoder
from llm.runtime.plugins import load_entry_point_hooks, load_entry_point_registry
from llm.runtime.registry import Registry
from llm.runtime.tokenizer_factory import TokenizerFactory

ensure_builtins_registered()

__all__ = [
    "MODEL_REGISTRY",
    "CheckpointContributor",
    "ModelFactory",
    "Registry",
    "TokenizerFactory",
    "build_decoder",
    "collect_extra_state",
    "ensure_builtins_registered",
    "load_entry_point_hooks",
    "load_entry_point_registry",
    "load_extra_state",
]
