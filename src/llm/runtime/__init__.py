"""Runtime plugin kernel: registries and shared factories."""

from llm.runtime.bootstrap import ensure_builtins_registered
from llm.runtime.model_factory import MODEL_REGISTRY, ModelFactory, build_decoder
from llm.runtime.registry import Registry
from llm.runtime.tokenizer_factory import TokenizerFactory

ensure_builtins_registered()

__all__ = [
    "MODEL_REGISTRY",
    "ModelFactory",
    "Registry",
    "TokenizerFactory",
    "build_decoder",
    "ensure_builtins_registered",
]
