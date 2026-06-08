__version__ = "0.1.0"

# Core components (import to trigger registration)
import llm.core.attn.mha
import llm.core.mlp
import llm.core.moe  # noqa: F401

# Public API exports
from llm.inference import batch_generate, generate, stream_generate
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

__all__ = [
    "DecoderModel",
    "SimpleCharacterTokenizer",
    "__version__",
    "batch_generate",
    "generate",
    "stream_generate",
]
