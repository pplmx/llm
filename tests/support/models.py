"""Decoder model factory defaults for model-layer tests."""

import torch

DECODER_BATCH_SIZE = 2
DECODER_SEQ_LEN = 10

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

DEFAULT_DECODER_KWARGS: dict = {
    "vocab_size": 500,
    "hidden_size": 64,
    "num_layers": 2,
    "num_heads": 4,
    "max_seq_len": 128,
    "intermediate_size": 64 * 4,
    "pos_encoding_learned": False,
    "embedding_dropout_p": 0.0,
    "attn_dropout_p": 0.0,
    "mlp_dropout_p": 0.0,
    "mlp_activation": "gelu",
    "norm_eps": 1e-5,
    "norm_first": True,
    "is_causal": True,
    "padding_idx": None,
    "qkv_bias": True,
    "mlp_bias": True,
    "lm_head_bias": True,
}


def decoder_model_kwargs(**overrides) -> dict:
    """Return a copy of default DecoderModel kwargs with optional overrides."""
    kwargs = DEFAULT_DECODER_KWARGS.copy()
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("dtype", torch.float32)
    kwargs.update(overrides)
    return kwargs
