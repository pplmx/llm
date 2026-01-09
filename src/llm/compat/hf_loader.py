"""
HuggingFace Model Loader.

Provides `from_pretrained` functionality for loading HuggingFace models
into our DecoderModel format.
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch

from llm.compat.weight_mapping import (
    convert_hf_weights,
    detect_architecture,
    get_config_mapping,
)
from llm.models.decoder import DecoderModel

logger = logging.getLogger(__name__)


def from_pretrained(
    model_path: str | Path,
    device: str | torch.device = "auto",
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
) -> DecoderModel:
    """
    Load a pretrained model from HuggingFace format.

    Supports loading from:
    - Local directory with config.json and model weights
    - HuggingFace Hub model ID (requires huggingface_hub)

    Args:
        model_path: Local path or HuggingFace model ID.
        device: Device to load model on ("auto", "cuda", "cpu").
        dtype: Data type for model weights.
        trust_remote_code: Whether to trust remote code (for HF Hub).

    Returns:
        Loaded DecoderModel.
    """
    model_path = Path(model_path)

    # Determine if local or Hub
    if model_path.exists():
        return _load_from_local(model_path, device, dtype)
    else:
        return _load_from_hub(str(model_path), device, dtype, trust_remote_code)


def _load_from_local(
    model_dir: Path,
    device: str | torch.device,
    dtype: torch.dtype | None,
) -> DecoderModel:
    """Load model from local directory."""
    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open() as f:
        hf_config = json.load(f)

    logger.info(f"Loading model from {model_dir}")
    logger.info(f"Model type: {hf_config.get('model_type', 'unknown')}")

    # Detect architecture and get our config
    architecture = detect_architecture(hf_config)
    our_config = get_config_mapping(hf_config)

    logger.info(f"Detected architecture: {architecture}")

    # Determine dtype
    if dtype is None:
        torch_dtype = hf_config.get("torch_dtype", "float16")
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = DecoderModel(
        vocab_size=our_config["vocab_size"],
        hidden_size=our_config["hidden_size"],
        num_layers=our_config["num_layers"],
        num_heads=our_config["num_heads"],
        max_seq_len=our_config["max_seq_len"],
        num_kv_heads=our_config.get("num_kv_heads"),
        intermediate_size=our_config.get("intermediate_size"),
        norm_eps=our_config.get("rms_norm_eps", 1e-5),
        attn_impl="gqa" if our_config.get("num_kv_heads") else "mha",
        mlp_impl="swiglu",  # Most modern LLMs use SwiGLU
        device=device,
        dtype=dtype,
    )

    # Load weights
    state_dict = _load_weights(model_dir)

    # Convert weight names
    converted_state_dict = convert_hf_weights(
        state_dict,
        architecture=architecture,
        num_layers=our_config["num_layers"],
    )

    # Load into model
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    logger.info(f"Model loaded successfully to {device} with dtype {dtype}")

    return model


def _load_from_hub(
    model_id: str,
    device: str | torch.device,
    dtype: torch.dtype | None,
    trust_remote_code: bool,  # noqa: ARG001 - reserved for future HF Hub auth
) -> DecoderModel:
    """Load model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to load models from Hub. Install with: pip install huggingface_hub"
        )

    logger.info(f"Downloading model from Hub: {model_id}")

    # Download model files
    local_dir = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.json", "*.safetensors", "*.bin"],
    )

    return _load_from_local(Path(local_dir), device, dtype)


def _load_weights(model_dir: Path) -> dict[str, Any]:
    """
    Load weights from model directory.

    Supports both safetensors and PyTorch bin formats.
    """
    # Check for safetensors first (preferred)
    safetensor_files = list(model_dir.glob("*.safetensors"))
    if safetensor_files:
        return _load_safetensors(safetensor_files)

    # Fall back to PyTorch bin
    bin_files = list(model_dir.glob("*.bin"))
    if bin_files:
        return _load_pytorch_bins(bin_files)

    raise FileNotFoundError(f"No weight files found in {model_dir}")


def _load_safetensors(files: list[Path]) -> dict[str, Any]:
    """Load weights from safetensors files."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors is required to load .safetensors files. Install with: pip install safetensors")

    state_dict = {}
    for file_path in files:
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 - safetensors API requires .keys()
                state_dict[key] = f.get_tensor(key)

    logger.info(f"Loaded {len(state_dict)} tensors from {len(files)} safetensors files")
    return state_dict


def _load_pytorch_bins(files: list[Path]) -> dict[str, Any]:
    """Load weights from PyTorch bin files."""
    state_dict = {}
    for file_path in files:
        partial = torch.load(file_path, map_location="cpu", weights_only=True)
        state_dict.update(partial)

    logger.info(f"Loaded {len(state_dict)} tensors from {len(files)} bin files")
    return state_dict


def list_supported_architectures() -> list[str]:
    """List supported model architectures."""
    return ["llama", "llama2", "llama3", "mistral", "mixtral", "qwen", "qwen2"]
