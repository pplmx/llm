"""High-level GGUF → model loader (closes the export-only loop, round 71).

The format layer (reader/writer/quant/metadata) stays torch-free; this
module is deliberately high-level: it parses a GGUF file with
:class:`GGUFReader`, rebuilds the exact ``llm`` model from the
``general.llm_model_config`` JSON blob the exporter writes, and fills its
state dict from the dequantized tensors.

Typical round trip::

    from llm.export.gguf import export_to_gguf, load_gguf_model

    export_to_gguf(model, "m.gguf", quantize="f16", model_config=cfg.model_dump())
    restored = load_gguf_model("m.gguf")

A GGUF without the config blob (e.g. a third-party llama.cpp file) is
refused with a clear error — mapping foreign tensor names into ``llm``
state-dict keys (via ``llm.compat.weight_mapping``) is a separate future
milestone, not v1 of this loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from llm.export.gguf.reader import GGUFReader
from llm.export.gguf.spec import GGUFError
from llm.runtime.model_factory import ModelFactory
from llm.training.core.config import ModelConfig

__all__ = ["load_gguf_model"]


def load_gguf_model(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Rebuild and load a model from a GGUF file produced by this exporter.

    Args:
        path: GGUF file path.
        device: Optional target device (default: CPU).

    Returns:
        The rebuilt model in ``eval()`` mode with the exported weights.

    Raises:
        GGUFError: If the file is malformed or carries no
            ``general.llm_model_config`` metadata (i.e. was not written by
            ``export_to_gguf(..., model_config=...)``).
        RuntimeError: If the tensor names/shapes in the file do not match the
            model rebuilt from the embedded config (strict ``load_state_dict``).

    Note:
        F32/F16 exports round-trip exactly (both widened to float32); *block-
        quantized* exports (Q4_0/Q8_0) come back dequantized and therefore
        approximately, within the quantizer's expected error.
    """
    reader = GGUFReader(path)

    raw_config = reader.metadata.get("general.llm_model_config")
    if not isinstance(raw_config, str):
        raise GGUFError(
            f"{reader.path}: no 'general.llm_model_config' metadata — export with "
            "export_to_gguf(..., model_config=<ModelConfig>.model_dump()) to make "
            "the file model-loader-loadable"
        )
    try:
        cfg = ModelConfig.model_validate(json.loads(raw_config))
    except Exception as exc:
        raise GGUFError(f"{reader.path}: invalid 'general.llm_model_config' JSON: {exc}") from exc

    model = ModelFactory.from_config(cfg)

    state: dict[str, torch.Tensor] = {}
    for name in reader.tensors:
        # ``read_tensor`` returns a read-only view over the file bytes for
        # F32/F16 (``np.frombuffer``); torch.from_numpy on a non-writable array
        # emits a UB warning, so take a writable copy before conversion.
        state[name] = torch.from_numpy(np.ascontiguousarray(reader.read_tensor(name)).copy())

    model.load_state_dict(state, strict=True)

    if device is not None:
        model.to(device)
    model.eval()
    return model
