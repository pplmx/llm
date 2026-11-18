from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from llm.export._wrapper import ExportCacheWrapper, clamp_input_shape, dummy_token_ids, model_vocab_size

#: Default ``(batch, seq_len)`` dummy shape for trace-based exports. The
#: sequence axis is clamped to the model's own context window at call time.
_DEFAULT_INPUT_SHAPE: tuple[int, int] = (1, 32)

logger = logging.getLogger(__name__)


def _normalize_layer_norm_dtypes(onnx_path: str | Path) -> Path:
    """Fix TorchScript ONNX LayerNormalization type binding for half-precision
    models.

    ``torch.onnx.export(..., dynamo=False)`` fuses a hand-written
    :class:`llm.core.layer_norm.LayerNorm` into an ONNX
    ``LayerNormalization`` node, but for fp16/bf16 models it binds the Norm's
    *weight/bias* inputs to the (fp16) initializer dtype while leaving the
    *X* input at fp32 — the exporter's value_info inference disagrees with
    the actual traced types. onnxruntime rejects the artifact at load time
    with ``Type parameter (T) of Optype (LayerNormalization) bound to
    different types``. The model itself is dtype-consistent in eager mode;
    only the fused graph is mislabeled.

    Fix: run ONNX shape inference, and for every ``LayerNormalization`` whose
    X-input inferred type differs from its weight dtype, insert a ``Cast``
    on X (to the weight dtype) before the node and a ``Cast`` on the output
    back to X's original dtype, keeping downstream consumers well-typed.

    Returns the (possibly rewritten) path. Pure post-processing: safe to run
    on any exported graph; no-ops when no type mismatch is found.
    """
    import onnx
    from onnx import shape_inference

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))

    # Populate value_info with inferred element types — the TorchScript
    # exporter leaves intermediate tensors untyped, and that is exactly why
    # onnxruntime cannot resolve the LayerNormalization type parameter.
    model = shape_inference.infer_shapes(model, strict_mode=False)

    # Type composition: value_info + input/outputs + initializers.
    elem_type_of: dict[str, int] = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("elem_type"):
            elem_type_of[vi.name] = vi.type.tensor_type.elem_type
    init_type_of = {i.name: i.data_type for i in model.graph.initializer}

    def _effective(name: str) -> int | None:
        return init_type_of.get(name) or elem_type_of.get(name)

    fixed = 0
    new_nodes: list[onnx.NodeProto] = []
    for node in list(model.graph.node):
        if node.op_type != "LayerNormalization" or len(node.input) < 2 or not node.output[0]:
            continue
        weight_ty = _effective(node.input[1])
        x_name = node.input[0]
        x_ty = elem_type_of.get(x_name)
        if weight_ty is None or x_ty is None or weight_ty == x_ty:
            continue
        # X input mislabeled: cast X to the weight dtype, run LN, cast back.
        cast_in_name = f"{x_name}__ln_x"
        new_nodes.append(onnx.helper.make_node("Cast", [x_name], [cast_in_name], to=weight_ty))
        out_name = node.output[0]
        cast_out_name = f"{out_name}__ln_out"
        node.input[0] = cast_in_name
        node.output[0] = cast_out_name
        new_nodes.append(onnx.helper.make_node("Cast", [cast_out_name], [out_name], to=x_ty))
        fixed += 1

    if fixed == 0:
        return onnx_path

    model.graph.node.extend(new_nodes)
    onnx.save(model, str(onnx_path))
    logger.info("Rewrote %d LayerNormalization node(s) to fix fp16 type binding.", fixed)
    return onnx_path


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int] | None = None,
    opset_version: int = 17,
    dynamic_axes: dict | None = None,
    verbose: bool = False,
) -> Path:
    """
    Export a model to ONNX format.

    Args:
        model: The model to export (e.g., DecoderModel)
        output_path: Path to save the ONNX file
        input_shape: (batch_size, seq_len) for the dummy input. ``None``
            (the default) traces at ``(1, 32)`` clamped to the model's own
            context window, so a model trained with a short ``max_seq_len``
            exports without arguments instead of failing the positional
            encoding's range check.
        opset_version: ONNX opset version (default: 17)
        dynamic_axes: Dynamic axes for variable-length inputs
        verbose: Print export details

    Returns:
        Path to the exported ONNX file

    Example::

        >>> model = DecoderModel(vocab_size=1000, hidden_size=64, num_layers=2, num_heads=4)  # doctest: +SKIP
        >>> export_to_onnx(model, "model.onnx", input_shape=(1, 32))  # doctest: +SKIP
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    # Wrap model to fix use_cache=False (avoids TracerWarning)
    wrapped_model = ExportCacheWrapper(model)
    wrapped_model.eval()

    # Create dummy input — bounded by the REAL vocab so small-vocab models
    # don't crash the embedding with out-of-range ids, and by the model's own
    # context window when the caller did not pin a shape.
    resolved_shape = input_shape if input_shape is not None else clamp_input_shape(model, _DEFAULT_INPUT_SHAPE)
    dummy_input = dummy_token_ids(model, resolved_shape, device=device)

    # Default dynamic axes for variable batch and sequence length
    if dynamic_axes is None:
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        }

    # Suppress expected warnings:
    # - TracerWarning from positional encoding bounds check
    # - DeprecationWarning from legacy TorchScript ONNX exporter (PyTorch 2.9+)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*TorchScript.*ONNX.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*feature will be removed.*")
        torch.onnx.export(
            wrapped_model,
            (dummy_input,),
            str(output_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=verbose,
            dynamo=False,
        )

    # fp16/bf16: TorchScript ONNX fusion mislabels LayerNormalization X-input
    # types, producing an artifact onnxruntime cannot load.
    # Pure post-processing fixes the type binding in place.
    model_dtype = next(model.parameters()).dtype
    if model_dtype in (torch.float16, torch.bfloat16):
        _normalize_layer_norm_dtypes(output_path)

    if verbose:
        logger.info("Exported model to %s", output_path)

    return output_path


def verify_onnx(
    onnx_path: str | Path,
    model: nn.Module | None = None,
    input_shape: tuple[int, int] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Verify ONNX model correctness by comparing with PyTorch output.

    Args:
        onnx_path: Path to ONNX file
        model: Original PyTorch model (optional, for comparison)
        input_shape: Input shape for verification
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if verification passes

    Raises:
        ImportError: If onnxruntime is not installed
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("onnxruntime is required: pip install onnxruntime") from e

    onnx_path = Path(onnx_path)

    # Create ONNX Runtime session
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Create test input — bounded by the real vocab when a model is supplied
    # so small-vocab models don't crash the embedding.
    resolved_shape = (
        input_shape
        if input_shape is not None
        else (clamp_input_shape(model, _DEFAULT_INPUT_SHAPE) if model is not None else _DEFAULT_INPUT_SHAPE)
    )
    if model is not None:
        test_input = dummy_token_ids(model, resolved_shape)
    else:
        batch_size, seq_len = resolved_shape
        test_input = torch.randint(0, 100, (batch_size, seq_len))

    # Run ONNX inference
    onnx_outputs = session.run(None, {"input_ids": test_input.numpy()})

    if model is not None:
        # Compare with PyTorch output.
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            # Run the comparison input on the MODEL's device. The ONNX
            # session always executes on CPU, but ``model(test_input)``
            # must be fed a tensor on the same device as the model —
            # otherwise a CUDA-resident model crashes with a device
            # mismatch. Then detach/move the result to CPU as float so
            # it can be compared against the (CPU, fp32) ONNX output
            # regardless of the model's native dtype.
            pt_input = test_input.to(device)
            # Handle tuple return (logits, kv_cache) or just logits
            pt_output = model(pt_input)
            if isinstance(pt_output, tuple):
                pt_output = pt_output[0]
            pt_output = pt_output.float().detach().cpu().numpy()

        # Compare
        import numpy as np

        return np.allclose(np.asarray(onnx_outputs[0]), pt_output, rtol=rtol, atol=atol)

    return True


def get_onnx_info(onnx_path: str | Path) -> dict:
    """
    Get information about an ONNX model.

    Args:
        onnx_path: Path to ONNX file

    Returns:
        Dictionary with model info (inputs, outputs, opset)

    Raises:
        ImportError: If onnx is not installed
    """
    try:
        import onnx
    except ImportError as e:
        raise ImportError("onnx is required: pip install onnx") from e

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))

    return {
        "opset_version": model.opset_import[0].version,
        "inputs": [
            {"name": inp.name, "shape": [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]}
            for inp in model.graph.input
        ],
        "outputs": [
            {"name": out.name, "shape": [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]}
            for out in model.graph.output
        ],
        "file_size_mb": onnx_path.stat().st_size / (1024 * 1024),
    }


class _CacheIOExporter(nn.Module):
    """Present the decoder as a pure function of ``(ids, past_k, past_v)``.

    The model's own cache is a mutable object (``KVCache`` writes into a
    pre-allocated buffer), which an exported graph cannot express — the graph is
    stateless. This adapter therefore rebuilds a :class:`FunctionalKVCache` per
    layer from TENSORS, runs the model, and returns the updated cache as tensors
    too. See ADR-016.

    Cache tensors are stacked over layers as ``[layers, batch, num_kv_heads,
    seq_len, head_dim]``, so one input pair describes the whole model.
    """

    def __init__(self, model: nn.Module, *, max_seq_len: int) -> None:
        super().__init__()
        self.model = model
        self.max_seq_len = max_seq_len

    def forward(self, ids: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor):
        from llm.core.kv_cache import FunctionalKVCache

        caches = []
        for layer in range(past_k.size(0)):
            cache = FunctionalKVCache(past_k.size(1), self.max_seq_len, past_k.size(2), past_k.size(4))
            cache._k_parts = [past_k[layer]]
            cache._v_parts = [past_v[layer]]
            cache._seq_len = past_k[layer].size(2)
            caches.append(cache)

        output = self.model(ids, use_cache=True, kv_caches=caches)
        logits = output[0] if isinstance(output, tuple) else output
        present_k = torch.stack([torch.cat(cache._k_parts, dim=2) for cache in caches])
        present_v = torch.stack([torch.cat(cache._v_parts, dim=2) for cache in caches])
        return logits, present_k, present_v


def _reject_rope_models_for_generation_export(model: nn.Module) -> None:
    """Raise a clear error if any attention module uses RoPE.

    See :func:`export_decoder_for_generation` for why the two-graph scheme
    cannot handle RoPE yet. The check walks the transformer blocks rather than
    trusting a single flag, because the attribute lives on each attention
    module.
    """
    blocks = getattr(model, "transformer_blocks", None) or ()
    for layer_index, block in enumerate(blocks):
        attention = getattr(block, "attention", None) or getattr(block, "self_attn", None)
        if attention is not None and getattr(attention, "use_rope", False):
            raise ValueError(
                "export_decoder_for_generation does not support RoPE models yet: "
                f"layer {layer_index} uses rotary position embedding. The traced decode graph "
                "is correct for the FIRST step and drifts on later ones (measured: step 1 "
                "matches eager to 2.4e-07, step 2 to 7.8e-03), because the absolute position "
                "is derived from the cache length inside the traced model. Export a non-RoPE "
                "model, or use export_to_onnx (single graph) for a forward pass."
            )


def export_decoder_for_generation(
    model: nn.Module,
    output_dir: str | Path,
    *,
    prompt_len: int = 8,
    max_seq_len: int | None = None,
    opset_version: int = 17,
    verbose: bool = False,
) -> dict[str, Path]:
    """Export a decoder as a PREFILL graph and a DECODE graph (ADR-016).

    A single graph cannot serve both phases: the model derives ``start_pos`` and
    ``has_past`` from ``kv_caches[0].seq_len``, which are PYTHON values, so a
    tracer freezes them into constants with the traced shape baked in. Measured:
    a graph traced on a prefill never correctly decodes (steps diverge by ~0.34),
    while a graph traced at its own phase's shape matches eager to <5e-7. Two
    graphs, each traced at its characteristic shape, is the standard ONNX/TensorRT
    decoder pattern and is what this function writes.

    Both graphs take the same three inputs — ``ids``, ``past_k``, ``past_v`` — and
    return ``logits``, ``present_k``, ``present_v``, with the cache stacked over
    layers. Callers chain them: run prefill once with an empty cache, then pass
    each step's ``present_*`` back as the next ``past_*``.

    Args:
        model: The decoder, in eval mode.
        output_dir: Directory to write ``prefill.onnx`` and ``decode.onnx`` into.
        prompt_len: Sequence length the prefill graph is traced at. Prefill uses
            a causal mask internally, so any length works — this only fixes the
            traced dummy.
        max_seq_len: Cache capacity. Defaults to the model's context window.
        opset_version: ONNX opset (17 matches the rest of this module).
        verbose: Forwarded to the exporter.

    Returns:
        ``{"prefill": Path, "decode": Path}``.

    Raises:
        ValueError: If the model exposes nothing that looks like a context window
            and ``max_seq_len`` is not given.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    # RoPE models cannot be exported by this two-graph scheme yet. The decode
    # graph's `has_past` is derived from `kv_cache.seq_len > seq_len`, a PYTHON
    # comparison that the tracer freezes into a constant; with RoPE the
    # positions are then built from that stale constant and the attention
    # reshape fails with `shape '[1, 1, 32]' is invalid for input of size 512`
    # — an error from inside `core/attn/mha.py` that names neither the export
    # nor the cause. Measured: every `use_rope=True` configuration fails
    # (1-2 layers, 2-4 heads), while every `use_rope=False` one succeeds.
    # Refuse up front with the actual limitation instead of surfacing that
    # downstream traceback; real DeepSeek-style/decoupled-RoPE export is a
    # separate slice.
    _reject_rope_models_for_generation_export(model)

    if max_seq_len is None:
        max_seq_len = _model_context_window(model)
        if max_seq_len is None:
            raise ValueError(
                "cannot infer the model's context window; pass max_seq_len explicitly. "
                "The cache capacity has to be known at export time."
            )

    # Resolve the cache geometry from the model rather than guessing: a wrong
    # num_kv_heads silently produces a graph whose attention shapes disagree.
    num_kv_heads, head_dim = _cache_geometry(model)
    layers = len(getattr(model, "transformer_blocks", []))

    adapter = _CacheIOExporter(model, max_seq_len=max_seq_len).eval()

    dynamic_axes = {
        "ids": {0: "batch", 1: "seq_len"},
        "past_k": {1: "batch", 3: "past_len"},
        "past_v": {1: "batch", 3: "past_len"},
        "logits": {0: "batch", 1: "seq_len"},
        "present_k": {1: "batch", 3: "total_len"},
        "present_v": {1: "batch", 3: "total_len"},
    }

    written: dict[str, Path] = {}
    # (past_len, new_len) per graph: prefill starts empty and takes the prompt;
    # decode has a past and appends exactly one token.
    phases = {"prefill": (0, prompt_len), "decode": (prompt_len, 1)}
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*TorchScript.*ONNX.*")
        for name, (past_len, new_len) in phases.items():
            path = directory / f"{name}.onnx"
            ids = torch.zeros(1, new_len, dtype=torch.long)
            empty_k = torch.zeros(layers, 1, num_kv_heads, past_len, head_dim)
            empty_v = torch.zeros(layers, 1, num_kv_heads, past_len, head_dim)
            torch.onnx.export(
                adapter,
                (ids, empty_k, empty_v),
                str(path),
                input_names=["ids", "past_k", "past_v"],
                output_names=["logits", "present_k", "present_v"],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=verbose,
                dynamo=False,
            )
            written[name] = path

    if verbose:
        logger.info("Exported prefill/decode graphs to %s", directory)
    return written


def _model_context_window(model: nn.Module) -> int | None:
    """Best-effort read of the model's ``max_seq_len``."""
    for holder in (model, getattr(model, "model", None)):
        if holder is None:
            continue
        for attr in ("max_seq_len",):
            value = getattr(holder, attr, None)
            if isinstance(value, int):
                return value
        config = getattr(holder, "config", None)
        value = getattr(config, "max_seq_len", None) if config is not None else None
        if isinstance(value, int):
            return value
    return None


def _cache_geometry(model: nn.Module) -> tuple[int, int]:
    """Resolve ``(num_kv_heads, head_dim)`` from the model's attention module."""
    blocks = getattr(model, "transformer_blocks", None)
    if not blocks:
        raise ValueError("export_decoder_for_generation needs a decoder exposing transformer_blocks")
    attention = getattr(blocks[0], "attention", None) or getattr(blocks[0], "self_attn", None)
    if attention is None:
        raise ValueError("the first transformer block exposes no attention module to size the KV cache")

    num_kv_heads = getattr(attention, "num_kv_heads", None)
    embedding = getattr(model, "embedding_layer", None)
    token_embeddings = getattr(embedding, "token_embeddings", None)
    hidden_size = getattr(token_embeddings, "embedding_dim", None)
    num_heads = getattr(attention, "num_heads", None)
    if num_kv_heads is None or hidden_size is None or num_heads is None:
        raise ValueError(
            "cannot resolve the KV cache geometry: attention needs num_kv_heads/num_heads and the "
            f"embedding needs embedding_dim (got num_kv_heads={num_kv_heads}, "
            f"embedding_dim={hidden_size}, num_heads={num_heads})"
        )
    return int(num_kv_heads), int(hidden_size) // int(num_heads)


def verify_decoder_generation_graphs(
    graph_dir: str | Path,
    model: nn.Module,
    *,
    prompt_len: int = 8,
    new_tokens: int = 3,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """Check an exported prefill+decode pair by decoding with it end to end.

    :func:`verify_onnx` cannot check these graphs: it feeds ``input_ids`` only,
    while the cache-aware pair takes ``ids`` / ``past_k`` / ``past_v`` and is
    meaningless without them (measured: ``ValueError: Required inputs (['ids',
    'past_k', 'past_v']) are missing from input feed (['input_ids'])``). Without
    this function a caller could export the pair but not verify it, which is the
    shape of gap that let the GGUF offsets ship broken.

    The check is the real workload: run the prefill graph once with an EMPTY
    cache, feed each decode step's ``present_*`` back as the next ``past_*``, and
    compare every step against the model decoding with its own in-place cache.
    Comparing only prefill would miss the decode graph entirely — and prefill is
    the phase that already worked before the pair existed.

    Args:
        graph_dir: Directory written by :func:`export_decoder_for_generation`.
        model: The model the graphs were exported from (eval mode not required).
        prompt_len: Prompt length to feed the prefill graph.
        new_tokens: Decode steps to run after the prefill.
        rtol: Relative tolerance for the per-step comparison.
        atol: Absolute tolerance for the per-step comparison.

    Returns:
        True when every step matches within tolerance.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.
        FileNotFoundError: If either graph is missing from ``graph_dir``.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:  # pragma: no cover - onnxruntime is a test dep
        raise ImportError("onnxruntime is required: pip install onnxruntime") from e

    directory = Path(graph_dir)
    prefill_path = directory / "prefill.onnx"
    decode_path = directory / "decode.onnx"
    for path in (prefill_path, decode_path):
        if not path.exists():
            raise FileNotFoundError(f"expected {path} — export the pair with export_decoder_for_generation first")

    prefill = ort.InferenceSession(str(prefill_path), providers=["CPUExecutionProvider"])
    decode = ort.InferenceSession(str(decode_path), providers=["CPUExecutionProvider"])

    num_kv_heads, head_dim = _cache_geometry(model)
    layers = len(getattr(model, "transformer_blocks", []))
    max_seq_len = _model_context_window(model) or (prompt_len + new_tokens)

    # Reference: the model decoding with its own in-place cache.
    from llm.core.kv_cache import KVCache

    torch.manual_seed(0)
    vocab = model_vocab_size(model) or 100
    prompt = torch.randint(0, vocab, (1, prompt_len))
    steps = [torch.randint(0, vocab, (1, 1)) for _ in range(new_tokens)]

    caches = [KVCache(1, max_seq_len, num_kv_heads, head_dim) for _ in range(layers)]
    expected: list[Any] = []
    with torch.no_grad():
        out = model(prompt, use_cache=True, kv_caches=caches)
        out = out[0] if isinstance(out, tuple) else out
        expected.append(out[:, -1:].float().numpy())
        for token in steps:
            out = model(token, use_cache=True, kv_caches=caches)
            out = out[0] if isinstance(out, tuple) else out
            expected.append(out.float().numpy())

    # ``InferenceSession.run`` is typed as a union (sparse / dict / list outputs
    # are all possible). These graphs return three dense float arrays, so the
    # results are narrowed once here instead of casting at every use site.
    def _run_dense(session: Any, ids: Any, past_k: Any, past_v: Any) -> tuple[Any, Any, Any]:
        logits, present_k, present_v = session.run(None, {"ids": ids, "past_k": past_k, "past_v": past_v})
        return np.asarray(logits), np.asarray(present_k), np.asarray(present_v)

    past_k = np.zeros((layers, 1, num_kv_heads, 0, head_dim), dtype=np.float32)
    past_v = np.zeros_like(past_k)
    logits, past_k, past_v = _run_dense(prefill, prompt.numpy(), past_k, past_v)
    got: list[Any] = [logits[:, -1:]]
    for token in steps:
        logits, past_k, past_v = _run_dense(decode, token.numpy(), past_k, past_v)
        got.append(logits)

    if past_k.shape[3] != prompt_len + new_tokens:
        return False

    return all(np.allclose(want, actual, rtol=rtol, atol=atol) for want, actual in zip(expected, got, strict=True))
