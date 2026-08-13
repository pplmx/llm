from __future__ import annotations

import logging
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from llm.export._wrapper import ExportCacheWrapper, dummy_token_ids

logger = logging.getLogger(__name__)


def _normalize_layer_norm_dtypes(onnx_path: str | Path) -> Path:
    """Fix TorchScript ONNX LayerNormalization type binding for half-precision
    models (RIL ISS-067).

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
    logger.info("Rewrote %d LayerNormalization node(s) to fix fp16 type binding (RIL ISS-067).", fixed)
    return onnx_path


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int] = (1, 32),
    opset_version: int = 17,
    dynamic_axes: dict | None = None,
    verbose: bool = False,
) -> Path:
    """
    Export a model to ONNX format.

    Args:
        model: The model to export (e.g., DecoderModel)
        output_path: Path to save the ONNX file
        input_shape: (batch_size, seq_len) for dummy input
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
    # don't crash the embedding with out-of-range ids (RIL ISS-058).
    dummy_input = dummy_token_ids(model, input_shape, device=device)

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
    # types, producing an artifact onnxruntime cannot load (RIL ISS-067).
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
    input_shape: tuple[int, int] = (1, 32),
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
    # so small-vocab models don't crash the embedding (RIL ISS-058).
    if model is not None:
        test_input = dummy_token_ids(model, input_shape)
    else:
        batch_size, seq_len = input_shape
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
