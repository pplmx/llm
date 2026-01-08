"""
ONNX Export Utilities.

Provides functions to export DecoderModel to ONNX format for deployment.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


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

    Example:
        >>> model = DecoderModel(vocab_size=1000, hidden_size=64, num_layers=2, num_heads=4)
        >>> export_to_onnx(model, "model.onnx", input_shape=(1, 32))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    # Create dummy input
    batch_size, seq_len = input_shape
    dummy_input = torch.randint(0, 100, (batch_size, seq_len), device=device)

    # Default dynamic axes for variable batch and sequence length
    if dynamic_axes is None:
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        }

    # Export using legacy API (dynamo=False avoids onnxscript dependency)
    with torch.no_grad():
        torch.onnx.export(
            model,
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

    if verbose:
        print(f"Exported model to {output_path}")

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

    # Create test input
    batch_size, seq_len = input_shape
    test_input = torch.randint(0, 100, (batch_size, seq_len))

    # Run ONNX inference
    onnx_outputs = session.run(None, {"input_ids": test_input.numpy()})

    if model is not None:
        # Compare with PyTorch output
        model.eval()
        with torch.no_grad():
            # Handle tuple return (logits, kv_cache) or just logits
            pt_output = model(test_input)
            if isinstance(pt_output, tuple):
                pt_output = pt_output[0]
            pt_output = pt_output.numpy()

        # Compare
        import numpy as np

        return np.allclose(onnx_outputs[0], pt_output, rtol=rtol, atol=atol)

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
