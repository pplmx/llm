"""GPU regression: quantized layers must stay on the model's device.

All three PTQ methods (GPTQ / AWQ / SmoothQuant) construct replacement
modules from intermediate tensors and insert them into a model that was
moved to CUDA beforehand. If the new module is not explicitly moved, its
``scales`` / ``input_scales`` buffers end up on CPU while ``weight_packed``
is on CUDA — a mixed-device module whose forward crashes on GPU.
"""

import pytest
import torch

from llm.models.decoder import DecoderModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("method", "quantize_fn", "config"),
    [
        pytest.param(
            "gptq",
            "quantize_model_gptq",
            None,
            id="gptq",
        ),
        pytest.param(
            "awq",
            "quantize_model_awq",
            None,
            id="awq",
        ),
        pytest.param(
            "smoothquant",
            "quantize_model_smoothquant",
            None,
            id="smoothquant",
        ),
    ],
)
def test_quantized_buffers_stay_on_cuda(method, quantize_fn, config):
    """Quantizing on CUDA leaves every buffer on CUDA and forward finite."""
    import llm.quantization as q

    quantize = getattr(q, quantize_fn)
    model = DecoderModel(vocab_size=1024, hidden_size=64, num_layers=2, num_heads=4, max_seq_len=128)
    calib = [torch.randint(0, 1024, (2, 16)) for _ in range(4)]

    quantized = quantize(model, iter(calib), config, device="cuda")
    quantized.eval()

    devices = {buffer.device.type for buffer in quantized.buffers()}
    assert devices == {"cuda"}, f"{method}: mixed-device buffers {devices}"

    ids = torch.randint(0, 1024, (2, 16), device="cuda")
    with torch.no_grad():
        out = quantized(ids)
    assert torch.isfinite(out).all().item(), f"{method}: forward produced non-finite logits"
