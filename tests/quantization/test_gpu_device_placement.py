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
from tests.support.devices import DEFAULT_DEVICE, all_gpu_devices, cuda_usable


@pytest.mark.skipif(not cuda_usable(), reason="requires CUDA with >= 512 MiB free VRAM")
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

    # Use the fattest usable GPU (repo ``DEFAULT_DEVICE`` convention) rather
    # than literal ``"cuda"`` (cuda:0), which can be an occupied / low-VRAM
    # device on a shared host and makes the test flaky (RIL ISS-046).
    device = str(DEFAULT_DEVICE) if all_gpu_devices() else "cpu"
    quantized = quantize(model, iter(calib), config, device=device)
    quantized.eval()

    devices = {buffer.device.type for buffer in quantized.buffers()}
    assert devices == {"cuda"}, f"{method}: mixed-device buffers {devices}"

    ids = torch.randint(0, 1024, (2, 16), device=device)
    with torch.no_grad():
        out = quantized(ids)
    assert torch.isfinite(out).all().item(), f"{method}: forward produced non-finite logits"
