import pytest
import torch
import torch.nn as nn

from llm.core.mlp import MLP
from llm.core.moe.moe import MoE


class TestMoE:
    @pytest.fixture(
        params=[
            (64, 4, 2),  # hidden_size, num_experts, top_k
            (128, 8, 1),
            (32, 2, 2),
        ]
    )
    def moe_params(self, request):
        return request.param

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def dtype(self):
        return torch.float32

    def test_moe_init(self, moe_params, device, dtype):
        hidden_size, num_experts, top_k = moe_params
        moe_layer = MoE(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k, device=device, dtype=dtype)
        assert isinstance(moe_layer, MoE)
        assert moe_layer.hidden_size == hidden_size
        assert moe_layer.num_experts == num_experts
        assert moe_layer.top_k == top_k
        assert isinstance(moe_layer.gate, nn.Linear)
        assert moe_layer.gate.in_features == hidden_size
        assert moe_layer.gate.out_features == num_experts
        assert len(moe_layer.experts) == num_experts
        for expert in moe_layer.experts:
            assert isinstance(expert, MLP)
            assert expert.hidden_size == hidden_size

    @pytest.mark.parametrize(
        "top_k, num_experts",
        [
            (5, 4),  # top_k > num_experts
            (0, 4),  # top_k = 0
        ],
    )
    def test_moe_init_invalid_params(self, top_k, num_experts):
        with pytest.raises(ValueError):
            MoE(hidden_size=64, num_experts=num_experts, top_k=top_k)

    @pytest.mark.parametrize(
        "batch_size, seq_len",
        [
            (2, 10),
            (1, 1),
            (4, 20),
        ],
    )
    def test_moe_forward(self, moe_params, batch_size, seq_len, device, dtype):
        hidden_size, num_experts, top_k = moe_params
        moe_layer = MoE(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k, device=device, dtype=dtype)
        moe_layer.eval()  # For consistent behavior

        input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
        output_tensor = moe_layer(input_tensor)

        assert output_tensor.shape == input_tensor.shape
        # Normalize device comparison: torch.device('cuda') and torch.device('cuda:0')
        # both should be considered equivalent for these tests. Compare by device.type
        expected_device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]
        assert output_tensor.device.type == expected_device_type
        assert output_tensor.dtype == dtype

        # Basic check: output should not be all zeros unless input is all zeros
        if not torch.allclose(input_tensor, torch.zeros_like(input_tensor)):
            assert not torch.allclose(output_tensor, torch.zeros_like(output_tensor))

        # Check for NaN or Inf values in the output
        assert not torch.isnan(output_tensor).any(), "Output contains NaN values"
        assert not torch.isinf(output_tensor).any(), "Output contains Inf values"
