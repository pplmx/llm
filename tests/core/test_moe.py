import pytest
import torch
import torch.nn as nn

from llm.core.mlp import MLP
from llm.core.moe.moe import MoE


@pytest.mark.slow
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
        ("top_k", "num_experts"),
        [
            (5, 4),  # top_k > num_experts
            (0, 4),  # top_k = 0
        ],
    )
    def test_moe_init_invalid_params(self, top_k, num_experts):
        with pytest.raises(ValueError, match=r"top_k|num_experts"):
            MoE(hidden_size=64, num_experts=num_experts, top_k=top_k)

    @pytest.mark.parametrize(
        ("batch_size", "seq_len"),
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
        expected_device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
        assert output_tensor.device.type == expected_device_type
        assert output_tensor.dtype == dtype

        # Basic check: output should not be all zeros unless input is all zeros
        if not torch.allclose(input_tensor, torch.zeros_like(input_tensor)):
            assert not torch.allclose(output_tensor, torch.zeros_like(output_tensor))

        # Check for NaN or Inf values in the output
        assert not torch.isnan(output_tensor).any(), "Output contains NaN values"
        assert not torch.isinf(output_tensor).any(), "Output contains Inf values"


class TestExpertChoiceRouting:
    """Expert Choice Routing research slice (ROADMAP 15.4)."""

    def test_each_expert_selects_exactly_k_tokens(self):
        from llm.core.moe.expert_choice import expert_choice_assignment

        torch.manual_seed(0)
        gate = torch.randn(8, 4)
        tokens, _ = expert_choice_assignment(gate, k=3)
        assert tokens.shape == (4, 3)
        # Load balance by construction: every expert's selection has exactly k
        # (possibly overlapping) tokens — no expert is starved or overloaded.
        for e in range(4):
            assert tokens[e].unique().numel() == 3
        # A token may legitimately be chosen by >1 expert (or by none).
        flat = tokens.reshape(-1)
        assert flat.max().item() < 8
        assert flat.min().item() >= 0

    def test_assignment_matches_topk_reference(self):
        from llm.core.moe.expert_choice import expert_choice_assignment

        torch.manual_seed(0)
        gate = torch.randn(8, 4)
        tokens, scores = expert_choice_assignment(gate, k=3)
        ref_scores, ref_tokens = torch.topk(gate.transpose(0, 1), 3, dim=1)
        assert torch.equal(tokens, ref_tokens)
        assert torch.allclose(scores, ref_scores, atol=1e-6)

    def test_weights_normalize_per_expert(self):
        from llm.core.moe.expert_choice import expert_choice_assignment, expert_choice_weights

        torch.manual_seed(1)
        gate = torch.randn(8, 4)
        _, scores = expert_choice_assignment(gate, k=3)
        weights = expert_choice_weights(scores)
        assert weights.shape == (4, 3)
        assert torch.allclose(weights.sum(-1), torch.ones(4), atol=1e-5)
        assert torch.isfinite(weights).all()

    def test_combine_equals_weighted_sum_reference(self):
        from llm.core.moe.expert_choice import expert_choice_assignment, expert_choice_output, expert_choice_weights

        torch.manual_seed(2)
        n_tok, n_exp, d, k = 8, 4, 6, 3
        x = torch.randn(n_tok, d)
        gate = torch.randn(n_tok, n_exp)

        def expert_fn(sel):
            return sel * 2.0

        out = expert_choice_output(x, gate, expert_fn, k)

        tokens, scores = expert_choice_assignment(gate, k)
        weights = expert_choice_weights(scores)
        ref = torch.zeros_like(x)
        for ex in range(n_exp):
            ref = ref.index_add(0, tokens[ex], (x[tokens[ex]] * 2.0) * weights[ex].unsqueeze(-1))
        assert torch.allclose(out, ref, atol=1e-5)

    def test_gradient_flow_finite(self):
        from llm.core.moe.expert_choice import expert_choice_output

        torch.manual_seed(3)
        n_tok, n_exp, d, k = 6, 3, 5, 2
        x = torch.randn(n_tok, d, requires_grad=True)
        gate = torch.randn(n_tok, n_exp, requires_grad=True)

        def expert_fn(sel):
            return sel * 1.5

        out = expert_choice_output(x, gate, expert_fn, k)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert x.grad is not None
        assert bool(torch.isfinite(x.grad).all())
        assert gate.grad is not None
        assert bool(torch.isfinite(gate.grad).all())

    def test_k_bounds_rejected(self):
        from llm.core.moe.expert_choice import expert_choice_assignment

        gate = torch.randn(4, 3)
        with pytest.raises(ValueError, match="k"):
            expert_choice_assignment(gate, k=0)
        with pytest.raises(ValueError, match="num_tokens"):
            expert_choice_assignment(gate, k=5)
