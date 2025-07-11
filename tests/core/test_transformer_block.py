import pytest
import torch
import torch.nn as nn

from llm.core.attn.mha import MultiHeadAttention
from llm.core.mlp import MLP
from llm.core.moe.moe import MoE
from llm.core.transformer_block import TransformerBlock

# Test Constants
HIDDEN_SIZE = 64
NUM_HEADS = 8
MLP_INTERMEDIATE_SIZE = HIDDEN_SIZE * 4
ATTN_DROPOUT_P = 0.0  # Default to 0 for deterministic tests unless specified
MLP_DROPOUT_P = 0.0  # Default to 0 for deterministic tests unless specified
NORM_EPS = 1e-5
BATCH_SIZE = 2
SEQ_LEN = 10

# Available devices for testing
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

DTYPES = [torch.float32]  # Keep it simple for most tests, can expand if needed


@pytest.fixture
def block_kwargs(request):
    """Fixture to provide default and overridable kwargs for TransformerBlock."""
    default = {
        "hidden_size": HIDDEN_SIZE,
        "num_heads": NUM_HEADS,
        "mlp_intermediate_size": MLP_INTERMEDIATE_SIZE,
        "attn_dropout_p": ATTN_DROPOUT_P,
        "mlp_dropout_p": MLP_DROPOUT_P,
        "mlp_activation": "gelu",
        "norm_eps": NORM_EPS,
        "norm_first": True,
        "is_causal": False,
        "qkv_bias": True,
        "mlp_bias": True,
        "use_moe": False,  # Default to not use MoE
        "num_experts": 4,  # Default for MoE if enabled
        "top_k": 2,  # Default for MoE if enabled
        "device": "cpu",  # Default device
        "dtype": torch.float32,  # Default dtype
    }
    if hasattr(request, "param"):
        default.update(request.param)
    return default


@pytest.fixture
def transformer_block(block_kwargs):
    """Creates a TransformerBlock instance based on block_kwargs."""
    block = TransformerBlock(**block_kwargs)
    block.eval()  # Default to eval mode
    return block


@pytest.fixture
def input_tensor(block_kwargs):
    """Creates a dummy input tensor based on block_kwargs."""
    return torch.randn(
        BATCH_SIZE, SEQ_LEN, block_kwargs["hidden_size"], device=block_kwargs["device"], dtype=block_kwargs["dtype"]
    )


class TestTransformerBlockInitialization:
    def test_submodule_types(self, transformer_block):
        assert isinstance(transformer_block.self_attn, MultiHeadAttention)
        # Check type of mlp based on whether MoE is used
        if (
            transformer_block.mlp.__class__.__name__ == "MoE"
        ):  # Check by name to avoid circular import issues if MoE is not directly imported
            assert isinstance(transformer_block.mlp, MoE)
        else:
            assert isinstance(transformer_block.mlp, MLP)
        assert isinstance(transformer_block.norm1, nn.LayerNorm)
        assert isinstance(transformer_block.norm2, nn.LayerNorm)

    @pytest.mark.parametrize(
        "block_kwargs",
        [
            {"use_moe": False},
            {"use_moe": True, "num_experts": 4, "top_k": 2},
        ],
        indirect=True,
    )
    def test_mlp_or_moe_instance(self, transformer_block, block_kwargs):
        if block_kwargs["use_moe"]:
            assert isinstance(transformer_block.mlp, MoE)
            assert transformer_block.mlp.num_experts == block_kwargs["num_experts"]
            assert transformer_block.mlp.top_k == block_kwargs["top_k"]
        else:
            assert isinstance(transformer_block.mlp, MLP)

    def test_mha_mlp_internal_norm_residual_disabled(self, transformer_block):
        assert transformer_block.self_attn.include_norm_residual is False, (
            "MHA submodule should have include_norm_residual=False"
        )
        # This assertion needs to be conditional for MLP vs MoE
        if isinstance(transformer_block.mlp, MLP):
            assert transformer_block.mlp.include_norm_residual is False, (
                "MLP submodule should have include_norm_residual=False"
            )
        # MoE's experts are MLPs with include_norm_residual=False, so this is implicitly covered.

    def test_parameter_propagation(self, transformer_block, block_kwargs):
        assert transformer_block.self_attn.hidden_size == block_kwargs["hidden_size"]
        assert transformer_block.self_attn.num_heads == block_kwargs["num_heads"]
        assert transformer_block.self_attn.p == block_kwargs["attn_dropout_p"]
        assert transformer_block.self_attn.is_causal == block_kwargs["is_causal"]

        if isinstance(transformer_block.mlp, MLP):
            assert transformer_block.mlp.hidden_size == block_kwargs["hidden_size"]
            # MLP's intermediate_size calculation: if None, it's 4*hidden_size
            expected_mlp_intermediate = block_kwargs["mlp_intermediate_size"] or 4 * block_kwargs["hidden_size"]
            assert transformer_block.mlp.intermediate_size == expected_mlp_intermediate
            assert transformer_block.mlp.dropout.p == block_kwargs["mlp_dropout_p"]  # MLP stores dropout layer
        elif isinstance(transformer_block.mlp, MoE):
            assert transformer_block.mlp.hidden_size == block_kwargs["hidden_size"]
            assert transformer_block.mlp.num_experts == block_kwargs["num_experts"]
            assert transformer_block.mlp.top_k == block_kwargs["top_k"]
            # Check parameters of experts within MoE
            for expert in transformer_block.mlp.experts:
                assert expert.hidden_size == block_kwargs["hidden_size"]
                expected_mlp_intermediate = block_kwargs["mlp_intermediate_size"] or 4 * block_kwargs["hidden_size"]
                assert expert.intermediate_size == expected_mlp_intermediate
                assert expert.dropout.p == block_kwargs["mlp_dropout_p"]

        assert transformer_block.norm1.eps == block_kwargs["norm_eps"]
        assert transformer_block.norm2.eps == block_kwargs["norm_eps"]
        assert transformer_block.norm_first == block_kwargs["norm_first"]


class TestTransformerBlockForwardPass:
    @pytest.mark.parametrize(
        "block_kwargs",
        [
            {"norm_first": True, "use_moe": False},
            {"norm_first": False, "use_moe": False},
            {"norm_first": True, "use_moe": True, "num_experts": 4, "top_k": 2},
            {"norm_first": False, "use_moe": True, "num_experts": 4, "top_k": 2},
        ],
        indirect=True,
    )
    def test_forward_pass_shape(self, transformer_block, input_tensor):
        output = transformer_block(input_tensor)
        assert output.shape == input_tensor.shape, (
            f"Output shape {output.shape} does not match input shape {input_tensor.shape}"
        )

    def test_causality_propagation(self, block_kwargs, input_tensor):
        # Test with block's default is_causal = True
        block_kwargs["is_causal"] = True
        block = TransformerBlock(**block_kwargs)
        block.eval()

        # Test with causal mask (default behavior for is_causal=True)
        output_causal = block(input_tensor)
        assert output_causal.shape == input_tensor.shape

        # Test with is_causal=False override
        output_non_causal_override = block(input_tensor, is_causal=False)
        assert output_non_causal_override.shape == input_tensor.shape

        # Test with is_causal=True override (explicitly same as default)
        output_causal_override = block(input_tensor, is_causal=True)
        assert output_causal_override.shape == input_tensor.shape

    def test_attention_mask_propagation(self, transformer_block, input_tensor):
        dummy_mask = torch.ones(
            BATCH_SIZE,
            1,
            SEQ_LEN,
            SEQ_LEN,  # MHA expects [B, N, S, S] or broadcastable
            device=input_tensor.device,
            dtype=torch.bool,
        )

        output_masked = transformer_block(input_tensor, attn_mask=dummy_mask)
        assert output_masked.shape == input_tensor.shape

    @pytest.mark.parametrize(
        "block_kwargs",
        [
            {"attn_dropout_p": 0.5, "mlp_dropout_p": 0.0, "norm_first": True, "use_moe": False},
            {"attn_dropout_p": 0.0, "mlp_dropout_p": 0.5, "norm_first": True, "use_moe": False},
            {"attn_dropout_p": 0.5, "mlp_dropout_p": 0.5, "norm_first": True, "use_moe": False},
            {"attn_dropout_p": 0.5, "mlp_dropout_p": 0.5, "norm_first": False, "use_moe": False},
            {"attn_dropout_p": 0.0, "mlp_dropout_p": 0.0, "norm_first": True, "use_moe": False},  # Control: no dropout
            {
                "attn_dropout_p": 0.5,
                "mlp_dropout_p": 0.0,
                "norm_first": True,
                "use_moe": True,
                "num_experts": 4,
                "top_k": 2,
            },
            {
                "attn_dropout_p": 0.0,
                "mlp_dropout_p": 0.5,
                "norm_first": True,
                "use_moe": True,
                "num_experts": 4,
                "top_k": 2,
            },
        ],
        indirect=True,
    )
    def test_transformer_block_dropout_train_eval_modes(self, transformer_block, input_tensor, block_kwargs):
        """
        Tests TransformerBlock dropout behavior in train vs eval modes.
        Dropout is applied in MHA (attention and output projection) and MLP/MoE.
        """
        attn_p = block_kwargs["attn_dropout_p"]
        mlp_p = block_kwargs["mlp_dropout_p"]
        norm_f = block_kwargs["norm_first"]
        use_moe = block_kwargs["use_moe"]

        # Eval mode: Dropout should be disabled, outputs should be identical.
        transformer_block.eval()  # Fixture already sets to eval, but explicit here
        with torch.no_grad():
            output_eval_1 = transformer_block(input_tensor)
            output_eval_2 = transformer_block(input_tensor)
        assert torch.allclose(output_eval_1, output_eval_2, atol=1e-7), (
            f"Outputs in eval mode should be identical (attn_p={attn_p}, mlp_p={mlp_p}, norm_first={norm_f}, use_moe={use_moe})"
        )

        # Train mode: Dropout should be active if p > 0.
        transformer_block.train()
        with torch.no_grad():
            # Multiple calls to dropout layers within MHA and MLP/MoE should use different masks.
            output_train_1 = transformer_block(input_tensor)
            output_train_2 = transformer_block(input_tensor)

        if attn_p > 0 or mlp_p > 0:
            assert not torch.allclose(output_train_1, output_train_2, atol=1e-6), (
                f"Outputs in train mode should differ due to dropout (attn_p={attn_p}, mlp_p={mlp_p}, norm_first={norm_f}, use_moe={use_moe})"
            )
        else:
            # If both dropout probabilities are 0, outputs in train mode should also be identical.
            assert torch.allclose(output_train_1, output_train_2, atol=1e-7), (
                f"Outputs in train mode should be identical if all dropout_p are 0 (attn_p={attn_p}, mlp_p={mlp_p}, norm_first={norm_f}, use_moe={use_moe})"
            )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype_str", ["torch.float32"])  # Could add 'torch.float64'
class TestDeviceAndDtypePropagation:
    def test_block_device_dtype(self, device, dtype_str, block_kwargs):
        dtype = getattr(torch, dtype_str.replace("torch.", ""))
        if device == "cuda" and dtype == torch.float64:
            if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7):
                pytest.skip("CUDA float64 support not adequate or device not capable.")

        block_kwargs.update({"device": device, "dtype": dtype})
        block = TransformerBlock(**block_kwargs)
        block.to(device, dtype)  # Ensure module is moved

        # Check parameters of norm layers
        assert block.norm1.weight.device.type == device.split(":")[0]
        assert block.norm1.weight.dtype == dtype
        assert block.norm2.weight.device.type == device.split(":")[0]
        assert block.norm2.weight.dtype == dtype

        # Check parameters of MHA (qkv_proj, out_proj)
        assert block.self_attn.qkv_proj.weight.device.type == device.split(":")[0]
        assert block.self_attn.qkv_proj.weight.dtype == dtype
        assert block.self_attn.qkv_proj.weight.dtype == dtype
        if block_kwargs["qkv_bias"]:
            assert block.self_attn.qkv_proj.bias.dtype == dtype

        # Check parameters of MLP/MoE (fc1, fc2 or experts)
        if isinstance(block.mlp, MLP):
            assert block.mlp.fc1.weight.device.type == device.split(":")[0]
            assert block.mlp.fc1.weight.dtype == dtype
            assert block.mlp.fc2.weight.device.type == device.split(":")[0]
            assert block.mlp.fc2.weight.dtype == dtype
            if block_kwargs["mlp_bias"]:
                assert block.mlp.fc1.bias.dtype == dtype
        elif isinstance(block.mlp, MoE):
            assert block.mlp.gate.weight.device.type == device.split(":")[0]
            assert block.mlp.gate.weight.dtype == dtype
            for expert in block.mlp.experts:
                assert expert.fc1.weight.device.type == device.split(":")[0]
                assert expert.fc1.weight.dtype == dtype
                assert expert.fc2.weight.device.type == device.split(":")[0]
                assert expert.fc2.weight.dtype == dtype
                if block_kwargs["mlp_bias"]:
                    assert expert.fc1.bias.dtype == dtype

        # Check output tensor device and dtype
        current_input_tensor = torch.randn(BATCH_SIZE, SEQ_LEN, block_kwargs["hidden_size"], device=device, dtype=dtype)
        output = block(current_input_tensor)
        assert output.device.type == device.split(":")[0]
        assert output.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize("norm_first_val", [True, False])
@pytest.mark.parametrize("qkv_bias_val", [True, False])
@pytest.mark.parametrize("mlp_bias_val", [True, False])
@pytest.mark.parametrize(
    "use_moe_val, num_experts_val, top_k_val",
    [
        (False, 0, 0),
        (True, 4, 2),
    ],
)
def test_transformer_block_gradient_computation(
    norm_first_val, qkv_bias_val, mlp_bias_val, use_moe_val, num_experts_val, top_k_val, block_kwargs
):
    """Tests if gradients are computed correctly for all trainable parameters."""
    torch.manual_seed(42)

    current_block_kwargs = block_kwargs.copy()
    current_block_kwargs.update(
        {
            "norm_first": norm_first_val,
            "qkv_bias": qkv_bias_val,
            "mlp_bias": mlp_bias_val,
            "attn_dropout_p": 0.0,  # Disable dropout for deterministic gradient check
            "mlp_dropout_p": 0.0,
            "use_moe": use_moe_val,
            "num_experts": num_experts_val,
            "top_k": top_k_val,
            # Use a smaller hidden_size for faster test if desired, but ensure it's divisible by num_heads
            # For this test, we'll use the fixture's default HIDDEN_SIZE
        }
    )

    # Create a new block and input tensor with potentially different device/dtype from fixture
    # The input_tensor fixture already uses block_kwargs for device/dtype
    # We need to ensure the block and tensor are on the same device/dtype
    device = current_block_kwargs["device"]
    dtype = current_block_kwargs["dtype"]

    block = TransformerBlock(**current_block_kwargs)
    block.to(device, dtype)  # Ensure model is on correct device/dtype
    block.train()  # Ensure model is in training mode

    # Create input tensor on the same device/dtype as the block
    current_input_tensor = torch.randn(
        BATCH_SIZE, SEQ_LEN, current_block_kwargs["hidden_size"], device=device, dtype=dtype, requires_grad=True
    )

    # Forward pass
    output = block(current_input_tensor)

    # Compute a dummy loss and backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients for all parameters that should have them
    for name, param in block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN values"
            assert not torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf values"
            # It's possible for some gradients to be zero, especially with simple inputs or architectures.
            # A stricter check might be `(param.grad != 0).any()` if non-zero grads are guaranteed.
            # For now, just ensure they exist and are finite.
        else:
            assert param.grad is None or (param.grad == 0).all(), f"Unexpected gradient for non-trainable param {name}"

    # Check input tensor gradient
    assert current_input_tensor.grad is not None, "Input tensor gradient is None"
    assert not torch.isnan(current_input_tensor.grad).any(), "Input tensor gradient contains NaN values"
