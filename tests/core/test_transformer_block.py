from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from llm.core.attn.mha import MultiHeadAttention
from llm.core.mlp import MLP
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
        assert isinstance(transformer_block.mlp, MLP)
        assert isinstance(transformer_block.norm1, nn.LayerNorm)
        assert isinstance(transformer_block.norm2, nn.LayerNorm)

    def test_mha_mlp_internal_norm_residual_disabled(self, transformer_block):
        assert transformer_block.self_attn.include_norm_residual is False, (
            "MHA submodule should have include_norm_residual=False"
        )
        assert transformer_block.mlp.include_norm_residual is False, (
            "MLP submodule should have include_norm_residual=False"
        )

    def test_parameter_propagation(self, transformer_block, block_kwargs):
        assert transformer_block.self_attn.hidden_size == block_kwargs["hidden_size"]
        assert transformer_block.self_attn.num_heads == block_kwargs["num_heads"]
        assert transformer_block.self_attn.p == block_kwargs["attn_dropout_p"]
        assert transformer_block.self_attn.is_causal == block_kwargs["is_causal"]

        assert transformer_block.mlp.hidden_size == block_kwargs["hidden_size"]
        # MLP's intermediate_size calculation: if None, it's 4*hidden_size
        expected_mlp_intermediate = block_kwargs["mlp_intermediate_size"] or 4 * block_kwargs["hidden_size"]
        assert transformer_block.mlp.intermediate_size == expected_mlp_intermediate
        assert transformer_block.mlp.dropout.p == block_kwargs["mlp_dropout_p"]  # MLP stores dropout layer

        assert transformer_block.norm1.eps == block_kwargs["norm_eps"]
        assert transformer_block.norm2.eps == block_kwargs["norm_eps"]
        assert transformer_block.norm_first == block_kwargs["norm_first"]


class TestTransformerBlockForwardPass:
    @pytest.mark.parametrize("block_kwargs", [{"norm_first": True}, {"norm_first": False}], indirect=True)
    def test_forward_pass_shape(self, transformer_block, input_tensor):
        output = transformer_block(input_tensor)
        assert output.shape == input_tensor.shape, (
            f"Output shape {output.shape} does not match input shape {input_tensor.shape}"
        )

    @patch("llm.core.transformer_block.MultiHeadAttention.forward")
    @patch("llm.core.transformer_block.MLP.forward")
    @patch("torch.nn.LayerNorm.forward")
    def test_pre_ln_path_logic(
        self, mock_layernorm_forward, mock_mlp_forward, mock_mha_forward, block_kwargs, input_tensor
    ):
        # Ensure norm_first is True for this test
        block_kwargs["norm_first"] = True
        block_kwargs["attn_dropout_p"] = 0.0  # Disable dropout for easier mocking if needed
        block_kwargs["mlp_dropout_p"] = 0.0

        # Make mocks return tensors of correct shape to allow arithmetic operations
        mock_mha_forward.return_value = torch.zeros_like(input_tensor)
        mock_mlp_forward.return_value = torch.zeros_like(input_tensor)
        # LayerNorm mock needs to handle multiple calls (norm1, norm2)
        # It should return a tensor of the same shape as its input.
        mock_layernorm_forward.side_effect = lambda x: x  # Identity, or torch.zeros_like(x)

        block = TransformerBlock(**block_kwargs)
        block.eval()

        # Use specific mocks for norm1 and norm2 if LayerNorm.forward is too general
        # For simplicity, we assume the single mock_layernorm_forward can be inspected for calls
        # Or, mock block.norm1.forward and block.norm2.forward specifically.

        with (
            patch.object(block.norm1, "forward", wraps=block.norm1.forward) as spy_norm1_fwd,
            patch.object(block.self_attn, "forward", wraps=block.self_attn.forward) as spy_attn_fwd,
            patch.object(block.norm2, "forward", wraps=block.norm2.forward) as spy_norm2_fwd,
            patch.object(block.mlp, "forward", wraps=block.mlp.forward) as spy_mlp_fwd,
        ):
            # Set return values for wrapped (actual) MHA and MLP if they don't already return zeros
            spy_attn_fwd.return_value = torch.zeros_like(input_tensor)
            spy_mlp_fwd.return_value = torch.zeros_like(input_tensor)

            _ = block(input_tensor)

            spy_norm1_fwd.assert_called_once()
            spy_attn_fwd.assert_called_once()
            spy_norm2_fwd.assert_called_once()
            spy_mlp_fwd.assert_called_once()

            # Check call order by asserting inputs if possible, or by inspecting call_args_list
            # This is tricky with generic mocks. A simpler check:
            # Ensure norm1 output is fed to MHA, MHA_res_sum output to norm2, norm2 output to MLP.
            # For now, call count is a basic check.
            # More detailed checks (e.g. with call_args) are possible but verbose.

    @patch("llm.core.transformer_block.MultiHeadAttention.forward")
    @patch("llm.core.transformer_block.MLP.forward")
    @patch("torch.nn.LayerNorm.forward")
    def test_post_ln_path_logic(
        self, mock_layernorm_forward, mock_mlp_forward, mock_mha_forward, block_kwargs, input_tensor
    ):
        block_kwargs["norm_first"] = False
        block_kwargs["attn_dropout_p"] = 0.0
        block_kwargs["mlp_dropout_p"] = 0.0

        mock_mha_forward.return_value = torch.zeros_like(input_tensor)
        mock_mlp_forward.return_value = torch.zeros_like(input_tensor)
        mock_layernorm_forward.side_effect = lambda x: x

        block = TransformerBlock(**block_kwargs)
        block.eval()

        with (
            patch.object(block.self_attn, "forward", wraps=block.self_attn.forward) as spy_attn_fwd,
            patch.object(block.norm1, "forward", wraps=block.norm1.forward) as spy_norm1_fwd,
            patch.object(block.mlp, "forward", wraps=block.mlp.forward) as spy_mlp_fwd,
            patch.object(block.norm2, "forward", wraps=block.norm2.forward) as spy_norm2_fwd,
        ):
            spy_attn_fwd.return_value = torch.zeros_like(input_tensor)
            spy_mlp_fwd.return_value = torch.zeros_like(input_tensor)

            _ = block(input_tensor)

            spy_attn_fwd.assert_called_once()
            spy_norm1_fwd.assert_called_once()
            spy_mlp_fwd.assert_called_once()
            spy_norm2_fwd.assert_called_once()
            # Order for Post-LN: attn -> norm1 -> mlp -> norm2

    def test_causality_propagation(self, block_kwargs, input_tensor):
        # Test with block's default is_causal = True
        block_kwargs["is_causal"] = True
        block = TransformerBlock(**block_kwargs)
        block.eval()

        with patch.object(block.self_attn, "forward", wraps=block.self_attn.forward) as spy_attn_fwd:
            spy_attn_fwd.return_value = torch.zeros_like(input_tensor)  # Ensure it returns something

            expected_attn_input = block.norm1(input_tensor) if block.norm_first else input_tensor

            # 1. Call without overriding is_causal in forward -> MHA should use block's default (True set by is_causal=True in block_kwargs)
            #    The is_causal=None in block() call means MHA's own self.is_causal is used by SDPA if MHA's is_causal=None.
            #    TransformerBlock passes its 'is_causal' (from init or forward override) to MHA's forward.
            #    MHA's forward then passes this to SDPA.
            #    So, for block.self_attn.is_causal = True (from block_kwargs["is_causal"] = True for the block's init)
            #    If block(..., is_causal=None) is called, MHA.forward gets is_causal=None from TransformerBlock.
            #    Then MHA.forward passes its own self.is_causal (True) to SDPA.
            #    The spy_attn_fwd will be called with is_causal=None from TransformerBlock's forward.
            _ = block(input_tensor, is_causal=None)  # is_causal=None in forward call
            spy_attn_fwd.assert_called_once()
            args, kwargs = spy_attn_fwd.call_args
            assert torch.equal(args[0], expected_attn_input)
            assert kwargs.get("attn_mask") is None
            assert kwargs.get("is_causal") is None  # MHA.forward gets None
            spy_attn_fwd.reset_mock()
            spy_attn_fwd.return_value = torch.zeros_like(input_tensor)

            # 2. Call with is_causal=False override
            _ = block(input_tensor, is_causal=False)
            spy_attn_fwd.assert_called_once()
            args, kwargs = spy_attn_fwd.call_args
            assert torch.equal(args[0], expected_attn_input)
            assert kwargs.get("attn_mask") is None
            assert kwargs.get("is_causal") is False
            spy_attn_fwd.reset_mock()
            spy_attn_fwd.return_value = torch.zeros_like(input_tensor)

            # 3. Call with is_causal=True override (explicitly same as default)
            _ = block(input_tensor, is_causal=True)
            spy_attn_fwd.assert_called_once()
            args, kwargs = spy_attn_fwd.call_args
            assert torch.equal(args[0], expected_attn_input)
            assert kwargs.get("attn_mask") is None
            assert kwargs.get("is_causal") is True
            spy_attn_fwd.reset_mock()
            spy_attn_fwd.return_value = torch.zeros_like(input_tensor)

    def test_attention_mask_propagation(self, transformer_block, input_tensor):
        dummy_mask = torch.ones(
            BATCH_SIZE,
            1,
            SEQ_LEN,
            SEQ_LEN,  # MHA expects [B, N, S, S] or broadcastable
            device=input_tensor.device,
            dtype=torch.bool,
        )

        with patch.object(
            transformer_block.self_attn, "forward", wraps=transformer_block.self_attn.forward
        ) as spy_attn_fwd:
            spy_attn_fwd.return_value = torch.zeros_like(input_tensor)
            _ = transformer_block(input_tensor, attn_mask=dummy_mask)

            # Check if the mask was passed to MHA's forward
            # Need to access call_args. Get the positional arguments.
            args, kwargs = spy_attn_fwd.call_args
            assert "attn_mask" in kwargs
            assert torch.equal(kwargs["attn_mask"], dummy_mask)
            # Or if passed positionally: assert torch.equal(args[1], dummy_mask) if arg order is known

    @pytest.mark.parametrize(
        "block_kwargs",
        [
            {"attn_dropout_p": 0.5, "mlp_dropout_p": 0.0, "norm_first": True},
            {"attn_dropout_p": 0.0, "mlp_dropout_p": 0.5, "norm_first": True},
            {"attn_dropout_p": 0.5, "mlp_dropout_p": 0.5, "norm_first": True},
            {"attn_dropout_p": 0.5, "mlp_dropout_p": 0.5, "norm_first": False},
            {"attn_dropout_p": 0.0, "mlp_dropout_p": 0.0, "norm_first": True},  # Control: no dropout
        ],
        indirect=True,
    )
    def test_transformer_block_dropout_train_eval_modes(self, transformer_block, input_tensor, block_kwargs):
        """
        Tests TransformerBlock dropout behavior in train vs eval modes.
        Dropout is applied in MHA (attention and output projection) and MLP.
        """
        attn_p = block_kwargs["attn_dropout_p"]
        mlp_p = block_kwargs["mlp_dropout_p"]
        norm_f = block_kwargs["norm_first"]

        # Eval mode: Dropout should be disabled, outputs should be identical.
        transformer_block.eval()  # Fixture already sets to eval, but explicit here
        with torch.no_grad():
            output_eval_1 = transformer_block(input_tensor)
            output_eval_2 = transformer_block(input_tensor)
        assert torch.allclose(output_eval_1, output_eval_2, atol=1e-7), (
            f"Outputs in eval mode should be identical (attn_p={attn_p}, mlp_p={mlp_p}, norm_first={norm_f})"
        )

        # Train mode: Dropout should be active if p > 0.
        transformer_block.train()
        with torch.no_grad():
            # Multiple calls to dropout layers within MHA and MLP should use different masks.
            output_train_1 = transformer_block(input_tensor)
            output_train_2 = transformer_block(input_tensor)

        if attn_p > 0 or mlp_p > 0:
            assert not torch.allclose(output_train_1, output_train_2, atol=1e-6), (
                f"Outputs in train mode should differ due to dropout (attn_p={attn_p}, mlp_p={mlp_p}, norm_first={norm_f})"
            )
        else:
            # If both dropout probabilities are 0, outputs in train mode should also be identical.
            assert torch.allclose(output_train_1, output_train_2, atol=1e-7), (
                f"Outputs in train mode should be identical if all dropout_p are 0 (attn_p={attn_p}, mlp_p={mlp_p}, norm_first={norm_f})"
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
        assert block.self_attn.out_proj.weight.device.type == device.split(":")[0]
        assert block.self_attn.out_proj.weight.dtype == dtype
        if block_kwargs["qkv_bias"]:
            assert block.self_attn.qkv_proj.bias.dtype == dtype

        # Check parameters of MLP (fc1, fc2)
        assert block.mlp.fc1.weight.device.type == device.split(":")[0]
        assert block.mlp.fc1.weight.dtype == dtype
        assert block.mlp.fc2.weight.device.type == device.split(":")[0]
        assert block.mlp.fc2.weight.dtype == dtype
        if block_kwargs["mlp_bias"]:
            assert block.mlp.fc1.bias.dtype == dtype

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
def test_transformer_block_gradient_computation(norm_first_val, qkv_bias_val, mlp_bias_val, block_kwargs):
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
