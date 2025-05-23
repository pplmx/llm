import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Adjust path to import from src
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from llm.core.transformer_block import TransformerBlock
from llm.core.attn.mha import MultiHeadAttention
from llm.core.mlp import MLP

# Test Constants
HIDDEN_SIZE = 64
NUM_HEADS = 8
MLP_INTERMEDIATE_SIZE = HIDDEN_SIZE * 4
ATTN_DROPOUT_P = 0.0 # Default to 0 for deterministic tests unless specified
MLP_DROPOUT_P = 0.0  # Default to 0 for deterministic tests unless specified
NORM_EPS = 1e-5
BATCH_SIZE = 2
SEQ_LEN = 10

# Available devices for testing
DEVICES = ['cpu']
if torch.cuda.is_available():
    DEVICES.append('cuda')

DTYPES = [torch.float32] # Keep it simple for most tests, can expand if needed


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
        "device": "cpu", # Default device
        "dtype": torch.float32 # Default dtype
    }
    if hasattr(request, "param"):
        default.update(request.param)
    return default

@pytest.fixture
def transformer_block(block_kwargs):
    """Creates a TransformerBlock instance based on block_kwargs."""
    block = TransformerBlock(**block_kwargs)
    block.eval() # Default to eval mode
    return block

@pytest.fixture
def input_tensor(block_kwargs):
    """Creates a dummy input tensor based on block_kwargs."""
    return torch.randn(
        BATCH_SIZE, SEQ_LEN, block_kwargs["hidden_size"],
        device=block_kwargs["device"], dtype=block_kwargs["dtype"]
    )


class TestTransformerBlockInitialization:
    def test_submodule_types(self, transformer_block):
        assert isinstance(transformer_block.self_attn, MultiHeadAttention)
        assert isinstance(transformer_block.mlp, MLP)
        assert isinstance(transformer_block.norm1, nn.LayerNorm)
        assert isinstance(transformer_block.norm2, nn.LayerNorm)

    def test_mha_mlp_internal_norm_residual_disabled(self, transformer_block):
        assert transformer_block.self_attn.include_norm_residual is False, \
            "MHA submodule should have include_norm_residual=False"
        assert transformer_block.mlp.include_norm_residual is False, \
            "MLP submodule should have include_norm_residual=False"

    def test_parameter_propagation(self, transformer_block, block_kwargs):
        assert transformer_block.self_attn.hidden_size == block_kwargs["hidden_size"]
        assert transformer_block.self_attn.num_heads == block_kwargs["num_heads"]
        assert transformer_block.self_attn.p == block_kwargs["attn_dropout_p"]
        assert transformer_block.self_attn.is_causal == block_kwargs["is_causal"]
        
        assert transformer_block.mlp.hidden_size == block_kwargs["hidden_size"]
        # MLP's intermediate_size calculation: if None, it's 4*hidden_size
        expected_mlp_intermediate = block_kwargs["mlp_intermediate_size"] or 4 * block_kwargs["hidden_size"]
        assert transformer_block.mlp.intermediate_size == expected_mlp_intermediate
        assert transformer_block.mlp.dropout.p == block_kwargs["mlp_dropout_p"] # MLP stores dropout layer
        
        assert transformer_block.norm1.eps == block_kwargs["norm_eps"]
        assert transformer_block.norm2.eps == block_kwargs["norm_eps"]
        assert transformer_block.norm_first == block_kwargs["norm_first"]

class TestTransformerBlockForwardPass:
    @pytest.mark.parametrize("block_kwargs", [{"norm_first": True}, {"norm_first": False}], indirect=True)
    def test_forward_pass_shape(self, transformer_block, input_tensor):
        output = transformer_block(input_tensor)
        assert output.shape == input_tensor.shape, \
            f"Output shape {output.shape} does not match input shape {input_tensor.shape}"

    @patch('llm.core.transformer_block.MultiHeadAttention.forward')
    @patch('llm.core.transformer_block.MLP.forward')
    @patch('torch.nn.LayerNorm.forward')
    def test_pre_ln_path_logic(self, mock_layernorm_forward, mock_mlp_forward, mock_mha_forward, block_kwargs, input_tensor):
        # Ensure norm_first is True for this test
        block_kwargs["norm_first"] = True
        block_kwargs["attn_dropout_p"] = 0.0 # Disable dropout for easier mocking if needed
        block_kwargs["mlp_dropout_p"] = 0.0

        # Make mocks return tensors of correct shape to allow arithmetic operations
        mock_mha_forward.return_value = torch.zeros_like(input_tensor)
        mock_mlp_forward.return_value = torch.zeros_like(input_tensor)
        # LayerNorm mock needs to handle multiple calls (norm1, norm2)
        # It should return a tensor of the same shape as its input.
        mock_layernorm_forward.side_effect = lambda x: x # Identity, or torch.zeros_like(x)

        block = TransformerBlock(**block_kwargs)
        block.eval()
        
        # Use specific mocks for norm1 and norm2 if LayerNorm.forward is too general
        # For simplicity, we assume the single mock_layernorm_forward can be inspected for calls
        # Or, mock block.norm1.forward and block.norm2.forward specifically.
        
        with patch.object(block.norm1, 'forward', wraps=block.norm1.forward) as spy_norm1_fwd, \
             patch.object(block.self_attn, 'forward', wraps=block.self_attn.forward) as spy_attn_fwd, \
             patch.object(block.norm2, 'forward', wraps=block.norm2.forward) as spy_norm2_fwd, \
             patch.object(block.mlp, 'forward', wraps=block.mlp.forward) as spy_mlp_fwd:
            
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

    @patch('llm.core.transformer_block.MultiHeadAttention.forward')
    @patch('llm.core.transformer_block.MLP.forward')
    @patch('torch.nn.LayerNorm.forward')
    def test_post_ln_path_logic(self, mock_layernorm_forward, mock_mlp_forward, mock_mha_forward, block_kwargs, input_tensor):
        block_kwargs["norm_first"] = False
        block_kwargs["attn_dropout_p"] = 0.0
        block_kwargs["mlp_dropout_p"] = 0.0

        mock_mha_forward.return_value = torch.zeros_like(input_tensor)
        mock_mlp_forward.return_value = torch.zeros_like(input_tensor)
        mock_layernorm_forward.side_effect = lambda x: x 

        block = TransformerBlock(**block_kwargs)
        block.eval()

        with patch.object(block.self_attn, 'forward', wraps=block.self_attn.forward) as spy_attn_fwd, \
             patch.object(block.norm1, 'forward', wraps=block.norm1.forward) as spy_norm1_fwd, \
             patch.object(block.mlp, 'forward', wraps=block.mlp.forward) as spy_mlp_fwd, \
             patch.object(block.norm2, 'forward', wraps=block.norm2.forward) as spy_norm2_fwd:
            
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
        
        with patch.object(block.self_attn, 'forward', wraps=block.self_attn.forward) as spy_attn_fwd:
            spy_attn_fwd.return_value = torch.zeros_like(input_tensor) # Ensure it returns something
            # 1. Call without overriding is_causal in forward -> MHA should use block's default (True)
            _ = block(input_tensor, is_causal=None)
            spy_attn_fwd.assert_called_with(input_tensor, attn_mask=None, is_causal=None) 
            # MHA's forward will then use its own self.is_causal which was set to True

            # 2. Call with is_causal=False override
            _ = block(input_tensor, is_causal=False)
            spy_attn_fwd.assert_called_with(input_tensor, attn_mask=None, is_causal=False)

            # 3. Call with is_causal=True override (explicitly same as default)
            _ = block(input_tensor, is_causal=True)
            spy_attn_fwd.assert_called_with(input_tensor, attn_mask=None, is_causal=True)

    def test_attention_mask_propagation(self, transformer_block, input_tensor):
        dummy_mask = torch.ones(BATCH_SIZE, 1, SEQ_LEN, SEQ_LEN, # MHA expects [B, N, S, S] or broadcastable
                                device=input_tensor.device, dtype=torch.bool)
        
        with patch.object(transformer_block.self_attn, 'forward', wraps=transformer_block.self_attn.forward) as spy_attn_fwd:
            spy_attn_fwd.return_value = torch.zeros_like(input_tensor)
            _ = transformer_block(input_tensor, attn_mask=dummy_mask)
            
            # Check if the mask was passed to MHA's forward
            # Need to access call_args. Get the positional arguments.
            args, kwargs = spy_attn_fwd.call_args
            assert 'attn_mask' in kwargs
            assert torch.equal(kwargs['attn_mask'], dummy_mask)
            # Or if passed positionally: assert torch.equal(args[1], dummy_mask) if arg order is known

    @pytest.mark.parametrize("block_kwargs", [{"attn_dropout_p": 0.5, "mlp_dropout_p": 0.5}], indirect=True)
    def test_dropout_behavior_integration(self, transformer_block, input_tensor):
        # Eval mode (default from fixture) - dropout disabled
        output_eval_1 = transformer_block(input_tensor)
        output_eval_2 = transformer_block(input_tensor)
        assert torch.allclose(output_eval_1, output_eval_2, atol=1e-7), \
            "Outputs in eval mode should be identical."

        # Train mode - dropout should be active
        transformer_block.train()
        torch.manual_seed(0) # For reproducibility of this test
        output_train_1 = transformer_block(input_tensor)
        torch.manual_seed(1) # Ensure different dropout masks if code relies on global seed per call
        output_train_2 = transformer_block(input_tensor)
        
        # If dropout is > 0, outputs should differ
        if transformer_block.self_attn.p > 0 or transformer_block.mlp.dropout.p > 0:
            assert not torch.allclose(output_train_1, output_train_2, atol=1e-6), \
                "Outputs in train mode should differ due to dropout."


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype_str", ['torch.float32']) # Could add 'torch.float64'
class TestDeviceAndDtypePropagation:
    def test_block_device_dtype(self, device, dtype_str, block_kwargs):
        dtype = getattr(torch, dtype_str)
        if device == 'cuda' and dtype == torch.float64:
            if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7):
                pytest.skip("CUDA float64 support not adequate or device not capable.")
        
        block_kwargs.update({"device": device, "dtype": dtype})
        block = TransformerBlock(**block_kwargs)
        block.to(device, dtype) # Ensure module is moved

        # Check parameters of norm layers
        assert block.norm1.weight.device.type == device.split(':')[0]
        assert block.norm1.weight.dtype == dtype
        assert block.norm2.weight.device.type == device.split(':')[0]
        assert block.norm2.weight.dtype == dtype

        # Check parameters of MHA (qkv_proj, out_proj)
        assert block.self_attn.qkv_proj.weight.device.type == device.split(':')[0]
        assert block.self_attn.qkv_proj.weight.dtype == dtype
        assert block.self_attn.out_proj.weight.device.type == device.split(':')[0]
        assert block.self_attn.out_proj.weight.dtype == dtype
        if block_kwargs["qkv_bias"]:
            assert block.self_attn.qkv_proj.bias.dtype == dtype

        # Check parameters of MLP (fc1, fc2)
        assert block.mlp.fc1.weight.device.type == device.split(':')[0]
        assert block.mlp.fc1.weight.dtype == dtype
        assert block.mlp.fc2.weight.device.type == device.split(':')[0]
        assert block.mlp.fc2.weight.dtype == dtype
        if block_kwargs["mlp_bias"]:
            assert block.mlp.fc1.bias.dtype == dtype
            
        # Check output tensor device and dtype
        current_input_tensor = torch.randn(
            BATCH_SIZE, SEQ_LEN, block_kwargs["hidden_size"],
            device=device, dtype=dtype
        )
        output = block(current_input_tensor)
        assert output.device.type == device.split(':')[0]
        assert output.dtype == dtype

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
```
