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

from llm.models.decoder import DecoderModel
from llm.core.embedding import EmbeddingLayer
from llm.core.transformer_block import TransformerBlock

# Test Constants
VOCAB_SIZE = 500
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_HEADS = 4
MAX_SEQ_LEN = 128
MLP_INTERMEDIATE_SIZE = HIDDEN_SIZE * 4
DROPOUT_P = 0.0 # Default to 0 for deterministic tests unless specified
NORM_EPS = 1e-5
BATCH_SIZE = 2
SEQ_LEN = 10

# Available devices for testing
DEVICES = ['cpu']
if torch.cuda.is_available():
    DEVICES.append('cuda')

DTYPES = [torch.float32]


@pytest.fixture
def model_kwargs(request):
    """Fixture to provide default and overridable kwargs for DecoderModel."""
    default = {
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "max_seq_len": MAX_SEQ_LEN,
        "mlp_intermediate_size": MLP_INTERMEDIATE_SIZE,
        "pos_encoding_learned": False,
        "embedding_dropout_p": DROPOUT_P,
        "attn_dropout_p": DROPOUT_P,
        "mlp_dropout_p": DROPOUT_P,
        "mlp_activation": "gelu",
        "norm_eps": NORM_EPS,
        "norm_first": True,
        "is_causal": True, # Default for DecoderModel
        "padding_idx": None,
        "qkv_bias": True,
        "mlp_bias": True,
        "lm_head_bias": True,
        "device": "cpu",
        "dtype": torch.float32
    }
    if hasattr(request, "param"):
        default.update(request.param)
    return default

@pytest.fixture
def decoder_model(model_kwargs):
    """Creates a DecoderModel instance based on model_kwargs."""
    model = DecoderModel(**model_kwargs)
    model.eval() # Default to eval mode
    return model

@pytest.fixture
def input_ids_tensor(model_kwargs):
    """Creates dummy input_ids based on model_kwargs."""
    return torch.randint(
        0, model_kwargs["vocab_size"], (BATCH_SIZE, SEQ_LEN),
        device=model_kwargs["device"], dtype=torch.long
    )

@pytest.fixture
def attention_mask_tensor(model_kwargs):
    """Creates a dummy attention mask (padding mask)."""
    # True means masked (ignored by attention)
    mask = torch.zeros(BATCH_SIZE, SEQ_LEN, device=model_kwargs["device"], dtype=torch.bool)
    # Example: mask the last token for the first batch item
    if SEQ_LEN > 1:
        mask[0, -1] = True
    # Reshape for MHA: [B, 1, 1, S_key] or [B, 1, S_q, S_k] for SDPA
    # For simplicity, let's use a mask that can be broadcasted by SDPA: [B, 1, 1, S]
    return mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, S]


class TestDecoderModelInitialization:
    def test_submodule_types_and_counts(self, decoder_model, model_kwargs):
        assert isinstance(decoder_model.embedding_layer, EmbeddingLayer)
        assert isinstance(decoder_model.transformer_blocks, nn.ModuleList)
        assert len(decoder_model.transformer_blocks) == model_kwargs["num_layers"]
        for block in decoder_model.transformer_blocks:
            assert isinstance(block, TransformerBlock)
        
        if model_kwargs["norm_first"]:
            assert isinstance(decoder_model.final_norm, nn.LayerNorm)
        else:
            assert decoder_model.final_norm is None
            
        assert isinstance(decoder_model.lm_head, nn.Linear)
        assert decoder_model.lm_head.out_features == model_kwargs["vocab_size"]

    def test_transformer_block_is_causal_setting(self, decoder_model, model_kwargs):
        expected_causality = model_kwargs.get("is_causal", True) # Default is True for DecoderModel
        for block in decoder_model.transformer_blocks:
            # TransformerBlock's __init__ takes is_causal and passes it to MHA
            # MHA's __init__ stores it as self.is_causal
            assert block.self_attn.is_causal == expected_causality

class TestDecoderModelForwardPass:
    @pytest.mark.parametrize("model_kwargs", [{"norm_first": True}, {"norm_first": False}], indirect=True)
    def test_forward_pass_shape(self, decoder_model, input_ids_tensor, model_kwargs):
        output = decoder_model(input_ids_tensor)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, model_kwargs["vocab_size"]), \
            f"Output shape {output.shape} is incorrect."

    def test_final_norm_application(self, model_kwargs, input_ids_tensor):
        # Test Pre-LN: final_norm should be called
        model_kwargs["norm_first"] = True
        pre_ln_model = DecoderModel(**model_kwargs)
        pre_ln_model.eval()
        assert pre_ln_model.final_norm is not None
        
        with patch.object(pre_ln_model.final_norm, 'forward', wraps=pre_ln_model.final_norm.forward) as spy_final_norm_fwd:
            _ = pre_ln_model(input_ids_tensor)
            spy_final_norm_fwd.assert_called_once()

        # Test Post-LN: final_norm should be None and thus not called
        model_kwargs["norm_first"] = False
        post_ln_model = DecoderModel(**model_kwargs)
        post_ln_model.eval()
        assert post_ln_model.final_norm is None
        # No need to mock if it's None, it won't be called.

    def test_padding_mask_handling(self, decoder_model, input_ids_tensor, attention_mask_tensor):
        # Spy on the forward method of the first TransformerBlock
        # We want to see what attn_mask it receives
        # Note: The TransformerBlock itself passes this mask to its MHA.
        # MHA's scaled_dot_product_attention then combines it with causal mask.
        
        # We can mock all blocks or just one. Let's mock the first one.
        # The mock should still return a tensor of the correct shape.
        original_block_forward = decoder_model.transformer_blocks[0].forward
        
        def block_forward_spy(hidden_states, attn_mask=None, is_causal=None):
            # This is where we can assert or store the attn_mask
            block_forward_spy.called_attn_mask = attn_mask
            # Call the original forward method to ensure model runs
            return original_block_forward(hidden_states, attn_mask, is_causal)
        
        block_forward_spy.called_attn_mask = None # Initialize
        
        with patch.object(decoder_model.transformer_blocks[0], 'forward', side_effect=block_forward_spy) as mock_block_fwd:
            _ = decoder_model(input_ids_tensor, attn_mask=attention_mask_tensor)
            
            mock_block_fwd.assert_called() # Ensure the block's forward was called
            # Check the attn_mask received by the block's forward method
            assert torch.equal(block_forward_spy.called_attn_mask, attention_mask_tensor)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype_str", ['torch.float32'])
class TestDeviceAndDtypePropagation:
    def test_model_device_dtype(self, device, dtype_str, model_kwargs, input_ids_tensor):
        dtype = getattr(torch, dtype_str)
        if device == 'cuda' and dtype == torch.float64: # Assuming float64 might be tested later
            if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7):
                pytest.skip("CUDA float64 support not adequate or device not capable.")
        
        model_kwargs.update({"device": device, "dtype": dtype})
        model = DecoderModel(**model_kwargs)
        model.to(device, dtype) # Ensure model is moved
        model.eval()

        # Check parameters of embedding_layer
        assert model.embedding_layer.token_embeddings.weight.device.type == device.split(':')[0]
        assert model.embedding_layer.token_embeddings.weight.dtype == dtype
        if model.embedding_layer.positional_encoding.learned:
            pe_weight = model.embedding_layer.positional_encoding.pos_embedding.weight
            assert pe_weight.device.type == device.split(':')[0]
            assert pe_weight.dtype == dtype
        else: # Sinusoidal
            pe_buffer = model.embedding_layer.positional_encoding.pe
            assert pe_buffer.device.type == device.split(':')[0]
            assert pe_buffer.dtype == dtype


        # Check parameters of a TransformerBlock (e.g., first one)
        block = model.transformer_blocks[0]
        assert block.norm1.weight.device.type == device.split(':')[0]
        assert block.norm1.weight.dtype == dtype
        assert block.self_attn.qkv_proj.weight.device.type == device.split(':')[0]
        assert block.self_attn.qkv_proj.weight.dtype == dtype
        assert block.mlp.fc1.weight.device.type == device.split(':')[0]
        assert block.mlp.fc1.weight.dtype == dtype

        # Check final_norm if it exists
        if model.final_norm:
            assert model.final_norm.weight.device.type == device.split(':')[0]
            assert model.final_norm.weight.dtype == dtype

        # Check lm_head
        assert model.lm_head.weight.device.type == device.split(':')[0]
        assert model.lm_head.weight.dtype == dtype
            
        # Check output tensor device and dtype
        # Input tensor needs to be on the correct device/dtype for the model
        current_input_ids = torch.randint(
            0, model_kwargs["vocab_size"], (BATCH_SIZE, SEQ_LEN),
            device=device, dtype=torch.long # input_ids are long
        )
        output = model(current_input_ids)
        assert output.device.type == device.split(':')[0]
        assert output.dtype == dtype

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
```
