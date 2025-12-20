from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from llm.core.embedding import EmbeddingLayer
from llm.core.transformer_block import TransformerBlock
from llm.models.decoder import DecoderModel

# Test Constants
VOCAB_SIZE = 500
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_HEADS = 4
MAX_SEQ_LEN = 128
MLP_INTERMEDIATE_SIZE = HIDDEN_SIZE * 4
DROPOUT_P = 0.0  # Default to 0 for deterministic tests unless specified
NORM_EPS = 1e-5
BATCH_SIZE = 2
SEQ_LEN = 10

# Available devices for testing
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

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
        "is_causal": True,  # Default for DecoderModel
        "padding_idx": None,
        "qkv_bias": True,
        "mlp_bias": True,
        "lm_head_bias": True,
        "device": "cpu",
        "dtype": torch.float32,
    }
    if hasattr(request, "param"):
        default.update(request.param)
    return default


@pytest.fixture
def decoder_model(model_kwargs):
    """Creates a DecoderModel instance based on model_kwargs."""
    model = DecoderModel(**model_kwargs)
    model.eval()  # Default to eval mode
    return model


@pytest.fixture
def input_ids_tensor(model_kwargs):
    """Creates dummy input_ids based on model_kwargs."""
    return torch.randint(
        0, model_kwargs["vocab_size"], (BATCH_SIZE, SEQ_LEN), device=model_kwargs["device"], dtype=torch.long
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
    return mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]


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
        expected_causality = model_kwargs.get("is_causal", True)  # Default is True for DecoderModel
        for block in decoder_model.transformer_blocks:
            # TransformerBlock's __init__ takes is_causal and passes it to MHA
            # MHA's __init__ stores it as self.is_causal
            assert block.self_attn.is_causal == expected_causality

    @pytest.mark.parametrize("model_kwargs", [{"lm_head_bias": True}, {"lm_head_bias": False}], indirect=True)
    def test_lm_head_bias(self, decoder_model, model_kwargs):
        """Tests if the lm_head bias is correctly set based on lm_head_bias kwarg."""
        if model_kwargs["lm_head_bias"]:
            assert decoder_model.lm_head.bias is not None, "LM head bias should exist when lm_head_bias=True"
        else:
            assert decoder_model.lm_head.bias is None, "LM head bias should be None when lm_head_bias=False"


class TestDecoderModelForwardPass:
    @pytest.mark.parametrize("model_kwargs", [{"norm_first": True}, {"norm_first": False}], indirect=True)
    def test_forward_pass_shape(self, decoder_model, input_ids_tensor, model_kwargs):
        output = decoder_model(input_ids_tensor)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, model_kwargs["vocab_size"]), (
            f"Output shape {output.shape} is incorrect."
        )

    def test_final_norm_application(self, model_kwargs, input_ids_tensor):
        # Test Pre-LN: final_norm should be called
        model_kwargs["norm_first"] = True
        pre_ln_model = DecoderModel(**model_kwargs)
        pre_ln_model.eval()
        assert pre_ln_model.final_norm is not None

        with patch.object(
            pre_ln_model.final_norm, "forward", wraps=pre_ln_model.final_norm.forward
        ) as spy_final_norm_fwd:
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

        def block_forward_spy(hidden_states, attn_mask=None, is_causal=None, **kwargs):
            # This is where we can assert or store the attn_mask
            block_forward_spy.called_attn_mask = attn_mask
            # Call the original forward method to ensure model runs
            return original_block_forward(hidden_states, attn_mask, is_causal, **kwargs)

        block_forward_spy.called_attn_mask = None  # Initialize

        with patch.object(
            decoder_model.transformer_blocks[0], "forward", side_effect=block_forward_spy
        ) as mock_block_fwd:
            _ = decoder_model(input_ids_tensor, attn_mask=attention_mask_tensor)

            mock_block_fwd.assert_called()  # Ensure the block's forward was called
            # Check the attn_mask received by the block's forward method
            assert torch.equal(block_forward_spy.called_attn_mask, attention_mask_tensor)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype_str", ["torch.float32"])
class TestDeviceAndDtypePropagation:
    def test_model_device_dtype(self, device, dtype_str, model_kwargs, input_ids_tensor):
        dtype = getattr(torch, dtype_str.replace("torch.", ""))
        if (
            device == "cuda"
            and dtype == torch.float64
            and not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7)
        ):
            pytest.skip("CUDA float64 support not adequate or device not capable.")

        model_kwargs.update({"device": device, "dtype": dtype})
        model = DecoderModel(**model_kwargs)
        model.to(device, dtype)  # Ensure model is moved
        model.eval()

        # Check parameters of embedding_layer
        assert model.embedding_layer.token_embeddings.weight.device.type == device.split(":")[0]
        assert model.embedding_layer.token_embeddings.weight.dtype == dtype
        if model.embedding_layer.positional_encoding.learned:
            pe_weight = model.embedding_layer.positional_encoding.pos_embedding.weight
            assert pe_weight.device.type == device.split(":")[0]
            assert pe_weight.dtype == dtype
        else:  # Sinusoidal
            pe_buffer = model.embedding_layer.positional_encoding.pe
            assert pe_buffer.device.type == device.split(":")[0]
            assert pe_buffer.dtype == dtype

        # Check parameters of a TransformerBlock (e.g., first one)
        block = model.transformer_blocks[0]
        assert block.norm1.weight.device.type == device.split(":")[0]
        assert block.norm1.weight.dtype == dtype
        assert block.self_attn.qkv_proj.weight.device.type == device.split(":")[0]
        assert block.self_attn.qkv_proj.weight.dtype == dtype
        assert block.mlp.fc1.weight.device.type == device.split(":")[0]
        assert block.mlp.fc1.weight.dtype == dtype

        # Check final_norm if it exists
        if model.final_norm:
            assert model.final_norm.weight.device.type == device.split(":")[0]
            assert model.final_norm.weight.dtype == dtype

        # Check lm_head
        assert model.lm_head.weight.device.type == device.split(":")[0]
        assert model.lm_head.weight.dtype == dtype

        # Check output tensor device and dtype
        # Input tensor needs to be on the correct device/dtype for the model
        current_input_ids = torch.randint(
            0,
            model_kwargs["vocab_size"],
            (BATCH_SIZE, SEQ_LEN),
            device=device,
            dtype=torch.long,  # input_ids are long
        )
        output = model(current_input_ids)
        assert output.device.type == device.split(":")[0]
        assert output.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize("norm_first_val", [True, False])
@pytest.mark.parametrize("qkv_bias_val", [True, False])
@pytest.mark.parametrize("mlp_bias_val", [True, False])
@pytest.mark.parametrize("lm_head_bias_val", [True, False])
@pytest.mark.parametrize("pos_encoding_learned_val", [True, False])
def test_decoder_model_gradient_computation(
    norm_first_val,
    qkv_bias_val,
    mlp_bias_val,
    lm_head_bias_val,
    pos_encoding_learned_val,
    model_kwargs,
    input_ids_tensor,
):
    """Tests if gradients are computed correctly for all trainable parameters of DecoderModel."""
    torch.manual_seed(42)

    current_model_kwargs = model_kwargs.copy()
    current_model_kwargs.update(
        {
            "norm_first": norm_first_val,
            "qkv_bias": qkv_bias_val,
            "mlp_bias": mlp_bias_val,
            "lm_head_bias": lm_head_bias_val,
            "pos_encoding_learned": pos_encoding_learned_val,
            "embedding_dropout_p": 0.0,  # Disable all dropouts for deterministic gradient check
            "attn_dropout_p": 0.0,
            "mlp_dropout_p": 0.0,
            # Using default hidden_size, vocab_size etc. from fixtures for this test
        }
    )

    device = current_model_kwargs["device"]
    dtype = current_model_kwargs["dtype"]

    model = DecoderModel(**current_model_kwargs)
    model.to(device, dtype)
    model.train()  # Ensure model is in training mode

    # input_ids_tensor fixture uses model_kwargs, so it's already on the correct device
    # but it's for torch.long, which is fine.
    # For DecoderModel, input_ids don't require gradients.

    # Forward pass
    output_logits = model(input_ids_tensor)

    # Compute a dummy loss and backward pass
    # Loss is typically calculated on logits vs target_ids (not input_ids for next token pred)
    # For simplicity, sum all logits.
    loss = output_logits.sum()
    loss.backward()

    # Check gradients for all parameters that should have them
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN values"
            assert not torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf values"
            # Check if any part of the gradient is non-zero.
            # Some params like biases or specific weights in embeddings might have zero grads
            # if not exercised by the input/loss. A simple sum of logits might not exercise all.
            # However, for a sufficiently complex model and sum over all outputs, most should be non-zero.
            # This check can be very strict; for now, ensure they are not all zero if the parameter is not trivial.
            if param.numel() > 1:  # Avoid issues with single-element tensors if they are zero
                assert (param.grad != 0).any(), (
                    f"Gradient for {name} is all zeros (potential issue for non-trivial param)"
                )
        else:
            assert param.grad is None or (param.grad == 0).all(), f"Unexpected gradient for non-trainable param {name}"

    # Input_ids typically do not require gradients in a language model.
    # If they did, we would check:
    # assert input_ids_tensor.grad is None # or check its properties if it had grads


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {
            "embedding_dropout_p": 0.5,
            "attn_dropout_p": 0.0,
            "mlp_dropout_p": 0.0,
            "norm_first": True,
            "pos_encoding_learned": False,
        },
        {
            "embedding_dropout_p": 0.0,
            "attn_dropout_p": 0.5,
            "mlp_dropout_p": 0.0,
            "norm_first": True,
            "pos_encoding_learned": False,
        },
        {
            "embedding_dropout_p": 0.0,
            "attn_dropout_p": 0.0,
            "mlp_dropout_p": 0.5,
            "norm_first": True,
            "pos_encoding_learned": False,
        },
        {
            "embedding_dropout_p": 0.5,
            "attn_dropout_p": 0.5,
            "mlp_dropout_p": 0.5,
            "norm_first": True,
            "pos_encoding_learned": False,
        },
        {
            "embedding_dropout_p": 0.5,
            "attn_dropout_p": 0.5,
            "mlp_dropout_p": 0.5,
            "norm_first": False,
            "pos_encoding_learned": True,  # Also test with post-LN and learned PE
        },
        {  # Control: no dropout
            "embedding_dropout_p": 0.0,
            "attn_dropout_p": 0.0,
            "mlp_dropout_p": 0.0,
            "norm_first": True,
            "pos_encoding_learned": False,
        },
    ],
    indirect=True,
)
def test_decoder_model_dropout_train_eval_modes(decoder_model, input_ids_tensor, model_kwargs):
    """
    Tests DecoderModel dropout behavior in train vs eval modes.
    Dropout can occur in EmbeddingLayer, MHA (attention and output projection), and MLP.
    """
    emb_p = model_kwargs["embedding_dropout_p"]
    attn_p = model_kwargs["attn_dropout_p"]
    mlp_p = model_kwargs["mlp_dropout_p"]
    norm_f = model_kwargs["norm_first"]
    pos_learned = model_kwargs["pos_encoding_learned"]

    # Eval mode: Dropout should be disabled, outputs should be identical.
    decoder_model.eval()  # Fixture already sets to eval, but explicit here
    with torch.no_grad():
        output_eval_1 = decoder_model(input_ids_tensor)
        output_eval_2 = decoder_model(input_ids_tensor)
    assert torch.allclose(output_eval_1, output_eval_2, atol=1e-7), (
        f"Outputs in eval mode should be identical (emb_p={emb_p}, attn_p={attn_p}, mlp_p={mlp_p}, "
        f"norm_first={norm_f}, pos_learned={pos_learned})"
    )

    # Train mode: Dropout should be active if any p > 0.
    decoder_model.train()
    with torch.no_grad():
        output_train_1 = decoder_model(input_ids_tensor)
        output_train_2 = decoder_model(input_ids_tensor)

    any_dropout_active = emb_p > 0 or attn_p > 0 or mlp_p > 0
    if any_dropout_active:
        assert not torch.allclose(output_train_1, output_train_2, atol=1e-6), (
            f"Outputs in train mode should differ due to dropout (emb_p={emb_p}, attn_p={attn_p}, mlp_p={mlp_p}, "
            f"norm_first={norm_f}, pos_learned={pos_learned})"
        )
    else:
        # If all dropout probabilities are 0, outputs in train mode should also be identical.
        assert torch.allclose(output_train_1, output_train_2, atol=1e-7), (
            f"Outputs in train mode should be identical if all dropout_p are 0 (emb_p={emb_p}, attn_p={attn_p}, mlp_p={mlp_p}, "
            f"norm_first={norm_f}, pos_learned={pos_learned})"
        )
