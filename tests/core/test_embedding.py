import math

import pytest
import torch
import torch.nn as nn

from llm.core.embedding import EmbeddingLayer
from llm.core.positional_encoding import PositionalEncoding

# Test constants
VOCAB_SIZE = 20  # Increased for padding_idx tests
HIDDEN_SIZE = 64
MAX_SEQ_LEN = 128
DROPOUT_P_TEST = 0.15  # Distinct from default 0.1 to ensure it's passed
BATCH_SIZE = 4
TEST_SEQ_LEN = MAX_SEQ_LEN // 2

# Available devices and dtypes for testing
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

DTYPES = [torch.float32]
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:  # Check for float64 support on CUDA
    # Some GPUs might have limited float64 support, but generally it's available.
    # For simplicity, we'll only add float64 if not causing issues in typical CI/dev envs.
    # DTYPES.append(torch.float64) # float64 can be slow, stick to float32 for typical tests
    pass


class TestEmbeddingLayer:
    @pytest.mark.parametrize("pos_learned", [True, False])
    @pytest.mark.parametrize("padding_idx_val", [None, 0, 5])
    def test_initialization(self, pos_learned, padding_idx_val):
        """Test initialization of EmbeddingLayer."""
        if padding_idx_val is not None and padding_idx_val >= VOCAB_SIZE:
            pytest.skip("padding_idx_val must be less than VOCAB_SIZE")

        layer = EmbeddingLayer(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            pos_encoding_learned=pos_learned,
            dropout_p=DROPOUT_P_TEST,
            padding_idx=padding_idx_val,
        )

        assert isinstance(layer.token_embeddings, nn.Embedding)
        assert layer.token_embeddings.num_embeddings == VOCAB_SIZE
        assert layer.token_embeddings.embedding_dim == HIDDEN_SIZE
        assert layer.token_embeddings.padding_idx == padding_idx_val

        assert isinstance(layer.positional_encoding, PositionalEncoding)
        assert layer.positional_encoding.learned == pos_learned
        assert math.isclose(layer.positional_encoding.dropout.p, DROPOUT_P_TEST)
        assert layer.positional_encoding.max_seq_len == MAX_SEQ_LEN
        assert layer.positional_encoding.hidden_size == HIDDEN_SIZE
        assert layer.hidden_size == HIDDEN_SIZE

    @pytest.mark.parametrize("device", DEVICES)
    def test_forward_pass_shape(self, device):
        """Test the output shape of the forward pass."""
        layer = EmbeddingLayer(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, max_seq_len=MAX_SEQ_LEN, device=device)
        layer.to(device)  # Ensure layer is on the specified device

        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, TEST_SEQ_LEN), device=device, dtype=torch.long)
        output = layer(input_ids)

        assert output.shape == (BATCH_SIZE, TEST_SEQ_LEN, HIDDEN_SIZE)
        assert str(output.device) == str(device)  # Check device of output tensor

    @pytest.mark.parametrize("padding_idx_val", [None, 0, 5])  # VOCAB_SIZE = 20
    def test_padding_idx_effect(self, padding_idx_val):
        """Test the effect of padding_idx on token_embeddings output."""
        if padding_idx_val is not None and padding_idx_val >= VOCAB_SIZE:
            pytest.skip("padding_idx_val must be less than VOCAB_SIZE")

        layer = EmbeddingLayer(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, padding_idx=padding_idx_val)

        if padding_idx_val is not None:
            # Create input that includes the padding index
            input_ids_list = [[padding_idx_val, 1, 2], [3, padding_idx_val, padding_idx_val]]
            # Ensure other tokens are not padding_idx_val if they are not meant to be
            for r_idx, row in enumerate(input_ids_list):
                for c_idx, val in enumerate(row):
                    if val != padding_idx_val:
                        if val == 0 and padding_idx_val != 0:  # if val is 0, ensure it's not the padding_idx
                            input_ids_list[r_idx][c_idx] = (
                                val + 1 + padding_idx_val
                            ) % VOCAB_SIZE  # make it non-zero and not padding_idx
                        elif val == padding_idx_val:  # this should not happen here
                            input_ids_list[r_idx][c_idx] = (val + 1) % VOCAB_SIZE

            input_ids = torch.tensor(input_ids_list, dtype=torch.long)

            # Get only the token embedding part (before scaling and PE)
            token_embs = layer.token_embeddings(input_ids)

            padding_mask = input_ids == padding_idx_val
            assert torch.all(token_embs[padding_mask] == 0.0), (
                f"Embedding vectors for padding_idx {padding_idx_val} should be all zeros."
            )
        else:
            # If padding_idx is None, no specific token should be zeroed out by default.
            # This part of the test can be minimal or skipped.
            input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, TEST_SEQ_LEN), dtype=torch.long)
            token_embs = layer.token_embeddings(input_ids)  # Just ensure it runs
            assert token_embs.shape == (BATCH_SIZE, TEST_SEQ_LEN, HIDDEN_SIZE)

    @pytest.mark.parametrize("pos_learned", [True, False])
    @pytest.mark.parametrize("device", DEVICES)
    def test_output_values_and_scaling(self, pos_learned, device):
        """Test the output values, including scaling and positional encoding addition."""
        layer = EmbeddingLayer(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            pos_encoding_learned=pos_learned,
            dropout_p=0.0,  # Disable dropout for deterministic output
            padding_idx=None,  # Simpler to check without padding_idx effects here
            device=device,
        )
        layer.eval()  # Also ensures dropout is off
        layer.to(device)

        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, TEST_SEQ_LEN), device=device, dtype=torch.long)

        actual_output = layer(input_ids)

        # Manual calculation
        expected_token_embs = layer.token_embeddings(input_ids)
        expected_scaled_embs = expected_token_embs * math.sqrt(layer.hidden_size)

        # PositionalEncoding.forward() adds PE to its input.
        # So, the output of layer.positional_encoding(expected_scaled_embs) IS the final expected output.
        expected_final_output = layer.positional_encoding(expected_scaled_embs)

        assert torch.allclose(actual_output, expected_final_output, atol=1e-6), (
            "Output does not match manually calculated expected output."
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_str", ["torch.float32", "torch.float64"])  # Test with float32 and float64
    @pytest.mark.parametrize("pos_learned", [True, False])
    def test_device_and_dtype_propagation(self, device, dtype_str, pos_learned):
        """Test if device and dtype are correctly propagated to submodules."""
        dtype = eval(dtype_str)

        if device == "cuda" and dtype == torch.float64:
            if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7):
                pytest.skip("CUDA float64 support not adequate or device not capable.")
        if device == "cpu" and dtype == torch.float64:  # float64 on CPU is generally fine
            pass

        factory_kwargs = {"device": device, "dtype": dtype}

        layer = EmbeddingLayer(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            pos_encoding_learned=pos_learned,
            **factory_kwargs,
        )
        # The layer itself should be on the device/dtype due to factory_kwargs application in __init__
        # For parameters, this means they are created on that device/dtype.
        # For buffers, they are registered and will be moved when the module is moved.
        # The nn.Module.to() call would also ensure this.
        layer.to(device=device, dtype=dtype)

        # Check token_embeddings
        assert layer.token_embeddings.weight.device.type == device.split(":")[0]  # device can be 'cuda:0'
        assert layer.token_embeddings.weight.dtype == dtype

        # Check positional_encoding components
        pe_module = layer.positional_encoding
        assert pe_module.learned == pos_learned

        if pos_learned:
            # Learned PE uses nn.Embedding
            assert hasattr(pe_module, "pos_embedding"), "PositionalEncoding (learned) should have 'pos_embedding'"
            assert pe_module.pos_embedding.weight.device.type == device.split(":")[0]
            assert pe_module.pos_embedding.weight.dtype == dtype
        else:
            # Sinusoidal PE uses a buffer 'pe'
            assert hasattr(pe_module, "pe"), "PositionalEncoding (sinusoidal) should have 'pe' buffer"
            assert pe_module.pe.device.type == device.split(":")[0]
            assert pe_module.pe.dtype == dtype

        # Test forward pass to ensure output tensor is also on correct device/dtype
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, TEST_SEQ_LEN), device=device, dtype=torch.long)
        output = layer(input_ids)
        assert output.device.type == device.split(":")[0]
        assert output.dtype == dtype


if __name__ == "__main__":
    # This allows running tests with `python tests/core/test_embedding.py`
    # Add further arguments to pytest.main as needed, e.g., '-v' for verbose
    pytest.main([__file__, "-v"])
