import math

import pytest
import torch
import torch.nn as nn

# Attempt to import PositionalEncoding from the location it should be
try:
    from src.llm.core.positional_encoding import PositionalEncoding
except ImportError:
    # Fallback for local testing if PYTHONPATH is not set, e.g. when running pytest from root
    # This assumes the script is run from a context where 'src' is a direct subdir
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from src.llm.core.positional_encoding import PositionalEncoding


# Test constants
HIDDEN_SIZE = 64
MAX_SEQ_LEN = 128
BATCH_SIZE = 4
TEST_SEQ_LEN = MAX_SEQ_LEN // 2  # Test with a sequence length smaller than max
DROPOUT_P = 0.1


@pytest.fixture
def dummy_input():
    return torch.randn(BATCH_SIZE, TEST_SEQ_LEN, HIDDEN_SIZE)


@pytest.fixture
def dummy_zero_input():
    return torch.zeros(BATCH_SIZE, TEST_SEQ_LEN, HIDDEN_SIZE)


def test_sinusoidal_encoding_initialization():
    """Test initialization of Sinusoidal PositionalEncoding."""
    model = PositionalEncoding(HIDDEN_SIZE, MAX_SEQ_LEN, learned=False)

    assert hasattr(model, "pe")
    assert isinstance(model.pe, torch.Tensor)
    assert model.pe.shape == (1, MAX_SEQ_LEN, HIDDEN_SIZE)

    # Check characteristic values for sinusoidal encoding at pos=0
    # PE(0, 2i) = sin(0 / ...) = 0
    # PE(0, 2i+1) = cos(0 / ...) = 1
    assert torch.allclose(model.pe[0, 0, 0::2], torch.zeros(HIDDEN_SIZE // 2), atol=1e-7)
    assert torch.allclose(model.pe[0, 0, 1::2], torch.ones(HIDDEN_SIZE // 2), atol=1e-7)

    # Check some other position, e.g. pos=1
    pos1_encoding = model.pe[0, 1, :]
    manual_pos1 = torch.zeros(HIDDEN_SIZE)
    position = torch.tensor([[1.0]])  # pos = 1
    div_term = torch.exp(torch.arange(0, HIDDEN_SIZE, 2).float() * (-math.log(10000.0) / HIDDEN_SIZE))
    manual_pos1[0::2] = torch.sin(position * div_term)
    manual_pos1[1::2] = torch.cos(position * div_term)
    assert torch.allclose(pos1_encoding, manual_pos1, atol=1e-6)


def test_learned_encoding_initialization():
    """Test initialization of Learned PositionalEncoding."""
    model = PositionalEncoding(HIDDEN_SIZE, MAX_SEQ_LEN, learned=True)

    assert hasattr(model, "pos_embedding")
    assert isinstance(model.pos_embedding, nn.Embedding)
    assert model.pos_embedding.weight.shape == (MAX_SEQ_LEN, HIDDEN_SIZE)

    # Check that weights are not all zeros (default is xavier_uniform)
    assert not torch.all(model.pos_embedding.weight == 0)


def test_forward_pass_sinusoidal(dummy_zero_input):
    """Test forward pass for Sinusoidal PositionalEncoding."""
    model = PositionalEncoding(HIDDEN_SIZE, MAX_SEQ_LEN, dropout_p=0.0, learned=False)  # Dropout 0 for exact check
    model.eval()  # Ensure dropout is off

    output = model(dummy_zero_input)

    assert output.shape == (BATCH_SIZE, TEST_SEQ_LEN, HIDDEN_SIZE)

    expected_pe = model.pe[:, :TEST_SEQ_LEN, :]
    # Output should be input (zeros) + positional encoding
    assert torch.allclose(output, expected_pe.expand_as(dummy_zero_input), atol=1e-6)


def test_forward_pass_learned(dummy_zero_input):
    """Test forward pass for Learned PositionalEncoding."""
    model = PositionalEncoding(HIDDEN_SIZE, MAX_SEQ_LEN, dropout_p=0.0, learned=True)  # Dropout 0 for exact check
    model.eval()  # Ensure dropout is off

    output = model(dummy_zero_input)
    assert output.shape == (BATCH_SIZE, TEST_SEQ_LEN, HIDDEN_SIZE)

    # Expected embeddings for positions 0 to TEST_SEQ_LEN-1
    positions = torch.arange(0, TEST_SEQ_LEN, dtype=torch.long, device=dummy_zero_input.device)
    expected_learned_pe = model.pos_embedding(positions).unsqueeze(0)  # Shape (1, TEST_SEQ_LEN, HIDDEN_SIZE)

    # Output should be input (zeros) + learned positional encoding
    assert torch.allclose(output, expected_learned_pe.expand_as(dummy_zero_input), atol=1e-6)


def test_dropout_application(dummy_input):
    """Test if dropout is applied during training and not during eval."""
    # Test Sinusoidal
    model_sin = PositionalEncoding(HIDDEN_SIZE, MAX_SEQ_LEN, dropout_p=DROPOUT_P, learned=False)

    # Training mode - dropout should be active
    model_sin.train()
    output_train_sin = model_sin(dummy_input)

    # Eval mode - dropout should be inactive
    model_sin.eval()
    output_eval_sin = model_sin(dummy_input)

    # Check output shapes are correct
    assert output_train_sin.shape == dummy_input.shape
    assert output_eval_sin.shape == dummy_input.shape

    # In eval mode, output must be input + PE
    expected_output_eval_sin = dummy_input + model_sin.pe[:, :TEST_SEQ_LEN, :]
    assert torch.allclose(output_eval_sin, expected_output_eval_sin, atol=1e-6), (
        "Sinusoidal output in eval mode should be input + PE without dropout"
    )

    # In train mode, with non-zero input and non-zero dropout_p, output should differ from eval output
    # (unless by extreme chance dropout zeros out nothing or the exact same elements)
    # This is a probabilistic test; could fail if dropout happens to be a no-op for a specific run.
    # A more robust check is if at least some elements differ.
    if DROPOUT_P > 0:
        assert not torch.allclose(output_train_sin, output_eval_sin), (
            "Sinusoidal output in train mode should differ from eval mode due to dropout"
        )

    # Test Learned
    model_lrn = PositionalEncoding(HIDDEN_SIZE, MAX_SEQ_LEN, dropout_p=DROPOUT_P, learned=True)

    # Training mode
    model_lrn.train()
    output_train_lrn = model_lrn(dummy_input)

    # Eval mode
    model_lrn.eval()
    output_eval_lrn = model_lrn(dummy_input)

    assert output_train_lrn.shape == dummy_input.shape
    assert output_eval_lrn.shape == dummy_input.shape

    positions = torch.arange(0, TEST_SEQ_LEN, dtype=torch.long, device=dummy_input.device)
    learned_pe = model_lrn.pos_embedding(positions).unsqueeze(0)  # (1, TEST_SEQ_LEN, HIDDEN_SIZE)
    expected_output_eval_lrn = dummy_input + learned_pe
    assert torch.allclose(output_eval_lrn, expected_output_eval_lrn, atol=1e-6), (
        "Learned output in eval mode should be input + PE without dropout"
    )

    if DROPOUT_P > 0:
        assert not torch.allclose(output_train_lrn, output_eval_lrn), (
            "Learned output in train mode should differ from eval mode due to dropout"
        )


def test_sequence_length_handling(dummy_input):
    """Test correct handling of sequences shorter than max_seq_len."""
    # Sinusoidal
    model_sin = PositionalEncoding(HIDDEN_SIZE, MAX_SEQ_LEN, dropout_p=0.0, learned=False)
    model_sin.eval()

    output_sin = model_sin(dummy_input)  # dummy_input has seq_len = TEST_SEQ_LEN

    assert output_sin.shape == (BATCH_SIZE, TEST_SEQ_LEN, HIDDEN_SIZE)
    # Ensure only the part of pe corresponding to TEST_SEQ_LEN is used
    expected_pe_slice = model_sin.pe[:, :TEST_SEQ_LEN, :]
    assert torch.allclose(output_sin, dummy_input + expected_pe_slice, atol=1e-6)

    # Learned
    model_lrn = PositionalEncoding(HIDDEN_SIZE, MAX_SEQ_LEN, dropout_p=0.0, learned=True)
    model_lrn.eval()

    output_lrn = model_lrn(dummy_input)  # dummy_input has seq_len = TEST_SEQ_LEN

    assert output_lrn.shape == (BATCH_SIZE, TEST_SEQ_LEN, HIDDEN_SIZE)
    # Ensure embeddings for indices up to TEST_SEQ_LEN-1 are used
    positions = torch.arange(0, TEST_SEQ_LEN, dtype=torch.long, device=dummy_input.device)
    expected_learned_slice = model_lrn.pos_embedding(positions).unsqueeze(0)
    assert torch.allclose(output_lrn, dummy_input + expected_learned_slice, atol=1e-6)


def test_seq_len_exceeds_max_len(dummy_input):
    """Test that an error is raised if seq_len > max_seq_len."""
    model = PositionalEncoding(HIDDEN_SIZE, MAX_SEQ_LEN, learned=False)
    too_long_input = torch.randn(BATCH_SIZE, MAX_SEQ_LEN + 1, HIDDEN_SIZE)
    with pytest.raises(ValueError, match=r"Sequence length \d+ exceeds maximum sequence length \d+"):
        model(too_long_input)

    model_learned = PositionalEncoding(HIDDEN_SIZE, MAX_SEQ_LEN, learned=True)
    with pytest.raises(ValueError, match=r"Sequence length \d+ exceeds maximum sequence length \d+"):
        model_learned(too_long_input)


# It might be good to also test that learned embeddings are actually learning if we had a training loop
# But for unit testing the module itself, checking the structure and forward pass is key.

if __name__ == "__main__":
    # This allows running the tests directly with `python tests/core/test_positional_encoding.py`
    # For more comprehensive test runs, use `pytest`
    pytest.main([__file__])
