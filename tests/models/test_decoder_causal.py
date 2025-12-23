import torch

from llm.models.decoder import DecoderModel


def test_causal_masking_property():
    """
    Verify that changing a future token does not affect the hidden states
    or logits of previous tokens in a causal decoder model.
    """
    vocab_size = 100
    hidden_size = 64
    num_layers = 2
    num_heads = 4
    seq_len = 10

    model = DecoderModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        is_causal=True,
        pos_encoding_learned=True,
        mlp_dropout_p=0.0,
        attn_dropout_p=0.0,
        embedding_dropout_p=0.0,
    )
    model.eval()

    # Original sequence
    input_ids = torch.randint(0, vocab_size, (1, seq_len))

    with torch.no_grad():
        original_output = model(input_ids)

    # Modified sequence: change the last token
    modified_input_ids = input_ids.clone()
    modified_input_ids[0, -1] = (modified_input_ids[0, -1] + 1) % vocab_size

    with torch.no_grad():
        modified_output = model(modified_input_ids)

    # The output for tokens 0 to seq_len-2 should be identical
    # original_output shape: [1, seq_len, vocab_size]
    assert torch.allclose(original_output[:, :-1, :], modified_output[:, :-1, :], atol=1e-5), (
        "Causal masking failed: modification of future token affected past outputs."
    )


def test_causal_masking_with_padding():
    """
    Verify causal masking works correctly even when a padding mask is applied.
    """
    vocab_size = 100
    hidden_size = 32
    num_layers = 1
    num_heads = 2
    seq_len = 8

    model = DecoderModel(
        vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, is_causal=True
    )
    model.eval()

    input_ids = torch.randint(0, vocab_size, (1, seq_len))

    # Mask out the last two tokens as padding
    attn_mask = torch.zeros(1, 1, 1, seq_len, dtype=torch.bool)
    attn_mask[0, 0, 0, -2:] = True

    with torch.no_grad():
        output_masked = model(input_ids, attn_mask=attn_mask)

    # Change one of the padded tokens
    input_ids_mod = input_ids.clone()
    input_ids_mod[0, -1] = (input_ids_mod[0, -1] + 1) % vocab_size

    with torch.no_grad():
        output_masked_mod = model(input_ids_mod, attn_mask=attn_mask)

    # Hidden states for non-padded tokens should be the same
    assert torch.allclose(output_masked[:, :-2, :], output_masked_mod[:, :-2, :], atol=1e-5)
