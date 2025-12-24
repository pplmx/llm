import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from llm.models.decoder import DecoderModel


@pytest.mark.heavy
def test_training_convergence_overfit():
    """
    A functional "smoke test" to ensure that gradients flow correctly and
    the model can overfit a single repeating sequence.
    """
    torch.manual_seed(42)

    vocab_size = 50
    hidden_size = 32
    num_layers = 1
    num_heads = 4

    model = DecoderModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        is_causal=True,
        # Use simple configuration for convergence test
        pos_encoding_learned=True,
        attn_dropout_p=0.0,
        mlp_dropout_p=0.0,
        embedding_dropout_p=0.0,
    )

    # Overfit on a single sequence
    target_sequence = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    optimizer = optim.AdamW(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    initial_loss = None
    final_loss = None

    # Perform a few steps of optimization
    for i in range(150):
        optimizer.zero_grad()

        # input: [1, 2, ..., 7], target: [2, 3, ..., 8]
        input_ids = target_sequence[:, :-1]
        labels = target_sequence[:, 1:]

        logits = model(input_ids)

        # logits shape: [1, seq_len-1, vocab_size] -> [seq_len-1, vocab_size]
        # labels shape: [1, seq_len-1] -> [seq_len-1]
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

        if i == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        final_loss = loss.item()

        if final_loss < 0.05:  # Early stop if sufficiently overfitted
            break

    print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")

    # Verify that loss decreased significantly
    assert final_loss < initial_loss
    assert final_loss < 0.2, f"Model failed to overfit tiny sequence. Final loss: {final_loss:.4f}"


@pytest.mark.heavy
def test_gradient_flow():
    """
    Verify that all parameters receive gradients during a backward pass.
    """
    vocab_size = 50
    hidden_size = 32
    num_layers = 1
    num_heads = 4
    seq_len = 5

    model = DecoderModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        use_glu=True,  # Test SwiGLU gradient flow too
    )

    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    logits = model(input_ids)

    loss = logits.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients."
        assert torch.abs(param.grad).sum() > 0, f"Parameter {name} has zero gradients."
