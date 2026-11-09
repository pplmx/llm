import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from llm.models.decoder import DecoderModel


@pytest.mark.heavy
def test_decoder_overfits_single_sequence():
    """Requirement: DecoderModel can overfit a tiny repeating sequence."""
    torch.manual_seed(42)

    model = DecoderModel(
        vocab_size=50,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        is_causal=True,
        pos_encoding_learned=True,
        attn_dropout_p=0.0,
        mlp_dropout_p=0.0,
        embedding_dropout_p=0.0,
    )

    target_sequence = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    optimizer = optim.AdamW(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    initial_loss = None
    final_loss = None

    for step in range(150):
        optimizer.zero_grad()
        input_ids = target_sequence[:, :-1]
        labels = target_sequence[:, 1:]
        logits = model(input_ids)
        loss = criterion(logits.view(-1, 50), labels.view(-1))

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        final_loss = loss.item()

        if final_loss < 0.05:
            break

    assert final_loss < initial_loss
    assert final_loss < 0.2
