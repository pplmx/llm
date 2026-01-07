"""
End-to-End Pipeline Test.

Tests the complete workflow: train → evaluate → inference.
This is marked as e2e and will only run with `make test` (all tests)
or `pytest -m e2e`.
"""

import pytest
import torch

from llm.inference import generate
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.training.core.config import Config, ModelConfig, TrainingConfig


@pytest.mark.e2e
class TestE2EPipeline:
    """End-to-end test for train → evaluate → inference pipeline."""

    @pytest.fixture
    def small_config(self):
        """Minimal config for fast E2E testing."""
        config = Config()
        config.model = ModelConfig(
            hidden_size=64,
            num_layers=1,
            num_heads=2,
            intermediate_size=128,
            vocab_size=128,
            max_seq_len=32,
        )
        config.training = TrainingConfig(
            batch_size=4,
            epochs=1,
            lr=1e-3,
            num_samples=100,
        )
        return config

    @pytest.fixture
    def tokenizer(self):
        """Create a simple tokenizer for testing."""
        corpus = ["hello world", "the quick brown fox", "testing one two three"]
        return SimpleCharacterTokenizer(corpus)

    def test_e2e_train_evaluate_inference(self, small_config, tokenizer):
        """
        Full pipeline test:
        1. Train model for a few steps
        2. Verify training loss decreases
        3. Generate text and verify output is valid
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === 1. Build Model ===
        model = DecoderModel(
            vocab_size=tokenizer.vocab_size,
            hidden_size=small_config.model.hidden_size,
            num_layers=small_config.model.num_layers,
            num_heads=small_config.model.num_heads,
            max_seq_len=small_config.model.max_seq_len,
        ).to(device)

        # === 2. Train for a few steps ===
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=small_config.training.lr)
        criterion = torch.nn.CrossEntropyLoss()

        # Generate synthetic batch
        batch_size = small_config.training.batch_size
        seq_len = 16
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)

        losses = []
        num_steps = 10

        for _ in range(num_steps):
            optimizer.zero_grad()
            logits = model(input_ids)
            # Handle tuple return (logits, kv_cache)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Verify loss decreased
        first_loss = losses[0]
        last_loss = losses[-1]
        assert last_loss < first_loss, f"Training should reduce loss: {first_loss:.4f} → {last_loss:.4f}"

        # === 3. Evaluate (compute perplexity) ===
        model.eval()
        with torch.no_grad():
            eval_logits = model(input_ids)
            if isinstance(eval_logits, tuple):
                eval_logits = eval_logits[0]
            eval_loss = criterion(eval_logits.view(-1, tokenizer.vocab_size), labels.view(-1))
            perplexity = torch.exp(eval_loss).item()

        assert perplexity > 0, "Perplexity should be positive"
        assert perplexity < float("inf"), "Perplexity should be finite"

        # === 4. Inference ===
        model.eval()
        prompt = "hel"
        generated = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=5,
            temperature=0,  # Greedy for determinism
        )

        # Verify output
        assert len(generated) > len(prompt), "Should generate more tokens than prompt"
        assert generated.startswith(prompt), "Generated text should start with prompt"

        # All generated characters should be in tokenizer's vocab
        for char in generated:
            try:
                tokenizer.encode(char)
            except KeyError:
                pytest.fail(f"Generated character '{char}' not in vocabulary")

    def test_e2e_with_streaming(self, small_config, tokenizer):
        """Test streaming generation produces same result as non-streaming."""
        from llm.inference import stream_generate

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = DecoderModel(
            vocab_size=tokenizer.vocab_size,
            hidden_size=small_config.model.hidden_size,
            num_layers=small_config.model.num_layers,
            num_heads=small_config.model.num_heads,
            max_seq_len=small_config.model.max_seq_len,
        ).to(device)
        model.eval()

        prompt = "the"
        max_tokens = 3

        # Non-streaming
        full_output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=0,
        )

        # Streaming
        streamed_parts = list(
            stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0,
            )
        )
        streamed_output = prompt + "".join(streamed_parts)

        assert full_output == streamed_output, "Streaming and non-streaming should produce same output"
