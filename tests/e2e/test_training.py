"""
End-to-End Pipeline Test.

Tests the complete workflow: train → evaluate → inference.
Uses core functions from llm.utils.e2e module.
"""

import pytest
import torch

from llm.inference import generate, stream_generate
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer
from llm.utils.e2e import E2EConfig, run_e2e_pipeline


@pytest.mark.e2e
class TestE2EPipeline:
    """End-to-end test for train → evaluate → inference pipeline."""

    @pytest.fixture
    def small_config(self) -> E2EConfig:
        """Minimal config for fast E2E testing."""
        return E2EConfig(
            hidden_size=64,
            num_layers=1,
            num_heads=2,
            max_seq_len=32,
            epochs=2,
            batch_size=4,
            lr=1e-3,
            num_samples=50,
            prompt="hel",
            max_new_tokens=5,
        )

    @pytest.fixture
    def tokenizer(self) -> SimpleCharacterTokenizer:
        """Create a simple tokenizer for testing."""
        corpus = ["hello world", "the quick brown fox", "testing one two three"]
        return SimpleCharacterTokenizer(corpus)

    @pytest.fixture
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_e2e_train_evaluate_inference(self, small_config: E2EConfig, tokenizer, device):
        """
        Full pipeline test using shared E2E functions.
        Verifies: training loss decreases, perplexity is valid, inference works.
        """
        result = run_e2e_pipeline(small_config, device, tokenizer)

        assert result.loss_decreased, (
            f"Training should reduce loss: {result.initial_loss:.4f} → {result.final_loss:.4f}"
        )
        assert result.perplexity > 0, "Perplexity should be positive"
        assert result.perplexity < float("inf"), "Perplexity should be finite"
        assert result.inference_ok, "Should generate at least one character"
        assert result.all_passed, "All E2E checks should pass"

    def test_e2e_result_properties(self, small_config: E2EConfig, tokenizer, device):
        """Test that E2EResult properties work correctly."""
        result = run_e2e_pipeline(small_config, device, tokenizer)

        assert isinstance(result.initial_loss, float)
        assert isinstance(result.final_loss, float)
        assert isinstance(result.val_loss, float)
        assert isinstance(result.perplexity, float)
        assert isinstance(result.generated_text, str)
        assert isinstance(result.training_time, float)

    def test_e2e_with_streaming(self, small_config: E2EConfig, tokenizer, device):
        """Test streaming generation produces same result as non-streaming."""
        model = DecoderModel(
            vocab_size=tokenizer.vocab_size,
            hidden_size=small_config.hidden_size,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            max_seq_len=small_config.max_seq_len,
        ).to(device)
        model.eval()

        prompt = "the"
        max_tokens = 3

        full_output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=0,
        )

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
