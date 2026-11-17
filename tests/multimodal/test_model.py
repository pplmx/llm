"""Tests for the standalone MultimodalModel / ModalityFusion."""

from __future__ import annotations

import pytest
import torch

from llm.runtime import ModelFactory
from llm.training.core.config import ModelConfig


def _decoder(use_rope: bool = True):
    return ModelFactory.from_config(
        ModelConfig(vocab_size=32, hidden_size=24, num_layers=2, num_heads=2, max_seq_len=24, use_rope=use_rope)
    )


def test_modality_fusion_projects_to_hidden_and_handles_2d():
    from llm.multimodal.model import ModalityFusion

    fusion = ModalityFusion(modal_dim=16, hidden_size=24)
    # 2D sample-level embedding -> (B, 1, hidden)
    out = fusion(torch.randn(4, 16))
    assert out.shape == (4, 1, 24)
    # 3D prefix embeddings -> (B, M, hidden)
    out3 = fusion(torch.randn(4, 3, 16))
    assert out3.shape == (4, 3, 24)


def test_multimodal_model_returns_text_logits_of_input_length():
    from llm.multimodal.model import MultimodalModel

    decoder = _decoder(use_rope=True)
    model = MultimodalModel(decoder, modal_dim=16)
    input_ids = torch.randint(0, 32, (3, 10))
    modal_embeds = torch.randn(3, 2, 16)  # 2 modal prefix tokens
    logits = model(input_ids, modal_embeds)
    assert logits.shape == (3, 10, 32)  # text logits only, prefix excluded


def test_multimodal_model_prefix_width_is_encoder_driven():
    """Regression: the old ``num_modal_tokens``
    constructor knob was stored but never read — forward/generate derive the
    prefix width from ``modal_embeds`` itself, so a 'knob' value could not
    take effect (setting 1 vs 9 produced bit-identical output). The knob is
    removed; the prefix width follows the encoder's ``num_tokens`` and the
    model simply slices off whatever the encoder emitted."""
    from llm.multimodal.model import MultimodalModel

    decoder = _decoder(use_rope=True)
    model = MultimodalModel(decoder, modal_dim=16)
    input_ids = torch.randint(0, 32, (2, 8))
    for n_prefix in (1, 4):
        modal_embeds = torch.randn(2, n_prefix, 16)
        logits = model(input_ids, modal_embeds)
        assert logits.shape == (2, 8, 32)  # text logits only, prefix stripped
    with pytest.raises(TypeError):
        # The constructor no longer accepts the dead knob.
        MultimodalModel(decoder, num_modal_tokens=1)


def test_multimodal_model_attn_mask_2d_bool_honored():
    """Regression: ``MultimodalModel.forward`` accepts an
    ``attn_mask`` and threads it into the decoder blocks, but a natural
    ``[B, fused_len]`` bool mask crashed both sdpa paths (fast: no
    unsqueeze; complex: OR against the rank-2 causal mask). The mask must
    be honored: masking everything except the first position forces the
    decoder to attend only the modal prefix (positions 0..M-1), which must
    change the text logits vs the unmasked (every-position visible) run."""
    from llm.multimodal.model import MultimodalModel

    decoder = _decoder(use_rope=True)
    model = MultimodalModel(decoder, modal_dim=16)
    input_ids = torch.randint(0, 32, (2, 8))
    modal_embeds = torch.randn(2, 2, 16)  # 2 modal prefix tokens
    fused_len = input_ids.size(1) + modal_embeds.size(1)  # 10

    with torch.no_grad():
        full = model(input_ids, modal_embeds)

        # Keep only position 0 (first modal prefix token), mask the rest.
        mask = torch.zeros(2, fused_len, dtype=torch.bool)
        mask[:, 1:] = True  # True = mask out
        masked = model(input_ids, modal_embeds, attn_mask=mask)

    assert full.shape == (2, 8, 32)
    assert masked.shape == (2, 8, 32)
    assert not torch.allclose(full, masked), "attn_mask had no effect"


def test_multimodal_model_does_not_patch_decoder():
    """The wrapped DecoderModel stays text-only and patched-free."""
    from llm.multimodal.model import MultimodalModel

    decoder = _decoder(use_rope=True)
    _ = MultimodalModel(decoder, modal_dim=16)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        text_logits = decoder(input_ids)
    assert text_logits.shape == (2, 8, 32)  # unchanged text-only forward


class TestGenerateConditionedOnAModalPrefix:
    """``MultimodalModel.generate`` is the inference seam; most of it had no test.

    Only ``tests/multimodal/test_asr.py`` called ``generate``, inside a
    ``heavy`` end-to-end ASR run, and only with a non-empty instruction prompt.
    That left three branches unexercised: the empty-prompt path (seeded with a
    neutral token whose output is stripped afterwards — this is the "pure
    audio->text, no instruction" case the ASR tower is meant to serve), the
    2-D ``modal_embeds`` accepted by ``ModalityFusion``, and every refusal path.

    These are cheap CPU tests; the ASR e2e they complement costs a training run.
    """

    @staticmethod
    def _model():
        from llm.multimodal.model import build_multimodal_model

        return build_multimodal_model(_decoder(), modal_dim=8).eval()

    def test_a_nonempty_prompt_is_preserved_and_extended(self):
        model = self._model()
        modal = torch.randn(2, 3, 8)
        prompt = torch.full((2, 4), 1, dtype=torch.long)

        out = model.generate(prompt, modal_embeds=modal, max_new_tokens=5)

        assert out.shape == (2, 4 + 5), f"expected prompt + 5 new tokens, got {tuple(out.shape)}"
        assert torch.equal(out[:, :4], prompt), "the prompt must be retained verbatim in the output"

    def test_an_empty_prompt_returns_only_generated_tokens(self):
        """The neutral seeding token must be stripped, not returned.

        ``generate`` seeds a zero-length prompt with one token so the decoder has
        something to condition on, then slices it off. Returning it would make
        audio-only transcription emit a spurious leading token.
        """
        model = self._model()
        modal = torch.randn(2, 3, 8)
        empty = torch.zeros(2, 0, dtype=torch.long)

        out = model.generate(empty, modal_embeds=modal, max_new_tokens=4)

        assert out.shape == (2, 4), f"expected exactly the 4 generated tokens, got {tuple(out.shape)}"

    def test_2d_modal_embeds_are_accepted(self):
        """A single embedding per sample is a legal prefix (``ModalityFusion`` accepts 2-D)."""
        model = self._model()
        modal_2d = torch.randn(2, 8)
        prompt = torch.full((2, 3), 1, dtype=torch.long)

        out = model.generate(prompt, modal_embeds=modal_2d, max_new_tokens=2)

        assert out.shape == (2, 5)

    def test_no_modality_at_all_is_refused(self):
        model = self._model()
        prompt = torch.full((1, 3), 1, dtype=torch.long)
        with pytest.raises(ValueError, match="needs modal_embeds or modal_samples"):
            model.generate(prompt)

    def test_zero_new_tokens_is_refused(self):
        """``max_new_tokens=0`` would return the prompt unchanged while looking like a decode."""
        model = self._model()
        prompt = torch.full((1, 3), 1, dtype=torch.long)
        with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
            model.generate(prompt, modal_embeds=torch.randn(1, 8), max_new_tokens=0)

    def test_raw_samples_without_an_encoder_are_refused(self):
        """``modal_samples`` requires the encoder that turns them into embeddings."""
        model = self._model()  # built with modal_dim only, so encoder is None
        assert model.encoder is None
        prompt = torch.full((1, 3), 1, dtype=torch.long)
        with pytest.raises(ValueError, match="needs modal_embeds or a raw modal sample with an encoder"):
            model.generate(prompt, modal_samples=torch.randn(1, 4, 16))

    def test_generation_is_deterministic_for_a_fixed_model(self):
        """Greedy decode: the same inputs must give the same tokens."""
        model = self._model()
        prompt = torch.full((1, 3), 1, dtype=torch.long)
        modal = torch.randn(1, 3, 8)
        first = model.generate(prompt, modal_embeds=modal, max_new_tokens=4)
        second = model.generate(prompt, modal_embeds=modal, max_new_tokens=4)
        assert torch.equal(first, second)
