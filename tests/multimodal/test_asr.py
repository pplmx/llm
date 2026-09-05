"""Tests for the audio-to-text (ASR) path (ROADMAP 12.2 slice 2).

Covers the transcript<->spectrogram codec (exact invertibility, determinism,
fit validation), the ASR corpus inside ``MultimodalDataModule`` (masked
``[instruction | transcript]`` layout where the audio encodes the transcript),
and a CPU end-to-end: train a trainable audio tower + decoder on random
transcripts encoded into spectrograms, then greedily decode a *held-out*
(hitherto-unseen) transcript from its audio — proving the model reads text out
of audio rather than memorising a fixed pattern.
"""

from __future__ import annotations

import pytest
import torch

from llm.multimodal import MultimodalDataModule
from llm.multimodal.asr import spectrogram_to_tokens, tokens_to_spectrogram

# ---------------------------------------------------------------------------
# transcript <-> spectrogram codec
# ---------------------------------------------------------------------------


def test_codec_roundtrip_exact_for_random_transcripts():
    """Encode-then-decode reconstructs the transcript exactly, for several
    (vocab, length, spectrogram geometry) combos and for batched inputs."""
    torch.manual_seed(0)
    cases = [
        (32, 8, 64, 64, 8),  # (vocab, L, frames, mels, slot_h)
        (32, 8, 64, 40, 8),
        (16, 3, 32, 32, 8),
        (8, 16, 128, 16, 8),
    ]
    for vocab, length, frames, mels, slot_h in cases:
        tokens = torch.randint(1, vocab, (4, length))
        spec = tokens_to_spectrogram(tokens, n_frames=frames, n_mels=mels, slot_h=slot_h, vocab_size=vocab)
        decoded = spectrogram_to_tokens(spec, n_tokens=length, slot_h=slot_h, n_mels=mels)
        assert torch.equal(decoded, tokens), (
            f"roundtrip failed for {cases.index((vocab, length, frames, mels, slot_h))}"
        )


def test_codec_shape_deterministic_and_finite():
    torch.manual_seed(1)
    tokens = torch.randint(1, 32, (3, 8))
    spec = tokens_to_spectrogram(tokens, n_frames=64, n_mels=64, slot_h=8, vocab_size=32)
    assert spec.shape == (3, 1, 64, 64)
    assert torch.isfinite(spec).all()
    assert torch.equal(spec, tokens_to_spectrogram(tokens, n_frames=64, n_mels=64, slot_h=8, vocab_size=32))
    # Energy stays inside [0, 1] (amplitude coded in [0.5, 1]).
    assert bool((spec >= 0.0).all())
    assert bool((spec <= 1.0).all())


def test_codec_roundtrip_exact_at_boundary_vocab():
    """Max exactly-invertible vocab is ``n_mels - 2`` (RIL TASK-310).

    Token ``t`` encodes to freq bin ``2 + (t % (n_mels - 2))``. For a
    permitted vocab the top valid token is ``vocab - 1``. At ``vocab ==
    n_mels - 1`` that top token is ``n_mels - 2``, which lands on bin 2 and
    decodes back to id ``0`` — a silent label corruption the old bound
    allowed. Here the max SAFE vocab (``n_mels - 2``) round-trips every valid
    token, including the top one ``n_mels - 3``.
    """
    torch.manual_seed(3)
    mels, slot_h, frames = 32, 8, 64
    vocab = mels - 2  # 30: the exact-inversion boundary
    tokens = torch.randint(1, vocab, (4, 6))  # ids in [1, vocab-1] = [1, 29]
    tokens[0, 0] = vocab - 1  # force the top valid token id 29 into the batch
    spec = tokens_to_spectrogram(tokens, n_frames=frames, n_mels=mels, slot_h=slot_h, vocab_size=vocab)
    decoded = spectrogram_to_tokens(spec, n_tokens=tokens.shape[-1], slot_h=slot_h, n_mels=mels)
    assert torch.equal(decoded, tokens)


def test_codec_boundary_vocab_roundtrip_is_corrupted():
    """The old permitted ``vocab == n_mels - 1`` breaks the roundtrip: the top
    valid token ``n_mels - 2`` lands on freq bin 2 and decodes as id ``0``.
    Pins the exact failure the DataModule bound now rejects (RIL TASK-310)."""
    mels, slot_h, frames = 32, 8, 64
    vocab = mels - 1  # 31 — previously allowed, silently corrupt
    t = torch.tensor([[mels - 2]])  # top valid token, decodes to 0
    spec = tokens_to_spectrogram(t, n_frames=frames, n_mels=mels, slot_h=slot_h, vocab_size=vocab)
    decoded = spectrogram_to_tokens(spec, n_tokens=1, slot_h=slot_h, n_mels=mels)
    assert not torch.equal(decoded, t)
    assert int(decoded[0, 0]) == 0


def test_codec_distinct_transcripts_produce_distinct_spectrograms():
    torch.manual_seed(2)
    a = torch.randint(1, 32, (1, 8))
    b = a.clone()
    b[0, 3] = (b[0, 3].item() % 28) + 2  # flip one token to a different id
    assert not torch.equal(b, a)
    spec_a = tokens_to_spectrogram(a, n_frames=64, n_mels=64, slot_h=8, vocab_size=32)
    spec_b = tokens_to_spectrogram(b, n_frames=64, n_mels=64, slot_h=8, vocab_size=32)
    # A single token change must move the peak in slot 3 -> spectrograms differ.
    assert not torch.allclose(spec_a, spec_b, atol=1e-6)


def test_codec_fit_and_token_range_validated():
    with pytest.raises(ValueError, match="slot"):
        tokens_to_spectrogram(
            torch.randint(1, 32, (1, 9)), n_frames=64, n_mels=64, slot_h=8, vocab_size=32
        )  # 9*8 = 72 > 64 frames
    with pytest.raises(ValueError, match="vocab"):
        tokens_to_spectrogram(torch.tensor([[0]]), n_frames=64, n_mels=64, slot_h=8, vocab_size=32)
    with pytest.raises(ValueError, match="vocab"):
        tokens_to_spectrogram(torch.tensor([[32]]), n_frames=64, n_mels=64, slot_h=8, vocab_size=32)


# ---------------------------------------------------------------------------
# ASR corpus inside MultimodalDataModule
# ---------------------------------------------------------------------------


def _config(num_samples: int = 32):
    from llm.training.core.config import Config, DistributedConfig, ModelConfig, OptimizationConfig, TrainingConfig

    return Config(
        model=ModelConfig(vocab_size=32, hidden_size=24, num_layers=2, num_heads=2, max_seq_len=16, use_rope=True),
        training=TrainingConfig(batch_size=8, epochs=1, num_samples=num_samples, lr=1e-3, warmup_epochs=0),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(parallel_strategy="ddp"),
    )


def test_datamodule_asr_batch_masks_instruction_and_matches_audio():
    """audio_asr=True: batch is [instruction | transcript] with -100 on the
    instruction, and the transcript decodes exactly out of the sample's
    spectrogram (audio is the source of the label)."""
    config = _config()
    module = MultimodalDataModule(
        config,
        modality="audio",
        audio_frames=64,
        audio_mels=64,
        patch_size=8,
        vit_layers=2,
        vit_heads=4,
        train_encoder=True,
        audio_asr=True,
        vit_instruction_len=8,
    )
    module.prepare_data()
    module.setup()
    loader, _ = module.train_dataloader(rank=0, world_size=1)
    batch = next(iter(loader))
    assert set(batch) == {"input_ids", "labels", "modal_samples"}
    assert batch["input_ids"].shape == (8, 16)
    inst = 8
    assert (batch["labels"][:, :inst] == -100).all()
    assert (batch["labels"][:, inst:] != -100).all()
    assert batch["modal_samples"].shape == (8, 1, 64, 64)
    # The spectrogram deterministically encodes the response (transcript).
    decoded = spectrogram_to_tokens(batch["modal_samples"], n_tokens=16 - inst, slot_h=64 // (16 - inst), n_mels=64)
    assert torch.equal(decoded, batch["labels"][:, inst:])


def test_datamodule_asr_requires_trainable_encoder():
    """ASR is only meaningful with a trainable tower (raw audio in the batch);
    the frozen precompute path must refuse loudly."""
    config = _config()
    module = MultimodalDataModule(
        config,
        modality="audio",
        audio_frames=64,
        audio_mels=64,
        patch_size=8,
        vit_layers=2,
        vit_heads=4,
        train_encoder=False,
        audio_asr=True,
    )
    module.prepare_data()
    with pytest.raises(ValueError, match="train_encoder"):
        module.setup()


def test_datamodule_asr_validates_vocab_fits_mel_bins():
    """asr_vocab >= n_mels - 1 would lose the exact codec roundtrip -> fail fast."""
    config = _config()
    module = MultimodalDataModule(
        config,
        modality="audio",
        audio_frames=64,
        audio_mels=32,  # only 30 usable freq bins (n_mels - 2)
        patch_size=8,
        vit_layers=2,
        vit_heads=4,
        train_encoder=True,
        audio_asr=True,
        asr_vocab=32,
    )
    module.prepare_data()
    with pytest.raises(ValueError, match=r"asr_vocab|freq"):
        module.setup()


def test_datamodule_asr_rejects_boundary_vocab():
    """Regression (RIL TASK-310): ``asr_vocab == n_mels - 1`` used to PASS
    validation yet silently corrupt the codec (token ``n_mels - 2`` round-trips
    to 0). Now rejected; ``n_mels - 2`` (the exact-inversion max) passes."""
    config = _config()
    boundary = MultimodalDataModule(
        config,
        modality="audio",
        audio_frames=64,
        audio_mels=32,
        patch_size=8,
        vit_layers=2,
        vit_heads=4,
        train_encoder=True,
        audio_asr=True,
        asr_vocab=31,  # n_mels - 1 == old silent-corruption boundary
    )
    boundary.prepare_data()
    with pytest.raises(ValueError, match=r"asr_vocab|n_mels"):
        boundary.setup()

    # The max SAFE vocab (n_mels - 2) is accepted and round-trips exactly.
    module = MultimodalDataModule(
        config,
        modality="audio",
        audio_frames=64,
        audio_mels=32,
        patch_size=8,
        vit_layers=2,
        vit_heads=4,
        train_encoder=True,
        audio_asr=True,
        asr_vocab=30,
    )
    module.prepare_data()
    module.setup()


# ---------------------------------------------------------------------------
# CPU end-to-end: transcribe a held-out spectrogram
# ---------------------------------------------------------------------------


def _asr_config(num_samples: int, hidden: int, layers: int, heads: int):
    from llm.training.core.config import Config, DistributedConfig, ModelConfig, OptimizationConfig, TrainingConfig

    return Config(
        model=ModelConfig(
            # vocabulary is 32 but the *ASR alphabet* is ``asr_vocab`` (below);
            # dropout=0 keeps the trajectory deterministic regardless of the
            # global torch RNG that pytest seeds via conftest.
            vocab_size=32,
            hidden_size=hidden,
            num_layers=layers,
            num_heads=heads,
            max_seq_len=16,
            use_rope=True,
            dropout=0.0,
        ),
        training=TrainingConfig(batch_size=16, epochs=1, num_samples=num_samples, lr=2e-3, warmup_epochs=0),
        optimization=OptimizationConfig(use_compile=False, use_amp=False, num_workers=0),
        distributed=DistributedConfig(parallel_strategy="ddp"),
    )


def _transcribe_held_out(engine, *, vocab: int, resp_len: int, inst_len: int, n_frames: int, n_mels: int, batch: int):
    """Greedy-decode ``batch`` held-out random transcripts from audio alone."""
    engine.model.eval()
    device = next(engine.model.parameters()).device
    gen = torch.Generator()
    gen.manual_seed(12345)
    held_out = torch.randint(1, vocab, (batch, resp_len), generator=gen)
    spec = tokens_to_spectrogram(
        held_out, n_frames=n_frames, n_mels=n_mels, slot_h=n_frames // resp_len, vocab_size=vocab
    ).to(device)
    prompt = torch.full((batch, inst_len), 1, dtype=torch.long, device=device)
    generated = engine.model.generate(prompt, modal_samples=spec, max_new_tokens=resp_len)
    assert generated.shape == (batch, inst_len + resp_len)
    pred = generated[:, inst_len:]
    return (pred == held_out.to(device)).float().mean().item()


def test_asr_e2e_trainable_audio_tower_and_decoder_transcribe_heldout_audio():
    """End-to-end: a trainable audio tower + decoder learn to **transcribe**
    audio into text, and greedy-decode a held-out (never-trained) random
    transcript from its spectrogram.

    The transcript alphabet is deliberately small (``asr_vocab=8`` -> 8 well
    separated frequency peaks): at this size the inverse code is learned almost
    instantly and the run generalises robustly on CPU across seeds/threads —
    unlike a 32-token alphabet where the synthetic model sat on a
    memorisation-vs-generalisation knife edge. Held-out accuracy is ~1.0
    (chance is 1/8 per token). Runs in the default suite (~50s CPU).
    """
    from llm.multimodal import MultimodalModel
    from llm.multimodal.task import MultimodalTask
    from llm.training.core.engine import TrainingEngine

    inst_len, resp_len = 8, 8
    n_frames, n_mels = 64, 64
    vocab, num_samples, epochs = 8, 256, 60

    prev = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        config = _asr_config(num_samples=num_samples, hidden=64, layers=4, heads=4)
        module = MultimodalDataModule(
            config,
            modality="audio",
            audio_frames=n_frames,
            audio_mels=n_mels,
            patch_size=8,
            vit_layers=2,
            vit_heads=4,
            train_encoder=True,
            audio_asr=True,
            asr_vocab=vocab,
            vit_instruction_len=inst_len,
        )
        module.prepare_data()
        module.setup()
        task = MultimodalTask(config, module)
        engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)
        assert isinstance(engine.model, MultimodalModel)
        assert engine.model.encoder is not None

        losses = [engine._run_epoch(epoch) for epoch in range(epochs)]
        assert all(loss == loss for loss in losses)
        assert losses[-1] < 1.5

        # Held-out transcription from audio alone (never-trained transcripts).
        acc = _transcribe_held_out(
            engine, vocab=vocab, resp_len=resp_len, inst_len=inst_len, n_frames=n_frames, n_mels=n_mels, batch=64
        )
        assert acc > 0.85, f"held-out transcript accuracy too low: {acc:.3f} (chance 1/8; tuned config reaches ~1.0)"

        # A short probe backward on the trained model proves gradients reach the
        # audio tower (the raw-spectrogram -> token path is jointly trainable).
        device = next(engine.model.parameters()).device
        loader, _ = module.train_dataloader(rank=0, world_size=1)
        batch = next(iter(loader))
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        engine.model.zero_grad(set_to_none=True)
        engine.model.train()
        engine.model(batch["input_ids"], modal_samples=batch["modal_samples"]).float().sum().backward()
        encoder = engine.model.encoder
        assert all(p.grad is not None for p in encoder.parameters())
        assert all(bool(torch.isfinite(p.grad).all()) for p in encoder.parameters() if p.grad is not None)
    finally:
        torch.set_num_threads(prev)


@pytest.mark.e2e
@pytest.mark.heavy
def test_asr_e2e_heldout_transcription_fluent_regime():
    """Larger heavy/e2e-marker variant: same task at higher capacity and more
    distinct transcripts, so the model sits deep in the generalisation regime
    and transcribes ~all held-out tokens (runs in ``make test-e2e``)."""
    from llm.multimodal.task import MultimodalTask
    from llm.training.core.engine import TrainingEngine

    inst_len, resp_len = 8, 8
    n_frames, n_mels = 64, 64
    vocab, num_samples, epochs = 8, 1024, 80

    prev = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        config = _asr_config(num_samples=num_samples, hidden=128, layers=4, heads=8)
        module = MultimodalDataModule(
            config,
            modality="audio",
            audio_frames=n_frames,
            audio_mels=n_mels,
            patch_size=8,
            vit_layers=2,
            vit_heads=4,
            train_encoder=True,
            audio_asr=True,
            asr_vocab=vocab,
            vit_instruction_len=inst_len,
        )
        module.prepare_data()
        module.setup()
        task = MultimodalTask(config, module)
        engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=module)
        losses = [engine._run_epoch(epoch) for epoch in range(epochs)]
        assert all(loss == loss for loss in losses)
        assert losses[-1] < 1.0
        acc = _transcribe_held_out(
            engine, vocab=vocab, resp_len=resp_len, inst_len=inst_len, n_frames=n_frames, n_mels=n_mels, batch=128
        )
        assert acc > 0.95, f"held-out transcript accuracy too low in fluent regime: {acc:.3f}"
    finally:
        torch.set_num_threads(prev)
