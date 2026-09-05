"""Audio-to-text (ASR) transcript <-> log-mel spectrogram codec (ROADMAP 12.2 slice 2).

The 12.2 slice 1 audio tower consumes raw log-mel spectrograms ``[B, 1, T, F]``.
To make *speech recognition* (audio -> text) genuinely learnable on CPU — as
opposed to the model memorising a fixed text pattern while ignoring the audio —
each synthetic training sample carries a **random** transcript whose tokens are
deterministically encoded into its spectrogram. The tower must invert this code
to reproduce the transcript, so audio-conditioned decoding is the only way to
hit high accuracy.

Encoding (a frequency-peak time-slot code):

1. the ``n_tokens`` response positions map to ``n_tokens`` contiguous time
   slots of ``slot_h`` frames each (``n_tokens * slot_h <= n_frames``);
2. token ``t`` lands in slot ``t`` as a smooth Gaussian energy bump centred on
   frequency ``2 + (t % (n_mels - 2))``, scaled by the token's amplitude
   ``0.5 + 0.5 * t / (vocab_size - 1)``, spread across every frame of the slot
   (with a sine envelope peaking at the slot centre) so each patch row in the
   band sees the peak.

The code is a bijection when the vocab fits the usable freq bins
(``vocab_size <= n_mels - 2``): :func:`spectrogram_to_tokens` recovers the
exact transcript by slot-energy argmax over frequency (``freq - 2``). The
DataModule (MultimodalDataModule with ``audio_asr=True``) validates that
bound. (RIL TASK-310: ``n_mels - 1`` is deliberately rejected — the top token
``n_mels - 2`` would land on bin 2 and decode as ``0``, silently corrupting
that label.)
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["spectrogram_to_tokens", "tokens_to_spectrogram"]


def tokens_to_spectrogram(
    tokens: Tensor,
    *,
    n_frames: int,
    n_mels: int,
    slot_h: int,
    vocab_size: int,
    sigma: float = 1.6,
) -> Tensor:
    """Encode integer transcript tokens into log-mel spectrogram(s).

    Args:
        tokens: token ids in ``[1, vocab_size - 1]``, shape ``[..., n_tokens]``.
        n_frames/n_mels: spectrogram geometry ``[1, n_frames, n_mels]``.
        slot_h: frames per token time-slot (``n_tokens * slot_h <= n_frames``).
        vocab_size: token vocabulary; tokens must satisfy ``0 < t < vocab_size``.
        sigma: Gaussian width (in mel bins) of each token's energy bump.

    Returns:
        Spectrogram ``[..., 1, n_frames, n_mels]`` in ``[0, 1]``, deterministic.
    """
    if not torch.is_tensor(tokens) or tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"tokens must be an integer tensor, got {type(tokens).__name__}")
    if tokens.numel() == 0 or (tokens <= 0).any() or (tokens >= vocab_size).any():
        raise ValueError(
            f"tokens must be non-empty and in [1, vocab_size-1], got {tokens.min().item()}..{tokens.max().item()}"
        )
    if n_frames <= 0 or n_mels <= 0 or slot_h <= 0:
        raise ValueError(f"n_frames/n_mels/slot_h must be positive, got {n_frames}/{n_mels}/{slot_h}")
    if n_mels < 3:
        raise ValueError(f"n_mels must be >= 3 to host a peak frequency, got {n_mels}")
    n_tokens = tokens.shape[-1]
    if n_tokens * slot_h > n_frames:
        raise ValueError(
            f"n_tokens*slot_h={n_tokens * slot_h} exceeds n_frames={n_frames}; shorten the transcript or reduce slot_h"
        )

    intrinsic = tokens.shape[:-1]
    flat = tokens.reshape(-1, n_tokens)  # [K, n_tokens]
    device = tokens.device

    # Frequency profile per token: Gaussian over mel bins centred at the token's
    # frequency bin, normalised to the token's amplitude.
    freq_idx = (2 + flat % (n_mels - 2)).float()  # [K, L] in [2, n_mels-1]
    amp = 0.5 + 0.5 * flat.float() / (vocab_size - 1)  # [K, L] in [0.5, 1]
    freq_cols = torch.arange(n_mels, device=device).float()  # [F]
    gauss = torch.exp(-0.5 * ((freq_cols.unsqueeze(0) - freq_idx.unsqueeze(-1)) / sigma) ** 2)
    band = amp.unsqueeze(-1) * gauss  # [K, L, F]

    # Time-slot layout: frame r belongs to slot r // slot_h; a sine envelope
    # peaks at each slot centre so patch rows at band edges see less bleed.
    frames = torch.arange(n_frames, device=device)  # [T]
    slot_of_frame = frames // slot_h
    valid = slot_of_frame < n_tokens
    clamped_slot = slot_of_frame.clamp(max=n_tokens - 1)  # [T]
    gather_idx = clamped_slot.unsqueeze(0).expand(flat.shape[0], -1).unsqueeze(-1).expand(-1, -1, n_mels)  # [K, T, F]
    band_for_frame = band.gather(1, gather_idx)  # [K, T, F]
    envelope = torch.sin(torch.pi * ((frames % slot_h).float() + 0.5) / slot_h)  # [T], 1 at slot centre
    out = band_for_frame * envelope.view(1, n_frames, 1)  # [K, T, F]
    out = out * valid.view(1, n_frames, 1).float()  # silence beyond the transcript

    return out.reshape(*intrinsic, 1, n_frames, n_mels)


def spectrogram_to_tokens(spec: Tensor, *, n_tokens: int, slot_h: int, n_mels: int) -> Tensor:
    """Inverse of :func:`tokens_to_spectrogram` for vocab ``<= n_mels - 2``.

    Per time-slot, averages energy over the slot's frames and takes the argmax
    frequency bin; since the code maps each token bijectively to ``2 + ...``,
    ``freq - 2`` recovers the token id exactly.
    """
    if spec.dim() < 4:
        raise ValueError(f"expected [..., 1, n_frames, n_mels], got shape {tuple(spec.shape)}")
    if spec.shape[-3] != 1:
        raise ValueError(f"expected single channel, got {spec.shape[-3]}")
    n_frames = spec.shape[-2]
    if n_tokens * slot_h > n_frames:
        raise ValueError(f"n_tokens*slot_h={n_tokens * slot_h} exceeds n_frames={n_frames}")
    if slot_h < 1:
        raise ValueError(f"slot_h must be >= 1, got {slot_h}")

    intrinsic = spec.shape[:-3]  # batch dims before [1, T, F]
    channel = spec.reshape(-1, n_frames, n_mels)  # [K, T, F]
    frames = torch.arange(n_frames, device=spec.device)
    slot_of_frame = frames // slot_h
    valid = (slot_of_frame < n_tokens).float()  # [T]
    energy = channel * valid.view(1, -1, 1)  # zero frames beyond the transcript
    # Per-slot mean energy over frequency -> [K, n_tokens, n_mels].
    pooled = torch.zeros(channel.shape[0], n_tokens, n_mels, device=spec.device)
    pooled.scatter_add_(
        1,
        slot_of_frame.clamp(max=n_tokens - 1)
        .unsqueeze(0)
        .expand(channel.shape[0], -1)
        .unsqueeze(-1)
        .expand(-1, -1, n_mels),
        energy,
    )
    freq_argmax = pooled.argmax(-1)  # [K, L]
    tokens = freq_argmax - 2
    return tokens.reshape(*intrinsic, n_tokens).long()
