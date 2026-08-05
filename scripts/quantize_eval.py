"""Quantization quality gate: perplexity before/after GPTQ on a real model.

Usage:
    uv run python scripts/quantize_eval.py \
        --model checkpoints_wiki/epoch_10 \
        --tokenizer-path /abs/path/to/tokenizer.pt \
        --device cuda:2

The script:
  1. loads a trained ``DecoderModel`` checkpoint (v2 stem / v2 sidecar /
     legacy ``.pt`` / bare quantized blob — same loader as ``llm-serve``);
  2. evaluates perplexity on ``wikitext-2`` (test split);
  3. GPTQ-quantizes (4-bit, group_size=128) with calibration sampled from
     ``wikitext-2`` (train split);
  4. evaluates perplexity again and prints the delta.

This is the reproducible "how much does quantization cost" gate that the
framework's quantization suite was missing: unit tests assert reconstruction
error on random weights, but only a real trained model run can answer the
question a user actually asks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm.evaluation.metrics.perplexity import PerplexityMetric
from llm.quantization import GPTQConfig, quantize_model_gptq


def _load_model_and_tokenizer(model_path: str | Path, tokenizer_path: str | Path):
    """Reuse the serving loader so any servable checkpoint works here."""
    from llm.serving.config import ServingConfig
    from llm.serving.loader import load_model_and_tokenizer

    config = ServingConfig(
        model_path=str(model_path),
        tokenizer_path=str(tokenizer_path),
        tokenizer_type="hf",
    )
    return load_model_and_tokenizer(config)


def _wikitext2_batches(split: str, tokenizer, seq_len: int, batch_size: int, max_batches: int | None):
    """Yield token-id batches from wikitext-2, mirroring training chunking.

    Wikitext lines are short, so per-line padding would fill most positions
    with pad tokens the model never saw during training (training builds
    exact ``max_seq_len`` chunks from consecutive lines).  To keep eval on
    the same distribution, consecutive lines are concatenated and cut into
    exact ``seq_len`` chunks; the trailing partial chunk is dropped.
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    rows = [r["text"] for r in ds if r["text"].strip()]

    buffer: list[int] = []
    chunks: list[list[int]] = []
    for row in rows:
        buffer.extend(tokenizer.encode(row))
        while len(buffer) >= seq_len:
            chunks.append(buffer[:seq_len])
            buffer = buffer[seq_len:]
    chunks = [c for c in chunks if len(c) > 1]
    for i in range(0, len(chunks), batch_size):
        if max_batches is not None and i // batch_size >= max_batches:
            break
        yield torch.tensor(chunks[i : i + batch_size], dtype=torch.long)


def _evaluate_ppl(
    model, tokenizer, device, split: str, seq_len: int, batch_size: int, max_batches: int | None
) -> float:
    """Average perplexity over capped wikitext-2 batches."""
    model.to(device).eval()
    metric = PerplexityMetric(ignore_index=None)
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in _wikitext2_batches(split, tokenizer, seq_len, batch_size, max_batches):
            logits = model(batch.to(device))
            score = metric.compute(logits.cpu().float(), batch)["perplexity"]
            if torch.isfinite(torch.tensor(score)):
                total += float(score)
                count += 1
    if count == 0:
        raise RuntimeError("no finite perplexity samples evaluated")
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Checkpoint path (servable by llm-serve).")
    parser.add_argument("--tokenizer-path", required=True, help="Serialized tokenizer file.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--calib-batches", type=int, default=16, help="Cap for calibration batches (train split).")
    parser.add_argument("--eval-batches", type=int, default=None, help="Cap for eval batches (test split).")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, tokenizer = _load_model_and_tokenizer(args.model, args.tokenizer_path)
    model.to(device).eval()

    print(f"Model: {args.model}")
    print(f"Device: {device}")

    ppl_before = _evaluate_ppl(model, tokenizer, device, "test", args.seq_len, args.batch_size, args.eval_batches)
    print(f"Perplexity (before quantization): {ppl_before:.3f}")

    # Quantize with calibration from the train split (same tokenizer).
    calib = list(_wikitext2_batches("train", tokenizer, args.seq_len, args.batch_size, args.calib_batches))
    if not calib:
        raise RuntimeError("no calibration batches produced")
    quantized = quantize_model_gptq(
        model,
        iter(calib),
        GPTQConfig(bits=args.bits, group_size=args.group_size),
        device=device,
    )
    quantized.to(device).eval()

    ppl_after = _evaluate_ppl(quantized, tokenizer, device, "test", args.seq_len, args.batch_size, args.eval_batches)
    print(f"Perplexity (after  GPTQ-{args.bits}bit): {ppl_after:.3f}")
    print(f"Delta: {ppl_after - ppl_before:+.3f} ({(ppl_after / ppl_before - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
