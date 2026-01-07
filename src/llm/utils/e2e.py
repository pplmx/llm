"""
End-to-End Pipeline Utilities.

Core functions for running train → evaluate → inference workflow.
Used by both scripts/e2e_pipeline.py and tests.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from llm.inference import generate
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer


@dataclass
class E2EConfig:
    """Configuration for E2E pipeline."""

    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    max_seq_len: int = 32
    epochs: int = 3
    batch_size: int = 16
    lr: float = 1e-3
    num_samples: int = 200
    prompt: str = "hello"
    max_new_tokens: int = 10


@dataclass
class E2EResult:
    """Results from E2E pipeline run."""

    initial_loss: float
    final_loss: float
    val_loss: float
    perplexity: float
    generated_text: str
    training_time: float

    @property
    def loss_decreased(self) -> bool:
        return self.final_loss < self.initial_loss

    @property
    def inference_ok(self) -> bool:
        return len(self.generated_text) > 0

    @property
    def all_passed(self) -> bool:
        return self.loss_decreased and self.inference_ok


def create_synthetic_data(
    vocab_size: int, num_samples: int, seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic training data."""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
    return input_ids, labels


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    vocab_size: int,
    epochs: int,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
) -> list[float]:
    """Train model and return per-epoch losses."""
    model.train()
    losses = []
    num_batches = (input_ids.size(0) + batch_size - 1) // batch_size

    for _ in range(epochs):
        epoch_loss = 0.0
        for i in range(0, input_ids.size(0), batch_size):
            batch_input = input_ids[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            optimizer.zero_grad()
            logits = model(batch_input)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = criterion(logits.view(-1, vocab_size), batch_labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / num_batches)
        if scheduler:
            scheduler.step()

    return losses


def evaluate_model(
    model: nn.Module,
    criterion: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    vocab_size: int,
) -> tuple[float, float]:
    """Evaluate model and return loss and perplexity."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, input_ids.size(0), batch_size):
            batch_input = input_ids[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            logits = model(batch_input)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = criterion(logits.view(-1, vocab_size), batch_labels.view(-1))
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, perplexity


def run_inference(
    model: nn.Module,
    tokenizer: SimpleCharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0,
) -> str:
    """Run inference and return generated text (excluding prompt)."""
    model.eval()
    generated = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return generated[len(prompt) :]


def run_e2e_pipeline(
    config: E2EConfig,
    device: torch.device | None = None,
    tokenizer: SimpleCharacterTokenizer | None = None,
) -> E2EResult:
    """
    Run the complete E2E pipeline: Train → Evaluate → Inference.

    Args:
        config: E2E configuration
        device: Device to run on (auto-detect if None)
        tokenizer: Tokenizer to use (create default if None)

    Returns:
        E2EResult with all metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer is None:
        corpus = ["hello world", "the quick brown fox", "testing one two three", "abcdefghijklmnopqrstuvwxyz"]
        tokenizer = SimpleCharacterTokenizer(corpus)

    model = DecoderModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
    ).to(device)

    train_inputs, train_labels = create_synthetic_data(
        tokenizer.vocab_size, config.num_samples, config.max_seq_len, device
    )
    val_inputs, val_labels = create_synthetic_data(
        tokenizer.vocab_size, config.num_samples // 5, config.max_seq_len, device
    )

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_start = time.time()
    losses = train_model(
        model,
        optimizer,
        criterion,
        train_inputs,
        train_labels,
        config.batch_size,
        tokenizer.vocab_size,
        config.epochs,
        scheduler,
    )
    training_time = time.time() - train_start

    val_loss, perplexity = evaluate_model(
        model, criterion, val_inputs, val_labels, config.batch_size, tokenizer.vocab_size
    )

    generated = run_inference(model, tokenizer, config.prompt, config.max_new_tokens)

    return E2EResult(
        initial_loss=losses[0],
        final_loss=losses[-1],
        val_loss=val_loss,
        perplexity=perplexity,
        generated_text=generated,
        training_time=training_time,
    )
