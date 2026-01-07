#!/usr/bin/env python3
"""
End-to-End Pipeline Script.

Automates the complete "Train → Evaluate → Inference" workflow.
This script serves as a smoke test to verify the entire LLM framework works correctly.

Usage:
    uv run scripts/e2e_pipeline.py
    uv run scripts/e2e_pipeline.py --epochs 5 --hidden-size 128
"""

import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from llm.inference import generate, stream_generate
from llm.models.decoder import DecoderModel
from llm.tokenization.simple_tokenizer import SimpleCharacterTokenizer

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


def create_synthetic_data(tokenizer: SimpleCharacterTokenizer, num_samples: int, seq_len: int, device: torch.device):
    """Generate synthetic training data."""
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
    return input_ids, labels


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    vocab_size: int,
    progress: Progress,
    task_id,
) -> float:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = (input_ids.size(0) + batch_size - 1) // batch_size

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

        total_loss += loss.item()
        progress.update(task_id, advance=1)

    return total_loss / num_batches


def evaluate(
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


@app.command()
def main(
    hidden_size: int = typer.Option(64, help="Model hidden size"),
    num_layers: int = typer.Option(2, help="Number of transformer layers"),
    num_heads: int = typer.Option(2, help="Number of attention heads"),
    max_seq_len: int = typer.Option(32, help="Maximum sequence length"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(16, help="Training batch size"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    num_samples: int = typer.Option(200, help="Number of synthetic training samples"),
    device: str = typer.Option("auto", help="Device: 'cpu', 'cuda', or 'auto'"),
    prompt: str = typer.Option("hello", help="Prompt for inference test"),
    max_new_tokens: int = typer.Option(10, help="Max tokens to generate during inference"),
):
    """
    End-to-End Pipeline: Train → Evaluate → Inference.

    This script validates the complete LLM framework workflow.
    """
    console.print(
        Panel.fit("[bold blue]LLM End-to-End Pipeline[/bold blue]\nTrain → Evaluate → Inference", border_style="blue")
    )

    # === Device Setup ===
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
    console.print(f"📱 Device: [cyan]{device_obj}[/cyan]")

    # === 1. Initialize Tokenizer ===
    console.print("\n[bold]1️⃣  Initializing Tokenizer[/bold]")
    corpus = ["hello world", "the quick brown fox", "testing one two three", "abcdefghijklmnopqrstuvwxyz"]
    tokenizer = SimpleCharacterTokenizer(corpus)
    console.print(f"   Vocabulary size: [green]{tokenizer.vocab_size}[/green]")

    # === 2. Initialize Model ===
    console.print("\n[bold]2️⃣  Initializing Model[/bold]")
    model = DecoderModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
    ).to(device_obj)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(
        f"   Parameters: [green]{total_params:,}[/green] total, [green]{trainable_params:,}[/green] trainable"
    )

    # === 3. Prepare Data ===
    console.print("\n[bold]3️⃣  Preparing Synthetic Data[/bold]")
    train_inputs, train_labels = create_synthetic_data(tokenizer, num_samples, max_seq_len, device_obj)
    val_inputs, val_labels = create_synthetic_data(tokenizer, num_samples // 5, max_seq_len, device_obj)
    console.print(
        f"   Train samples: [green]{train_inputs.size(0)}[/green], Val samples: [green]{val_inputs.size(0)}[/green]"
    )

    # === 4. Training ===
    console.print("\n[bold]4️⃣  Training[/bold]")
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    num_batches = (train_inputs.size(0) + batch_size - 1) // batch_size
    losses = []

    train_start = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for epoch in range(epochs):
            task_id = progress.add_task(f"Epoch {epoch + 1}/{epochs}", total=num_batches)
            avg_loss = train_epoch(
                model,
                optimizer,
                criterion,
                train_inputs,
                train_labels,
                batch_size,
                tokenizer.vocab_size,
                progress,
                task_id,
            )
            losses.append(avg_loss)
            scheduler.step()
            progress.update(task_id, description=f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    train_time = time.time() - train_start

    # Verify loss decreased
    loss_decreased = losses[-1] < losses[0]
    status = "[green]✓ PASS[/green]" if loss_decreased else "[red]✗ FAIL[/red]"
    console.print(f"   Loss: {losses[0]:.4f} → {losses[-1]:.4f} {status}")
    console.print(f"   Training time: [cyan]{train_time:.2f}s[/cyan]")

    # === 5. Evaluation ===
    console.print("\n[bold]5️⃣  Evaluation[/bold]")
    val_loss, perplexity = evaluate(model, criterion, val_inputs, val_labels, batch_size, tokenizer.vocab_size)
    console.print(f"   Validation Loss: [green]{val_loss:.4f}[/green]")
    console.print(f"   Perplexity: [green]{perplexity:.2f}[/green]")

    # === 6. Inference ===
    console.print("\n[bold]6️⃣  Inference[/bold]")

    # Non-streaming generation
    console.print(f"   Prompt: [yellow]{prompt!r}[/yellow]")
    generated = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0,
    )
    console.print(f"   Generated: [green]{generated!r}[/green]")

    # Streaming generation
    console.print("   Streaming: ", end="")
    for token in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=5,
    ):
        console.print(f"[cyan]{token}[/cyan]", end="")
    console.print()

    # === Summary ===
    console.print("\n")
    table = Table(title="Pipeline Summary", show_header=True, header_style="bold")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    table.add_row("Training", "[green]✓[/green]", f"Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    table.add_row("Evaluation", "[green]✓[/green]", f"PPL: {perplexity:.2f}")
    inference_ok = len(generated) > len(prompt)
    table.add_row(
        "Inference",
        "[green]✓[/green]" if inference_ok else "[red]✗[/red]",
        f"Generated {len(generated) - len(prompt)} chars",
    )

    console.print(table)

    all_passed = loss_decreased and inference_ok
    if all_passed:
        console.print(Panel.fit("[bold green]All E2E checks passed! ✓[/bold green]", border_style="green"))
    else:
        console.print(Panel.fit("[bold red]Some checks failed! ✗[/bold red]", border_style="red"))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
