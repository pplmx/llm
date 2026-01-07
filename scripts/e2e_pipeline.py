#!/usr/bin/env python3
"""
End-to-End Pipeline Script.

Automates the complete "Train → Evaluate → Inference" workflow.
Uses core functions from llm.utils.e2e module.

Usage:
    uv run scripts/e2e_pipeline.py
    uv run scripts/e2e_pipeline.py --epochs 5 --hidden-size 128
"""

import torch
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm.utils.e2e import E2EConfig, run_e2e_pipeline

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


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
    """End-to-End Pipeline: Train → Evaluate → Inference."""
    console.print(
        Panel.fit("[bold blue]LLM End-to-End Pipeline[/bold blue]\nTrain → Evaluate → Inference", border_style="blue")
    )

    # Device setup
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
    console.print(f"📱 Device: [cyan]{device_obj}[/cyan]")

    # Create config and run pipeline
    config = E2EConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_samples=num_samples,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    console.print("\n[bold]Running E2E Pipeline...[/bold]")
    result = run_e2e_pipeline(config, device_obj)

    # Display results
    console.print("\n[bold]Training[/bold]")
    status = "[green]✓ PASS[/green]" if result.loss_decreased else "[red]✗ FAIL[/red]"
    console.print(f"   Loss: {result.initial_loss:.4f} → {result.final_loss:.4f} {status}")
    console.print(f"   Time: [cyan]{result.training_time:.2f}s[/cyan]")

    console.print("\n[bold]Evaluation[/bold]")
    console.print(f"   Validation Loss: [green]{result.val_loss:.4f}[/green]")
    console.print(f"   Perplexity: [green]{result.perplexity:.2f}[/green]")

    console.print("\n[bold]Inference[/bold]")
    console.print(f"   Prompt: [yellow]{prompt!r}[/yellow]")
    console.print(f"   Generated: [green]{result.generated_text!r}[/green]")

    # Summary table
    table = Table(title="Pipeline Summary", show_header=True, header_style="bold")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details")
    table.add_row(
        "Training",
        "[green]✓[/green]" if result.loss_decreased else "[red]✗[/red]",
        f"Loss: {result.initial_loss:.4f} → {result.final_loss:.4f}",
    )
    table.add_row("Evaluation", "[green]✓[/green]", f"PPL: {result.perplexity:.2f}")
    table.add_row(
        "Inference",
        "[green]✓[/green]" if result.inference_ok else "[red]✗[/red]",
        f"Generated {len(result.generated_text)} chars",
    )
    console.print("\n")
    console.print(table)

    if result.all_passed:
        console.print(Panel.fit("[bold green]All E2E checks passed! ✓[/bold green]", border_style="green"))
    else:
        console.print(Panel.fit("[bold red]Some checks failed! ✗[/bold red]", border_style="red"))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
