"""``llm-quantize`` CLI — currently supports the ``gptq`` subcommand.

This CLI closes the gap between the Python API
(:func:`llm.quantization.gptq.quantize_model_gptq`) and the command line.
The same validation rules that the Python API enforces (``GPTQConfig``
``__post_init__``) are also enforced here so users get early, clear errors
on bad input rather than a stack trace halfway through Hessian
accumulation.

Subcommand surface (matches
``docs/superpowers/plans/2026-07-22-gptq-integration.md`` § Task 10):

    llm-quantize gptq \\
        --model PATH                 # torch.save blob with a DecoderModel \\
        --output PATH                # where to write the quantized model \\
        --calib-data PATH            # raw text (one sample per line) — needs --tokenizer \\
        --calib-data-tokens PATH     # pre-tokenized tensor file — mutually exclusive with --calib-data \\
        --tokenizer PATH             # HF tokenizer dir; required with --calib-data \\
        --bits {4,8}                 # default 4 \\
        --group-size N|-1            # default 128; -1 = per-channel \\
        [--sym|--asym]               # default sym (4-bit packing assumes sym) \\
        [--act-order|--no-act-order] # default off \\
        --percdamp F                 # default 0.01 \\
        --blocksize N                # default 128 \\
        --target-modules m1,m2,...   # default: all nn.Linear layers

Exit codes:

    0 — quantization succeeded
    1 — argument validation failed (bad bits, missing tokenizer, etc.)
    2 — runtime failure (model load, tokenization, etc.)
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    add_completion=False,
    help=("Model quantization CLI. Currently supports `gptq` (Frantar 2022, Hessian-aware 4-bit / 8-bit)."),
)


@app.callback()
def _root_callback() -> None:
    """Root callback — keeps the app in "subcommand" mode.

    Without a callback, typer's behaviour depends on the number of
    commands: a single ``@app.command()`` makes the function the only
    top-level command, and ``llm-quantize gptq --flags`` complains
    about an extra argument. Adding an empty callback switches the app
    to "multi-command" mode so ``llm-quantize gptq --flags`` works as
    expected. ``llm-quantize --flags`` (no subcommand) prints help.
    """


def _die(message: str) -> typer.Exit:
    """Print a one-line error to stderr and exit 1.

    Used for argument validation errors that should never reach the
    quantization kernel — they are user mistakes, not runtime failures.
    """
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _validate_calib_inputs(
    calib_data: Path | None,
    calib_data_tokens: Path | None,
    tokenizer: Path | None,
) -> None:
    """Reject mutually-exclusive / missing calibration inputs.

    Rules (mirroring plan § Task 10):

    - Exactly one of ``--calib-data`` / ``--calib-data-tokens`` must be set.
    - ``--calib-data`` requires ``--tokenizer`` (raw text needs tokenization).
    """
    if calib_data is not None and calib_data_tokens is not None:
        _die("--calib-data and --calib-data-tokens are mutually exclusive. Choose one calibration source.")
    if calib_data is None and calib_data_tokens is None:
        _die(
            "must supply calibration data via either --calib-data "
            "(raw text + --tokenizer) or --calib-data-tokens (pre-tokenized .pt)."
        )
    if calib_data is not None and tokenizer is None:
        _die(
            "--calib-data requires --tokenizer PATH "
            "(raw text needs to be tokenized before calibration). "
            "Use --calib-data-tokens for pre-tokenized data instead."
        )


def _validate_model_path(model: Path) -> None:
    """The model file must exist and be a regular file.

    Empty files would torch.load successfully but produce a malformed
    state_dict downstream — fail fast instead.
    """
    if not model.exists():
        _die(f"--model {model} does not exist.")
    if not model.is_file():
        _die(f"--model {model} is not a regular file.")


def _validate_quant_params(
    bits: int,
    group_size: int,
    percdamp: float,
    blocksize: int,
) -> None:
    """Reject CLI-level quantization param violations.

    These duplicate ``GPTQConfig.__post_init__`` so that users see a
    CLI-friendly error message (not a deep dataclass stack frame). The
    config itself re-validates at construction as a defense-in-depth.
    """
    if bits not in (4, 8):
        _die(f"--bits must be 4 or 8 (GPTQ supports these two widths); got {bits}.")
    if group_size != -1 and group_size <= 0:
        _die(f"--group-size must be -1 (per-channel) or positive; got {group_size}.")
    if not (0.0 < percdamp < 1.0):
        _die(f"--percdamp must be in (0, 1); got {percdamp}.")
    if blocksize <= 0:
        _die(f"--blocksize must be positive; got {blocksize}.")
    if group_size > 0 and blocksize % group_size != 0:
        _die(
            f"--blocksize ({blocksize}) must be divisible by --group-size ({group_size}) for correct packing alignment."
        )


def _load_calibration_batches(
    calib_data: Path | None,
    calib_data_tokens: Path | None,
    tokenizer: Path | None,
) -> list:
    """Materialize calibration batches from the chosen source.

    Returns a list of tensors (one per calibration sample). Each tensor
    shape is ``[1, seq_len]`` for raw text or ``[batch, seq_len]`` for
    pre-tokenized .pt files (the algorithm accepts any iterable of
    tensors).
    """
    import torch  # local import — torch is heavy, defer to runtime

    if calib_data is not None:
        typer.echo(f"Loading tokenizer from {tokenizer}...")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(tokenizer))
        text_lines = calib_data.read_text().splitlines()
        batches: list = []
        for line in text_lines:
            if not line.strip():
                continue
            ids = tok(line, return_tensors="pt").input_ids
            batches.append(ids)
        return batches

    # calib_data_tokens path
    typer.echo(f"Loading pre-tokenized calibration from {calib_data_tokens}...")
    # ``weights_only=False`` — calibration .pt files are user-supplied
    # and may contain custom token tensors (e.g. wrappers from a
    # different framework's tokenizer pipeline). PyTorch 2.6's default
    # ``weights_only=True`` would reject anything beyond plain tensors.
    loaded = torch.load(calib_data_tokens, map_location="cpu", weights_only=False)
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, torch.Tensor):
        # Single tensor → wrap in list so the iterable contract is uniform.
        return [loaded]
    # Unknown shape — let downstream fail with a clearer error.
    return loaded


def _resolve_target_modules(target_modules: str | None) -> list[str] | None:
    """Parse the comma-separated ``--target-modules`` flag into a list.

    Empty / whitespace-only entries are dropped. ``None`` (the default)
    means "quantize every ``nn.Linear``" — the algorithm's default
    behaviour, no filter applied.
    """
    if target_modules is None:
        return None
    return [m.strip() for m in target_modules.split(",") if m.strip()]


@app.command()
def gptq(
    model: Path = typer.Option(
        ...,
        "--model",
        help="Path to model checkpoint (.pt with DecoderModel state_dict).",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        help="Output path for quantized model (torch.save blob).",
    ),
    calib_data: Path | None = typer.Option(
        None,
        "--calib-data",
        help="Path to raw text file (one sample per line). Requires --tokenizer.",
    ),
    calib_data_tokens: Path | None = typer.Option(
        None,
        "--calib-data-tokens",
        help="Path to pre-tokenized .pt file (tensor or list of tensors). Mutually exclusive with --calib-data.",
    ),
    tokenizer: Path | None = typer.Option(
        None,
        "--tokenizer",
        help="Path to HF tokenizer (required when --calib-data is set).",
    ),
    bits: int = typer.Option(
        4,
        "--bits",
        help="Quantization bit width (4 or 8).",
    ),
    group_size: int = typer.Option(
        128,
        "--group-size",
        help="Group size for per-group scales (-1 = per-channel).",
    ),
    sym: bool = typer.Option(
        True,
        "--sym/--asym",
        help="Symmetric (default) vs asymmetric quantization.",
    ),
    percdamp: float = typer.Option(
        0.01,
        "--percdamp",
        help="Hessian damping fraction (must be in (0, 1)).",
    ),
    blocksize: int = typer.Option(
        128,
        "--blocksize",
        help="Column block size for the GPTQ outer loop.",
    ),
    act_order: bool = typer.Option(
        False,
        "--act-order/--no-act-order",
        help="Sort weight columns by diag(H) descending (better accuracy, slower).",
    ),
    target_modules: str | None = typer.Option(
        None,
        "--target-modules",
        help="Comma-separated layer names to quantize (default: all nn.Linear).",
    ),
) -> None:
    """Quantize a model with GPTQ (Frantar 2022, arXiv:2210.17323)."""
    # --- 1. Argument validation (fail fast, exit 1) ---------------------
    _validate_quant_params(bits, group_size, percdamp, blocksize)
    _validate_calib_inputs(calib_data, calib_data_tokens, tokenizer)
    _validate_model_path(model)

    # --- 2. Late imports (keep startup cheap) ---------------------------
    import torch

    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    # --- 3. Load inputs -------------------------------------------------
    try:
        typer.echo(f"Loading model from {model}...")
        # ``weights_only=False`` is required: DecoderModel is a custom
        # class that PyTorch 2.6's default ``weights_only=True`` would
        # reject (it only allows a small allow-list of stdlib + torch
        # types). The model file is user-supplied and the user owns the
        # trust boundary — same trust model as ``pickle.load``.
        model_obj = torch.load(model, map_location="cpu", weights_only=False)
    except Exception as exc:
        typer.echo(f"Error: failed to load model {model}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        batches = _load_calibration_batches(calib_data, calib_data_tokens, tokenizer)
    except Exception as exc:
        typer.echo(f"Error: failed to load calibration data: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    # --- 4. Build config + run quantization -----------------------------
    target_list = _resolve_target_modules(target_modules)

    config = GPTQConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        percdamp=percdamp,
        blocksize=blocksize,
        act_order=act_order,
    )

    typer.echo(
        f"Quantizing model with GPTQ (bits={bits}, group_size={group_size}, "
        f"sym={sym}, act_order={act_order}, target_modules={target_list or 'all'})..."
    )
    try:
        quantized = quantize_model_gptq(model_obj, iter(batches), config, target_list)
    except Exception as exc:
        typer.echo(f"Error: quantization failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    # --- 5. Save --------------------------------------------------------
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized, output)
    except Exception as exc:
        typer.echo(f"Error: failed to save quantized model to {output}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"Quantized model saved to {output}")


def main() -> None:
    """Entry point for ``llm-quantize = llm.cli.quantize:main`` in pyproject.toml."""
    app()


if __name__ == "__main__":
    main()
