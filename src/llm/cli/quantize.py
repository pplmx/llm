"""``llm quantize`` CLI — currently supports the ``gptq`` subcommand.

This CLI closes the gap between the Python API
(:func:`llm.quantization.gptq.quantize_model_gptq`) and the command line.
The same validation rules that the Python API enforces (``GPTQConfig``
``__post_init__``) are also enforced here so users get early, clear errors
on bad input rather than a stack trace halfway through Hessian
accumulation.

Subcommand surface (matches
``docs/superpowers/plans/2026-07-22-gptq-integration.md`` § Task 10):

    llm quantize gptq \\
        --model PATH                 # checkpoint stem / .pt / .safetensors / saved module \\
        --output PATH                # where to write the quantized model \\
        --calib-data PATH            # raw text (one sample per line) — needs --tokenizer \\
        --calib-data-tokens PATH     # pre-tokenized tensor file — mutually exclusive with --calib-data \\
        --tokenizer PATH             # HF tokenizer dir; required with --calib-data \\
        --bits {4,8}                 # default 4 \\
        --group-size N|-1            # default 128; -1 = per-channel \\
        [--sym|--asym]               # default sym (--asym rejected: not implemented) \\
        [--act-order|--no-act-order] # default off \\
        --percdamp F                 # default 0.01 \\
        --blocksize N                # default 128 \\
        --target-modules m1,m2,...   # default: all nn.Linear layers

Exit codes:

    0 — quantization succeeded
    1 — argument validation failed (bad bits, missing tokenizer, etc.)
    2 — runtime failure (model load, tokenization, etc.)
    (Note: typer itself exits 2 — the same as a runtime failure — for
    argparse-level usage errors like a missing required --model/--output,
    so a bare mis-typed invocation is not distinguishable from an in-run
    crash by exit code alone; the stderr message differs.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import typer

from llm.cli._compression_args import die, reject_clobbering_output, validate_model_path

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    no_args_is_help=True,
    add_completion=False,
    help=(
        "Model quantization CLI. Subcommands: `gptq` (Frantar 2022, Hessian-aware "
        "4-bit / 8-bit) and `fp8` (E4M3/E5M2 weight+activation PTQ)."
    ),
)


@app.callback()
def _root_callback() -> None:
    """Root callback — keeps the app in "subcommand" mode.

    Without a callback, typer's behaviour depends on the number of
    commands: a single ``@app.command()`` makes the function the only
    top-level command, and ``llm quantize gptq --flags`` complains
    about an extra argument. Adding an empty callback switches the app
    to "multi-command" mode so ``llm quantize gptq --flags`` works as
    expected. ``llm quantize --flags`` (no subcommand) prints help.
    """


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
        die("--calib-data and --calib-data-tokens are mutually exclusive. Choose one calibration source.")
    if calib_data is None and calib_data_tokens is None:
        die(
            "must supply calibration data via either --calib-data "
            "(raw text + --tokenizer) or --calib-data-tokens (pre-tokenized .pt)."
        )
    if calib_data is not None and tokenizer is None:
        die(
            "--calib-data requires --tokenizer PATH "
            "(raw text needs to be tokenized before calibration). "
            "Use --calib-data-tokens for pre-tokenized data instead."
        )


def _atomic_save_quantized(model_obj, output: Path) -> None:
    """Write ``model_obj`` to ``output`` atomically (temp + fsync + rename).

    Mirrors the checkpoint manager's ``_atomic_write_bytes`` pattern: a direct ``torch.save`` to the final path leaves a corrupt
    partial file if it is interrupted (ENOSPC, crash, Ctrl-C). ``torch.save``
    has no bytes-level hook here (the payload is a pickled module, not a
    bytes buffer we control), so write to a sibling temp file first, fsync it,
    then rename into place — a crash before the rename leaves the temp behind
    but never a truncated file under the final name. Also removes a stale
    temp from a previous interrupted run before re-saving.
    """
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    torch.save(model_obj, tmp)
    # fsync so the rename only lands after the bytes are durable. Must open the
    # temp READ-WRITE: Windows `os.fsync` requires a write-access handle and
    # raises EBADF on a read-only one (POSIX tolerates read-only). Opening "rb+"
    # keeps the fsync-before-rename durability on both platforms.
    import os

    with tmp.open("rb+") as fh:
        os.fsync(fh.fileno())
    tmp.replace(output)


def _validate_quant_params(
    bits: int,
    group_size: int,
    percdamp: float,
    blocksize: int,
    sym: bool = True,
) -> None:
    """Reject CLI-level quantization param violations.

    These duplicate ``GPTQConfig.__post_init__`` so that users see a
    CLI-friendly error message (not a deep dataclass stack frame). The
    config itself re-validates at construction as a defense-in-depth.
    """
    if not sym:
        # Asymmetric GPTQ is unimplemented end-to-end (the quantizer raises
        # NotImplementedError mid-loop). Reject it here as an argument error
        # (exit 1) instead of letting it surface as an exit-2 "quantization
        # failed" deep in the Hessian loop.
        die("--asym asymmetric GPTQ is not yet implemented; use symmetric quantization (default) or drop --asym.")
    if bits not in (4, 8):
        die(f"--bits must be 4 or 8 (GPTQ supports these two widths); got {bits}.")
    if group_size != -1 and group_size <= 0:
        die(f"--group-size must be -1 (per-channel) or positive; got {group_size}.")
    if not (0.0 < percdamp < 1.0):
        die(f"--percdamp must be in (0, 1); got {percdamp}.")
    if blocksize <= 0:
        die(f"--blocksize must be positive; got {blocksize}.")
    if group_size > 0 and blocksize % group_size != 0:
        die(
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

        # `tokenizer` is a local calibration-tokenizer Path (not a Hub repo),
        # so a pinned HF `revision` is inapplicable here (bandit B615 false +).
        tok = AutoTokenizer.from_pretrained(str(tokenizer))  # nosec B615 - local path, not Hub download
        if tok is None:
            raise RuntimeError(f"failed to load tokenizer from {tokenizer}")
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
    # Calibration files are a tensor or a list of tensors — plain payloads
    # that ``weights_only=True`` loads fine. Loading with the old
    # ``weights_only=False`` executed arbitrary ``__reduce__`` code from a
    # third-party/shared calibration file, matching the
    # arbitrary-pickle RCE the repo closed elsewhere.
    if calib_data_tokens is None:
        raise RuntimeError("pre-tokenized calibration path is required")
    loaded = torch.load(calib_data_tokens, map_location="cpu", weights_only=True)
    if isinstance(loaded, torch.Tensor):
        # Single tensor → wrap in list so the iterable contract is uniform.
        return [loaded]
    if isinstance(loaded, list):
        if all(isinstance(t, torch.Tensor) for t in loaded):
            return loaded
        bad = next((t for t in loaded if not isinstance(t, torch.Tensor)), None)
        raise RuntimeError(
            f"{calib_data_tokens} contains a non-tensor entry "
            f"({type(bad).__name__}); expected a single tensor or a list of "
            "tensors (one per calibration sample)."
        )
    # Anything else (dict, tuple, scalar) would make the quantizer iterate
    # dict keys / fail deep inside Hessian capture — reject it at load time
    # with a clear message instead.
    raise RuntimeError(
        f"{calib_data_tokens} has an unexpected payload ({type(loaded).__name__}); "
        "expected a single tensor or a list of tensors (one per calibration sample)."
    )


def _resolve_target_modules(target_modules: str | None) -> list[str] | None:
    """Parse the comma-separated ``--target-modules`` flag into a list.

    Empty / whitespace-only entries are dropped. ``None`` (the default)
    means "quantize every ``nn.Linear``" — the algorithm's default
    behaviour, no filter applied.
    """
    if target_modules is None:
        return None
    parsed = [m.strip() for m in target_modules.split(",") if m.strip()]
    # Empty / whitespace-only input → None ("all"), so the CLI summary echoes
    # ``all`` only when the algorithm actually quantizes every nn.Linear
    # layer. The old ``[]`` filtered to ZERO layers and then failed at runtime
    # with "target_modules matched no nn.Linear".
    return parsed or None


@app.command()
def gptq(
    model: Path = typer.Option(
        ...,
        "--model",
        help="Checkpoint to load: v2 split stem (e.g. 'checkpoints/epoch_5'), the '.pt' spelling, a '.safetensors' sidecar, a legacy single-file .pt blob, or a bare torch.save'd module.",
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
    _validate_quant_params(bits, group_size, percdamp, blocksize, sym=sym)
    _validate_calib_inputs(calib_data, calib_data_tokens, tokenizer)
    validate_model_path(model)
    reject_clobbering_output(output, model)

    # --- 2. Late imports (keep startup cheap) ---------------------------

    from llm.quantization.gptq import GPTQConfig, quantize_model_gptq

    # --- 3. Load inputs -------------------------------------------------
    try:
        typer.echo(f"Loading model from {model}...")
        # ``load_trained_model`` accepts the v2 split layout this repo writes
        # (ADR-006) as well as a legacy single-file blob / a bare saved
        # ``nn.Module``. Its bare-module probe keeps the hardened
        # ``weights_only=True`` + framework-allowlist loading, so a shared
        # ``.pt`` still cannot smuggle a ``__reduce__``.
        from llm.cli._model_loading import load_trained_model

        model_obj = load_trained_model(model)
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
        _atomic_save_quantized(quantized, output)
    except Exception as exc:
        typer.echo(f"Error: failed to save quantized model to {output}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"Quantized model saved to {output}")


@app.command()
def fp8(
    model: Path = typer.Option(
        ...,
        "--model",
        help="Checkpoint to load: v2 split stem (e.g. 'checkpoints/epoch_5'), the '.pt' spelling, a '.safetensors' sidecar, a legacy single-file .pt blob, or a bare torch.save'd module.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        help="Output path for quantized model (torch.save blob).",
    ),
    weight_dtype: str = typer.Option(
        "e4m3",
        "--weight-dtype",
        help="FP8 weight format: 'e4m3' (E4M3FN, default) or 'e5m2' (wider range).",
    ),
    per_channel: bool = typer.Option(
        True,
        "--per-channel/--per-tensor",
        help="Per-output-row (default) vs per-tensor weight scaling.",
    ),
    activation: str = typer.Option(
        "dynamic",
        "--activation",
        help="Activation scaling: 'dynamic' (per forward; default, no calibration needed) or 'static' (from calibration).",
    ),
    calib_data: Path | None = typer.Option(
        None,
        "--calib-data",
        help="Path to raw text file (one sample per line). Requires --tokenizer. Needed only for static activation.",
    ),
    calib_data_tokens: Path | None = typer.Option(
        None,
        "--calib-data-tokens",
        help="Path to pre-tokenized .pt file (tensor or list of tensors). Needed only for static activation.",
    ),
    tokenizer: Path | None = typer.Option(
        None,
        "--tokenizer",
        help="Path to HF tokenizer (required when --calib-data is set).",
    ),
    target_modules: str | None = typer.Option(
        None,
        "--target-modules",
        help="Comma-separated layer names to quantize (default: all nn.Linear).",
    ),
) -> None:
    """Quantize a model's Linear weights + activations with FP8 (E4M3/E5M2).

    Stores real float8 weights (1 byte/weight) and simulates the fp8 matmul
    in fp32. Static activation scaling captures per-layer abs-max over the
    calibration batches; dynamic computes it per forward and needs no
    calibration data.
    """
    if weight_dtype not in ("e4m3", "e5m2"):
        die(f"--weight-dtype must be 'e4m3' or 'e5m2'; got {weight_dtype}.")
    if activation not in ("static", "dynamic"):
        die(f"--activation must be 'static' or 'dynamic'; got {activation}.")

    if activation == "static":
        if calib_data is None and calib_data_tokens is None:
            # Static is opt-in now (dynamic is the default), so when the user
            # asks for it without data the message should name the only thing
            # they need to add (a calib source) AND the escape hatch: ``--activation dynamic`` needs no calibration at all.
            die(
                "activation='static' requires calibration data via --calib-data/"
                "--calib-data-tokens; use --activation dynamic to quantize "
                "without calibration."
            )
        _validate_calib_inputs(calib_data, calib_data_tokens, tokenizer)
    elif calib_data is not None or calib_data_tokens is not None or tokenizer is not None:
        # Calibration is ignored under dynamic activation — refuse rather than
        # silently drop the user's data.
        die("activation='dynamic' needs no calibration; drop --calib-data / --calib-data-tokens / --tokenizer.")
    validate_model_path(model)
    reject_clobbering_output(output, model)

    from llm.cli._model_loading import load_trained_model
    from llm.quantization.fp8 import quantize_model_fp8

    try:
        typer.echo(f"Loading model from {model}...")
        # Same layout handling as the GPTQ command: the v2 split layout this
        # repo writes (ADR-006) as well as a legacy blob / bare saved module.
        # Loading with ``torch.load`` here rejected every v2 spelling.
        model_obj = load_trained_model(model)
    except Exception as exc:
        typer.echo(f"Error: failed to load model {model}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    batches = None
    if activation == "static":
        try:
            batches = _load_calibration_batches(calib_data, calib_data_tokens, tokenizer)
        except Exception as exc:
            typer.echo(f"Error: failed to load calibration data: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    else:
        # Dynamic activation: quantize_model_fp8 accepts None calib_iter.
        batches = None

    target_list = _resolve_target_modules(target_modules)
    typer.echo(
        f"Quantizing model with FP8 (weight_dtype={weight_dtype}, "
        f"per_channel={per_channel}, activation={activation}, "
        f"target_modules={target_list or 'all'})..."
    )
    try:
        from llm.quantization.fp8 import Fp8Config

        fp8_cfg = Fp8Config(
            weight_dtype=cast(Literal["e4m3", "e5m2"], weight_dtype),
            per_channel=per_channel,
            activation=cast(Literal["static", "dynamic"], activation),
        )
        quantized = quantize_model_fp8(
            model_obj,
            iter(batches) if batches is not None else None,
            fp8_cfg,
            target_modules=target_list,
        )
    except Exception as exc:
        typer.echo(f"Error: quantization failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        _atomic_save_quantized(quantized, output)
    except Exception as exc:
        typer.echo(f"Error: failed to save quantized model to {output}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"Quantized model saved to {output}")


def main() -> None:
    """Entry point for ``llm quantize = llm.cli.quantize:main`` in pyproject.toml."""
    app()


if __name__ == "__main__":
    main()
