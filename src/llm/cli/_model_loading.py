"""Shared helpers for the ``llm quantize`` / ``prune`` / ``decompose`` CLIs.

Those three commands all take a trained checkpoint and hand the resulting
``nn.Module`` to an in-place transform. They used to load it with a bare
``torch.load(model)``, which only understands the v0.0.5 single-file layout —
so pointing them at a checkpoint this repo actually produces today (the v2
split layout from ADR-006: ``<stem>.safetensors`` + ``.meta.json`` +
``.extra_state.pt``) failed inside the transform with
``'dict' object has no attribute 'named_modules'``, and the documented
``--model <stem>.pt`` path did not exist at all.

:func:`load_trained_model` resolves BOTH layouts through the public
:func:`llm.training.core.checkpoint.load_checkpoint_payload` helper and
rebuilds the model from the checkpoint's own ``model_config`` sidecar, which
is what ``llm serve`` already does.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class CheckpointModelError(RuntimeError):
    """Raised when a checkpoint cannot be turned back into a live model.

    Typed so the CLIs can map it to their documented exit code 2
    (runtime failure) instead of a raw traceback.
    """


def load_trained_model(model_path: str | Path) -> nn.Module:
    """Load a trained model from a checkpoint in either on-disk layout.

    Accepts a legacy v0.0.5 single-file ``.pt`` blob, the v2 split layout
    (the checkpoint stem, the ``.pt`` spelling of the stem, or the
    ``.safetensors`` sidecar directly), or a bare ``torch.save``'d
    :class:`~torch.nn.Module` blob.

    Args:
        model_path: Path to the checkpoint (any of the spellings above).

    Returns:
        The reconstructed ``nn.Module`` on CPU.

    Raises:
        CheckpointModelError: the path holds no usable checkpoint, or the
            weights cannot be mapped onto the architecture recorded in the
            checkpoint's ``model_config`` sidecar.
    """
    path = Path(model_path)

    module = _try_load_bare_module(path)
    if module is not None:
        logger.info("Loading bare model blob from %s", path)
        return module

    # Lazy imports: these pull in the training stack, which the CLI should
    # only pay for on the code paths that need it.
    from llm.training.core.checkpoint import load_checkpoint_payload

    try:
        payload = load_checkpoint_payload(path)
    except Exception as exc:  # CheckpointIncompleteError / OSError / pickle errors
        raise CheckpointModelError(f"checkpoint at {path} could not be read: {exc}") from exc
    if payload is None:
        raise CheckpointModelError(
            f"no checkpoint found at {path} (checked the legacy single-file layout "
            "and the v2 split layout: <stem>.safetensors / .meta.json / .extra_state.pt)"
        )

    model_state = payload.get("model_state")
    if not model_state:
        raise CheckpointModelError(f"checkpoint at {path} has no model_state to load")

    model = _build_model(payload.get("model_config"), path)
    model.load_state_dict(model_state)
    model.eval()
    logger.info("Rebuilt %s from %s (%d tensors)", type(model).__name__, path, len(model_state))
    return model


def _try_load_bare_module(path: Path) -> nn.Module | None:
    """Return the module if ``path`` is a ``torch.save``'d ``nn.Module`` blob.

    The quantize / prune / decompose CLIs also accept their OWN output as
    input (a plain saved module). Those files are pickles, not checkpoints, so
    they are probed first — under the same ``weights_only`` allowlist the rest
    of the repo uses — and any failure falls through to the checkpoint loader.
    """
    if not path.is_file():
        return None
    try:
        from llm.utils.serialization import register_framework_safe_globals

        register_framework_safe_globals()
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - probe only; the checkpoint path reports properly
        return None
    return obj if isinstance(obj, nn.Module) else None


def _build_model(model_config: dict | None, path: Path) -> nn.Module:
    """Rebuild the architecture from the checkpoint's ``model_config`` sidecar."""
    if not model_config:
        raise CheckpointModelError(
            f"checkpoint at {path} has no model_config sidecar, so the architecture cannot be "
            "reconstructed. Re-save it with CheckpointManager (which records model_config), "
            "or pass a bare saved nn.Module blob instead."
        )
    from llm.runtime.model_factory import ModelConfig, ModelFactory

    try:
        config = ModelConfig(**model_config)
        return ModelFactory.from_config(config)
    except Exception as exc:
        raise CheckpointModelError(
            f"checkpoint at {path} recorded a model_config that cannot be rebuilt: {exc}"
        ) from exc


def checkpoint_exists(model_path: str | Path) -> bool:
    """True when ``model_path`` names a loadable checkpoint in ANY layout.

    The pre-flight path check used to require a regular file, which rejected
    every spelling of the v2 split layout except the ``.safetensors``
    sidecar — including the bare stem and the documented ``<stem>.pt`` form
    (whose ``.pt`` no longer exists once a checkpoint has been migrated).
    Accept a regular file OR a stem that has any split sidecar beside it.
    """
    path = Path(model_path)
    if path.is_file():
        return True
    if path.is_dir():
        # A directory can never be a checkpoint; say so explicitly rather than
        # falling through to the sidecar probe (which would produce a
        # confusing "does not exist" for a path that plainly does).
        return False
    from llm.training.core.checkpoint import _resolve_checkpoint_paths

    _, safetensors_path, meta_path, extra_state_path = _resolve_checkpoint_paths(path)
    return any(p.is_file() for p in (safetensors_path, meta_path, extra_state_path))
