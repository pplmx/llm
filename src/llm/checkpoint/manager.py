"""Training-side checkpoint manager: retention, best-loss, epoch snapshots.

:class:`CheckpointManager` is what a training run talks to. It decides *when*
to save (periodic / best / final), *what* to keep (retention policy removing
old epoch snapshots), and owns the resume bookkeeping (``best_loss``,
``loaded_extra_state``). The actual bytes on disk are written by
:mod:`llm.checkpoint.format`.

Split from the format layer because the two have different audiences: the
format is read by `serving`, `runtime` and `cli`, which never train; this
module is imported only by the training layer.
"""

from __future__ import annotations

import logging
import pickle
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

from llm.checkpoint.format import (
    CHECKPOINT_FORMAT_VERSION,
    EXTRA_STATE_SUFFIX,
    LEGACY_SUFFIX,
    META_SUFFIX,
    SAFETENSORS_SUFFIX,
    _discover_epoch_snapshots,
    _model_config_mismatches,
    _resolve_checkpoint_paths,
    _safetensors_available,
    _save_extra_state_pt,
    _save_metadata_json,
    _save_weights_safetensors,
    load_checkpoint_payload,
)
from llm.training.core.config import CheckpointConfig
from llm.training.distributed import is_fsdp, is_pp, is_pp3d, is_tp, load_model_state_dict, model_state_dict

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Save/load checkpoints with retention and atomic-write semantics.

    Writes the v2 split layout (three sidecar files per checkpoint
    name). :meth:`load_checkpoint` accepts both the new layout AND
    the legacy single-file ``.pt`` layout — auto-detected on read.
    """

    def __init__(self, config: CheckpointConfig, rank: int, logger):
        self.config = config
        self.rank = rank
        self.logger = logger
        self.best_loss = float("inf")
        self.loaded_extra_state: dict | None = None
        self.checkpoints_saved: list[Path] = []
        if self.rank == 0:
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ---- save side --------------------------------------------------------

    def _save_split(
        self,
        *,
        name: str,
        model_state: dict[str, torch.Tensor],
        optimizer_state: dict[str, Any] | None,
        scheduler_state: dict[str, Any] | None,
        scaler_state: dict[str, Any] | None,
        epoch: int,
        loss: float,
        best_loss: float,
        model_config: dict | None,
        extra_state: dict | None,
    ) -> tuple[Path, Path, Path]:
        """Write all three sidecars for ``name``; return the paths."""
        if not _safetensors_available():
            raise ImportError(
                f"Saving checkpoint {name} requires the 'safetensors' package. "
                "Install with `uv sync --extra compat` or `pip install llm[compat]`."
            )
        base = Path(self.config.checkpoint_dir) / name
        weights_path = base.with_name(base.name + SAFETENSORS_SUFFIX)
        meta_path = base.with_name(base.name + META_SUFFIX)
        extra_state_path = base.with_name(base.name + EXTRA_STATE_SUFFIX)

        # Per-save generation marker: written into BOTH the
        # meta and the extra_state sidecars so the loader can prove the trio
        # came from one atomic save. A crash between writing the weights and
        # the extra_state would otherwise leave meta/extra from a previous
        # save paired with fresh weights — silently re-training from a stale
        # optimizer/epoch. ``epoch`` is cross-checked the same way.
        save_id = uuid.uuid4().hex

        # Stamp the save_id into the weights sidecar FIRST: the
        # three sidecars only prove they came from one atomic save if the
        # weights file also carries the generation marker — a crash between a
        # weights write and the meta/extra write would otherwise leave a fresh
        # weights file beside a stale but mutually-consistent meta/extra pair
        # that the meta↔extra check cannot detect.
        _save_weights_safetensors(model_state, weights_path, save_id=save_id, epoch=epoch)
        _save_metadata_json(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "epoch": epoch,
                "loss": loss,
                "best_loss": best_loss,
                "model_config": model_config,
                "save_id": save_id,
            },
            meta_path,
        )
        _save_extra_state_pt(
            optimizer_state,
            scheduler_state,
            scaler_state,
            extra_state,
            extra_state_path,
            save_id=save_id,
            epoch=epoch,
        )
        return weights_path, meta_path, extra_state_path

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer | None,
        scheduler: LRScheduler | None,
        scaler: torch.amp.GradScaler | None,
        loss: float,
        extra_state: dict | None = None,
        model_config: dict | None = None,
    ):
        collective = is_fsdp(model) or is_tp(model) or is_pp(model) or is_pp3d(model)
        # PP stages own DISJOINT parameter slices, so each rank's optimizer
        # state only covers its own stage. Rank 0's state dict
        # alone cannot replay the other stages' moments at resume, so gather
        # EVERY rank's stage optimizer state over the WORLD before the write
        # and persist them keyed by GLOBAL rank. The gather is world-wide (not
        # PP-group-wide) deliberately: under PP+DP different DP
        # groups hold DISTINCT moments, and a PP-group-only gather would drop
        # the other groups from rank 0's file. Every rank must enter it (it is
        # a collective) even though non-zero ranks discard the result.
        pp_optimizer_state: dict[str, Any] | None = None
        if collective and (is_pp(model) or is_pp3d(model)) and optimizer is not None and dist.is_initialized():
            mine = {str(dist.get_rank()): optimizer.state_dict()}
            gathered: list[dict[str, Any]] = [{} for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, mine)
            merged: dict[str, Any] = {}
            for other in gathered:
                merged.update(other)
            pp_optimizer_state = {"__pp_stage_states__": merged}

        if self.rank != 0:
            # FSDP's FULL_STATE_DICT state_dict() and the TP full-state-dict
            # gather are cross-rank collectives:
            # every shard must enter them or rank 0 blocks forever. Non-zero
            # ranks enter the gather and discard the result (they never
            # write); rank 0 below does the disk write.
            if collective:
                model_state_dict(model)
            if collective and dist.is_initialized():
                # Durable-return contract: a collective save
                # must not return on non-zero ranks before rank 0's disk write
                # is complete — a rank that reads the checkpoint right after
                # the call (last-epoch engine e2e, any resume-from-disk on a
                # non-zero rank) would otherwise race the write and see a
                # missing / stale / partially-flushed file. Rank 0 reaches the
                # same barrier after writing, so an all-world barrier here
                # makes "save_checkpoint returned" mean "checkpoint is on disk".
                dist.barrier()
            return

        model_state_to_save = model_state_dict(model)
        opt_state = (
            optimizer.state_dict() if optimizer is not None and pp_optimizer_state is None else pp_optimizer_state
        )

        # ``save_best`` updates best_loss before writing the rest of the
        # meta, so the .meta.json best_loss is consistent across files.
        if self.config.save_best and loss < self.best_loss:
            self.best_loss = loss
            self.logger.info(f"🏆 New best model saved with loss {loss:.4f}")
            self._save_split(
                name="best",
                model_state=model_state_to_save,
                optimizer_state=opt_state,
                scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                scaler_state=scaler.state_dict() if scaler is not None else None,
                epoch=epoch,
                loss=loss,
                best_loss=self.best_loss,
                model_config=model_config,
                extra_state=extra_state,
            )

        self._save_split(
            name="latest",
            model_state=model_state_to_save,
            optimizer_state=opt_state,
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            scaler_state=scaler.state_dict() if scaler is not None else None,
            epoch=epoch,
            loss=loss,
            best_loss=self.best_loss,
            model_config=model_config,
            extra_state=extra_state,
        )

        if (epoch + 1) % self.config.save_interval == 0:
            epoch_name = f"epoch_{epoch + 1}"
            _, _, _ = self._save_split(
                name=epoch_name,
                model_state=model_state_to_save,
                optimizer_state=opt_state,
                scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                scaler_state=scaler.state_dict() if scaler is not None else None,
                epoch=epoch,
                loss=loss,
                best_loss=self.best_loss,
                model_config=model_config,
                extra_state=extra_state,
            )
            epoch_pt = Path(self.config.checkpoint_dir) / f"{epoch_name}{LEGACY_SUFFIX}"
            self.checkpoints_saved.append(epoch_pt)
            self._cleanup_old_checkpoints()
            self.logger.debug(f"Checkpoint saved to {epoch_name}{SAFETENSORS_SUFFIX}")

        if collective and dist.is_initialized():
            # Match the non-zero-rank barrier above: every rank
            # waits here until rank 0's disk writes are durable, so no rank
            # can return from a collective save ahead of the on-disk state.
            dist.barrier()

    def _cleanup_old_checkpoints(self):
        # ``checkpoints_saved`` only records THIS process's writes; a resumed
        # run starts with it empty, so basing retention on the list alone left
        # prior runs' ``epoch_N`` trios to pile up on disk forever. Prune from what is actually on disk: keep the newest
        # ``keep_last_n`` epoch snapshots by save time (mtime; epoch number
        # breaks same-second ties). The in-memory list is kept coherent as the
        # current run's record (engine tests/introspection read it).
        keep = self.config.keep_last_n
        by_epoch = _discover_epoch_snapshots(Path(self.config.checkpoint_dir))
        ordered = sorted(by_epoch.items(), key=lambda kv: (-kv[1].stat().st_mtime, -kv[0]))
        for epoch_no, _ in ordered[keep:]:
            self._remove_epoch_snapshot(epoch_no)
        while len(self.checkpoints_saved) > keep:
            self.checkpoints_saved.pop(0)

    def _remove_epoch_snapshot(self, epoch_no: int) -> None:
        """Remove every sidecar of the ``epoch_<N>`` snapshot best-effort."""
        stem = Path(self.config.checkpoint_dir) / f"epoch_{epoch_no}"
        for suffix in (SAFETENSORS_SUFFIX, META_SUFFIX, EXTRA_STATE_SUFFIX, LEGACY_SUFFIX):
            target = stem.with_name(stem.name + suffix)
            if target.exists():
                try:
                    target.unlink()
                    self.logger.debug(f"Removed old checkpoint sidecar: {target}")
                except OSError as e:
                    self.logger.warning(f"Could not remove {target}: {e}")

    # ---- load side --------------------------------------------------------

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None,
        scheduler: LRScheduler | None,
        scaler: torch.amp.GradScaler | None,
        device: torch.device,
        expected_model_config: dict | None = None,
    ) -> tuple[int, float]:
        if not self.config.resume_from_checkpoint:
            return 0, float("inf")

        ckp_path = Path(self.config.resume_from_checkpoint)
        # Probe both layouts at this stem: the legacy .pt (if the
        # caller pointed there explicitly) AND the v2 split trio
        # (the v2 default). ``load_checkpoint_payload`` handles the
        # priority — legacy wins when the .pt exists, split otherwise.
        legacy_path, safetensors_path, meta_path, extra_state_path = _resolve_checkpoint_paths(ckp_path)
        if not (legacy_path.exists() or safetensors_path.exists() or meta_path.exists() or extra_state_path.exists()):
            self.logger.warning(
                f"Checkpoint file not found: {ckp_path} (checked legacy + split layouts). Starting from scratch."
            )
            return 0, float("inf")

        try:
            payload = load_checkpoint_payload(ckp_path)
        except (OSError, EOFError, pickle.UnpicklingError, TypeError) as e:
            # File-level corruption: the payload itself could not be read
            # (truncated pickle stream, wrong-kind ``.pt``, ``[None]*n``
            # padding) — the designed degrade-to-scratch path.
            # Nothing has been mutated yet, so degrading to scratch here is
            # GENUINE. Structural mismatches (model shape, optimizer /
            # scheduler / scaler layout) NEVER land here — they are raised
            # after payload load, below. ``pickle`` is imported at module top
            # for the weights_only fix.
            self.logger.error(f"Failed to read checkpoint from {ckp_path}: {e}")
            self.logger.warning("Starting from scratch due to checkpoint read error.")
            return 0, float("inf")

        if payload is None:
            self.logger.warning(
                f"Checkpoint at {ckp_path} exists but no recognized layout "
                "(neither split nor legacy). Starting from scratch."
            )
            return 0, float("inf")

        # Resume-config compatibility check: a checkpoint
        # whose model_config differs from the current run's in any
        # architecture-defining field means the user changed config (or
        # pointed at the wrong checkpoint). Weight-shape mismatches are
        # already caught loudly by load_model_state_dict below; but
        # non-tensor differences the state dict CANNOT expose — a changed
        # scheduler_type/epochs/warmup whose state dict loads fine into a
        # differently-shaped scheduler, a changed tokenizer re-tokenizing
        # with new ids over old weights, a changed max_seq_len — would
        # silently corrupt or mis-anneal the resume. Compare the fields
        # that define the model/schedule and refuse loudly.
        if expected_model_config and payload.get("model_config"):
            mismatches = _model_config_mismatches(expected_model_config, payload["model_config"])
            if mismatches:
                raise ValueError(
                    f"Checkpoint at {ckp_path} was saved with a different model/schedule "
                    f"configuration: {', '.join(f'{k} (ckpt {v[0]}, now {v[1]})' for k, v in mismatches.items())}. "
                    "Refusing to silently resume with mismatched weights — fix the config or "
                    "point resume_from_checkpoint at a compatible checkpoint."
                )

        # ---- State loads (outside the corrupt-file swallow) ----
        # The payload is on hand, so a failure to load its state is a
        # STRUCTURAL mismatch (changed scheduler_type, changed param-group
        # count, resized model), NOT a corrupt file. These must fail loud:
        # the pre-fix code swallowed them and returned ``(0, inf)`` while the
        # model (and maybe the optimizer) already held checkpoint weights — a
        # silent hybrid resume that trains a fresh epoch counter / LR
        # schedule over resumed weights. Data-integrity errors
        # from the payload read are also re-raised by the
        # loader and must NOT be swallowed here. Optimizer / scheduler /
        # scaler load BEFORE the model so a mismatch leaves the model
        # untouched — never a partially-resumed model behind a scratch notice.
        # Advance the ordering so the no-op note is unambiguous: PP matters
        # for the optimizer branch but is_pp is already imported at module top.
        try:
            if (
                (is_pp(model) or is_pp3d(model))
                and optimizer is not None
                and payload.get("optimizer_state") is not None
            ):
                opt_blob = payload["optimizer_state"]
                if isinstance(opt_blob, dict) and "__pp_stage_states__" in opt_blob:
                    #: per-stage optimizer states persisted by GLOBAL
                    # rank (under PP+DP, different DP groups hold
                    # distinct moments, so stage index alone would collide).
                    # Each rank restores its own slice — same topology on
                    # resume, so global keys still match the same stages.
                    my_state = opt_blob["__pp_stage_states__"].get(str(dist.get_rank()))
                    if my_state is None:
                        self.logger.warning(
                            f"PP resume: checkpoint has no optimizer state for rank {dist.get_rank()}; "
                            "the optimizer starts from fresh state."
                        )
                    else:
                        optimizer.load_state_dict(my_state)
                else:
                    # Pre checkpoint: the optimizer slot holds rank 0's
                    # partial state which cannot be replayed onto every stage.
                    # Restore weights + scheduler but reset the optimizer, with
                    # an explicit note — never a silent dropout.
                    self.logger.warning(
                        "PP resume: checkpoint predates per-stage optimizer checkpointing "
                        "— restoring weights and scheduler but starting the "
                        "optimizer from fresh state."
                    )
            elif optimizer is not None and payload.get("optimizer_state") is not None:
                optimizer.load_state_dict(payload["optimizer_state"])
            if scheduler is not None and payload.get("scheduler_state") is not None:
                scheduler.load_state_dict(payload["scheduler_state"])
            if scaler is not None and payload.get("scaler_state") is not None:
                scaler.load_state_dict(payload["scaler_state"])
            load_model_state_dict(model, payload["model_state"])
        except (
            OSError,
            RuntimeError,
            KeyError,
            ValueError,
            EOFError,
            pickle.UnpicklingError,
            TypeError,
        ) as e:
            # Any mismatch here is structural, not a corrupt file. A model
            # shape/key incompatibility (the most common) means the current
            # model architecture does NOT match the checkpoint. Fail loud
            # instead of silently training a full run that looks like a
            # resume but discards all prior state.
            message = str(e)
            if isinstance(e, RuntimeError) and any(
                marker in message
                for marker in ("size mismatch", "unexpected key", "missing key", "too many dimensions")
            ):
                raise RuntimeError(
                    f"Checkpoint at {ckp_path} does not match the current model architecture "
                    f"(state_dict mismatch: {message.splitlines()[0]}). Refusing to silently "
                    "restart from scratch — fix the model config or point resume_from_checkpoint "
                    "at a compatible checkpoint."
                ) from e
            if isinstance(e, ValueError) and ("inconsistent" in message or "Refusing to silently resume" in message):
                raise
            raise RuntimeError(
                f"Checkpoint at {ckp_path} state could not be loaded into the "
                f"current model/optimizer/scheduler/scaler ({type(e).__name__}: "
                f"{message.splitlines()[0]}). Refusing to silently resume with "
                "partially-loaded state — fix the config or point "
                "resume_from_checkpoint at a compatible checkpoint."
            ) from e

        start_epoch = payload["epoch"] + 1
        best_loss = payload.get("best_loss", float("inf"))
        self.best_loss = best_loss
        self.loaded_extra_state = payload.get("extra_state")
        self.logger.info(
            f"✅ Resumed training from epoch {start_epoch} using checkpoint {ckp_path} "
            f"(format={payload.get('format_version', 'legacy')})"
        )
        return start_epoch, best_loss
