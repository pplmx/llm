"""Pipeline parallelism (PP) training helpers.

``parallel_strategy="pp"`` partitions a :class:`~llm.models.DecoderModel`
across ``pp_size`` pipeline stages — one stage per rank — and drives the
training / validation schedule with
:class:`torch.distributed.pipelining.ScheduleGPipe` (RIL DEC-049 / TASK-210).

v1 layout is a **pure pipeline**: the whole world is the pipeline
(``pp_size == world_size``, no data-parallel dimension). Every stage rank
loads the *same* data shard (mirroring pure-TP v1), and the schedule feeds
each batch into stage 0 and flows activations forward / gradients backward
stage by stage. Only the LAST stage computes the loss (LM shift + CE, see
:func:`lm_shift_loss`); the engine broadcasts it so every rank holds the same
value for metric reduction and ``save_best``.

The model is split at ``transformer_blocks``:

* stage 0        = embedding layer + the first block chunk
* stages 1..N-2  = middle block chunks
* stage N-1      = last block chunk + final norm + lm_head

Each stage's forward is semantics-preserving — it uses the exact block call
the monolithic :meth:`DecoderModel.forward` uses (``use_cache=False``,
``attn_mask=None``, causal by the block's own default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn

__all__ = [
    "build_pipeline_model",
    "is_pp",
    "lm_shift_loss",
    "partition_decoder_model",
]


def lm_shift_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    scale: float = 1.0,
) -> torch.Tensor:
    """Causal-LM loss: ``logits[s]`` predicts ``labels[s + 1]``.

    Mirrors :meth:`llm.training.tasks.lm_task.LMTask.train_step` so a
    pipeline's last stage computes exactly the serial training loss
    (shift + ``CrossEntropyLoss(ignore_index=-100)``). ``scale`` carries the
    engine's gradient-accumulation factor (the schedule has already
    backpropagated by the time ``step()`` returns, so the 1/accum_steps
    scaling the standard loop applies to the loss must instead be folded
    into the schedule's own loss).
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if shift_labels.numel() == 0:
        raise ValueError("sequence length must be > 1 for next-token language modeling")
    return criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) / scale


class _PipelineStage(nn.Module):
    """One sequential chunk of a DecoderModel, with a bound ``forward``.

    ``blocks`` holds the stage's slice of ``transformer_blocks`` (assigned by
    :func:`partition_decoder_model`); ``embedding_layer`` (stage 0),
    ``final_norm`` and ``lm_head`` (last stage) are attached as ``None`` when
    not owned so the same forward body handles every stage. ``forward`` takes
    ``input_ids`` on stage 0 and hidden states elsewhere, and returns hidden
    states — or ``logits`` on the last stage.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embedding_layer: nn.Module | None = None
        self.final_norm: nn.Module | None = None
        self.lm_head: nn.Module | None = None
        self.blocks: nn.ModuleList = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding_layer(x, start_pos=0, position_ids=None) if self.embedding_layer is not None else x
        for _i, blk in enumerate(self.blocks):
            # The exact training call DecoderModel.forward makes with
            # use_cache=False: causal by the block's own is_causal.
            h = blk(h, attn_mask=None, is_causal=None, use_cache=False, start_pos=0, layer_idx=None)
        if self.final_norm is not None:
            h = self.final_norm(h)
        if self.lm_head is not None:
            h = self.lm_head(h)
        return h


def partition_decoder_model(model: nn.Module, num_stages: int) -> list[nn.Module]:
    """Split a DecoderModel-family model into ``num_stages`` sequential stage modules.

    The split happens at ``transformer_blocks`` (stage 0 embeds + owns the
    first chunk; the last stage owns the final norm and lm_head), so each
    stage's forward takes a hidden-state tensor and returns one — except
    stage 0 (takes ``input_ids``) and the last stage (returns ``logits``).
    The stage modules *reference* the original model's submodules (same
    parameter objects), so distributed initialisation via
    ``broadcast_parameters`` on the full model works unchanged.

    Raises:
        NotImplementedError: for models without ``transformer_blocks``, with
            ``use_alibi``, with ``attn_impl='flash_attn'``, or with gradient
            checkpointing enabled — PP v1 does not cover these (kept out; the
            bit-exact parity against a serial reference is verified at the
            dropout=0 test configuration — with dropout > 0 each stage draws
            its own RNG stream and the parity is approximate, which is the
            standard GPipe caveat).
    """
    am = cast(Any, model)  # nn.Module attr access types as Tensor|Module; we know what we built
    if not (isinstance(getattr(am, "transformer_blocks", None), nn.ModuleList) and len(am.transformer_blocks) > 0):
        raise NotImplementedError(
            "Pipeline parallelism v1 partitions DecoderModel-family models that expose "
            f"a non-empty 'transformer_blocks' ModuleList, got {type(model).__name__}."
        )
    if getattr(am, "alibi", None) is not None:
        raise NotImplementedError(
            "Pipeline parallelism v1 does not partition ALiBi models (the shared bias "
            "module would be duplicated across stage state dicts)."
        )
    if getattr(am, "gradient_checkpointing", False):
        raise NotImplementedError(
            "Pipeline parallelism v1 does not support gradient checkpointing (the "
            "checkpointed block call differs from the plain call the stage uses)."
        )
    if getattr(am, "attn_impl", None) == "flash_attn":
        # PP v1 is fp32-only (the engine refuses use_amp), and the flash_attn
        # kernel rejects fp32 — running it here would crash on the first stage
        # forward with an opaque torch backend error instead of this setup
        # refusal (review MEDIUM on DEC-049/TASK-210).
        raise NotImplementedError(
            "Pipeline parallelism v1 does not support attn_impl='flash_attn' (the flash kernel "
            "requires half precision, but PP v1 is fp32-only). Use attn_impl='mha', 'sdpa' or "
            "'mla' for now."
        )

    blocks = am.transformer_blocks
    num_layers = len(blocks)
    boundaries = [num_layers * s // num_stages for s in range(num_stages + 1)]

    stages: list[nn.Module] = []
    for s in range(num_stages):
        lo, hi = boundaries[s], boundaries[s + 1]
        stage = _PipelineStage()
        if s == 0:
            stage.embedding_layer = am.embedding_layer
        stage.blocks = blocks[lo:hi]  # fresh ModuleList; blocks keep their own refs
        if s == num_stages - 1:
            stage.final_norm = am.final_norm
            stage.lm_head = am.lm_head
        stages.append(stage)

    return stages


def _build_name_maps(stage_module: nn.Module, lo: int) -> dict[str, str]:
    """Map a stage's state-dict keys onto the original model's global names.

    Stage-local keys are ``embedding_layer.*`` / ``final_norm.*`` /
    ``lm_head.*`` verbatim, and ``blocks.{j}.*`` which become
    ``transformer_blocks.{lo + j}.*``.
    """
    mapping: dict[str, str] = {}
    for key in stage_module.state_dict():
        if key.startswith("blocks."):
            rest = key[len("blocks.") :]
            if "." not in rest:
                raise RuntimeError(
                    f"unexpected stage state-dict key {key!r}: a block key must carry a "
                    "parameter/buffer suffix, not name the block module itself."
                )
            idx_str, suffix = rest.split(".", 1)
            mapping[key] = f"transformer_blocks.{lo + int(idx_str)}.{suffix}"
        else:
            mapping[key] = key
    return mapping


@dataclass
class _PPRuntime:
    """Per-rank PP metadata attached to the stage module as ``model._pp``."""

    stage_index: int
    num_stages: int
    group: dist.ProcessGroup | None  # None = the world group
    schedule: Any
    stage_module: nn.Module
    local_to_global: dict[str, str] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def last_stage_rank(self) -> int:
        return self.num_stages - 1

    def broadcast_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Broadcast a loss scalar from the last stage to every stage rank.

        Only the last stage computes the loss (``step``/``eval`` return it
        there); every other rank holds a detached zero. The engine needs the
        same value on every rank so ``reduce_mean`` (divisor = world) and
        ``save_best`` see the true pipeline loss.
        """
        t = loss.detach().to(self.device)
        dist.broadcast(t, src=self.last_stage_rank, group=self.group)
        return t

    def full_state_dict(self) -> dict[str, Any]:
        """Cross-stage collect: every stage's owned params into one global-named dict.

        Each rank gathers its own slice (global names via ``local_to_global``)
        and ``all_gather_object`` over the PP group, then every rank merges
        the union — a collective exactly like the TP full-state-dict gather,
        so the checkpoint manager's "every shard must enter it" contract
        (RIL ISS-186) applies.
        """
        mine = {self.local_to_global[k]: v for k, v in self.stage_module.state_dict().items()}
        gathered: list[dict[str, Any]] = [{} for _ in range(self.num_stages)]
        dist.all_gather_object(gathered, mine, group=self.group)
        merged: dict[str, Any] = {}
        for d in gathered:
            for name, value in d.items():
                # Stages own DISJOINT global-name slices; a collision means a
                # stage emitted a non-owned (e.g. duplicated) name and one
                # copy would be silently dropped from the checkpoint — fail
                # loud instead (review LOW on DEC-049/TASK-210).
                if name in merged:
                    raise RuntimeError(
                        f"pipeline stage overlap: global name {name!r} was produced by more than one "
                        "stage; the model is not partitioned into disjoint slices."
                    )
                merged[name] = value
        return merged

    def scatter_load(self, state_dict: dict[str, Any]) -> None:
        """Load the global-named full state dict, keeping only this stage's slice."""
        sub = {local: state_dict[global_name] for local, global_name in self.local_to_global.items()}
        self.stage_module.load_state_dict(sub, strict=True)


def build_pipeline_model(model: nn.Module, num_stages: int, device: torch.device) -> nn.Module:
    """Partition ``model`` into this rank's pipeline stage and schedule.

    v1 maps stages one-to-one onto ranks: ``stage_index == dist.get_rank()``
    and ``num_stages == world_size`` (a pure pipeline, no PP+DP hybrid yet).
    Weight initialisation is handled by the caller (the engine broadcasts
    rank 0's full model before wrapping); the stage modules *share* the
    model's parameter objects, so the broadcast reaches them.

    Returns the per-rank stage module tagged with ``_pp`` metadata
    (:func:`is_pp`), ready to be driven by ``runtime.step`` /
    ``runtime.eval`` from the engine training loop.
    """
    if num_stages <= 1:
        raise ValueError("pipeline parallelism needs num_stages > 1")
    stage_index = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    partitions = partition_decoder_model(model, num_stages)
    stage_module = partitions[stage_index].to(device)

    from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

    # Dynamic metadata mode: pass no ``input_args`` example so the stage's
    # recv/send buffers are inferred from the first real microbatch — batch
    # sequence lengths are dataset-dependent and a static example would
    # over-constrain them (the metadata validation rejects shape drift).
    pipeline_stage = PipelineStage(stage_module, stage_index, num_stages, device, input_args=None)
    schedule = ScheduleGPipe(pipeline_stage, n_microbatches=1, loss_fn=lm_shift_loss)

    group = dist.group.WORLD if dist.is_available() and dist.is_initialized() else None  # type: ignore[attr-defined]
    runtime = _PPRuntime(
        stage_index=stage_index,
        num_stages=num_stages,
        group=group,
        schedule=schedule,
        stage_module=stage_module,
        device=device,
    )
    am = cast(Any, model)
    boundaries = [len(am.transformer_blocks) * s // num_stages for s in range(num_stages + 1)]
    runtime.local_to_global = _build_name_maps(stage_module, boundaries[stage_index])

    cast(Any, stage_module)._pp = runtime
    return stage_module


def is_pp(model: nn.Module) -> bool:
    """True when ``model`` is a pipeline stage (tagged by ``build_pipeline_model``).

    Makes checkpoint save/load a cross-stage collective, like
    :func:`llm.training.distributed.parallel.is_tp` for the TP identity.
    """
    return getattr(model, "_pp", None) is not None
