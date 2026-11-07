import logging
from pathlib import Path

import pytest
import torch

from llm.training.core.checkpoint import load_checkpoint_payload
from llm.training.core.config import CheckpointConfig
from llm.training.core.utils import CheckpointManager, Logger, LoggingConfig
from tests.support.devices import DEFAULT_DEVICE

# Use real components where easy, mocks where interface is all that matters for file I/O
# We can use TinyModel from conftest if we import it, or just a dummy state dict holder.


class DummyState:
    def state_dict(self):
        return {"a": 1}

    def load_state_dict(self, state):
        pass


@pytest.fixture
def checkpoint_manager(tmp_path):
    config = CheckpointConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"), save_interval=1, keep_last_n=2, save_best=True
    )
    logging_config = LoggingConfig(log_level="DEBUG")
    logger = Logger(rank=0, config=logging_config)

    return CheckpointManager(config, rank=0, logger=logger)


def test_checkpoint_manager_init(checkpoint_manager):
    assert Path(checkpoint_manager.config.checkpoint_dir).exists()


def test_save_checkpoint(checkpoint_manager):
    model = DummyState()
    optimizer = DummyState()
    scheduler = DummyState()
    scaler = DummyState()

    checkpoint_manager.save_checkpoint(
        epoch=0, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, loss=1.0
    )

    # v2 split layout: each checkpoint name writes three sidecars.
    ckpt_dir = Path(checkpoint_manager.config.checkpoint_dir)
    for stem in ("latest", "epoch_1", "best"):
        assert (ckpt_dir / f"{stem}.safetensors").exists()
        assert (ckpt_dir / f"{stem}.meta.json").exists()
        assert (ckpt_dir / f"{stem}.extra_state.pt").exists()


def test_checkpoint_rotation(checkpoint_manager):
    model = DummyState()
    optimizer = DummyState()
    scheduler = DummyState()
    scaler = DummyState()

    ckpt_dir = Path(checkpoint_manager.config.checkpoint_dir)

    # Save 3 checkpoints (keep_last_n=2)
    for i in range(3):
        checkpoint_manager.save_checkpoint(i, model, optimizer, scheduler, scaler, loss=1.0)

    # Expect: epoch_2 and epoch_3 sidecars. epoch_1 sidecars should be deleted.
    assert (ckpt_dir / "epoch_3.safetensors").exists()
    assert (ckpt_dir / "epoch_3.meta.json").exists()
    assert (ckpt_dir / "epoch_3.extra_state.pt").exists()
    assert (ckpt_dir / "epoch_2.safetensors").exists()
    assert not (ckpt_dir / "epoch_1.safetensors").exists()
    assert not (ckpt_dir / "epoch_1.meta.json").exists()
    assert not (ckpt_dir / "epoch_1.extra_state.pt").exists()


def test_rotation_prunes_prior_runs_epochs_on_resume(tmp_path):
    """``keep_last_n`` must bound the epoch snapshots on DISK,
    not just the current process's in-memory list.

    A fresh CheckpointManager (a resumed run) starts with an empty
    ``checkpoints_saved`` list, so it only ever prunes the epochs IT writes —
    prior runs' ``epoch_N`` trios were left to accumulate forever. Retention
    must consider everything already on disk.
    """
    from llm.training.core.utils import CheckpointManager, Logger, LoggingConfig

    config = CheckpointConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        save_interval=1,
        keep_last_n=2,
        save_best=False,
    )
    logger = Logger(rank=0, config=LoggingConfig(log_level="DEBUG"))
    model = DummyState()

    def _epoch_stems():
        return {p.name.split(".")[0] for p in (tmp_path / "checkpoints").iterdir() if p.name.startswith("epoch_")}

    # Run 1: three snapshots; keep_last_n=2 leaves epoch_2 + epoch_3 on disk.
    CheckpointManager(config, rank=0, logger=logger).save_checkpoint(0, model, None, None, None, loss=1.0)
    CheckpointManager(config, rank=0, logger=logger).save_checkpoint(1, model, None, None, None, loss=1.0)
    CheckpointManager(config, rank=0, logger=logger).save_checkpoint(2, model, None, None, None, loss=1.0)

    # "Run 2": a RESUME is a fresh process — a new manager, empty in-memory
    # list, continuing the epoch counter. One more snapshot must push the
    # total back down to keep_last_n=2, deleting run 1's stale trios.
    mgr2 = CheckpointManager(config, rank=0, logger=logger)
    assert len(mgr2.checkpoints_saved) == 0, "resumed manager must not inherit run 1's in-memory list"
    mgr2.save_checkpoint(3, model, None, None, None, loss=1.0)

    stems = _epoch_stems()
    assert len(stems) <= 2, f"resumed run left {len(stems)} epoch trios on disk (keep_last_n=2): {sorted(stems)}"
    assert "epoch_4" in stems, "newest epoch snapshot must survive retention"


def test_load_checkpoint_saves_extra_state(checkpoint_manager):
    model = DummyState()
    optimizer = DummyState()
    scheduler = DummyState()
    scaler = DummyState()

    checkpoint_manager.save_checkpoint(
        0,
        model,
        optimizer,
        scheduler,
        scaler,
        loss=0.25,
        extra_state={"stream_data": {"0": {"line_index": 42, "token_buffer": [1, 2]}}},
    )

    # Pass the legacy "latest.pt" path; the loader should resolve to
    # the split layout (latest.safetensors + latest.meta.json +
    # latest.extra_state.pt) when no legacy .pt exists at the path.
    checkpoint_manager.config.resume_from_checkpoint = str(Path(checkpoint_manager.config.checkpoint_dir) / "latest.pt")
    _, best_loss = checkpoint_manager.load_checkpoint(model, optimizer, scheduler, scaler, device=DEFAULT_DEVICE)

    assert best_loss == 0.25
    assert checkpoint_manager.loaded_extra_state["stream_data"]["0"]["line_index"] == 42


def test_save_checkpoint_includes_model_config(checkpoint_manager):
    model = DummyState()
    optimizer = DummyState()
    scheduler = DummyState()
    scaler = DummyState()

    checkpoint_manager.save_checkpoint(
        0,
        model,
        optimizer,
        scheduler,
        scaler,
        loss=0.5,
        model_config={"vocab_size": 100, "hidden_size": 16, "num_layers": 1, "num_heads": 2, "max_seq_len": 16},
    )

    # model_config lives in meta.json under the v2 layout.
    import json

    with (Path(checkpoint_manager.config.checkpoint_dir) / "latest.meta.json").open() as f:
        meta = json.load(f)
    assert meta["model_config"]["vocab_size"] == 100
    assert meta["format_version"] == "2.0"


def test_load_checkpoint(checkpoint_manager):
    # Setup: Save one
    model = DummyState()
    optimizer = DummyState()
    scheduler = DummyState()
    scaler = DummyState()

    checkpoint_manager.save_checkpoint(0, model, optimizer, scheduler, scaler, loss=0.5)

    # Enable resume — pass the legacy .pt stem; the manager finds
    # the split layout at the same stem.
    checkpoint_manager.config.resume_from_checkpoint = str(Path(checkpoint_manager.config.checkpoint_dir) / "latest.pt")

    start_epoch, best_loss = checkpoint_manager.load_checkpoint(
        model, optimizer, scheduler, scaler, device=DEFAULT_DEVICE
    )

    assert start_epoch == 1
    assert best_loss == 0.5


def test_load_checkpoint_mismatched_architecture_raises(checkpoint_manager):
    """Resume from a checkpoint whose model architecture differs from the
    current one must RAISE with a clear error, not silently restart from
    scratch.

    A shape/key mismatch means the user changed config (e.g. hidden_size) or
    pointed at the wrong checkpoint; silently discarding the resume trains a
    full run the user believes continued the old one.
    """
    import torch.nn as nn

    saved = nn.Linear(4, 4)
    checkpoint_manager.save_checkpoint(0, saved, None, None, None, loss=0.5)
    checkpoint_manager.config.resume_from_checkpoint = str(Path(checkpoint_manager.config.checkpoint_dir) / "latest.pt")

    # Different architecture (5x5 vs saved 4x4) → load_state_dict size mismatch.
    different_arch = nn.Linear(5, 5)
    with pytest.raises(RuntimeError, match=r"checkpoint|architecture|config"):
        checkpoint_manager.load_checkpoint(different_arch, None, None, None, device=DEFAULT_DEVICE)


def test_save_stamps_matching_save_id_into_meta_and_extra(checkpoint_manager):
    """every save stamps a shared ``save_id`` (and the epoch)
    into BOTH the meta.json and the extra_state.pt sidecar. The loader uses
    the pair to prove the three sidecars came from one atomic save."""
    import json

    checkpoint_manager.save_checkpoint(0, DummyState(), DummyState(), DummyState(), DummyState(), loss=1.0)
    ckpt_dir = Path(checkpoint_manager.config.checkpoint_dir)

    meta = json.loads((ckpt_dir / "latest.meta.json").read_text())
    extra = torch.load(ckpt_dir / "latest.extra_state.pt", map_location="cpu", weights_only=False)

    assert "save_id" in meta, "meta.json must carry the per-save save_id"
    assert meta["save_id"] == extra["save_id"], "meta + extra_state must share the save_id"
    assert meta["epoch"] == extra["epoch"], "meta + extra_state must share the epoch"


def test_load_rejects_inconsistent_trio_stale_extra(checkpoint_manager):
    """a crash interrupted between writing the (new) weights
    sidecar and the (old) extra_state/optimizer must NOT resume silently with
    a mismatched trio — new weights paired with stale optimizer/scheduler
    state would silently re-train/skip. Loading must raise instead."""
    import torch as _torch  # noqa: F401
    from torch import nn

    saved = nn.Linear(4, 4)
    checkpoint_manager.save_checkpoint(0, saved, DummyState(), DummyState(), DummyState(), loss=0.5)
    ckpt_dir = Path(checkpoint_manager.config.checkpoint_dir)

    # Simulate a crash-mid-save: rewrite the extra_state.pt from a DIFFERENT
    # save (a stale generation) — the loader must detect the save_id/epoch
    # mismatch and refuse to silently resume.
    stale_extra = ckpt_dir / "stale.extra_state.pt"
    torch.save(
        {"optimizer_state": None, "scheduler_state": None, "scaler_state": None, "extra_state": None}, stale_extra
    )
    stale = torch.load(stale_extra, map_location="cpu", weights_only=False)
    # Copy the stale blob into the real extra_state path but with a different
    # save_id + epoch than the meta sidecar.
    stale["save_id"] = "STALE-SAVE"
    stale["epoch"] = 999
    torch.save(stale, str(ckpt_dir / "latest.extra_state.pt"))

    checkpoint_manager.config.resume_from_checkpoint = str(ckpt_dir / "latest.pt")
    with pytest.raises(ValueError, match="inconsistent"):
        checkpoint_manager.load_checkpoint(saved, None, None, None, device=DEFAULT_DEVICE)


def test_load_optimizer_mismatch_raises_not_silent_hybrid_resume(checkpoint_manager):
    """a structural optimizer mismatch (changed param-group
    layout) must raise loudly instead of silently "starting from scratch"
    AFTER the model has already been mutated.

    The pre-fix code loaded model → optimizer → scheduler → scaler inside
    the same try that swallowed non-marker load errors, then returned
    ``(0, inf)``. A ``ValueError: loaded state dict has a different number
    of parameter groups`` from the optimizer was swallowed — so training
    ran "from scratch" with a fresh epoch counter over the ALREADY-RESUMED
    weights: a silent hybrid. The load must refuse loudly, and the model
    must NOT be mutated when the mismatch is in optimizer/scheduler/scaler.
    """
    from torch import nn
    from torch.optim import AdamW

    saved = nn.Linear(4, 1)
    # Save with a 2-group optimizer so the checkpoint carries two groups.
    save_opt = AdamW(
        [
            {"params": [saved.weight], "lr": 1e-3},
            {"params": [saved.bias], "lr": 1e-2},
        ]
    )
    save_opt.zero_grad()
    loss = saved(torch.randn(2, 4)).pow(2).mean()
    loss.backward()
    save_opt.step()
    checkpoint_manager.save_checkpoint(0, saved, save_opt, DummyState(), DummyState(), loss=0.5)
    checkpoint_manager.config.resume_from_checkpoint = str(Path(checkpoint_manager.config.checkpoint_dir) / "latest.pt")

    # Resume into a 1-group optimizer — structurally incompatible.
    target_model = nn.Linear(4, 1)
    target_opt = AdamW(target_model.parameters())
    # Snapshot the target's own (random) weights before the load attempt.
    target_model.eval()
    with torch.no_grad():
        initial_weight = target_model.weight.detach().clone()
        initial_bias = target_model.bias.detach().clone()

    with pytest.raises(RuntimeError, match="state could not be loaded"):
        checkpoint_manager.load_checkpoint(target_model, target_opt, None, None, device=DEFAULT_DEVICE)

    # The failure happened in the OPTIMIZER load, before the model load —
    # the model must be untouched (not the hybrid "scratch but resumed").
    assert torch.equal(target_model.weight.detach(), initial_weight), (
        "optimizer mismatch must not mutate the model into a hybrid resume state"
    )
    assert torch.equal(target_model.bias.detach(), initial_bias)


def test_load_rejects_resume_with_mismatched_model_config(checkpoint_manager):
    """resuming must not silently accept a checkpoint whose
       architecture-defining config differs from the current run.

       A changed ``vocab_size`` / ``hidden_size`` / ``max_seq_len`` / MoE knobs
       between runs means the (shape-compatible or not) weights were trained for
       a different model; resuming silently would either corrupt the run (when
       shapes happen to line up) or retrain from scratch. Fail loudly, like the
    shape guard and the trio guard.
    """
    import torch.nn as nn

    saved = nn.Linear(4, 4)
    checkpoint_manager.save_checkpoint(
        0, saved, None, None, None, loss=0.5, model_config={"hidden_size": 32, "num_layers": 2}
    )
    checkpoint_manager.config.resume_from_checkpoint = str(Path(checkpoint_manager.config.checkpoint_dir) / "latest.pt")

    # Current run uses a DIFFERENT arch (hidden_size 64) → refuse.
    expected = {"hidden_size": 64, "num_layers": 2}
    with pytest.raises(ValueError, match="different model"):
        checkpoint_manager.load_checkpoint(
            saved, None, None, None, device=DEFAULT_DEVICE, expected_model_config=expected
        )


def test_load_resume_matching_model_config_is_accepted(checkpoint_manager):
    """guard must not reject a compatible resume: identical
    arch-defining fields load normally, and non-arch drift (dropout etc.) is
    ignored."""
    import torch.nn as nn

    saved = nn.Linear(4, 4)
    checkpoint_manager.save_checkpoint(
        0,
        saved,
        None,
        None,
        None,
        loss=0.5,
        model_config={"hidden_size": 64, "num_layers": 2, "dropout": 0.3},
    )
    checkpoint_manager.config.resume_from_checkpoint = str(Path(checkpoint_manager.config.checkpoint_dir) / "latest.pt")

    # Identical arch + different (non-arch) dropout → still loads.
    expected = {"hidden_size": 64, "num_layers": 2, "dropout": 0.0}
    start_epoch, _ = checkpoint_manager.load_checkpoint(
        saved, None, None, None, device=DEFAULT_DEVICE, expected_model_config=expected
    )
    assert start_epoch == 1


def test_load_rejects_fresh_weights_with_stale_meta_extra(checkpoint_manager):
    """the meta↔extra cross-check alone could NOT catch the exact
    crash window a fresh weights file sits beside a stale-but-mutually-
    consistent meta/extra pair. The weights sidecar now carries the save_id,
    so a weights-vs-meta mismatch must raise instead of silently resuming with
    new weights + old optimizer/epoch."""
    from torch import nn

    saved = nn.Linear(4, 4)
    checkpoint_manager.save_checkpoint(0, saved, None, None, None, loss=0.5)
    ckpt_dir = Path(checkpoint_manager.config.checkpoint_dir)

    # Simulate "crash before meta/extra write": rewrite weights.safetensors
    # with a DIFFERENT save_id while leaving meta.json + extra_state.pt (which
    # still agree with each other) untouched. The meta↔extra check passes; the
    # new weights-stamp check must catch it.
    from safetensors.torch import load_file, save_file

    weights = load_file(str(ckpt_dir / "latest.safetensors"))
    save_file(weights, str(ckpt_dir / "latest.safetensors"), metadata={"save_id": "FRESH-SAVE", "epoch": "0"})

    checkpoint_manager.config.resume_from_checkpoint = str(ckpt_dir / "latest.pt")
    with pytest.raises(ValueError, match=r"weights\.safetensors save_id"):
        checkpoint_manager.load_checkpoint(saved, None, None, None, device=DEFAULT_DEVICE)


def test_load_corrupt_legacy_checkpoint_degrades_not_crashes(checkpoint_manager):
    """a truncated/corrupt ``.pt`` raises EOFError/
    UnpicklingError (NOT in the old OSError/RuntimeError/KeyError/ValueError
    except tuple), so it used to crash with a raw traceback instead of the
    designed degrade-to-scratch path. It must now log + start from scratch."""
    import torch.nn as nn

    saved = nn.Linear(4, 4)
    checkpoint_manager.save_checkpoint(0, saved, None, None, None, loss=0.5)
    ckpt_dir = Path(checkpoint_manager.config.checkpoint_dir)
    # Leave only a corrupt legacy .pt at the resume path.
    corrupt = ckpt_dir / "corrupt.pt"
    corrupt.write_bytes(b"this is not a valid pickle blob\x00\x01" + b"x" * 64)
    checkpoint_manager.config.resume_from_checkpoint = str(corrupt)

    # Loader resolution: corrupt.pt exists → _load_legacy_checkpoint raises
    # pickle.UnpicklingError → load_checkpoint catches it (via the extended
    # except tuple) → logs error and returns scratch (0, inf).
    start_epoch, best_loss = checkpoint_manager.load_checkpoint(saved, None, None, None, device=DEFAULT_DEVICE)
    assert start_epoch == 0
    assert best_loss == float("inf")


# --- Resume fidelity: what round-trips exactly, and what does not ---


def test_resume_restores_weights_and_optimizer_bit_exactly(tmp_path):
    """Weights, AdamW moments and param_group hyperparameters round-trip exactly.

    Kept as a contract because the *observable* effect of resume is that training
    continues from the same point: a bit of drift here would silently reset the
    optimizer (its moments are what carry momentum across the boundary).
    """
    import torch
    from safetensors.torch import load_file

    from llm.models.decoder import DecoderModel

    batch = torch.randint(0, 32, (2, 8))
    torch.manual_seed(0)
    model = DecoderModel(vocab_size=32, hidden_size=16, num_layers=1, num_heads=2, max_seq_len=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    config = CheckpointConfig(checkpoint_dir=str(tmp_path), save_interval=1)
    logger = logging.getLogger("test-resume")
    logger.addHandler(logging.NullHandler())
    manager = CheckpointManager(config, rank=0, logger=logger)
    manager.save_checkpoint(
        1,
        model,
        optimizer,
        None,
        None,
        1.0,
        model_config={"vocab_size": 32, "hidden_size": 16, "num_layers": 1, "num_heads": 2, "max_seq_len": 16},
    )

    payload = load_checkpoint_payload(tmp_path / "latest")
    saved_weights = load_file(str(tmp_path / "latest.safetensors"))
    for key, value in model.state_dict().items():
        assert torch.equal(value, saved_weights[key]), f"weight {key} did not round-trip bit-exactly"

    live = optimizer.state_dict()
    restored = payload["optimizer_state"]
    assert len(restored["state"]) == len(live["state"])
    for index, fields in live["state"].items():
        for field, value in fields.items():
            assert torch.equal(value, restored["state"][index][field]), (
                f"optimizer state[{index}][{field}] did not round-trip"
            )
    # Hyperparameters must survive too, or the resumed run uses a different lr/etc.
    for key, value in live["param_groups"][0].items():
        if key == "params":
            continue
        assert restored["param_groups"][0][key] == value, f"param_group[{key}] changed across the checkpoint"


def test_checkpoint_does_not_store_rng_state(tmp_path):
    """The RNG state is NOT part of a checkpoint — this is why resume is not bit-reproducible.

    ``CheckpointManager`` persists model / optimizer / scheduler / scaler /
    ``extra_state``. With no RNG state, dropout, shuffling and augmentation
    resume on a fresh random stream, so "3 steps + save + resume 3" differs
    from "6 steps straight". ``docs/guides/checkpoints.md`` states that
    limitation; this test keeps the claim honest by asserting the absence
    directly, so adding RNG persistence later will fail here and prompt a doc
    update rather than leaving a stale caveat.

    Verified behaviourally too: the same split run in ``eval()`` (dropout off)
    DOES match bit-for-bit, which is what pins the divergence on randomness
    rather than on the checkpoint.
    """
    import torch as _torch

    from llm.models.decoder import DecoderModel

    config = {"vocab_size": 32, "hidden_size": 16, "num_layers": 1, "num_heads": 2, "max_seq_len": 16}
    batches = [_torch.randint(0, 32, (2, 8)) for _ in range(6)]

    def _make():
        _torch.manual_seed(42)
        return DecoderModel(**config)

    def _step(model, optimizer, batch):
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _run(eval_mode: bool):
        """Run 6 steps straight, and 3+3 with a checkpoint round trip between."""
        _torch.manual_seed(7)
        straight = _make()
        straight_opt = _torch.optim.AdamW(straight.parameters(), lr=1e-3)
        if eval_mode:
            straight.eval()
        for batch in batches:
            _step(straight, straight_opt, batch)

        split_dir = tmp_path / ("eval" if eval_mode else "train")
        _torch.manual_seed(7)
        split = _make()
        split_opt = _torch.optim.AdamW(split.parameters(), lr=1e-3)
        if eval_mode:
            split.eval()
        for batch in batches[:3]:
            _step(split, split_opt, batch)

        logger = logging.getLogger("test-rng")
        logger.addHandler(logging.NullHandler())
        CheckpointManager(
            CheckpointConfig(checkpoint_dir=str(split_dir), save_interval=1), rank=0, logger=logger
        ).save_checkpoint(1, split, split_opt, None, None, 1.0, model_config=config)
        payload = load_checkpoint_payload(split_dir / "latest")

        # The checkpoint carries no RNG state.
        for key in payload:
            assert "rng" not in key.lower(), f"checkpoint unexpectedly carries RNG under {key!r}"

        _torch.manual_seed(999)  # a different stream on the resume path
        resumed = _make()
        resumed.load_state_dict(payload["model_state"])
        if eval_mode:
            resumed.eval()
        resumed_opt = _torch.optim.AdamW(resumed.parameters(), lr=1e-3)
        resumed_opt.load_state_dict(payload["optimizer_state"])
        for batch in batches[3:]:
            _step(resumed, resumed_opt, batch)

        return all(
            _torch.equal(a, b)
            for a, b in zip(straight.state_dict().values(), resumed.state_dict().values(), strict=True)
        )

    # Without dropout the two paths agree exactly: the checkpoint itself is faithful.
    assert _run(eval_mode=True), "eval-mode resume must reproduce the uninterrupted run bit-for-bit"
    # With dropout the resume path draws different masks, so it diverges — the
    # documented caveat. If this ever becomes True, RNG persistence was added.
    assert not _run(eval_mode=False), "dropout resume is expected to diverge; update docs/guides/checkpoints.md"
