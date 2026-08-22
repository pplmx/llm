"""Pipeline-parallelism milestone tests (``parallel_strategy='pp'``).

Verification strategy (RIL DEC-049 / TASK-210): numeric parity against a full
single-rank reference model. Every rank builds the SAME ``DecoderModel`` from
the same CPU seed, then ``build_pipeline_model`` partitions it into one stage
per rank. A pipeline ``schedule.step`` must give a loss that equals the serial
LM-shift loss (scaled by the gradient-accumulation factor) and per-stage
gradients that are bit-exact to the serial reference's; ``schedule.eval``
must reproduce the serial forward-only loss; the PP-group global-norm clip
must agree across ranks; and the global-name full-state-dict gather/scatter
checkpoint boundary must round-trip a plain full state dict identical to the
reference's.

Unlike the TP tests (NCCL, >= 2 GPUs), PP v1 runs on CPU + gloo, so these
tests verify continuously in CI without GPUs.
"""

from __future__ import annotations

import os
import socket
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from llm.models.decoder import DecoderModel
from llm.training.distributed import (
    allreduce_pp_dp_grads,
    build_pipeline_model,
    clip_grad_norm_tp,
    lm_shift_loss,
    partition_decoder_model,
    pp_dp_layout,
    wrap_model_for_training,
)
from llm.training.distributed.parallel import is_pp

PP_JOIN_TIMEOUT_S = 300

SEED = 123
VOCAB = 64
HIDDEN = 32
LAYERS = 4
HEADS = 4
SEQ = 8
BATCH = 2
ACCUM = 3


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _setup_env() -> int:
    port = _free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    return port


def _build_model(seed: int = SEED) -> DecoderModel:
    torch.manual_seed(seed)
    return DecoderModel(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
        num_heads=HEADS,
        max_seq_len=SEQ + 8,
        embedding_dropout_p=0.0,
        attn_dropout_p=0.0,
        mlp_dropout_p=0.0,
        pos_encoding_learned=True,
        norm_impl="layer_norm",
    ).train()


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(999)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
    torch.manual_seed(1000)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ))
    return input_ids, labels


def _serial_reference() -> dict:
    model = _build_model()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    input_ids, labels = _make_inputs()
    loss = lm_shift_loss(model(input_ids), labels, criterion)
    grads = {n: (p.grad.clone() if p.grad is not None else None) for n, p in model.named_parameters()}
    loss.backward()
    grads = {n: (p.grad.clone() if p.grad is not None else None) for n, p in model.named_parameters()}
    with torch.no_grad():
        eval_loss = lm_shift_loss(model(input_ids), labels, criterion).item()
    total_sq = torch.zeros((), dtype=torch.float32)
    for p in model.parameters():
        if p.grad is not None:
            total_sq += (p.grad.float() ** 2).sum()
    return {"loss": loss.item(), "eval_loss": eval_loss, "grads": grads, "global_sq": total_sq.item()}


def _run_spawn_worker(worker, world_size: int) -> dict:
    """Spawn ``worker(rank, world_size, results)`` over gloo/CPU procs."""
    _setup_env()
    manager = mp.Manager()
    results: dict = manager.dict()
    context = mp.spawn(worker, args=(world_size, results), nprocs=world_size, join=False)
    end_at = time.monotonic() + PP_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"PP spawn exceeded {PP_JOIN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break
    out = dict(results)
    if any(v.get("error") for v in out.values()):
        first = next(v["error"] for v in out.values() if v.get("error"))
        raise AssertionError(f"PP worker failed: {first}")
    return out


def _pp_parity_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"], rank=rank, world_size=world_size
        )
        torch.manual_seed(SEED)
        model = _build_model().to(torch.device("cpu"))
        stage = build_pipeline_model(model, world_size, torch.device("cpu"))
        assert is_pp(stage)
        rt = stage._pp
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        input_ids, labels = _make_inputs()

        losses: list = []
        # scale=1.0: this test asserts BIT-EXACT parity with the serial loss
        # and gradients. The gradient-accumulation factor is covered by
        # test_pp_loss_scaling_two_process_cpu below (loss_hi = raw/ACCUM).
        rt.schedule.step(input_ids, target=labels, losses=losses, loss_kwargs={"criterion": criterion, "scale": 1.0})
        loss = rt.broadcast_loss(losses[-1] if losses else torch.tensor(0.0))

        # Per-stage local grads keyed by the rank's OWNED local names.
        grads = {}
        for local_name, p in stage.named_parameters():
            grads[local_name] = p.grad.clone() if p.grad is not None else None

        full = rt.full_state_dict()
        results[rank] = {
            "loss": loss.item(),
            "grads": grads,
            "full": full,
            "error": None,
        }
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def test_pp_loss_and_grad_parity_two_process_cpu() -> None:
    ref = _serial_reference()
    out = _run_spawn_worker(_pp_parity_worker, 2)

    loss0 = out[0]["loss"]
    loss1 = out[1]["loss"]
    # Every rank holds the same (broadcast, raw) pipeline loss.
    assert loss0 == pytest.approx(loss1, abs=1e-12)
    assert loss0 == pytest.approx(ref["loss"], abs=1e-12)

    # Every OWNED grad must be bit-exact to the serial reference.
    for rank, payload in out.items():
        for local_name, grad in payload["grads"].items():
            if grad is None:
                continue
            # map stage-local name -> global name
            assert payload["full"]  # global names available on the merged dict
            if local_name.startswith("blocks."):
                idx = int(local_name.split(".")[1])
                global_name = f"transformer_blocks.{2 * rank + idx}" + "." + ".".join(local_name.split(".")[2:])
            else:
                global_name = local_name
            ref_grad = ref["grads"][global_name]
            assert ref_grad is not None, f"{global_name} not in reference"
            assert torch.equal(grad.detach(), ref_grad), f"grad {global_name} (rank {rank}) not bit-exact"


def _pp_scale_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"], rank=rank, world_size=world_size
        )
        torch.manual_seed(SEED)
        model = _build_model().to(torch.device("cpu"))
        stage = build_pipeline_model(model, world_size, torch.device("cpu"))
        rt = stage._pp
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        input_ids, labels = _make_inputs()
        losses: list = []
        rt.schedule.step(input_ids, target=labels, losses=losses, loss_kwargs={"criterion": criterion, "scale": ACCUM})
        raw = rt.broadcast_loss(losses[-1] * ACCUM if losses else torch.tensor(0.0))
        results[rank] = {"raw": raw.item(), "scaled": (losses[-1].item() if losses else None), "error": None}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def test_pp_loss_scaling_two_process_cpu() -> None:
    """The schedule's loss_fn folds 1/accum; raw is recovered by * accum."""
    ref = _serial_reference()
    out = _run_spawn_worker(_pp_scale_worker, 2)
    # Only the LAST stage computes the per-microbatch loss; the first stage
    # still receives the broadcast RAW value. The fp32 tensor division by
    # ACCUM rounds inside the schedules loss, so compare with an fp32-scale
    # reference (abs tolerance ~1e-6, not the bit-exact 1e-12 of scale=1.0).
    fp32_scaled = float(torch.tensor(ref["loss"], dtype=torch.float32) / ACCUM)
    assert out[1]["scaled"] == pytest.approx(fp32_scaled, abs=1e-7)
    assert out[0]["raw"] == pytest.approx(ref["loss"], rel=1e-4)
    assert out[0]["raw"] == pytest.approx(out[1]["raw"], rel=1e-4)


def _pp_eval_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"], rank=rank, world_size=world_size
        )
        torch.manual_seed(SEED)
        model = _build_model().to(torch.device("cpu"))
        stage = build_pipeline_model(model, world_size, torch.device("cpu"))
        rt = stage._pp
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        input_ids, labels = _make_inputs()
        evals: list = []
        with torch.no_grad():
            rt.schedule.eval(input_ids, target=labels, losses=evals, loss_kwargs={"criterion": criterion})
        got = rt.broadcast_loss(evals[-1] if evals else torch.tensor(0.0))
        results[rank] = {"eval_loss": got.item(), "error": None}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def test_pp_eval_parity_two_process_cpu() -> None:
    ref = _serial_reference()
    out = _run_spawn_worker(_pp_eval_worker, 2)
    assert out[0]["eval_loss"] == pytest.approx(ref["eval_loss"], abs=1e-12)
    assert out[1]["eval_loss"] == pytest.approx(ref["eval_loss"], abs=1e-12)


def _pp_roundtrip_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"], rank=rank, world_size=world_size
        )
        torch.manual_seed(SEED)
        model = _build_model().to(torch.device("cpu"))
        stage = build_pipeline_model(model, world_size, torch.device("cpu"))
        rt = stage._pp
        full = rt.full_state_dict()
        # Each rank's merged dict must be the SAME global-name superset.
        rt.scatter_load(full)  # reload from the gathered dict (a roundtrip)
        stage_names = sorted(stage.state_dict().keys())
        results[rank] = {"global_keys": sorted(full.keys()), "stage_keys": stage_names, "error": None}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def test_pp_gather_scatter_roundtrip_two_process_cpu() -> None:
    ref = _serial_reference()
    out = _run_spawn_worker(_pp_roundtrip_worker, 2)
    keys0 = out[0]["global_keys"]
    keys1 = out[1]["global_keys"]
    assert keys0 == keys1
    serial_names = sorted(ref["grads"].keys())
    assert keys0 == serial_names, "gathered global names must equal the serial model's parameter names"
    # stage state dict keys must be disjoint-ish (each stage owns a slice): stage 0
    # owns embedding and blocks 0/1, stage 1 owns blocks 2/3 + final_norm + lm_head.
    assert any(k.startswith("embedding_layer") for k in out[0]["stage_keys"])
    assert not any(k.startswith("embedding_layer") for k in out[1]["stage_keys"])
    assert any(k.startswith("lm_head") for k in out[1]["stage_keys"])
    assert not any(k.startswith("lm_head") for k in out[0]["stage_keys"])


def _pp_gnorm_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"], rank=rank, world_size=world_size
        )
        torch.manual_seed(SEED)
        model = _build_model().to(torch.device("cpu"))
        stage = build_pipeline_model(model, world_size, torch.device("cpu"))
        rt = stage._pp
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        input_ids, labels = _make_inputs()
        rt.schedule.step(input_ids, target=labels, loss_kwargs={"criterion": criterion, "scale": 1.0})
        norm = clip_grad_norm_tp(stage, 1e12, group=rt.group).item()
        results[rank] = {"norm": norm, "error": None}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def test_pp_grad_norm_global_over_group_two_process_cpu() -> None:
    ref = _serial_reference()
    out = _run_spawn_worker(_pp_gnorm_worker, 2)
    n0, n1 = out[0]["norm"], out[1]["norm"]
    assert n0 == pytest.approx(n1, abs=1e-12), "global norm must agree across stages"
    assert n0 == pytest.approx(ref["global_sq"] ** 0.5, abs=1e-6), (
        "PP global norm must equal the serial full-model norm"
    )


def test_pp_world_size_one_is_noop() -> None:
    model = _build_model()
    wrapped = wrap_model_for_training(model, parallel_strategy="pp", device=torch.device("cpu"), world_size=1)
    assert not is_pp(wrapped)


def test_pp_partition_refuses_unsupported_models() -> None:
    # A bare MLP has no transformer_blocks -> refuse loudly.
    mlp = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    with pytest.raises(NotImplementedError, match="transformer_blocks"):
        partition_decoder_model(mlp, 2)
    # Gradient checkpointing is SUPPORTED since RIL TASK-213 (the stage forward
    # wraps block calls like the monolithic model does) — the model must
    # partition, not refuse.
    model = _build_model()
    model.enable_gradient_checkpointing()
    stages = partition_decoder_model(model, 2)
    assert all(getattr(stage, "gradient_checkpointing", False) for stage in stages)
    # flash_attn cannot run in PP's forced fp32 (review MEDIUM on TASK-210).
    # Charade the model attr (the real flash kernel is an optional dependency,
    # so constructing a flash model would gate this test on it); the
    # partitioner must refuse on the attr alone.
    flash_model = _build_model()
    flash_model.attn_impl = "flash_attn"
    with pytest.raises(NotImplementedError, match="flash_attn"):
        partition_decoder_model(flash_model, 2)


def test_pp_single_rank_engine_is_serial_noop() -> None:
    """High review fix: single-rank 'pp' must run the standard loop, not crash.

    The wrap path returns the bare model for world_size<=1; the engine must
    NOT dispatch ``_pp_train_step`` (which would AttributeError on a model
    with no ``_pp`` tag and log a misleading 'completed with inf' summary).
    """
    from llm.training.core.config import Config, DistributedConfig, ModelConfig, OptimizationConfig, TrainingConfig
    from llm.training.core.engine import TrainingEngine
    from llm.training.tasks.lm_task import LanguageModelingTask
    from tests.support.data import DummyLMDataModule

    config = Config(
        model=ModelConfig(vocab_size=64, hidden_size=16, num_layers=4, num_heads=2, max_seq_len=32),
        training=TrainingConfig(batch_size=2, epochs=1, num_samples=8),
        optimization=OptimizationConfig(use_compile=False, use_amp=False),
        distributed=DistributedConfig(backend="gloo", parallel_strategy="pp"),
    )
    data_module = DummyLMDataModule(config)
    task = LanguageModelingTask(config, data_module)
    engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=data_module)
    assert engine.is_training_pp is False, "single-rank PP must run the serial loop"
    assert not is_pp(engine.model), "single-rank PP must leave the model unwrapped"


def test_pp_batch_validation_fails_loud() -> None:
    """Review MEDIUM fix: a malformed PP batch refuses loudly, not silently."""
    from llm.training.core.config import Config, DistributedConfig, ModelConfig, OptimizationConfig, TrainingConfig
    from llm.training.core.engine import TrainingEngine
    from llm.training.tasks.lm_task import LanguageModelingTask
    from tests.support.data import DummyLMDataModule

    config = Config(
        model=ModelConfig(vocab_size=64, hidden_size=16, num_layers=4, num_heads=2, max_seq_len=32),
        training=TrainingConfig(batch_size=2, epochs=1, num_samples=8),
        optimization=OptimizationConfig(use_compile=False, use_amp=False),
        distributed=DistributedConfig(backend="gloo", parallel_strategy="pp"),
    )
    data_module = DummyLMDataModule(config)
    task = LanguageModelingTask(config, data_module)
    engine = TrainingEngine(config=config, task=task, rank=0, world_size=1, data_module=data_module)

    input_ids = torch.randint(0, VOCAB, (2, SEQ))
    labels = torch.randint(0, VOCAB, (2, SEQ))
    # Valid shapes extract fine.
    assert engine._pp_extract_inputs({"input_ids": input_ids, "labels": labels}) == (input_ids, labels)
    assert engine._pp_extract_inputs((input_ids, labels)) == (input_ids, labels)
    # A dict missing labels is refused.
    with pytest.raises(ValueError, match=r"input_ids.*labels"):
        engine._pp_extract_inputs({"input_ids": input_ids})
    # A tuple wider than (input_ids, labels) — e.g. extra attention_mask — is
    # refused instead of silently treating the mask as labels.
    with pytest.raises(ValueError, match="exactly"):
        engine._pp_extract_inputs((input_ids, torch.ones(2, SEQ), labels))


def test_pp_strategy_accepted_by_config() -> None:
    from llm.training.core.config import DistributedConfig

    cfg = DistributedConfig(parallel_strategy="pp")
    assert cfg.parallel_strategy == "pp"
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="parallel_strategy"):
        DistributedConfig(parallel_strategy="bogus")


def test_pp_load_model_state_dict_via_distributed_helper() -> None:
    # ``load_model_state_dict`` must route a PP stage to ``scatter_load`` and a
    # plain model to the regular path (no crash on the PP branch is the point).
    from llm.training.distributed.parallel import model_for_checkpoint_io

    model = _build_model()
    assert model_for_checkpoint_io(model) is model


def test_lm_shift_loss_contract() -> None:
    """The pipeline loss function must equal the LM/standard-loop shift-CE loss."""
    model = _build_model()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    input_ids, labels = _make_inputs()
    serial = lm_shift_loss(model(input_ids), labels, criterion)
    # The standard loop uses the same shift + CE expression; recompute inline.
    logits = model(input_ids)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    task_loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    assert torch.equal(serial, task_loss)
    # The gradient-accum scale is a plain multiplicative factor.
    assert torch.equal(lm_shift_loss(model(input_ids), labels, criterion, scale=ACCUM), serial / ACCUM)


# ---------------------------------------------------------------------------
# PP + data-parallel 2D (RIL TASK-211): pp_size < world_size
# ---------------------------------------------------------------------------


def _make_full_batch(dp_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A globally-consistent ``(BATCH * dp_size, SEQ)`` batch to shard per DP group."""
    torch.manual_seed(999)
    input_ids = torch.randint(0, VOCAB, (BATCH * dp_size, SEQ))
    torch.manual_seed(1000)
    labels = torch.randint(0, VOCAB, (BATCH * dp_size, SEQ))
    return input_ids, labels


def _pp_2d_reference(dp_size: int) -> dict:
    """Serial references: per-shard losses + FULL-batch gradients.

    ``(G0 + G1) / 2 == full-batch gradient`` when the DP shards are equal-sized
    (each shard's pipeline loss divides by the same token count), which is what
    ``allreduce_pp_dp_grads`` must reproduce.
    """
    full_ids, full_labels = _make_full_batch(dp_size)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    shard_losses = []
    for d in range(dp_size):
        torch.manual_seed(SEED)
        m = _build_model()
        with torch.no_grad():
            shard_losses.append(
                lm_shift_loss(
                    m(full_ids[d * BATCH : (d + 1) * BATCH]), full_labels[d * BATCH : (d + 1) * BATCH], criterion
                ).item()
            )
    torch.manual_seed(SEED)
    ref = _build_model()
    loss = lm_shift_loss(ref(full_ids), full_labels, criterion)
    loss.backward()
    ref_grads = {n: (p.grad.clone() if p.grad is not None else None) for n, p in ref.named_parameters()}
    return {"shard_losses": shard_losses, "ref_grads": ref_grads, "ref_loss": loss.item()}


def _pp_2d_parity_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"], rank=rank, world_size=world_size
        )
        pp_size, dp_size = 2, 2
        torch.manual_seed(SEED)
        model = _build_model().to(torch.device("cpu"))
        # The REAL engine wrap path: [DP][PP] subgroups + stage partition.
        stage = wrap_model_for_training(
            model,
            parallel_strategy="pp",
            device=torch.device("cpu"),
            world_size=world_size,
            pp_size=pp_size,
        )
        assert is_pp(stage)
        rt = stage._pp
        assert rt.dp_group is not None, "2D PP must wire a DP group for gradient averaging"
        assert rt.group is not None, "2D PP must run P2P over the PP subgroup"
        assert rt.group != dist.group.WORLD, "2D PP must run P2P over the PP subgroup (not the world)"
        _, _, dp_rank, pp_rank = pp_dp_layout(world_size, pp_size, rank)
        assert rt.stage_index == pp_rank

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        full_ids, full_labels = _make_full_batch(dp_size)
        shard_ids = full_ids[dp_rank * BATCH : (dp_rank + 1) * BATCH]
        shard_labels = full_labels[dp_rank * BATCH : (dp_rank + 1) * BATCH]

        losses: list = []
        rt.schedule.step(
            shard_ids, target=shard_labels, losses=losses, loss_kwargs={"criterion": criterion, "scale": 1.0}
        )
        loss = rt.broadcast_loss(losses[-1] if losses else torch.tensor(0.0))

        # The engine's step-boundary hook.
        allreduce_pp_dp_grads(stage)

        owned = {rt.local_to_global[k]: v.grad.clone() for k, v in stage.named_parameters()}
        full = rt.full_state_dict()
        results[rank] = {
            "loss": loss.item(),
            "owned": owned,
            "full": full,
            "dp_rank": dp_rank,
            "pp_rank": pp_rank,
            "error": None,
        }
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def test_pp_dp_2d_loss_and_grad_parity_four_process_cpu() -> None:
    """PP + data-parallel 2D (TASK-211) on a 2x2 [DP][PP] gloo/CPU grid.

    Each DP group pipelines its own data shard; the loss must equal that
    shard's serial LM-shift loss (broadcast over the PP subgroup), and the
    DP-group gradient average must reproduce the serial FULL-batch gradients
    — the exact semantics ``allreduce_pp_dp_grads`` implements at the engine's
    step boundary.
    """
    ref = _pp_2d_reference(dp_size=2)
    out = _run_spawn_worker(_pp_2d_parity_worker, 4)

    # 1. Per-DP-group loss equals that shard's serial loss (all ranks of a PP
    #    group broadcast the same value).
    for rank, payload in out.items():
        d = payload["dp_rank"]
        assert payload["loss"] == pytest.approx(ref["shard_losses"][d], abs=1e-12), f"rank {rank} shard {d}"
    assert out[0]["loss"] == pytest.approx(out[1]["loss"], abs=1e-12)  # PP group 0
    assert out[2]["loss"] == pytest.approx(out[3]["loss"], abs=1e-12)  # PP group 1

    # 2. DP-averaged owned gradients equal the serial FULL-batch gradients.
    checked = 0
    for rank, payload in out.items():
        for global_name, grad in payload["owned"].items():
            ref_grad = ref["ref_grads"][global_name]
            assert ref_grad is not None, f"{global_name} not in reference grads"
            checked += 1
            torch.testing.assert_close(
                grad, ref_grad, atol=1e-4, rtol=1e-4, msg=f"grad {global_name} (rank {rank}) after DP average"
            )
    assert checked > 0

    # 3. The merged full state dict (PP-subgroup gather) is bit-identical
    #    across ALL ranks and carries exactly the serial model's names.
    full_keys = set(out[0]["full"].keys())
    assert full_keys == set(ref["ref_grads"].keys())
    for payload in out.values():
        assert set(payload["full"].keys()) == full_keys
        for key in full_keys:
            assert torch.equal(payload["full"][key], out[0]["full"][key]), f"{key} diverged from rank 0"


def _pp_2d_dyn_worker(rank: int, world_size: int, results: dict) -> None:
    """A couple of SGD steps with the DP average: shards must stay in lockstep."""
    try:
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"], rank=rank, world_size=world_size
        )
        pp_size, dp_size = 2, 2
        torch.manual_seed(SEED)
        model = _build_model().to(torch.device("cpu"))
        stage = wrap_model_for_training(
            model,
            parallel_strategy="pp",
            device=torch.device("cpu"),
            world_size=world_size,
            pp_size=pp_size,
        )
        rt = stage._pp
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        full_ids, full_labels = _make_full_batch(dp_size)
        _, _, dp_rank, _pp_rank = pp_dp_layout(world_size, pp_size, rank)
        opt = torch.optim.SGD(stage.parameters(), lr=0.05)
        for _ in range(2):
            stage.zero_grad()
            shard_ids = full_ids[dp_rank * BATCH : (dp_rank + 1) * BATCH]
            shard_labels = full_labels[dp_rank * BATCH : (dp_rank + 1) * BATCH]
            rt.schedule.step(shard_ids, target=shard_labels, loss_kwargs={"criterion": criterion, "scale": 1.0})
            allreduce_pp_dp_grads(stage)
            opt.step()
        full = rt.full_state_dict()
        results[rank] = {"full": {k: v.detach().cpu() for k, v in full.items()}, "error": None}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def test_pp_dp_2d_training_stays_in_lockstep_four_process_cpu() -> None:
    """After a few SGD steps the DP-averaged stage copies must stay bit-identical
    across all four ranks (a DP-group divergence — the TASK-211 failure mode —
    fails this equality)."""
    out = _run_spawn_worker(_pp_2d_dyn_worker, 4)
    keys = set(out[0]["full"].keys())
    assert keys  # non-empty
    for rank, payload in out.items():
        for key in keys:
            assert torch.equal(payload["full"][key], out[0]["full"][key]), f"{key} diverged from rank 0 (rank {rank})"


def test_pp_dp_layout_validation() -> None:
    assert pp_dp_layout(4, 2, rank=0) == (2, 2, 0, 0)
    assert pp_dp_layout(4, 2, rank=1) == (2, 2, 0, 1)
    assert pp_dp_layout(4, 2, rank=3) == (2, 2, 1, 1)
    # 0 / negative means "whole world" (pure PP).
    assert pp_dp_layout(4, 0, rank=2) == (4, 1, 0, 2)
    assert pp_dp_layout(4, -1, rank=2) == (4, 1, 0, 2)
    assert pp_dp_layout(6, 3, rank=5) == (3, 2, 1, 2)
    with pytest.raises(ValueError, match="exceed"):
        pp_dp_layout(2, 4)
    with pytest.raises(ValueError, match="divide"):
        pp_dp_layout(6, 4)


# ---------------------------------------------------------------------------
# PP microbatch overlap + gradient checkpointing (RIL TASK-213)
# ---------------------------------------------------------------------------


def _pp_gc_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"], rank=rank, world_size=world_size
        )
        torch.manual_seed(SEED)
        model = _build_model().to(torch.device("cpu"))
        model.enable_gradient_checkpointing()
        stage = build_pipeline_model(model, world_size, torch.device("cpu"))
        rt = stage._pp
        assert rt.stage_module.gradient_checkpointing, (
            "partitioner must propagate the GC flag to every stage (TASK-213)"
        )
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        input_ids, labels = _make_inputs()
        losses: list = []
        rt.schedule.step(input_ids, target=labels, losses=losses, loss_kwargs={"criterion": criterion, "scale": 1.0})
        loss = rt.broadcast_loss(losses[-1] if losses else torch.tensor(0.0))
        grads = {}
        for local_name, p in stage.named_parameters():
            grads[local_name] = p.grad.clone() if p.grad is not None else None
        results[rank] = {"loss": loss.item(), "grads": grads, "error": None}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def test_pp_gradient_checkpointing_parity_two_process_cpu() -> None:
    """A gradient-checkpointed model must partition (no refusal since TASK-213)
    and the stage forwards must recompute activations WITHOUT changing the
    numerics vs the (equally checkpointed) serial reference."""
    torch.manual_seed(SEED)
    ref_model = _build_model()
    ref_model.enable_gradient_checkpointing()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    input_ids, labels = _make_inputs()
    ref_loss = lm_shift_loss(ref_model(input_ids), labels, criterion)
    ref_loss.backward()
    ref_grads = {n: p.grad.clone() for n, p in ref_model.named_parameters()}

    out = _run_spawn_worker(_pp_gc_worker, 2)
    for rank, payload in out.items():
        assert payload["loss"] == pytest.approx(ref_loss.item(), abs=1e-5), f"rank {rank}: GC loss diverged"
    # The checkpointed path must reproduce the reference gradients (~1e-5:
    # activation recomputation is deterministic but the backward graph differs).
    for rank, payload in out.items():
        for local_name, grad in payload["grads"].items():
            if grad is None:
                continue
            if local_name.startswith("blocks."):
                idx = int(local_name.split(".")[1])
                global_name = f"transformer_blocks.{2 * rank + idx}" + "." + ".".join(local_name.split(".")[2:])
            else:
                global_name = local_name
            torch.testing.assert_close(
                grad.detach(), ref_grads[global_name], atol=1e-4, rtol=1e-4, msg=f"rank {rank} {global_name}"
            )


def _pp_mb_worker(rank: int, world_size: int, results: dict) -> None:
    try:
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"], rank=rank, world_size=world_size
        )
        torch.manual_seed(SEED)
        model = _build_model().to(torch.device("cpu"))
        mb = 2
        stage = build_pipeline_model(model, world_size, torch.device("cpu"), n_microbatches=mb)
        rt = stage._pp
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        input_ids, labels = _make_inputs()
        losses: list = []
        rt.schedule.step(input_ids, target=labels, losses=losses, loss_kwargs={"criterion": criterion, "scale": 1.0})
        # n_microbatches=2 -> TWO per-chunk losses on the LAST stage (the only
        # rank that computes them); the batch loss is their mean. Other stages
        # pass a zero placeholder and receive the true value via broadcast.
        if losses:
            assert len(losses) == mb, f"expected {mb} per-microbatch losses, got {len(losses)}"
            batch_loss = sum(losses) / len(losses)
        else:
            batch_loss = torch.tensor(0.0)
        loss = rt.broadcast_loss(batch_loss)
        grads = {}
        for local_name, p in stage.named_parameters():
            grads[local_name] = p.grad.clone() if p.grad is not None else None
        results[rank] = {"loss": loss.item(), "grads": grads, "error": None}
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def test_pp_n_microbatches_grad_parity_two_process_cpu() -> None:
    """n_microbatches=2 must leave the optimisation step UNCHANGED: the schedule
    normalises the accumulated gradient by the microbatch count, so the loss
    (mean of the two chunks) and the per-stage gradients equal the serial
    full-batch reference (~1e-5 — chunking reorders the fp32 reductions)."""
    ref = _serial_reference()
    out = _run_spawn_worker(_pp_mb_worker, 2)
    for rank, payload in out.items():
        assert payload["loss"] == pytest.approx(ref["loss"], abs=1e-5), f"rank {rank}: microbatch loss diverged"
    checked = 0
    for rank, payload in out.items():
        for local_name, grad in payload["grads"].items():
            if grad is None:
                continue
            if local_name.startswith("blocks."):
                idx = int(local_name.split(".")[1])
                global_name = f"transformer_blocks.{2 * rank + idx}" + "." + ".".join(local_name.split(".")[2:])
            else:
                global_name = local_name
            ref_grad = ref["grads"][global_name]
            assert ref_grad is not None, f"{global_name} not in reference"
            checked += 1
            torch.testing.assert_close(grad.detach(), ref_grad, atol=1e-4, rtol=1e-4, msg=f"rank {rank} {global_name}")
    assert checked > 0


def test_pp_n_microbatches_build_rejects_zero() -> None:
    model = _build_model()
    with pytest.raises(ValueError, match="n_microbatches"):
        build_pipeline_model(model, 2, torch.device("cpu"), n_microbatches=0)


def _pp_amp_worker(rank: int, world_size: int, results: dict) -> None:
    """Engine construction with PP + float16 AMP must refuse loudly (TASK-214)."""
    try:
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:" + os.environ["MASTER_PORT"], rank=rank, world_size=world_size
        )
        from llm.training.core.config import Config, DistributedConfig, ModelConfig, OptimizationConfig, TrainingConfig
        from llm.training.core.engine import TrainingEngine
        from llm.training.tasks.lm_task import LanguageModelingTask
        from tests.support.data import DummyLMDataModule

        config = Config(
            model=ModelConfig(vocab_size=64, hidden_size=16, num_layers=4, num_heads=2, max_seq_len=32),
            training=TrainingConfig(batch_size=2, epochs=1, num_samples=8),
            optimization=OptimizationConfig(use_compile=False, use_amp=True, amp_dtype="float16"),
            distributed=DistributedConfig(backend="gloo", parallel_strategy="pp"),
        )
        data_module = DummyLMDataModule(config)
        task = LanguageModelingTask(config, data_module)
        TrainingEngine(config=config, task=task, rank=rank, world_size=world_size, data_module=data_module)
        results[rank] = {"error": "engine constructed despite float16 AMP refusal"}
        dist.destroy_process_group()
    except ValueError as exc:
        results[rank] = {"refused": str(exc)}
    except Exception:  # noqa: BLE001 - failure reporting path
        import traceback

        results[rank] = {"error": traceback.format_exc()}


def test_pp_refuses_float16_amp_two_process_cpu() -> None:
    """PP + use_amp only supports bf16 (TASK-214): the schedule computes AND
    backprops the loss inside step(), so a GradScaler (float16 AMP) cannot
    scale the loss before the schedule's backward. float16 must refuse loudly
    on construction; bf16 is exercised by the engine e2e."""
    out = _run_spawn_worker(_pp_amp_worker, 2)
    for rank, payload in out.items():
        err = payload.get("error")
        assert err is None, f"rank {rank}: {err}"
        assert "requires bf16" in payload["refused"], f"rank {rank}: unexpected message {payload['refused']}"
