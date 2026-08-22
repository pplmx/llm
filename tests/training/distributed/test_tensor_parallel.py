"""Tensor-parallelism milestone tests (``parallel_strategy='tp'``).

Verification strategy (RIL TASK-200 / DEC-045): numeric parity against a full
single-rank reference model. Every rank builds the SAME ``DecoderModel`` from
the same CPU seed (deterministic init), then rank-slices it in place; a
forward with dropout disabled must produce bit-comparable logits, a single
backward must produce full-model-matching gradients (this is what proves the
``_BackwardAllReduce`` / ``_ForwardAllReduce`` wiring is right), and the
full-state-dict gather/scatter checkpoint boundary must roundtrip a plain
full state dict identical to the reference's.

These tests need >= 2 GPUs (NCCL). They are skipped automatically otherwise.
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
    allreduce_dp_grads,
    apply_tensor_parallel,
    clip_grad_norm_tp,
    is_tp,
    load_model_state_dict,
    model_state_dict,
    wrap_model_for_training,
)
from tests.support.devices import all_gpu_devices

TP_MIN_FREE_BYTES = 1 * 1024**3
# Generous-but-bounded: on a shared box hosting another GPU workload the
# multi-rank NCCL rendezvous can take minutes (4-GPU wrap timed out at 180s
# under load once; instant in isolation). A true deadlock still trips these.
TP_JOIN_TIMEOUT_S = 360


def _free_port() -> int:
    """Bind an ephemeral port so consecutive TP tests do not share MASTER_PORT."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _release_parent_cuda_caches() -> None:
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(index):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except RuntimeError, torch.AcceleratorError:
            continue


def _setup_tp_env() -> int:
    port = _free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    return port


def _build_model(
    seed: int = 7,
    device=None,
    layers: int = 2,
    num_kv_heads: int | None = None,
    attn_impl: str = "mha",
    use_rope: bool = False,
    num_experts: int = 0,
    top_k: int = 0,
    mlp_impl: str = "mlp",
) -> DecoderModel:
    torch.manual_seed(seed)
    return DecoderModel(
        vocab_size=64,
        hidden_size=32,
        num_layers=layers,
        num_heads=4,
        max_seq_len=24,
        intermediate_size=64,
        attn_dropout_p=0.0,
        mlp_dropout_p=0.0,
        embedding_dropout_p=0.0,
        num_kv_heads=num_kv_heads,
        qkv_bias=True,
        mlp_bias=True,
        lm_head_bias=True,
        attn_impl=attn_impl,
        use_rope=use_rope,
        num_experts=num_experts,
        top_k=top_k,
        mlp_impl=mlp_impl,  # num_experts > 0 only builds MoE with mlp_impl="moe"
        device=device,
    )


def _gather_tp_grads(model) -> dict[str, torch.Tensor]:
    """Assemble the full-model gradient from per-rank shards (test helper).

    Mirrors the checkpoint gather (``_TPState.gather_full_state_dict``): the
    fused-QKV column shards are stored in local [q,k,v]-block order and must
    be scattered back into the full q/k/v blocks via ``full_index``; MoE
    expert-parallel shards (TASK-207) are rebuilt rank-major by global expert
    index (dead local experts contribute zero).
    """
    tp = model._tp
    full: dict[str, torch.Tensor] = {}
    for key, param in model.named_parameters():
        if param.grad is None:
            continue
        if tp.is_expert_param(key):
            continue  # combined below (needs every rank's block)
        axis = tp.partition.get(key)
        if axis is None:
            full[key] = param.grad.detach().clone().contiguous()
            continue
        idx_all = tp.full_index.get(key)
        if idx_all is not None:
            pieces = [torch.empty_like(param.grad) for _ in range(tp.world_size)]
            dist.all_gather(pieces, param.grad.detach().contiguous(), group=tp.group)
            full_shape = list(param.grad.shape)
            full_shape[axis] = max(int(idx.max().item()) for idx in idx_all) + 1
            full_t = torch.zeros(full_shape, dtype=param.grad.dtype, device=param.grad.device)
            for r, piece in enumerate(pieces):
                full_t.index_copy_(axis, idx_all[r].to(param.grad.device), piece)
            full[key] = full_t
            continue
        pieces = [torch.empty_like(param.grad) for _ in range(tp.world_size)]
        dist.all_gather(pieces, param.grad.detach().contiguous(), group=tp.group)
        full[key] = torch.cat(pieces, dim=axis)
    for prefix, (total, n_local) in tp.expert_shards.items():
        marker = prefix + ".experts."
        by_suffix: dict[str, dict[int, torch.Tensor]] = {}
        for key, param in model.named_parameters():
            if not key.startswith(marker):
                continue
            li_str, suffix = key[len(marker) :].split(".", 1)
            grad = param.grad.detach() if param.grad is not None else torch.zeros_like(param)
            by_suffix.setdefault(suffix, {})[int(li_str)] = grad
        for suffix, grads_by_li in by_suffix.items():
            grads = [grads_by_li[li] for li in range(n_local)]
            block = grads[0].unsqueeze(0) if n_local == 1 else torch.stack(grads)
            pieces = [torch.empty_like(block) for _ in range(tp.world_size)]
            dist.all_gather(pieces, block.contiguous(), group=tp.group)
            for g in range(total):
                owner, li = divmod(g, n_local)
                full[f"{prefix}.experts.{g}.{suffix}"] = pieces[owner][li].detach().clone()
    return full


def _parity_worker(rank: int, world_size: int, device_indices: list[int], results) -> None:
    try:
        device_index = device_indices[rank]
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device_index)
        dev = torch.device(f"cuda:{device_index}")

        # Identical weights on every rank: same CPU seed -> same init.
        ref = _build_model().to(dev)  # full single-rank reference
        model = _build_model().to(dev)  # will be TP-partitioned in place
        model = apply_tensor_parallel(model, process_group=dist.group.WORLD)
        assert is_tp(model)

        x = torch.randint(0, 64, (2, 12), device=dev, dtype=torch.long)

        # --- forward parity (eval: dropout disabled) ---
        ref.eval()
        model.eval()
        with torch.no_grad():
            ref_logits = ref(x)
        train_logits = model(x)  # TP forward is collective; identical every rank
        torch.testing.assert_close(train_logits, ref_logits, atol=1e-5, rtol=1e-5)

        # --- gradient parity: one backward against the same loss ---
        model.train()
        ref.train()
        ref.zero_grad()
        model.zero_grad()
        tp_loss = model(x).float().mean()
        tp_loss.backward()
        # ref backward on identical logits — must run the reference AFTER the
        # TP all-gather is done (NCCL sync) to avoid interleaving collectives.
        ref_loss = ref(x).float().mean()
        ref_loss.backward()

        ref_grads = {k: v.grad for k, v in ref.named_parameters() if v.grad is not None}
        tp_grads = _gather_tp_grads(model)
        assert set(ref_grads.keys()) == set(tp_grads.keys()), (
            f"grad key sets differ: missing {set(ref_grads) - set(tp_grads)}, extra {set(tp_grads) - set(ref_grads)}"
        )
        for key in ref_grads:
            torch.testing.assert_close(
                tp_grads[key], ref_grads[key], atol=1e-4, rtol=1e-4, msg=f"grad mismatch on {key}"
            )

        # --- checkpoint boundary: full gather == reference state dict ---
        full = model_state_dict(model)  # collective; every rank enters
        for key, value in ref.state_dict().items():
            assert key in full, f"gathered state dict missing {key}"
            torch.testing.assert_close(full[key], value, atol=0, rtol=0, msg=f"state mismatch on {key}")

        # --- scatter load roundtrip: load ref full dict, forward again ---
        load_model_state_dict(model, ref.state_dict())
        model.eval()
        with torch.no_grad():
            reload_logits = model(x)
        torch.testing.assert_close(reload_logits, ref_logits, atol=1e-5, rtol=1e-5)

        results[rank] = {"success": True}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}


def _run_parity(world_size: int) -> None:
    _run_spawn_parity(_parity_worker, world_size=world_size)


def _run_spawn_parity(worker, world_size: int) -> None:
    """Spawn ``worker(rank, world_size, device_indices, results)`` over free GPUs.

    Shared harness for the TP parity tests: each worker (re)creates the group,
    runs forward/grad/checkpoint parity vs a full reference, and reports its
    own ``{"success": ..., "error": ...}``; the parent polls with a generous
    timeout (a shared box's NCCL rendezvous can be slow under load) and fails
    loudly on any rank error or timeout.
    """
    gpu_devices = all_gpu_devices(min_free_bytes=TP_MIN_FREE_BYTES)
    if len(gpu_devices) < world_size:
        pytest.skip(f"need at least {world_size} free GPUs")
    device_indices = [device.index for device in gpu_devices[:world_size]]
    _release_parent_cuda_caches()
    _setup_tp_env()

    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(
        worker,
        args=(world_size, device_indices, results),
        nprocs=world_size,
        join=False,
    )

    end_at = time.monotonic() + TP_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"TP parity spawn exceeded {TP_JOIN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break

    for rank in range(world_size):
        assert rank in results, f"rank {rank} produced no result"
        assert results[rank]["success"], f"rank {rank} failed: {results[rank].get('error')}"


@pytest.mark.need_gpu(2)
@pytest.mark.slow
def test_tp_forward_grad_checkpoint_parity_two_gpu():
    """Forward / gradient / checkpoint-gather parity vs a full reference (2 GPUs)."""
    _run_parity(2)


def _flash_parity_worker(
    rank: int, world_size: int, device_indices: list[int], results, use_rope: bool, num_kv_heads: int | None
) -> None:
    """FlashAttention TP parity (TASK-204 slice).

    The TP transform extends to ``attn_impl='flash_attn'`` because it shares
    MHA's projection surface (fused QKV column-parallel over heads + row
    out_proj). The forward runs under bf16 autocast because the flash kernel
    requires half precision, so logits/grads are compared with a bf16-scale
    tolerance rather than the fp32 one the MHA test uses — but the state-dict
    gather compares at atol=0: TP slicing of the fp32 weights must be exact
    regardless of the kernel dtype.
    """
    try:
        device_index = device_indices[rank]
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device_index)
        dev = torch.device(f"cuda:{device_index}")

        ref = _build_model(attn_impl="flash_attn", use_rope=use_rope, num_kv_heads=num_kv_heads).to(dev)
        model = _build_model(attn_impl="flash_attn", use_rope=use_rope, num_kv_heads=num_kv_heads).to(dev)
        model = apply_tensor_parallel(model, process_group=dist.group.WORLD)
        assert is_tp(model)

        x = torch.randint(0, 64, (2, 12), device=dev, dtype=torch.long)

        # --- forward parity under bf16 autocast (flash kernel requirement) ---
        ref.eval()
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ref_logits = ref(x)
            train_logits = model(x)  # TP collective; identical on every rank
        torch.testing.assert_close(train_logits, ref_logits, atol=1e-2, rtol=1e-2)

        # --- gradient parity: one backward against the same loss ---
        model.train()
        ref.train()
        ref.zero_grad()
        model.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tp_loss = model(x).float().mean()
        tp_loss.backward()
        # ref backward after the TP collectives have drained (NCCL ordering).
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ref_loss = ref(x).float().mean()
        ref_loss.backward()

        ref_grads = {k: v.grad for k, v in ref.named_parameters() if v.grad is not None}
        tp_grads = _gather_tp_grads(model)
        assert set(ref_grads.keys()) == set(tp_grads.keys()), (
            f"grad key sets differ: missing {set(ref_grads) - set(tp_grads)}, extra {set(tp_grads) - set(ref_grads)}"
        )
        for key in ref_grads:
            torch.testing.assert_close(
                tp_grads[key], ref_grads[key], atol=1e-2, rtol=1e-2, msg=f"grad mismatch on {key}"
            )

        # --- checkpoint boundary: full gather == reference state dict (fp32, exact) ---
        full = model_state_dict(model)  # collective; every rank enters
        for key, value in ref.state_dict().items():
            assert key in full, f"gathered state dict missing {key}"
            torch.testing.assert_close(full[key], value, atol=0, rtol=0, msg=f"state mismatch on {key}")

        # --- scatter load roundtrip: load ref full dict, forward again ---
        load_model_state_dict(model, ref.state_dict())
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            reload_logits = model(x)
        torch.testing.assert_close(reload_logits, ref_logits, atol=1e-2, rtol=1e-2)

        results[rank] = {"success": True}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}


def _run_flash_parity(world_size: int, *, use_rope: bool = False, num_kv_heads: int | None = None) -> None:
    gpu_devices = all_gpu_devices(min_free_bytes=TP_MIN_FREE_BYTES)
    if len(gpu_devices) < world_size:
        pytest.skip(f"need at least {world_size} free GPUs (flash TP parity)")
    device_indices = [device.index for device in gpu_devices[:world_size]]
    _release_parent_cuda_caches()

    port = _free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(
        _flash_parity_worker,
        args=(world_size, device_indices, results, use_rope, num_kv_heads),
        nprocs=world_size,
        join=False,
    )
    end_at = time.monotonic() + TP_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"flash TP parity spawn exceeded {TP_JOIN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break
    for rank in range(world_size):
        assert rank in results, f"rank {rank} produced no result"
        assert results[rank]["success"], f"rank {rank} failed: {results[rank].get('error')}"


@pytest.mark.need_gpu(2)
@pytest.mark.slow
def test_tp_flash_attn_parity_two_gpu():
    """FlashAttention TP forward/grad/checkpoint parity vs the single-GPU flash reference."""
    _run_flash_parity(2)


@pytest.mark.need_gpu(2)
@pytest.mark.slow
def test_tp_flash_attn_rope_parity_two_gpu():
    """FlashAttention + RoPE TP parity (RoPE rotates local head slices — position
    is token-local, so TP slicing is transparent to it)."""
    _run_flash_parity(2, use_rope=True)


@pytest.mark.need_gpu(2)
@pytest.mark.slow
def test_tp_flash_attn_gqa_parity_two_gpu():
    """FlashAttention + GQA TP parity (num_kv_heads=2 must divide tp_size)."""
    _run_flash_parity(2, num_kv_heads=2)


def _mla_parity_worker(rank: int, world_size: int, device_indices: list[int], results) -> None:
    """MultiLatentAttention TP parity (TASK-206, TASK-204 mla leg).

    MLA has a different head-slicing surface than the fused-QKV backends:
    latent_q_proj / latent_output_proj are column over hidden, input_kv_proj
    is column over the [K | V] block layout (block-interleaved full_index),
    latent_v_proj / out_proj are row over hidden, and the learnable
    ``latents`` vector is replicated. fp32 throughout (MLA's attention runs
    through the sdpa functional, not the flash kernel), so the MHA-style
    tolerances apply.
    """
    try:
        device_index = device_indices[rank]
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device_index)
        dev = torch.device(f"cuda:{device_index}")

        ref = _build_model(attn_impl="mla").to(dev)
        model = _build_model(attn_impl="mla").to(dev)
        model = apply_tensor_parallel(model, process_group=dist.group.WORLD)
        assert is_tp(model)

        x = torch.randint(0, 64, (2, 12), device=dev, dtype=torch.long)

        # --- forward parity (evel: dropout disabled) ---
        ref.eval()
        model.eval()
        with torch.no_grad():
            ref_logits = ref(x)
        train_logits = model(x)  # TP forward is collective; identical every rank
        torch.testing.assert_close(train_logits, ref_logits, atol=1e-5, rtol=1e-5)

        # --- gradient parity: one backward against the same loss ---
        model.train()
        ref.train()
        ref.zero_grad()
        model.zero_grad()
        tp_loss = model(x).float().mean()
        tp_loss.backward()
        # ref backward after the TP collectives have drained (NCCL ordering).
        ref_loss = ref(x).float().mean()
        ref_loss.backward()

        ref_grads = {k: v.grad for k, v in ref.named_parameters() if v.grad is not None}
        tp_grads = _gather_tp_grads(model)
        assert set(ref_grads.keys()) == set(tp_grads.keys()), (
            f"grad key sets differ: missing {set(ref_grads) - set(tp_grads)}, extra {set(tp_grads) - set(ref_grads)}"
        )
        for key in ref_grads:
            torch.testing.assert_close(
                tp_grads[key], ref_grads[key], atol=1e-4, rtol=1e-4, msg=f"grad mismatch on {key}"
            )

        # --- checkpoint boundary: full gather == reference state dict ---
        full = model_state_dict(model)  # collective; every rank enters
        for key, value in ref.state_dict().items():
            assert key in full, f"gathered state dict missing {key}"
            torch.testing.assert_close(full[key], value, atol=0, rtol=0, msg=f"state mismatch on {key}")

        # --- scatter load roundtrip: load ref full dict, forward again ---
        load_model_state_dict(model, ref.state_dict())
        model.eval()
        with torch.no_grad():
            reload_logits = model(x)
        torch.testing.assert_close(reload_logits, ref_logits, atol=1e-5, rtol=1e-5)

        results[rank] = {"success": True}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}


def _run_mla_parity(world_size: int) -> None:
    gpu_devices = all_gpu_devices(min_free_bytes=TP_MIN_FREE_BYTES)
    if len(gpu_devices) < world_size:
        pytest.skip(f"need at least {world_size} free GPUs (MLA TP parity)")
    device_indices = [device.index for device in gpu_devices[:world_size]]
    _release_parent_cuda_caches()
    _setup_tp_env()

    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(
        _mla_parity_worker,
        args=(world_size, device_indices, results),
        nprocs=world_size,
        join=False,
    )
    end_at = time.monotonic() + TP_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"MLA TP parity spawn exceeded {TP_JOIN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break
    for rank in range(world_size):
        assert rank in results, f"rank {rank} produced no result"
        assert results[rank]["success"], f"rank {rank} failed: {results[rank].get('error')}"


@pytest.mark.need_gpu(2)
@pytest.mark.slow
def test_tp_mla_parity_two_gpu():
    """MLA TP forward/grad/checkpoint parity vs the single-rank MLA reference (2 GPUs).

    Exercises the K/V-block column slice (block-interleaved full_index is
    shared with the fused-QKV path) plus the latent/proj row/column mix.
    """
    _run_mla_parity(2)


def _wrap_worker(rank: int, world_size: int, device_indices: list[int], results) -> None:
    """Engine entry path: ``wrap_model_for_training`` tp branch + full gather/load."""
    try:
        device_index = device_indices[rank]
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device_index)
        dev = torch.device(f"cuda:{device_index}")

        ref = _build_model().to(dev)
        model = _build_model().to(dev)
        wrapped = wrap_model_for_training(
            model,
            parallel_strategy="tp",
            device=dev,
            world_size=world_size,
            tp_size=world_size,
        )
        assert wrapped is model  # in-place mutation, not a wrapper
        assert is_tp(model)

        x = torch.randint(0, 64, (2, 12), device=dev)
        model.eval()
        with torch.no_grad():
            tp_logits = model(x)  # collective
        ref.eval()
        with torch.no_grad():
            ref_logits = ref(x)
        torch.testing.assert_close(tp_logits, ref_logits, atol=1e-5, rtol=1e-5)

        # Full-state-dict gather/scatter roundtrip through the public helper.
        full = model_state_dict(model)  # collective; every rank enters
        for key, value in ref.state_dict().items():
            torch.testing.assert_close(full[key], value, atol=0, rtol=0, msg=f"state mismatch on {key}")
        load_model_state_dict(model, ref.state_dict())
        model.eval()
        with torch.no_grad():
            reloaded = model(x)
        torch.testing.assert_close(reloaded, ref_logits, atol=1e-5, rtol=1e-5)

        results[rank] = {"success": True}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}


def _run_wrap(world_size: int) -> None:
    gpu_devices = all_gpu_devices(min_free_bytes=TP_MIN_FREE_BYTES)
    if len(gpu_devices) < world_size:
        pytest.skip(f"need at least {world_size} free GPUs")
    device_indices = [device.index for device in gpu_devices[:world_size]]
    _release_parent_cuda_caches()
    _setup_tp_env()

    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(_wrap_worker, args=(world_size, device_indices, results), nprocs=world_size, join=False)
    end_at = time.monotonic() + TP_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"TP wrap spawn exceeded {TP_JOIN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break
    for rank in range(world_size):
        assert rank in results, f"rank {rank} produced no result"
        assert results[rank]["success"], f"rank {rank} failed: {results[rank].get('error')}"


@pytest.mark.need_gpu(2)
@pytest.mark.slow
def test_tp_wrap_engine_path_roundtrip_two_gpu():
    """``wrap_model_for_training(tp)`` engine path + full state-dict roundtrip (2 GPUs)."""
    _run_wrap(2)


def _parity_2d_worker(rank: int, world_size: int, device_indices: list[int], results, tp_size: int) -> None:
    """TP + data-parallel 2D (TASK-202): tp_size x dp_size grid on ``world_size`` GPUs.

    Every rank builds the same full model (one CPU seed), wraps it with
    ``wrap_model_for_training(tp, tp_size=world_size//2)`` and compares
    against a full single-rank reference that sees the WHOLE batch:

    * forward parity per DP group's data shard (each TP group replicates its
      own shard internally);
    * gradient parity: shard-agnostic gradients AFTER the DP-group average
      (``allreduce_dp_grads`` — the exact engine step-boundary hook) must
      equal the reference's full-batch gradients;
    * checkpoint gather/scatter roundtrip (the TP group's full-state dict
      must reproduce the reference's, atol=0 absent any optimizer step);
    * training-dynamics parity: after a few SGD steps (with the DP average
      applied each step) the gathered full state dict must still match the
      reference stepped on the full batch.
    """
    try:
        device_index = device_indices[rank]
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device_index)
        dev = torch.device(f"cuda:{device_index}")

        ref = _build_model().to(dev)
        model = _build_model().to(dev)
        wrapped = wrap_model_for_training(
            model, parallel_strategy="tp", device=dev, world_size=world_size, tp_size=tp_size
        )
        assert wrapped is model  # in-place mutation, not a wrapper
        assert is_tp(model)
        tp = model._tp
        assert tp.world_size == tp_size  # group-LOCAL world (== tp_size)
        assert tp.rank == rank % tp_size
        dp_size = world_size // tp_size
        if dp_size > 1:
            assert tp.dp_group is not None  # 2D: DP averaging is wired
        else:
            assert tp.dp_group is None  # pure TP v1: no DP dimension

        dp_rank = rank // tp_size
        batch_per_dp = 2
        x_full = torch.randint(0, 64, (batch_per_dp * dp_size, 12), device=dev, dtype=torch.long)
        x_local = x_full[dp_rank * batch_per_dp : (dp_rank + 1) * batch_per_dp]

        # --- forward parity: this DP group's shard through the TP model ===  #
        # --- reference on the same shard rows of the FULL batch           ---#
        ref.eval()
        model.eval()
        with torch.no_grad():
            ref_logits = ref(x_full)[dp_rank * batch_per_dp : (dp_rank + 1) * batch_per_dp]
        tp_logits = model(x_local)  # collective within the TP group
        torch.testing.assert_close(tp_logits, ref_logits, atol=1e-5, rtol=1e-5)

        # --- gradient parity after the DP-group average ---
        model.train()
        ref.train()
        model.zero_grad()
        ref.zero_grad()
        tp_loss = model(x_local).float().mean()
        tp_loss.backward()
        # The engine's step-boundary hook: average grads across the strided DP
        # group so each shard/param converges to the full-batch gradient.
        allreduce_dp_grads(model)
        # Reference MUST run after the TP+DP collectives (NCCL ordering).
        ref_loss = ref(x_full).float().mean()
        ref_loss.backward()

        ref_grads = {k: v.grad for k, v in ref.named_parameters() if v.grad is not None}
        tp_grads = _gather_tp_grads(model)
        assert set(ref_grads.keys()) == set(tp_grads.keys()), (
            f"grad key sets differ: missing {set(ref_grads) - set(tp_grads)}, extra {set(tp_grads) - set(ref_grads)}"
        )
        for key in ref_grads:
            torch.testing.assert_close(
                tp_grads[key], ref_grads[key], atol=1e-4, rtol=1e-4, msg=f"grad mismatch on {key}"
            )

        # --- checkpoint boundary: full gather == reference state dict ---
        full = model_state_dict(model)  # collective; every rank enters
        for key, value in ref.state_dict().items():
            assert key in full, f"gathered state dict missing {key}"
            torch.testing.assert_close(full[key], value, atol=0, rtol=0, msg=f"state mismatch on {key}")

        # --- scatter load roundtrip + forward ---
        load_model_state_dict(model, ref.state_dict())
        model.eval()
        with torch.no_grad():
            reloaded = model(x_local)
        torch.testing.assert_close(reloaded, ref_logits, atol=1e-5, rtol=1e-5)

        # --- training-dynamics parity: SGD keeps shards consistent across ---
        # --- DP groups (would diverge without allreduce_dp_grads)        ---#
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        ref_opt = torch.optim.SGD(ref.parameters(), lr=0.05)
        for _ in range(2):
            model.train()
            ref.train()
            model.zero_grad()
            m_loss = model(x_local).float().mean()
            m_loss.backward()
            allreduce_dp_grads(model)
            opt.step()
            # Reference on the FULL batch — same schedule, after TP collectives.
            ref.zero_grad()
            r_loss = ref(x_full).float().mean()
            r_loss.backward()
            ref_opt.step()
        full = model_state_dict(model)
        for key, value in ref.state_dict().items():
            torch.testing.assert_close(full[key], value, atol=1e-5, rtol=1e-5, msg=f"post-step mismatch on {key}")

        results[rank] = {"success": True}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}


def _run_parity_2d(world_size: int, tp_size: int, min_free_bytes: int = TP_MIN_FREE_BYTES) -> None:
    gpu_devices = all_gpu_devices(min_free_bytes=min_free_bytes)
    if len(gpu_devices) < world_size:
        pytest.skip(f"need at least {world_size} free GPUs (TP+DP 2D)")
    device_indices = [device.index for device in gpu_devices[:world_size]]
    _release_parent_cuda_caches()
    _setup_tp_env()

    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(
        _parity_2d_worker,
        args=(world_size, device_indices, results, tp_size),
        nprocs=world_size,
        join=False,
    )
    end_at = time.monotonic() + TP_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"TP+DP 2D parity spawn exceeded {TP_JOIN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break
    for rank in range(world_size):
        assert rank in results, f"rank {rank} produced no result"
        assert results[rank]["success"], f"rank {rank} failed: {results[rank].get('error')}"


@pytest.mark.need_gpu(4)
@pytest.mark.slow
def test_tp_dp_2d_parity_four_gpu():
    """TP + data-parallel 2D numeric parity vs a full-batch reference (4 GPUs, tp=2 dp=2)."""
    _run_parity_2d(4, tp_size=2)


@pytest.mark.need_gpu(8)
@pytest.mark.slow
def test_tp_dp_2d_parity_square_grid_eight_gpu():
    """TP + data-parallel 2D parity, deep-TP square grid (8 GPUs, tp=4 dp=2).

    ``tp_size`` need not be 2: a square ``[DP][TP]`` layout with tp = 4
    exercises deeper tensor sharding (every partition axis ÷ 4) alongside
    the DP dimension.

    Requires a FRESH 8-GPU box (high per-GPU free-memory threshold): an
    8-rank spawn after a heavy GPU battery in the same pytest process can
    OOM on a device whose free memory dropped just below the default 1 GiB
    gate even though no test "owns" it. On a fresh 8x80GB box every GPU has
    well over this threshold, so the test runs there and skips on a depleted
    one — an environment gate, not a correctness guard.
    """
    _run_parity_2d(8, tp_size=4, min_free_bytes=16 * 1024**3)


@pytest.mark.need_gpu(6)
@pytest.mark.slow
def test_tp_dp_2d_parity_wide_grid_six_gpu():
    """TP + data-parallel 2D parity with a wide DP grid (6 GPUs, tp=2 dp=3)."""
    _run_parity_2d(6, tp_size=2)


@pytest.mark.need_gpu(4)
@pytest.mark.slow
def test_tp_deep_partition_parity_four_gpu():
    """Deep tensor partition parity: tp_size=4 over a whole 4-rank group (dp=1).

    ``apply_tensor_parallel`` v1 and the 2D milestone only exercised tp_size=2
    (2 blocked-QKV fragments, half-vocab heads). A tp=4 pure-TP group forces
    every partitioned axis ÷ 4 — the 4-fragment fused-QKV scatter, quarter
    vocab / heads / intermediate slices — through the same parity gate. Needs
    only 4 GPUs (robust on busy shared boxes), unlike the 8-GPU square-grid
    companion which additionally stacks a DP dimension.
    """
    _run_parity_2d(4, tp_size=4)


def _wrap_2d_worker(rank: int, world_size: int, device_indices: list[int], results) -> None:
    """2D wrap sanity: subgroup wiring + engine-path checkpoint roundtrip (tp=2 dp=dp_size)."""
    try:
        device_index = device_indices[rank]
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device_index)
        dev = torch.device(f"cuda:{device_index}")

        tp_size = world_size // 2
        ref = _build_model().to(dev)
        model = _build_model().to(dev)
        wrapped = wrap_model_for_training(
            model, parallel_strategy="tp", device=dev, world_size=world_size, tp_size=tp_size
        )
        assert wrapped is model
        assert is_tp(model)
        tp = model._tp
        dp_size = world_size // tp_size
        assert tp.dp_group is not None
        dp_rank = rank // tp_size

        # The strided DP communicator really spans the dp_size replicas that
        # hold this shard: a SUM over it equals dp_size (every member adds 1).
        card = torch.ones(1, device=dev, dtype=torch.long)
        dist.all_reduce(card, group=tp.dp_group)
        assert card.item() == dp_size

        x_full = torch.randint(0, 64, (2 * dp_size, 12), device=dev, dtype=torch.long)
        x_local = x_full[dp_rank * 2 : (dp_rank + 1) * 2]
        model.eval()
        with torch.no_grad():
            tp_logits = model(x_local)
        ref.eval()
        with torch.no_grad():
            ref_logits = ref(x_full)[dp_rank * 2 : (dp_rank + 1) * 2]
        torch.testing.assert_close(tp_logits, ref_logits, atol=1e-5, rtol=1e-5)

        # Full-state-dict gather/scatter roundtrip through the public helpers.
        full = model_state_dict(model)
        for key, value in ref.state_dict().items():
            torch.testing.assert_close(full[key], value, atol=0, rtol=0, msg=f"state mismatch on {key}")
        load_model_state_dict(model, ref.state_dict())
        model.eval()
        with torch.no_grad():
            reloaded = model(x_local)
        torch.testing.assert_close(reloaded, ref_logits, atol=1e-5, rtol=1e-5)

        results[rank] = {"success": True}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}


def _run_wrap_2d(world_size: int) -> None:
    gpu_devices = all_gpu_devices(min_free_bytes=TP_MIN_FREE_BYTES)
    if len(gpu_devices) < world_size:
        pytest.skip(f"need at least {world_size} free GPUs (TP+DP 2D wrap)")
    device_indices = [device.index for device in gpu_devices[:world_size]]
    _release_parent_cuda_caches()
    _setup_tp_env()

    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(_wrap_2d_worker, args=(world_size, device_indices, results), nprocs=world_size, join=False)
    end_at = time.monotonic() + TP_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError(f"TP+DP 2D wrap spawn exceeded {TP_JOIN_TIMEOUT_S}s")
        if context.join(timeout=remaining):
            break
    for rank in range(world_size):
        assert rank in results, f"rank {rank} produced no result"
        assert results[rank]["success"], f"rank {rank} failed: {results[rank].get('error')}"


@pytest.mark.need_gpu(4)
@pytest.mark.slow
def test_tp_dp_2d_wrap_roundtrip_four_gpu():
    """2D subgroup wiring + engine-path state-dict roundtrip (4 GPUs, tp=2 dp=2)."""
    _run_wrap_2d(4)


# ---------------------------------------------------------------------------
# Non-GPU / cheap unit checks
# ---------------------------------------------------------------------------


def test_tp_strategy_accepted_by_config():
    from pydantic import ValidationError

    from llm.training.core.config import DistributedConfig

    DistributedConfig(parallel_strategy="tp", tp_size=2)
    with pytest.raises(ValidationError):
        DistributedConfig(parallel_strategy="tp_and_bad")


def test_tp_world_size_one_is_noop():
    """Single-rank TP returns the model unchanged (mirrors DDP/FSDP)."""
    model = torch.nn.Linear(4, 2)
    wrapped = wrap_model_for_training(
        model,
        parallel_strategy="tp",
        device=torch.device("cpu"),
        world_size=1,
    )
    assert wrapped is model
    assert not is_tp(model)


def test_tp_rejects_tp_size_gt_world():
    with pytest.raises(ValueError, match="tp_size"):
        wrap_model_for_training(
            torch.nn.Linear(4, 2),
            parallel_strategy="tp",
            device=torch.device("cpu"),
            world_size=2,
            tp_size=4,
        )


def test_tp_rejects_tp_size_not_dividing_world():
    with pytest.raises(ValueError, match="divide world_size"):
        wrap_model_for_training(
            torch.nn.Linear(4, 2),
            parallel_strategy="tp",
            device=torch.device("cpu"),
            world_size=6,
            tp_size=4,
        )


def test_tp_mla_and_moe_scope_boundary(monkeypatch):
    """TP scope after the MLA + MoE slices (TASK-206 / TASK-207): MLA is
    accepted (its latent layout is head-sliced by the dedicated branch) and MoE
    is accepted (expert parallelism — replicated gate + rank-local experts).
    Only a num_experts that fails to divide evenly is rejected. The scope
    guards fire before any collective, so faking the 2-rank group accessors and
    passing any group object is enough — no NCCL, no GPUs."""
    from llm.training.distributed.tensor_parallel import (
        _ExpertParallelMoE,
        apply_tensor_parallel,
    )

    class _StubGroup:
        pass

    monkeypatch.setattr("torch.distributed.get_world_size", lambda _group: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda _group: 0)

    stub = _StubGroup()
    # MLA now partitions (heads divisible: 4 heads / 2 ranks). Assert the
    # transform not only runs but tags the model and slices the K/V block.
    mla_model = DecoderModel(vocab_size=32, hidden_size=16, num_layers=1, num_heads=4, attn_impl="mla")
    out = apply_tensor_parallel(mla_model, process_group=stub)
    assert is_tp(out)
    attn = out.transformer_blocks[0].self_attn
    # K/V block of the sliced input_kv_proj: 2 blocks (K, V) x 2 local heads x
    # head_dim 4 = 16 rows total on each rank.
    assert attn.input_kv_proj.weight.shape[0] == 16
    assert attn.num_heads == 2  # 4 heads / 2 ranks
    # MoE now partitions too: the block's MLP is replaced by the expert-parallel
    # wrapper holding only this rank's slice of the experts, and the full gate.
    moe_model = apply_tensor_parallel(
        DecoderModel(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            num_experts=4,
            top_k=2,
            mlp_impl="moe",  # num_experts only builds MoE with mlp_impl="moe"
        ),
        process_group=stub,
    )
    assert is_tp(moe_model)
    moe = moe_model.transformer_blocks[0].mlp
    assert isinstance(moe, _ExpertParallelMoE)
    assert moe.num_experts == 4  # full gate output dim kept
    assert len(moe.experts) == 2  # 4 experts / 2 ranks
    assert moe._n_local == 2
    assert moe._expert_offset == 0  # rank 0 owns global experts [0, 2)
    assert moe_model._tp.expert_shards["transformer_blocks.0.mlp"] == (4, 2)
    # An odd num_experts cannot split evenly — rejected loudly.
    with pytest.raises(ValueError, match="num_experts"):
        apply_tensor_parallel(
            DecoderModel(
                vocab_size=32,
                hidden_size=16,
                num_layers=1,
                num_heads=4,
                num_experts=3,
                top_k=2,
                mlp_impl="moe",
            ),
            process_group=stub,
        )


def test_tp_dp_layout_2d():
    """TP + DP 2D rank-layout math (row-major [DP][TP], ``rank`` injectable)."""
    from llm.training.distributed import tp_dp_layout

    # Pure TP: tp_size 0/None/world -> one TP group, dp_size 1.
    assert tp_dp_layout(8, 0, rank=3) == (8, 1, 0, 3)
    assert tp_dp_layout(8, 8, rank=5) == (8, 1, 0, 5)
    # 2D row-major [DP][TP]: rank = dp_rank * tp_size + tp_rank.
    assert tp_dp_layout(8, 4, rank=0) == (4, 2, 0, 0)
    assert tp_dp_layout(8, 4, rank=3) == (4, 2, 0, 3)
    assert tp_dp_layout(8, 4, rank=4) == (4, 2, 1, 0)
    assert tp_dp_layout(8, 4, rank=7) == (4, 2, 1, 3)
    assert tp_dp_layout(8, 2, rank=5) == (2, 4, 2, 1)
    assert tp_dp_layout(6, 2, rank=4) == (2, 3, 2, 0)
    # Validation.
    with pytest.raises(ValueError, match="divide world_size"):
        tp_dp_layout(8, 3, rank=0)
    with pytest.raises(ValueError, match="tp_size"):
        tp_dp_layout(8, 12, rank=0)


def test_three_d_layout_resolver():
    """DP + PP + TP 3D rank-layout math (row-major [DP][PP][TP], rank injectable).

    Rank = ((dp_rank * pp_size) + pp_rank) * tp_size + tp_rank, so every rank
    maps to exactly one grid point (TASK-215 / DEC-052).
    """
    from llm.training.distributed import three_d_layout

    # Pure / degenerate configurations: an unspecified (<=0) dim defaults to 1,
    # and the remaining dims must multiply to the world size.
    assert three_d_layout(8, 1, 8, 1, rank=3) == (1, 8, 1, 0, 3, 0)  # pure PP
    assert three_d_layout(8, 1, 1, 8, rank=5) == (1, 1, 8, 0, 0, 5)  # pure TP
    assert three_d_layout(8, 1, 0, 8, rank=7) == (1, 1, 8, 0, 0, 7)  # pp defaults to 1

    # 2x2x2 grid over 8 ranks: rank = ((dp*2)+pp)*2 + tp.
    assert three_d_layout(8, 2, 2, 2, rank=0) == (2, 2, 2, 0, 0, 0)
    assert three_d_layout(8, 2, 2, 2, rank=3) == (2, 2, 2, 0, 1, 1)
    assert three_d_layout(8, 2, 2, 2, rank=5) == (2, 2, 2, 1, 0, 1)
    assert three_d_layout(8, 2, 2, 2, rank=7) == (2, 2, 2, 1, 1, 1)

    # Uneven 3D: dp=2, pp=1, tp=2 over world 4 (rank = (dp*1 + pp)*2 + tp).
    assert three_d_layout(4, 2, 1, 2, rank=0) == (2, 1, 2, 0, 0, 0)
    assert three_d_layout(4, 2, 1, 2, rank=3) == (2, 1, 2, 1, 0, 1)

    # dp=3, pp=2, tp=2 over world 12.
    assert three_d_layout(12, 3, 2, 2, rank=0) == (3, 2, 2, 0, 0, 0)
    assert three_d_layout(12, 3, 2, 2, rank=10) == (3, 2, 2, 2, 1, 0)
    assert three_d_layout(12, 3, 2, 2, rank=11) == (3, 2, 2, 2, 1, 1)

    # Validation: a grid that does not tile the world exactly is rejected.
    with pytest.raises(ValueError, match="world_size"):
        three_d_layout(8, 3, 2, 2, rank=0)  # 3*2*2=12 != 8
    with pytest.raises(ValueError, match="world_size"):
        three_d_layout(9, 2, 2, 2, rank=0)  # 8 != 9


def test_three_d_groups_bijection_and_contiguity():
    """3D group membership tables: one group per family per rank, contiguous
    TP/PP runs and strided DP columns (TASK-215).
    """
    from llm.training.distributed import three_d_groups

    tp_groups, pp_groups, dp_groups = three_d_groups(8, 2, 2, 2)

    # TP groups: dp*pp = 4 stages, each a contiguous tp_size=2 run.
    assert tp_groups == [[0, 1], [2, 3], [4, 5], [6, 7]]
    for g in tp_groups:
        assert g == list(range(g[0], g[0] + 2)), "TP groups must be contiguous runs"

    # PP groups: dp=2 full pipeline columns of pp*tp = 4 contiguous ranks.
    assert pp_groups == [[0, 1, 2, 3], [4, 5, 6, 7]]
    for g in pp_groups:
        assert g == list(range(g[0], g[0] + 4)), "PP groups must be contiguous columns"

    # DP groups: pp*tp = 4 shards, each strided by pp*tp=4 across DP blocks.
    assert dp_groups == [[0, 4], [1, 5], [2, 6], [3, 7]]

    # Bijection: every rank is in exactly one group of each family.
    for family in (tp_groups, pp_groups, dp_groups):
        seen: set[int] = set()
        for g in family:
            for r in g:
                assert r not in seen, f"rank {r} appears in two groups of the same family"
                seen.add(r)
        assert seen == set(range(8)), "every rank must belong to exactly one group per family"

    # Union of all TP-stage params == the world, and PP columns partition it.
    assert set().union(*(set(g) for g in tp_groups)) == set(range(8))
    assert set().union(*(set(g) for g in pp_groups)) == set(range(8))


def _moe_roundtrip_worker(rank: int, world_size: int, results) -> None:
    """MoE expert-parallel checkpoint boundary on CPU (gloo, TASK-207)."""
    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        dev = torch.device("cpu")
        ref = _build_model(num_experts=4, top_k=2, mlp_impl="moe").to(dev)
        model = _build_model(num_experts=4, top_k=2, mlp_impl="moe").to(dev)
        model = apply_tensor_parallel(model, process_group=dist.group.WORLD)
        assert is_tp(model)

        # Shard layout: rank r owns global experts [r*n_local:(r+1)*n_local).
        moe = model.transformer_blocks[0].mlp
        n_local = moe._n_local
        assert n_local == 4 // world_size
        assert moe._expert_offset == rank * n_local
        assert moe.num_experts == 4  # full gate output dim retained
        tp = model._tp
        prefix = next(name for name, module in model.named_modules() if module is moe)
        assert tp.expert_shards[prefix] == (4, n_local)
        assert tp.is_expert_param(f"{prefix}.experts.0.fc1.weight")
        assert not tp.is_expert_param(f"{prefix}.gate.weight")  # replicated router

        x = torch.randint(0, 64, (2, 12), device=dev, dtype=torch.long)

        # Full gather must be bit-exact vs the unittest reference despite each
        # rank holding only half the experts.
        with torch.no_grad():
            full = model_state_dict(model)
        for key, value in ref.state_dict().items():
            assert key in full, f"gathered state dict missing {key}"
            torch.testing.assert_close(full[key], value, atol=0, rtol=0, msg=f"state mismatch on {key}")

        # Gradient parity (incl. the replicated gate + expert shards): the
        # engine's step-boundary reduction (allreduce_dp_grads) SUMs the
        # gate's per-rank partial weight grad over the TP group.
        model.train()
        ref.train()
        model.zero_grad()
        ref.zero_grad()
        tp_loss = model(x).float().mean()
        tp_loss.backward()
        allreduce_dp_grads(model)
        ref_loss = ref(x).float().mean()
        ref_loss.backward()
        ref_grads = {k: v.grad for k, v in ref.named_parameters() if v.grad is not None}
        tp_grads = _gather_tp_grads(model)
        assert set(ref_grads) <= set(tp_grads), f"tp missing grads: {set(ref_grads) - set(tp_grads)}"
        for key in ref_grads:
            torch.testing.assert_close(
                tp_grads[key], ref_grads[key], atol=2e-3, rtol=2e-3, msg=f"grad mismatch on {key}"
            )

        # Scatter roundtrip: load the reference full dict -> forward unchanged.
        load_model_state_dict(model, ref.state_dict())
        model.eval()
        ref.eval()
        with torch.no_grad():
            train_logits = model(x)
            ref_logits = ref(x)
        torch.testing.assert_close(train_logits, ref_logits, atol=1e-5, rtol=1e-5, msg="scatter roundtrip")

        results[rank] = {"success": True}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}


def _moe_dead_rank_worker(rank: int, world_size: int, results) -> None:
    """Backward with a rank that routes NO token to any local expert (TASK-207).

    A single token with top_k=1 hits exactly ONE expert globally, so exactly
    one rank's local experts receive zero routed tokens. That rank must still
    enter every collective in backward and ``allreduce_dp_grads`` (its
    ``_BackwardAllReduce`` fires through the zero-contributing ``x_ep * 0``
    term and its dead experts/routers contribute zero grads) — otherwise a peer
    with hits runs ``dist.all_reduce`` alone and the backward deadlocks (the
    hang this test protects against; collapsed routing is the MoE norm).
    """
    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        dev = torch.device("cpu")
        # 4 experts / 2 ranks, top_k=1, ONE token -> exactly one rank is dead.
        model = _build_model(num_experts=4, top_k=1, mlp_impl="moe").to(dev)
        model = apply_tensor_parallel(model, process_group=dist.group.WORLD)
        assert is_tp(model)
        x = torch.randint(0, 64, (1, 1), device=dev, dtype=torch.long)
        model.train()
        model.zero_grad()
        loss = model(x).float().mean()
        loss.backward()  # must NOT deadlock on the zero-hit rank
        allreduce_dp_grads(model)  # must NOT deadlock (zeros materialised)
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no grad after allreduce"
            assert torch.isfinite(param.grad).all(), f"{name} grad not finite"
        results[rank] = {"success": True}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}


@pytest.mark.quick
def test_tp_moe_dead_rank_no_deadlock_cpu():
    """A zero-hit MoE rank still enters every collective (no backward deadlock).

    Routes a single token (top_k=1) so exactly one of the two ranks has no
    local expert hit, then runs backward + ``allreduce_dp_grads`` — the graph
    and step-boundary collectives must complete uniformly across ranks.
    Regression guard for the expert-parallel deadlock (collapsed routing).
    """
    _setup_tp_env()
    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(_moe_dead_rank_worker, args=(2, results), nprocs=2, join=False)
    end_at = time.monotonic() + TP_JOIN_TIMEOUT_S
    while True:
        if context.join(timeout=1):
            break
        if time.monotonic() > end_at:
            for p in context.processes:
                p.terminate()
            pytest.fail("MoE dead-rank backward timed out (deadlock)")
    for rank in range(2):
        assert results[rank]["success"], f"rank {rank} failed: {results[rank].get('error')}"


@pytest.mark.quick
def test_tp_moe_state_dict_roundtrip_cpu():
    """MoE expert-parallel state-dict gather/scatter on CPU (gloo, 2 ranks).

    Each rank holds a DISJOINT expert subset and a replicated full gate; the
    gathered full dict must equal the reference bit-for-bit (this is what
    ``llm-serve`` / resume rely on) and loading it back must leave the TP
    forward unchanged (TASK-207).
    """
    _setup_tp_env()
    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(_moe_roundtrip_worker, args=(2, results), nprocs=2, join=False)
    end_at = time.monotonic() + TP_JOIN_TIMEOUT_S
    while True:
        if context.join(timeout=1):
            break
        if time.monotonic() > end_at:
            for p in context.processes:
                p.terminate()
            pytest.fail("MoE CPU roundtrip timed out (deadlock?)")
    for rank in range(2):
        assert results[rank]["success"], f"rank {rank} failed: {results[rank].get('error')}"


def _moe_parity_worker(rank: int, world_size: int, device_indices: list[int], results) -> None:
    """MoE expert-parallel forward / gradient / checkpoint parity (TASK-207).

    The EP forward restructures the dense MoE sum (masked per-local-expert
    ``index_add_`` + an all-reduce instead of the dense token loop), so
    forward/grad parity is asserted CLOSE (fp summation order) rather than the
    bit-exact the non-MoE tests reach; the state-dict gather is still exact.
    ``allreduce_dp_grads`` runs after the backward, mirroring the engine's
    step-boundary reduction (the replicated gate's weight grad is a per-rank
    partial that the TP-group SUM completes).
    """
    try:
        device_index = device_indices[rank]
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(device_index)
        dev = torch.device(f"cuda:{device_index}")

        ref = _build_model(num_experts=4, top_k=2, mlp_impl="moe").to(dev)
        model = _build_model(num_experts=4, top_k=2, mlp_impl="moe").to(dev)
        model = apply_tensor_parallel(model, process_group=dist.group.WORLD)
        assert is_tp(model)

        x = torch.randint(0, 64, (2, 12), device=dev, dtype=torch.long)

        # --- forward parity (eval: dropout disabled; MoE has no dropout) ---
        ref.eval()
        model.eval()
        with torch.no_grad():
            ref_logits = ref(x)
        train_logits = model(x)
        torch.testing.assert_close(train_logits, ref_logits, atol=1e-5, rtol=1e-5)

        # --- gradient parity ---
        model.train()
        ref.train()
        ref.zero_grad()
        model.zero_grad()
        tp_loss = model(x).float().mean()
        tp_loss.backward()
        allreduce_dp_grads(model)  # engine step-boundary reduction
        ref_loss = ref(x).float().mean()
        ref_loss.backward()

        ref_grads = {k: v.grad for k, v in ref.named_parameters() if v.grad is not None}
        tp_grads = _gather_tp_grads(model)
        assert set(ref_grads) <= set(tp_grads), f"tp missing grads: {set(ref_grads) - set(tp_grads)}"
        for key in ref_grads:
            torch.testing.assert_close(
                tp_grads[key], ref_grads[key], atol=2e-3, rtol=2e-3, msg=f"grad mismatch on {key}"
            )

        # --- checkpoint boundary: full gather == reference state dict ---
        full = model_state_dict(model)  # collective; every rank enters
        for key, value in ref.state_dict().items():
            assert key in full, f"gathered state dict missing {key}"
            torch.testing.assert_close(full[key], value, atol=0, rtol=0, msg=f"state mismatch on {key}")

        # --- scatter load roundtrip: load ref full dict, forward again ---
        load_model_state_dict(model, ref.state_dict())
        model.eval()
        with torch.no_grad():
            reload_logits = model(x)
        torch.testing.assert_close(reload_logits, ref_logits, atol=1e-5, rtol=1e-5)

        results[rank] = {"success": True}
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results[rank] = {"success": False, "error": repr(e)}


@pytest.mark.need_gpu(2)
@pytest.mark.slow
def test_tp_moe_forward_grad_checkpoint_parity_two_gpu():
    """MoE expert-parallel parity vs a full reference: forward, gradients
    (incl. the replicated gate + expert shards), and the full-state-dict
    gather/scatter boundary (2 GPUs, TASK-207)."""
    _run_spawn_parity(_moe_parity_worker, world_size=2)


def _clip_global_worker(rank: int, world_size: int, results) -> None:
    """Global-norm clip parity (RIL ISS-253): every rank clips by the FULL
    model norm, not its own shard's local norm."""
    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        class _Pair(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = nn.Parameter(torch.zeros(2))

        model = _Pair()
        # Each rank holds a [3.0, 4.0] gradient (local norm 5.0); the FULL
        # model norm across both ranks is sqrt(25 + 25) = sqrt(50) ~ 7.071.
        _ref = nn.Parameter(torch.zeros(4))
        _ref.grad = torch.tensor([3.0, 4.0, 3.0, 4.0])
        reference = torch.nn.utils.clip_grad_norm_([_ref], 6.0)
        coef = float(6.0 / (reference.item() + 1e-6))  # clip_grad_norm_'s own coef

        # 1. global norm clips BOTH shards with the shared coef.
        model.p.grad = torch.tensor([3.0, 4.0])
        with torch.no_grad():
            total = clip_grad_norm_tp(model, 6.0, group=dist.group.WORLD)
        assert abs(total.item() - reference.item()) < 1e-5, (total.item(), reference.item())
        assert torch.allclose(model.p.grad, torch.tensor([3.0, 4.0]) * coef, atol=1e-5)
        if rank == 0:
            results["clipped_norm"] = total.item()
            results["clipped_grad"] = model.p.grad.tolist()

        # 2. a capped max_norm leaves grads untouched and still reports the norm.
        model.p.grad = torch.tensor([3.0, 4.0])
        total = clip_grad_norm_tp(model, 1000.0, group=dist.group.WORLD)
        assert torch.equal(model.p.grad, torch.tensor([3.0, 4.0]))
        if rank == 0:
            results["uncapped_norm"] = total.item()

        # 3. a non-finite gradient propagates inf into the global norm.
        model.p.grad = torch.tensor([float("inf"), 4.0])
        total = clip_grad_norm_tp(model, 6.0, group=dist.group.WORLD)
        assert not torch.isfinite(total)
        if rank == 0:
            results["inf_norm"] = float(total.item())

        results["success"] = True
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 — report worker failure in the parent
        results["success"] = False
        results["error"] = repr(e)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.quick
def test_clip_grad_norm_tp_global_norm_over_two_ranks():
    """Global L2 clip parallels clip_grad_norm_ on the concatenated model."""

    _setup_tp_env()
    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(_clip_global_worker, args=(2, results), nprocs=2, join=False)
    end_at = time.monotonic() + TP_JOIN_TIMEOUT_S
    while True:
        remaining = end_at - time.monotonic()
        if remaining <= 0:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            raise TimeoutError("clip-grad-norm-tp spawn exceeded timeout")
        if context.join(timeout=remaining):
            break
    assert results["success"], results.get("error")
    # Compare the GLOBAL clip against clip_grad_norm_ on the FULL model.
    coef = 6.0 / (float(results["clipped_norm"]) + 1e-6)
    expected_full = torch.tensor([3.0, 4.0, 3.0, 4.0]) * coef
    assert torch.allclose(torch.tensor(results["clipped_grad"]), expected_full[:2], atol=1e-5)
    assert abs(float(results["clipped_norm"]) - (50.0**0.5)) < 1e-5
    assert results["uncapped_norm"] == pytest.approx(50.0**0.5, rel=1e-5)
    assert results["inf_norm"] == float("inf")
