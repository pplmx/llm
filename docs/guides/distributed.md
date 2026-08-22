---
tags:
  - 指南
  - 训练
  - 分布式
---

# Distributed Training Guide

This guide covers the distributed-training strategies the framework
supports: **DDP** (data-parallel, the default), **FSDP**
(fully-sharded data-parallel, opt-in via `parallel_strategy="fsdp"`),
**Tensor Parallelism** (`parallel_strategy="tp"`, optionally
combined with a data-parallel dimension — see the TP section) and
**Pipeline Parallelism** (`parallel_strategy="pp"` — different layers
on different devices, see the PP section).

Both strategies are exposed through a single entry point —
`llm.training.distributed.wrap_model_for_training` — so the trainer
loop doesn't have to branch on which strategy is in use.

## When to use which

|                        | DDP                                | FSDP                                 |
| ---------------------- | ---------------------------------- | ------------------------------------ |
| Memory per rank        | full model + grads + optim state   | shard of model + grads + optim state |
| Communication overhead | per-step all-reduce of gradients   | per-step all-gather + reduce-scatter |
| Best for               | models that already fit on one GPU | models that don't fit on one GPU     |
| Minimum world size     | 1 (effectively no-op)              | 2+ for actual sharding benefit       |

If your model fits on one GPU, **use DDP** — FSDP adds
communication overhead even when sharding isn't needed. Pick
FSDP when you're hitting OOM with `batch_size=1` and the model
parameters are the bottleneck, not the activations.

## DDP quick start

`llm-train` reads `distributed` from the YAML config and spawns one
worker per GPU internally (`torch.multiprocessing.spawn`) — there is
**no `torchrun` step**.

Single-node, multi-GPU:

```yaml
# configs/ddp.yaml
distributed:
  gpus_per_node: 2        # defaults to torch.cuda.device_count()
  parallel_strategy: ddp  # default; shown for clarity
  backend: nccl
```

```bash
uv run llm-train --task stream_lm --config-path configs/ddp.yaml
```

Multi-node:

```bash
# Node 0 (master)
MASTER_ADDR=192.168.1.1 MASTER_PORT=12355 \
NUM_NODES=2 NODE_RANK=0 GPUS_PER_NODE=4 \
uv run llm-train --task stream_lm --config-path configs/ddp.yaml

# Node 1
MASTER_ADDR=192.168.1.1 MASTER_PORT=12355 \
NUM_NODES=2 NODE_RANK=1 GPUS_PER_NODE=4 \
uv run llm-train --task stream_lm --config-path configs/ddp.yaml
```

The multi-node env vars are read by `DistributedConfig` at startup; the
YAML itself can stay identical on both nodes (leave `master_addr` /
`num_nodes` / `node_rank` / `gpus_per_node` unset there so the env vars
apply). For a single-node run none of them are needed.

## FSDP quick start

Set `parallel_strategy="fsdp"` in your config — that's it:

```yaml
# configs/fsdp-pretrain.yaml
distributed:
  parallel_strategy: fsdp
  # Optional FSDP knobs (defaults shown):
  fsdp_mixed_precision: bf16
  fsdp_auto_wrap_min_params: 10000000
  fsdp_cpu_offload: false
```

Launch the same way as DDP — `llm-train` spawns the FSDP workers:

```bash
uv run llm-train --task stream_lm --config-path configs/fsdp-pretrain.yaml
```

### FSDP configuration knobs

All three knobs live on `DistributedConfig` and are documented in
the config help string. The defaults are conservative and safe to
leave alone:

| Knob                        | Default      | What it does                                                                                                                                                              |
| --------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fsdp_mixed_precision`      | `"bf16"`     | Parameter / gradient / buffer dtype. `"bf16"` is recommended on modern GPUs. `"fp16"` needs a loss scaler. `"fp32"` skips mixed precision entirely.                       |
| `fsdp_auto_wrap_min_params` | `10_000_000` | Size-based auto-wrap threshold. Modules with at least this many parameters get their own FSDP unit. Set to `0` to disable auto-wrap and wrap the whole model as one unit. |
| `fsdp_cpu_offload`          | `false`      | Offload params to CPU when idle. Trades throughput for memory — only useful when the model is too big to fit even after BF16 sharding.                                    |

### Auto-wrap policy in detail

`fsdp_auto_wrap_min_params` controls the size-based
`auto_wrap_policy` that FSDP applies recursively. Concretely:

- `10_000_000` (default) — only modules with ≥10M params are
  wrapped as standalone FSDP units. For a typical transformer
  this means each transformer block is one FSDP unit, which
  gives most of the memory benefit while keeping the
  communication overhead low.
- `0` — disable auto-wrap; the whole model is one FSDP unit.
  Maximises communication efficiency but loses the per-layer
  granularity that's responsible for most of FSDP's memory
  benefit.
- `1` (or any tiny number) — wrap every leaf module. Maximum
  granularity, maximum overhead. Rarely useful.

The right value depends on the model size and the GPU memory
budget; the default is a good starting point for most
transformer models in the 1B–10B parameter range.

### Saving and loading FSDP checkpoints

`model_state_dict` and `load_model_state_dict` accept a
`state_dict_type` argument:

- `"full"` (default) — materialises the full state dict on rank
  0. Easier for single-host save/load and produces a file
  readable by any non-distributed code.
- `"sharded"` — each rank saves its own shard. Memory stays
  bounded by the shard size; the resulting checkpoint is only
  readable by a parallel run with the same world size.

```python
from llm.training.distributed import model_state_dict, load_model_state_dict

# Save: full state dict on rank 0
sd = model_state_dict(model, state_dict_type="full")
if rank == 0:
    torch.save(sd, "checkpoint.pt")

# Load: distribute the loaded state dict across the FSDP ranks
sd = torch.load("checkpoint.pt", map_location="cpu")
load_model_state_dict(model, sd, state_dict_type="full")
```

For large-scale resume, prefer `"sharded"` to avoid the rank-0
memory spike. The checkpoint manager in
`src/llm/training/core/checkpoint.py` is responsible for
writing the per-rank files.

### FSDP gotchas

- **FSDP needs CUDA + a process group.** On CPU or single-rank
  runs `wrap_model_for_training` returns the bare model
  unchanged — FSDP doesn't have a meaningful "shard across one
  GPU" mode.
- **Mixed precision interaction.** When
  `fsdp_mixed_precision="bf16"` the trainer's separate AMP config
  is effectively a no-op for FSDP-managed parameters (they're
  already BF16). The trainer still applies AMP to the optimiser
  / loss as usual.
- **Activation checkpointing** is orthogonal but complementary
  — combine both for the largest memory savings.

## Tensor Parallelism quick start

Tensor parallelism (`parallel_strategy="tp"`) partitions the model's
weights across a group of GPUs (Megatron-style column / row parallel
linears): attention heads, the fused QKV projection, the MLP
intermediate width and the vocabulary of the output head are sliced
across the group. It is the option when a single GPU cannot hold the
*weights* of the model and you want low communication overhead
(all-reduces only on the group's boundaries).

```yaml
# configs/tp-pretrain.yaml
distributed:
  parallel_strategy: tp
  backend: nccl
```

`tp_size: 0` (the default) means "use all ranks as one TP group".
Using the whole world is the *pure TP* mode: every rank processes the
**same** microbatches (replicated data), and each rank's optimizer
step moves only its own shards.

### TP + data-parallel 2D

Set `tp_size` smaller than the total number of ranks to add a
data-parallel dimension. Ranks are laid out in a row-major `[DP][TP]`
grid — **TP groups are contiguous rank ranges** (intra-node friendly),
DP groups are the strided columns that hold the *same* shard across TP
groups:

```yaml
# 8 GPUs, 4 TP groups of 2 — gradient-averaged across the 4 DP columns
distributed:
  parallel_strategy: tp
  tp_size: 2
  backend: nccl
```

- Each TP group partitions the model in parallel and sees **its own
  data shard** (the engine shards the dataset per DP group).
- After each step's backward, gradients are **averaged across the DP
  group** (DDP semantics) so every shard converges to the true
  full-batch gradient — plus the intra-group reduce the tensor
  parallelism already does.
- Checkpoints are the *full* model state dict (gathered) on rank 0,
  identical to a plain single-GPU checkpoint — `llm-serve` and resume
  need no special handling.

Requirements and constraints (all fail loudly, not silently wrong):

- `world_size` must divide evenly by `tp_size` (`n % tp == 0`).
- `tp_size` must divide `num_heads`, `num_kv_heads`, `vocab_size` and
  the MLP intermediate width evenly; with MoE it must also divide
  `num_experts` evenly.
- TP supports the `mha`, `flash_attn` and `mla` attention backends and,
  since TASK-207, MoE via expert parallelism (the gate stays replicated and
  the experts are split across ranks by expert index; the full state dict is
  rebuilt rank-major on save so `llm-serve` / resume need no special
  handling). `sdpa` is a *functional*, not a registered `attn_impl` — every
  supported backend runs its attention through it, so TP covers the sdpa
  kernel transitively. ALiBi and serving are out of scope (rejected at wrap
  time).

## Pipeline Parallelism quick start

Pipeline parallelism (`parallel_strategy="pp"`) places **different layers of
the model on different devices** and streams activations forward / gradients
backward between them (RIL DEC-049 / TASK-210). It is the option when a single
GPU cannot hold the model *at all* — TP shards individual weight matrices
across devices, PP chunks the layer stack itself, cutting each device's
activation + weight footprint to ~1/`pp_size` of the whole.

```yaml
# configs/pp-pretrain.yaml
distributed:
  parallel_strategy: pp
  backend: nccl
```

v1 lays the **whole world out as pipeline stages** (one stage per rank, so
`pp_size == world_size`). The model is split at `transformer_blocks`: stage 0
holds the embedding + the first block chunk, the last stage holds the final
norm + the LM head, every other stage holds a middle block chunk. Training is
driven by `torch.distributed.pipelining.ScheduleGPipe` (`n_microbatches=1`),
whose loss is computed on the last stage (the standard LM shift + cross
entropy) and broadcast back so metric reduction / save_best see the same value
on every rank.

Like pure TP v1, PP v1 **replicates the data shard** across every stage (all
stage ranks must pump the same microbatch sequence through the pipeline). The
engine wires the standard-loop PP step, the PP-group-aware global gradient-norm
clip (each rank holds a disjoint stage, so the full-model norm is summed over
the pipeline group), and the full model state dict on rank 0 (gathered
stage-by-stage under the original model's parameter names), so checkpoints,
`llm-serve` and resume are unchanged.

PP v1 refuses loudly rather than silently training the wrong loss:

- **Standard-loop LM tasks only** — the task must advertise
  `supports_pipeline_parallel()` (the `LMTask` family). SFT passes an
  `attention_mask` into the model that the stage forward would drop; PPO / DPO
  / reward use custom loops — all rejected at setup.
- **FP32 only** (`use_amp` is rejected): the schedule backprops inside
  `step()`, where the engine's autocast/GradScaler scaling cannot interact
  safely.
- **No `torch.compile`** (the schedule drives the stages with silent P2P
  send/recv ops a compile graph must not capture) and no TP/FSDP composition
  (3D parallel is a follow-up).

The 2-stage numeric parity vs a single-rank serial run (loss to 10 digits,
every owned stage gradient bit-exact) is a CI-enforced test on CPU + gloo with
zero GPUs.

## Single-rank and CPU behaviour

`wrap_model_for_training` short-circuits when `world_size <= 1`
or `device.type != "cuda"`:

```python
from llm.training.distributed import wrap_model_for_training

model = DecoderModel(...)
out = wrap_model_for_training(
    model,
    parallel_strategy="fsdp",
    device=torch.device("cpu"),
    world_size=1,
)
assert out is model  # bare model, no wrapping
```

This is intentional — wrapping a CPU model in DDP is a no-op
that can confuse some optimisers, and FSDP cannot run on CPU at
all.

## Configuration reference

### `DistributedConfig` fields

| Field                       | Default           | Description                                         |
| --------------------------- | ----------------- | --------------------------------------------------- |
| `master_addr`               | `"127.0.0.1"`     | Process-group master address                        |
| `master_port`               | `"12355"`         | Process-group master port                           |
| `num_nodes`                 | `1`               | Total number of nodes                               |
| `gpus_per_node`             | auto (CUDA count) | GPUs per node                                       |
| `node_rank`                 | `0`               | This node's rank                                    |
| `backend`                   | `"nccl"`          | `torch.distributed` backend                         |
| `parallel_strategy`         | `"ddp"`           | `"ddp"` / `"fsdp"` / `"tp"` / `"pp"`                |
| `tp_size`                   | `0` (= world)     | TP size for `"tp"`; `< world_size` enables TP+DP 2D |
| `fsdp_mixed_precision`      | `"bf16"`          | `"fp32"` / `"bf16"` / `"fp16"`                      |
| `fsdp_auto_wrap_min_params` | `10_000_000`      | Size-based auto-wrap threshold                      |
| `fsdp_cpu_offload`          | `false`           | Offload params to CPU when idle                     |

### Environment variables

| Variable             | Description          | Default |
| -------------------- | -------------------- | ------- |
| `NCCL_DEBUG`         | NCCL debug verbosity | `WARN`  |
| `NCCL_IB_DISABLE`    | Disable InfiniBand   | `0`     |
| `NCCL_NET_GDR_LEVEL` | RDMA level           | `2`     |

### Launching multi-node runs

`llm-train` does not use `torchrun`; distributed parameters come from
`DistributedConfig` (YAML or env vars). The env vars that matter are:

| Variable       | Description                      | Default     |
| -------------- | -------------------------------- | ----------- |
| `MASTER_ADDR`  | Master node address              | `127.0.0.1` |
| `MASTER_PORT`  | Master node port                 | `12355`     |
| `NUM_NODES`    | Total number of nodes            | `1`         |
| `NODE_RANK`    | This node's rank                 | `0`         |
| `GPUS_PER_NODE`| Processes (GPUs) per node        | CUDA count  |
| `BACKEND`      | `nccl` / `gloo`                  | `nccl`      |

Alternatively every field can be set in YAML under `distributed:`, or
overridden with the `LLM_DISTRIBUTED__<FIELD>` env-var convention (e.g.
`LLM_DISTRIBUTED__GPUS_PER_NODE=4`).

## Performance notes

- **Communication optimisation** — set
  `NCCL_NET_GDR_LEVEL=2` on hardware that supports GPUDirect
  RDMA; set `NCCL_IB_DISABLE=1` if InfiniBand is misbehaving.
- **DDP gradient sync** — `gradient_as_bucket_view=True` (set
  in the trainer) reduces memory by avoiding intermediate
  copies.
- **FSDP backoff** — `fsdp_forward_prefetch` /
  `backward_prefetch` aren't exposed as config yet; the default
  is fine for most workloads.

## Monitoring

```bash
watch -n 1 nvidia-smi
```

For NCCL debugging:

```bash
export NCCL_DEBUG=INFO
uv run llm-train --task stream_lm --config-path configs/ddp.yaml
```

## Troubleshooting

**Q: NCCL connection fails.**

- Verify GPUs are visible with `nvidia-smi`.
- Set `NCCL_DEBUG=INFO` for verbose logs.

**Q: Out of memory.**

- For DDP: reduce batch size, enable gradient accumulation,
  enable mixed precision.
- For FSDP: increase `fsdp_auto_wrap_min_params` (more
  aggressive sharding), enable `fsdp_cpu_offload`, or move to
  `fsdp_mixed_precision="fp16"` if you're on a hardware
  generation where BF16 isn't supported.

**Q: Training is slow.**

- Check network latency between ranks (`ethtool` / `ibstat`).
- Profile with `torch.profiler`.
- Confirm NCCL backend is in use (not Gloo).

## Related

- [Deep dive into DDP](../development/deep-dive-ddp.md)
- [Training flow guide](../development/training-flow.md)
- FSDP 的接线记录（config + state-dict helpers）见 `CHANGELOG.md`
  （Tier 3 #29）。
