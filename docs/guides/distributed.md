# Distributed Training Guide

This guide covers the two distributed-training strategies the
framework supports out of the box: **DDP** (data-parallel, the
default) and **FSDP** (fully-sharded data-parallel, opt-in via
`parallel_strategy="fsdp"`).

Both strategies are exposed through a single entry point —
`llm.training.distributed.wrap_model_for_training` — so the trainer
loop doesn't have to branch on which strategy is in use.

## When to use which

| | DDP | FSDP |
|---|---|---|
| Memory per rank | full model + grads + optim state | shard of model + grads + optim state |
| Communication overhead | per-step all-reduce of gradients | per-step all-gather + reduce-scatter |
| Best for | models that already fit on one GPU | models that don't fit on one GPU |
| Minimum world size | 1 (effectively no-op) | 2+ for actual sharding benefit |

If your model fits on one GPU, **use DDP** — FSDP adds
communication overhead even when sharding isn't needed. Pick
FSDP when you're hitting OOM with `batch_size=1` and the model
parameters are the bottleneck, not the activations.

## DDP quick start

Single-node, multi-GPU:

```bash
torchrun --nproc_per_node=2 scripts/train.py
```

Multi-node:

```bash
# Node 0 (master)
torchrun --nnodes=2 --nproc_per_node=4 \
    --master_addr=192.168.1.1 --master_port=29500 \
    scripts/train.py

# Node 1
torchrun --nnodes=2 --nproc_per_node=4 \
    --master_addr=192.168.1.1 --master_port=29500 \
    --node_rank=1 \
    scripts/train.py
```

The trainer picks up `parallel_strategy="ddp"` from the default
config; nothing to set.

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

Launch the same way as DDP — `torchrun` handles the process group
init:

```bash
torchrun --nproc_per_node=4 scripts/train.py
```

### FSDP configuration knobs

All three knobs live on `DistributedConfig` and are documented in
the config help string. The defaults are conservative and safe to
leave alone:

| Knob | Default | What it does |
|---|---|---|
| `fsdp_mixed_precision` | `"bf16"` | Parameter / gradient / buffer dtype. `"bf16"` is recommended on modern GPUs. `"fp16"` needs a loss scaler. `"fp32"` skips mixed precision entirely. |
| `fsdp_auto_wrap_min_params` | `10_000_000` | Size-based auto-wrap threshold. Modules with at least this many parameters get their own FSDP unit. Set to `0` to disable auto-wrap and wrap the whole model as one unit. |
| `fsdp_cpu_offload` | `false` | Offload params to CPU when idle. Trades throughput for memory — only useful when the model is too big to fit even after BF16 sharding. |

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

| Field | Default | Description |
|---|---|---|
| `master_addr` | `"127.0.0.1"` | Process-group master address |
| `master_port` | `"12355"` | Process-group master port |
| `num_nodes` | `1` | Total number of nodes |
| `gpus_per_node` | auto (CUDA count) | GPUs per node |
| `node_rank` | `0` | This node's rank |
| `backend` | `"nccl"` | `torch.distributed` backend |
| `parallel_strategy` | `"ddp"` | `"ddp"` or `"fsdp"` |
| `fsdp_mixed_precision` | `"bf16"` | `"fp32"` / `"bf16"` / `"fp16"` |
| `fsdp_auto_wrap_min_params` | `10_000_000` | Size-based auto-wrap threshold |
| `fsdp_cpu_offload` | `false` | Offload params to CPU when idle |

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `NCCL_DEBUG` | NCCL debug verbosity | `WARN` |
| `NCCL_IB_DISABLE` | Disable InfiniBand | `0` |
| `NCCL_NET_GDR_LEVEL` | RDMA level | `2` |

### `torchrun` flags

| Flag | Description |
|---|---|
| `--nnodes` | Total number of nodes |
| `--nproc_per_node` | Processes (GPUs) per node |
| `--node_rank` | This node's rank |
| `--master_addr` | Master node address |
| `--master_port` | Master node port |

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
torchrun scripts/train.py
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
- [Tier 3 ticket #2](../audits/2026-07-12-tickets/29-fsdp-e2e-docs.md)
  — the audit follow-up that wired FSDP through the config +
  state-dict helpers.
