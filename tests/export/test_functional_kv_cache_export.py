"""A KV cache can be exported as explicit tensors — the premise ADR-016 rests on.

ADR-016 (`docs/adr/016-tensorrt-coreml-feasibility.md`) found the real blocker for
a TensorRT/ONNX serving path is not operator coverage but that the export wrapper
forces ``use_cache=False``. Decoding without a cache recomputes the whole prefix:
measured 248 positions processed versus 24 with a cache at prompt=8/new=16, a
10.3x gap that grows quadratically with generation length.

The fix it proposes is to express the cache functionally — past K/V as INPUTS,
present K/V as OUTPUTS, concatenated rather than written into a mutable buffer —
so a tracer sees a pure function. This module proves that premise holds, using a
minimal attention block rather than the real model: the point is to pin the
mechanism, which does not depend on DecoderModel's plumbing.

Measured when this was written (torch 2.14.1, opset 17, onnxruntime CPU):

* the functional block traces to ONNX (21.4 KiB);
* the graph runs INCREMENTALLY — feeding each step's ``present_*`` back as the
  next ``past_*`` grows the cache ``(2,4,0,8) -> (2,4,3,8)``;
* outputs match eager-with-cache to ``max|d| = 1.788e-07``.

Why a synthetic block and not the real model: ``DecoderModel`` has no functional
entry point yet (that is the remaining work), so testing it would mean testing
code that does not exist. What can be pinned today is that the export MECHANISM
is sound, so nobody has to re-derive it when the model-side work starts.

A note on the numerics assertion: an early probe reported ``max|d| = 1.06`` and
looked like a tracer bug. It was the probe — two passes each called
``manual_seed`` but consumed the RNG in different orders. Sharing one set of input
tensors between both paths dropped the error to 1.8e-7. The test below fixes the
inputs for exactly that reason.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn

DIM = 32
HEADS = 4
HEAD_DIM = DIM // HEADS
BATCH = 2
STEPS = 3


class FunctionalCacheAttention(nn.Module):
    """Attention whose KV cache is an explicit input/output pair (no mutation)."""

    def __init__(self, dim: int = DIM, heads: int = HEADS) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor):
        batch, new_len, _ = x.shape
        head_dim = self.dim // self.heads
        q, k, v = self.qkv(x).split(self.dim, dim=-1)
        q = q.view(batch, new_len, self.heads, head_dim).transpose(1, 2)
        k = k.view(batch, new_len, self.heads, head_dim).transpose(1, 2)
        v = v.view(batch, new_len, self.heads, head_dim).transpose(1, 2)
        # Concatenate instead of writing into a buffer: a pure function of the
        # inputs, which is what lets the tracer record it.
        k_all = torch.cat([past_k, k], dim=2)
        v_all = torch.cat([past_v, v], dim=2)
        attn = torch.softmax(q @ k_all.transpose(-1, -2) / (head_dim**0.5), dim=-1)
        out = (attn @ v_all).transpose(1, 2).reshape(batch, new_len, self.dim)
        return self.out(out), k_all, v_all


def _export(tmp_path):
    torch.manual_seed(0)
    block = FunctionalCacheAttention().eval()
    path = tmp_path / "functional_cache.onnx"
    empty_k = torch.zeros(BATCH, HEADS, 0, HEAD_DIM)
    empty_v = torch.zeros(BATCH, HEADS, 0, HEAD_DIM)
    torch.onnx.export(
        block,
        (torch.randn(BATCH, 1, DIM), empty_k, empty_v),
        str(path),
        input_names=["hidden", "past_k", "past_v"],
        output_names=["out", "present_k", "present_v"],
        dynamic_axes={
            "hidden": {0: "batch", 1: "new_len"},
            "past_k": {0: "batch", 2: "past_len"},
            "past_v": {0: "batch", 2: "past_len"},
            "out": {0: "batch", 1: "new_len"},
            "present_k": {0: "batch", 2: "total_len"},
            "present_v": {0: "batch", 2: "total_len"},
        },
        opset_version=17,
        dynamo=False,
    )
    return block, path


@pytest.mark.quick
def test_a_functional_kv_cache_traces_with_cache_among_the_inputs(tmp_path):
    """The traced graph exposes past K/V as inputs and present K/V as outputs."""
    path = tmp_path / "functional_cache.onnx"
    _export(tmp_path)

    proto = onnx.load(str(path))
    inputs = {value.name for value in proto.graph.input}
    outputs = {value.name for value in proto.graph.output}

    assert {"past_k", "past_v"} <= inputs, (
        f"the cache must be an explicit INPUT for a tracer to see it; graph inputs are {sorted(inputs)}"
    )
    assert {"present_k", "present_v"} <= outputs, (
        f"the updated cache must be an explicit OUTPUT; graph outputs are {sorted(outputs)}"
    )
    # Anti-vacuity: a graph with no nodes would satisfy the name checks above.
    assert len(proto.graph.node) >= 5, f"expected real ops, found {len(proto.graph.node)} nodes"


@pytest.mark.quick
def test_the_exported_graph_grows_its_cache_incrementally(tmp_path):
    """Feeding present_* back as past_* grows the cache across decode steps."""
    import onnxruntime as ort

    _export(tmp_path)
    session = ort.InferenceSession(str(tmp_path / "functional_cache.onnx"), providers=["CPUExecutionProvider"])

    k = np.zeros((BATCH, HEADS, 0, HEAD_DIM), dtype=np.float32)
    v = np.zeros((BATCH, HEADS, 0, HEAD_DIM), dtype=np.float32)
    saw_lengths = []
    torch.manual_seed(7)
    for _ in range(STEPS):
        x = torch.randn(BATCH, 1, DIM)
        out, k, v = session.run(None, {"hidden": x.numpy(), "past_k": k, "past_v": v})
        saw_lengths.append(k.shape[2])
        assert out.shape == (BATCH, 1, DIM), f"unexpected output shape {out.shape}"

    assert saw_lengths == [1, 2, 3], (
        f"the cache must grow one position per step, saw lengths {saw_lengths}. "
        "A graph that ignored past_* would report the same length every time."
    )


@pytest.mark.quick
def test_the_exported_graph_matches_eager_with_a_carried_cache(tmp_path):
    """Incremental ONNX decoding equals eager decoding with the same cached state.

    Both paths consume the SAME input tensors: an earlier probe re-seeded each
    side and the RNG draw order differed, producing a bogus 1.06 mismatch.
    """
    import onnxruntime as ort

    block, _ = _export(tmp_path)
    session = ort.InferenceSession(str(tmp_path / "functional_cache.onnx"), providers=["CPUExecutionProvider"])

    # One set of inputs, shared by both paths.
    torch.manual_seed(11)
    steps = [torch.randn(BATCH, 1, DIM) for _ in range(STEPS)]

    # Eager with a hand-carried cache.
    k = torch.zeros(BATCH, HEADS, 0, HEAD_DIM)
    v = torch.zeros(BATCH, HEADS, 0, HEAD_DIM)
    eager = []
    with torch.no_grad():
        for x in steps:
            out, k, v = block(x, k, v)
            eager.append(out.numpy())
    expected = np.concatenate(eager, axis=1)

    # The traced graph, same inputs.
    kn = np.zeros((BATCH, HEADS, 0, HEAD_DIM), dtype=np.float32)
    vn = np.zeros((BATCH, HEADS, 0, HEAD_DIM), dtype=np.float32)
    got_steps = []
    for x in steps:
        out, kn, vn = session.run(None, {"hidden": x.numpy(), "past_k": kn, "past_v": vn})
        got_steps.append(out)
    got = np.concatenate(got_steps, axis=1)

    assert tuple(k.shape) == kn.shape, f"cache shapes diverged: eager {tuple(k.shape)} vs onnx {kn.shape}"
    max_diff = float(np.abs(expected - got).max())
    assert max_diff < 1e-5, (
        f"traced functional cache diverged from eager (max |d|={max_diff:.3e}); "
        "the whole point is that it is a faithful re-expression, not an approximation"
    )
