"""The decoder exports as a PREFILL + DECODE graph pair that decodes incrementally.

ADR-016 measured why the export path forces ``use_cache=False``: ``KVCache``
writes into a pre-allocated buffer, and an exported graph is stateless, so a
decoded token had to recompute the whole prefix (248 positions versus 24, a 10.3x
gap at prompt=8/new=16).

Two obstacles had to be found, not assumed:

1. **The cache must be a pure function** — ``FunctionalKVCache`` (added
   separately) returns the updated K/V instead of mutating a buffer.
2. **One graph cannot serve both phases.** ``DecoderModel.forward`` derives
   ``start_pos`` and ``has_past`` from ``kv_caches[0].seq_len`` — PYTHON values,
   which a tracer freezes into constants with the traced shape baked in. The
   frozen ``is_causal`` alone does not break a graph; what breaks it is using a
   graph traced at PREFILL shape to decode. Measured: such a graph diverges by
   ~0.34 on decode steps, while a graph traced at DECODE shape matches eager to
   <5e-7. So the fix is two graphs, each traced at its own phase's shape — the
   standard ONNX/TensorRT decoder pattern.

The pair is chained exactly as a runtime would: prefill once with an empty cache,
then feed each step's ``present_*`` back as the next ``past_*``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm.core.kv_cache import KVCache
from llm.export.onnx import export_decoder_for_generation
from llm.models.decoder import DecoderModel

VOCAB = 64
HIDDEN = 32
LAYERS = 2
HEADS = 2
MAX_SEQ = 32
PROMPT = 4
NEW_TOKENS = 4


def _model() -> DecoderModel:
    torch.manual_seed(0)
    return DecoderModel(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_layers=LAYERS, num_heads=HEADS, max_seq_len=MAX_SEQ
    ).eval()


def _geometry(model: DecoderModel) -> tuple[int, int]:
    block = model.transformer_blocks[0]
    attention = getattr(block, "attention", None) or getattr(block, "self_attn", None)
    assert attention is not None
    num_kv = getattr(attention, "num_kv_heads", HEADS)
    return num_kv, model.embedding_layer.token_embeddings.embedding_dim // HEADS


def _eager_reference(model, prompt, steps):
    """Prefill + decode with the production in-place cache (the ground truth)."""
    num_kv, head_dim = _geometry(model)
    caches = [KVCache(1, MAX_SEQ, num_kv, head_dim) for _ in range(LAYERS)]
    outputs = []
    with torch.no_grad():
        out = model(prompt, use_cache=True, kv_caches=caches)
        out = out[0] if isinstance(out, tuple) else out
        outputs.append(out[:, -1:])
        for token in steps:
            out = model(token, use_cache=True, kv_caches=caches)
            out = out[0] if isinstance(out, tuple) else out
            outputs.append(out)
    return outputs


@pytest.mark.quick
def test_export_writes_both_phase_graphs(tmp_path):
    """Export produces a prefill graph and a decode graph, each taking cache I/O."""
    import onnx

    paths = export_decoder_for_generation(_model(), tmp_path, prompt_len=PROMPT)

    assert set(paths) == {"prefill", "decode"}, f"expected both phase graphs, got {sorted(paths)}"
    for name, path in paths.items():
        proto = onnx.load(str(path))
        inputs = {value.name for value in proto.graph.input}
        outputs = {value.name for value in proto.graph.output}
        assert {"ids", "past_k", "past_v"} <= inputs, f"{name}: cache must be an INPUT, has {sorted(inputs)}"
        assert {"logits", "present_k", "present_v"} <= outputs, (
            f"{name}: cache must be an OUTPUT, has {sorted(outputs)}"
        )
        assert len(proto.graph.node) >= 5, f"{name}: expected real ops, found {len(proto.graph.node)} nodes"


@pytest.mark.quick
def test_the_two_graphs_decode_incrementally_and_match_eager(tmp_path):
    """Chaining prefill -> decode reproduces eager decoding to float32 noise.

    This is the whole point: without a working cache path, every decoded token
    recomputes the prefix. Agreement with the in-place-cache reference proves the
    exported pair is a faithful re-expression.
    """
    import onnxruntime as ort

    model = _model()
    paths = export_decoder_for_generation(model, tmp_path, prompt_len=PROMPT)
    prefill = ort.InferenceSession(str(paths["prefill"]), providers=["CPUExecutionProvider"])
    decode = ort.InferenceSession(str(paths["decode"]), providers=["CPUExecutionProvider"])

    torch.manual_seed(5)
    prompt = torch.randint(0, VOCAB, (1, PROMPT))
    steps = [torch.randint(0, VOCAB, (1, 1)) for _ in range(NEW_TOKENS)]
    expected = _eager_reference(model, prompt, steps)

    num_kv, head_dim = _geometry(model)
    past_k = np.zeros((LAYERS, 1, num_kv, 0, head_dim), dtype=np.float32)
    past_v = np.zeros_like(past_k)
    logits, past_k, past_v = prefill.run(None, {"ids": prompt.numpy(), "past_k": past_k, "past_v": past_v})
    got = [logits[:, -1:]]
    for token in steps:
        logits, past_k, past_v = decode.run(None, {"ids": token.numpy(), "past_k": past_k, "past_v": past_v})
        got.append(logits)

    assert past_k.shape[3] == PROMPT + NEW_TOKENS, (
        f"the cache must accumulate every token ({PROMPT}+{NEW_TOKENS}), got {past_k.shape[3]}"
    )
    for step, (want, actual) in enumerate(zip(expected, got, strict=True)):
        max_diff = float(np.abs(want.numpy() - actual).max())
        assert max_diff < 1e-5, (
            f"step {step}: exported decode diverged from eager (max |d|={max_diff:.3e}). "
            "A divergence here is what a single prefill-traced graph produces (~0.34)."
        )


@pytest.mark.quick
def test_a_single_graph_cannot_serve_both_phases(tmp_path):
    """Why two graphs exist: a prefill-traced graph mis-decodes.

    Pinning the negative result keeps the two-graph design justified — if this
    ever starts matching eager, the extra graph could be dropped.
    """
    import onnxruntime as ort

    model = _model()
    paths = export_decoder_for_generation(model, tmp_path, prompt_len=PROMPT)
    prefill = ort.InferenceSession(str(paths["prefill"]), providers=["CPUExecutionProvider"])

    torch.manual_seed(5)
    prompt = torch.randint(0, VOCAB, (1, PROMPT))
    token = torch.randint(0, VOCAB, (1, 1))
    expected = _eager_reference(model, prompt, [token])

    num_kv, head_dim = _geometry(model)
    past_k = np.zeros((LAYERS, 1, num_kv, 0, head_dim), dtype=np.float32)
    past_v = np.zeros_like(past_k)
    _, past_k, past_v = prefill.run(None, {"ids": prompt.numpy(), "past_k": past_k, "past_v": past_v})
    logits, _, _ = prefill.run(None, {"ids": token.numpy(), "past_k": past_k, "past_v": past_v})

    max_diff = float(np.abs(expected[-1].numpy() - logits).max())
    assert max_diff > 1e-3, (
        f"reusing the prefill graph for a decode step matched eager to {max_diff:.3e}. The frozen "
        "start_pos/has_past constants apparently no longer matter, so the separate decode graph may "
        "be unnecessary — re-measure before simplifying."
    )


@pytest.mark.quick
def test_the_pair_can_be_verified_after_export(tmp_path):
    """A caller must be able to check what it exported.

    ``verify_onnx`` cannot do it — it feeds ``input_ids`` only, while the pair
    takes ``ids``/``past_k``/``past_v``, so an export was previously unverifiable
    from outside this test module. That is the gap shape that let the GGUF offsets
    ship broken: a writer whose output nobody could independently check.
    """
    from llm.export.onnx import verify_decoder_generation_graphs

    model = _model()
    export_decoder_for_generation(model, tmp_path, prompt_len=PROMPT)
    assert verify_decoder_generation_graphs(tmp_path, model, prompt_len=PROMPT, new_tokens=3) is True


@pytest.mark.quick
def test_the_verifier_rejects_graphs_from_a_different_model(tmp_path):
    """The verifier must be able to say NO, or returning True proves nothing.

    Exports a pair from a DIFFERENT model (same architecture, different weights)
    and checks it against the first one: the decode must diverge.
    """
    from llm.export.onnx import verify_decoder_generation_graphs

    model = _model()
    torch.manual_seed(999)
    other = DecoderModel(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_layers=LAYERS, num_heads=HEADS, max_seq_len=MAX_SEQ
    ).eval()
    assert not torch.equal(model.transformer_blocks[0].mlp.fc1.weight, other.transformer_blocks[0].mlp.fc1.weight), (
        "the two models must actually differ for this test to mean anything"
    )

    export_decoder_for_generation(other, tmp_path, prompt_len=PROMPT)
    assert verify_decoder_generation_graphs(tmp_path, model, prompt_len=PROMPT, new_tokens=3) is False, (
        "the verifier accepted a pair exported from a different model; it cannot be trusted to detect a wrong export"
    )


@pytest.mark.quick
def test_the_verifier_names_the_missing_graph(tmp_path):
    """A half-exported directory must fail with the path, not a cryptic ORT error."""
    from llm.export.onnx import verify_decoder_generation_graphs

    model = _model()
    export_decoder_for_generation(model, tmp_path, prompt_len=PROMPT)
    (tmp_path / "decode.onnx").unlink()

    with pytest.raises(FileNotFoundError, match=r"decode\.onnx"):
        verify_decoder_generation_graphs(tmp_path, model, prompt_len=PROMPT)


@pytest.mark.quick
def test_verify_onnx_cannot_check_the_pair(tmp_path):
    """Pin WHY the dedicated verifier exists.

    ``verify_onnx`` is the single-graph checker and feeds ``input_ids``; using it
    on the pair raises rather than silently verifying the wrong thing. If this
    ever starts passing, the dedicated verifier may be redundant.
    """
    from llm.export.onnx import verify_onnx

    model = _model()
    paths = export_decoder_for_generation(model, tmp_path, prompt_len=PROMPT)
    with pytest.raises((ValueError, KeyError)):
        verify_onnx(paths["prefill"], model, input_shape=(1, PROMPT))
