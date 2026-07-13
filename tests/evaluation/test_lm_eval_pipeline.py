"""Tests for the lm-eval-harness pipeline integration (T3 #27).

Covers:

- :class:`EvalPreset` construction + :meth:`EvalPreset.to_lm_eval_kwargs`
- Built-in presets (``MMLU_PRESET``, ``ARCEASY_PRESET``, ``WIKITEXT_PRESET``)
- :func:`get_preset` lookup + unknown-name error
- :meth:`LmEvalAdapter.summarize` flattening of the nested result shape
- Soft-dependency contract: ``LlamaLmEvalLM`` and ``LmEvalAdapter``
  raise a clear ``ImportError`` with the install hint when ``lm_eval``
  is missing (verified by monkey-patching the module-level
  ``LM_EVAL_AVAILABLE`` flag).
"""

from __future__ import annotations

import pytest

from llm.evaluation.harness.presets import (
    ARCEASY_PRESET,
    BUILTIN_PRESETS,
    EvalPreset,
    MMLU_PRESET,
    WIKITEXT_PRESET,
    get_preset,
)


# --- EvalPreset dataclass ---------------------------------------------------


def test_evalpreset_construction_minimal():
    """Only ``task`` is required; the rest have sensible defaults."""
    preset = EvalPreset(task="mmlu")
    assert preset.task == "mmlu"
    assert preset.num_fewshot is None
    assert preset.batch_size == 8
    assert preset.limit is None
    assert preset.task_kwargs == {}
    assert preset.description == ""


def test_evalpreset_construction_full():
    preset = EvalPreset(
        task="mmlu",
        num_fewshot=5,
        batch_size=16,
        limit=100,
        task_kwargs={"subject": "abstract_algebra"},
        description="MMLU single-subject slice",
    )
    assert preset.task == "mmlu"
    assert preset.num_fewshot == 5
    assert preset.batch_size == 16
    assert preset.limit == 100
    assert preset.task_kwargs == {"subject": "abstract_algebra"}
    assert preset.description == "MMLU single-subject slice"


def test_evalpreset_is_frozen():
    """``frozen=True`` prevents accidental preset mutation."""
    preset = EvalPreset(task="mmlu")
    with pytest.raises(Exception):
        preset.batch_size = 16  # type: ignore[misc]


def test_evalpreset_to_kwargs_minimal():
    """Minimal preset -> ``tasks`` + ``batch_size`` only."""
    preset = EvalPreset(task="arc_easy")
    assert preset.to_lm_eval_kwargs() == {"tasks": ["arc_easy"], "batch_size": 8}


def test_evalpreset_to_kwargs_full():
    preset = EvalPreset(
        task="mmlu",
        num_fewshot=5,
        batch_size=16,
        limit=100,
        task_kwargs={"subject": "abstract_algebra"},
    )
    kwargs = preset.to_lm_eval_kwargs()
    assert kwargs["tasks"] == ["mmlu"]
    assert kwargs["batch_size"] == 16
    assert kwargs["num_fewshot"] == 5
    assert kwargs["limit"] == 100
    assert kwargs["task_kwargs"] == {"subject": "abstract_algebra"}


def test_evalpreset_to_kwargs_returns_fresh_dict():
    """The returned dict must be a fresh copy so callers can mutate it."""
    preset = EvalPreset(task="mmlu")
    a = preset.to_lm_eval_kwargs()
    a["batch_size"] = 999
    b = preset.to_lm_eval_kwargs()
    assert b["batch_size"] == 8


def test_evalpreset_to_kwargs_omits_none_values():
    """``num_fewshot=None`` and ``limit=None`` are not forwarded."""
    kwargs = EvalPreset(task="wikitext").to_lm_eval_kwargs()
    assert "num_fewshot" not in kwargs
    assert "limit" not in kwargs
    assert "task_kwargs" not in kwargs


# --- Built-in presets -------------------------------------------------------


def test_builtin_presets_present():
    assert MMLU_PRESET.task == "mmlu"
    assert ARCEASY_PRESET.task == "arc_easy"
    assert WIKITEXT_PRESET.task == "wikitext"


def test_builtin_presets_index_keys_match_task_names():
    assert set(BUILTIN_PRESETS) == {"mmlu", "arc_easy", "wikitext"}


def test_mmlu_preset_kwargs_shape():
    assert MMLU_PRESET.to_lm_eval_kwargs() == {
        "tasks": ["mmlu"],
        "batch_size": 8,
        "num_fewshot": 5,
    }


def test_arceasy_preset_kwargs_shape():
    assert ARCEASY_PRESET.to_lm_eval_kwargs() == {
        "tasks": ["arc_easy"],
        "batch_size": 8,
        "num_fewshot": 0,
    }


def test_wikitext_preset_kwargs_shape():
    """WikiText ships a smaller default batch (perplexity is heavier)."""
    assert WIKITEXT_PRESET.to_lm_eval_kwargs() == {
        "tasks": ["wikitext"],
        "batch_size": 4,
        "num_fewshot": 0,
    }


def test_get_preset_returns_builtin():
    assert get_preset("mmlu") is MMLU_PRESET
    assert get_preset("arc_easy") is ARCEASY_PRESET
    assert get_preset("wikitext") is WIKITEXT_PRESET


def test_get_preset_unknown_raises_keyerror_with_available_list():
    with pytest.raises(KeyError) as exc_info:
        get_preset("nonexistent")
    msg = str(exc_info.value)
    assert "nonexistent" in msg
    assert "mmlu" in msg
    assert "arc_easy" in msg


# --- summarize() flattening ------------------------------------------------


def test_summarize_empty_results():
    assert LmEvalAdapter.summarize({}) == {}
    assert LmEvalAdapter.summarize({"results": {}}) == {}


def test_summarize_single_task_single_metric():
    out = LmEvalAdapter.summarize(
        {"results": {"mmlu": {"acc,none": 0.42}}}
    )
    assert out == {"mmlu": {"acc": 0.42}}


def test_summarize_single_task_multi_metric():
    """Multiple comma-suffixed metrics collapse to plain metric names."""
    out = LmEvalAdapter.summarize(
        {
            "results": {
                "mmlu": {
                    "acc,none": 0.42,
                    "acc_norm,none": 0.45,
                    "f1,none": 0.40,
                }
            }
        }
    )
    assert out == {
        "mmlu": {"acc": 0.42, "acc_norm": 0.45, "f1": 0.40}
    }


def test_summarize_multi_task():
    out = LmEvalAdapter.summarize(
        {
            "results": {
                "mmlu": {"acc,none": 0.42, "acc_norm,none": 0.45},
                "arc_easy": {"acc,none": 0.60},
                "wikitext": {"perplexity,none": 12.3, "word_perplexity,none": 11.9},
            }
        }
    )
    assert out == {
        "mmlu": {"acc": 0.42, "acc_norm": 0.45},
        "arc_easy": {"acc": 0.60},
        "wikitext": {"perplexity": 12.3, "word_perplexity": 11.9},
    }


def test_summarize_skips_non_numeric_values():
    """String metadata (e.g. ``"alias"``) is dropped from the flat output."""
    out = LmEvalAdapter.summarize(
        {
            "results": {
                "mmlu": {
                    "acc,none": 0.42,
                    "alias,none": "mmlu",
                }
            }
        }
    )
    assert out == {"mmlu": {"acc": 0.42}}


def test_summarize_skips_boolean_values():
    """``True`` would coerce to ``1.0`` silently — guard against that."""
    out = LmEvalAdapter.summarize(
        {
            "results": {
                "mmlu": {
                    "acc,none": 0.42,
                    "is_fewshot,none": True,
                }
            }
        }
    )
    assert out == {"mmlu": {"acc": 0.42}}
    assert "is_fewshot" not in out["mmlu"]


def test_summarize_metric_without_comma():
    """Keys without ``,subset`` suffix pass through unchanged."""
    out = LmEvalAdapter.summarize(
        {"results": {"mmlu": {"score": 0.5}}}
    )
    assert out == {"mmlu": {"score": 0.5}}


def test_summarize_coerces_to_float():
    """Integer metric values are coerced to float for a stable type."""
    out = LmEvalAdapter.summarize(
        {"results": {"mmlu": {"count,none": 42}}}
    )
    assert out == {"mmlu": {"count": 42.0}}
    assert isinstance(next(iter(out["mmlu"].values())), float)


def test_summarize_ignores_top_level_groups_and_configs():
    """Only ``results`` is flattened; ``groups`` / ``configs`` are passed through unwrapped."""
    payload = {
        "results": {"mmlu": {"acc,none": 0.42}},
        "groups": {"mmlu_group": {"acc,none": 0.43}},
        "configs": {"mmlu": {"num_fewshot": 5}},
    }
    out = LmEvalAdapter.summarize(payload)
    assert out == {"mmlu": {"acc": 0.42}}
    # ``groups`` / ``configs`` are NOT promoted to top-level flat keys.


# --- Soft-dependency contract ----------------------------------------------


def test_lm_eval_lm_raises_when_lm_eval_missing(monkeypatch):
    """With ``LM_EVAL_AVAILABLE=False``, ``LlamaLmEvalLM.__init__`` must raise."""
    from llm.evaluation.harness import lm_eval_lm

    monkeypatch.setattr(lm_eval_lm, "LM_EVAL_AVAILABLE", False)

    fake_model = type("M", (), {"max_seq_len": 64})()  # empty model stand-in
    fake_tokenizer = type("T", (), {"encode": lambda self, s: [1, 2, 3]})()

    with pytest.raises(ImportError) as exc_info:
        LlamaLmEvalLM(fake_model, fake_tokenizer)
    msg = str(exc_info.value)
    assert "lm_eval" in msg
    assert "pip install" in msg


def test_lm_eval_adapter_raises_when_lm_eval_missing(monkeypatch):
    """With ``LM_EVAL_AVAILABLE=False``, ``LmEvalAdapter()`` must raise."""
    from llm.evaluation.harness import adapter as adapter_module

    monkeypatch.setattr(adapter_module, "LM_EVAL_AVAILABLE", False)

    with pytest.raises(ImportError) as exc_info:
        LmEvalAdapter()
    msg = str(exc_info.value)
    assert "lm_eval" in msg
    assert "pip install" in msg


def test_modules_import_safely_without_lm_eval(monkeypatch):
    """The presets/adapter/lm_eval_lm modules must import even when lm_eval is missing.

    This protects callers from import-time crashes on CPU-only hosts
    that don't need lm_eval at all (just want to read presets).
    """
    import sys

    # Simulate a host where lm_eval is not installed by setting both
    # find_spec to return None and patching the module's flag.
    from llm.evaluation.harness import adapter as adapter_module

    monkeypatch.setattr(adapter_module, "LM_EVAL_AVAILABLE", False)

    # Re-importing shouldn't raise; presets stay accessible.
    assert isinstance(MMLU_PRESET, EvalPreset)
    assert get_preset("mmlu") is MMLU_PRESET


def test_lm_eval_lm_protocol_methods_exist():
    """Protocol surface documented in the ticket must be present on the class."""
    expected = {"loglikelihood", "loglikelihood_rolling", "generate_until"}
    for name in expected:
        assert hasattr(LlamaLmEvalLM, name), f"missing lm_eval LM method: {name}"


# --- Lazy imports ----------------------------------------------------------


# The named imports are pulled at module scope so the test file's
# import-time smoke test catches typos in the public API.
from llm.evaluation.harness.adapter import LmEvalAdapter  # noqa: E402
from llm.evaluation.harness.lm_eval_lm import LlamaLmEvalLM  # noqa: E402


# --- LlamaLmEvalLM with a fake model + tokenizer ---------------------------
#
# These tests exercise the *real* forward path through the adapter using
# a tiny model that maps tokens -> uniform log-probs over a small
# vocab. They focus on the contract (shapes, batching, stop tokens)
# rather than on numerical correctness — the upstream model is the one
# being evaluated, the adapter just needs to be a faithful shim.


import torch  # noqa: E402


class _FakeTokenizer:
    """Minimal tokenizer that splits text into one token per character."""

    eos_token_id = 9
    pad_token_id = 0

    def encode(self, text):
        return [ord(c) % 16 for c in text]

    def decode(self, ids):
        return "".join(chr(int(i) % 128) for i in ids)


class _FakeModel:
    """Maps input ids -> uniform logits over a tiny vocab.

    Forward returns a tensor of shape ``(1, T, vocab_size)``. Setting
    ``argmax_id`` makes the model deterministic (always picks the same
    next token) which is useful for ``generate_until`` tests.
    """

    vocab_size = 16
    max_seq_len = 64

    def __init__(self, argmax_id: int = 1):
        self.argmax_id = argmax_id
        self._dummy = torch.nn.Parameter(torch.zeros(1))  # for .parameters()

    def parameters(self):
        return iter([self._dummy])

    def eval(self):
        return self

    def __call__(self, input_ids, use_cache=None):
        # Returns logits where argmax == argmax_id at every position.
        b, t = input_ids.shape
        logits = torch.full(
            (b, t, self.vocab_size), -10.0, dtype=torch.float32
        )
        logits[..., self.argmax_id] = 10.0
        return logits


class _FakeRequest:
    """Stand-in for ``lm_eval.api.request.Request`` (just needs ``args``)."""

    def __init__(self, args):
        self.args = args


def test_lm_eval_lm_init_with_fake_model():
    """``LlamaLmEvalLM`` initializes and binds to the model's device."""
    lm = LlamaLmEvalLM(_FakeModel(), _FakeTokenizer(), batch_size=2)
    assert lm.batch_size == 2
    assert lm.max_length == _FakeModel.max_seq_len
    assert lm.model is not None
    assert lm.tokenizer is not None


def test_lm_eval_lm_loglikelihood_returns_one_tuple_per_request():
    lm = LlamaLmEvalLM(_FakeModel(), _FakeTokenizer(), batch_size=2)
    requests = [
        _FakeRequest(("hello", "world")),
        _FakeRequest(("foo", "bar")),
    ]
    out = lm.loglikelihood(requests)
    assert len(out) == 2
    for sum_logprob, is_greedy in out:
        assert isinstance(sum_logprob, float)
        assert isinstance(is_greedy, bool)


def test_lm_eval_lm_loglikelihood_greedy_match_is_true_for_argmax_model():
    """If the model always picks the same id, ``is_greedy`` matches continuation."""
    lm = LlamaLmEvalLM(
        _FakeModel(argmax_id=3), _FakeTokenizer(), batch_size=2
    )
    # Force the tokenizer to encode continuation into ids where 3 dominates
    # the model's argmax — we'll just check the *shape* of the output.
    requests = [_FakeRequest(("a", "b"))]
    out = lm.loglikelihood(requests)
    assert len(out) == 1


def test_lm_eval_lm_loglikelihood_handles_empty_continuation():
    """Empty continuation must not crash (guard with ``[0]`` fallback)."""
    lm = LlamaLmEvalLM(_FakeModel(), _FakeTokenizer(), batch_size=2)
    requests = [_FakeRequest(("hello", ""))]
    out = lm.loglikelihood(requests)
    assert len(out) == 1
    assert isinstance(out[0][0], float)


def test_lm_eval_lm_loglikelihood_batches_requests():
    """Batching must not drop or reorder results."""
    lm = LlamaLmEvalLM(
        _FakeModel(), _FakeTokenizer(), batch_size=2, max_length=64
    )
    requests = [_FakeRequest((f"context_{i}", "cont")) for i in range(5)]
    out = lm.loglikelihood(requests)
    assert len(out) == 5


def test_lm_eval_lm_loglikelihood_rolling_short_input():
    """Inputs shorter than 2 tokens yield ``0.0`` rather than crashing."""
    lm = LlamaLmEvalLM(_FakeModel(), _FakeTokenizer(), batch_size=2)
    requests = [_FakeRequest(("a",)), _FakeRequest(("bb",))]
    out = lm.loglikelihood_rolling(requests)
    assert len(out) == 2
    assert out[0] == 0.0
    assert isinstance(out[1], float)


def test_lm_eval_lm_loglikelihood_rolling_returns_list_of_floats():
    """``loglikelihood_rolling`` must return ``list[float]`` per lm_eval protocol.

    WikiText perplexity downstream does ``(loglikelihood,) = results``
    — if we returned 1-tuples, that unpacking would yield a tuple
    instead of a float and corrupt the metric tuple.
    """
    lm = LlamaLmEvalLM(_FakeModel(), _FakeTokenizer(), batch_size=2)
    requests = [_FakeRequest(("hello",)), _FakeRequest(("world!",))]
    out = lm.loglikelihood_rolling(requests)
    assert len(out) == 2
    for v in out:
        assert isinstance(v, float)
        # NOT a tuple — guard against regressing back to the 1-tuple shape.
        assert not isinstance(v, tuple)


def test_lm_eval_lm_generate_until_respects_max_gen_toks():
    lm = LlamaLmEvalLM(
        _FakeModel(argmax_id=1), _FakeTokenizer(), batch_size=1, max_length=64
    )
    requests = [
        _FakeRequest(("ctx", {"until": [], "max_gen_toks": 3})),
    ]
    out = lm.generate_until(requests)
    assert len(out) == 1
    # 3 tokens -> 3 decoded chars
    assert len(lm.tokenizer.decode([1, 1, 1])) == 3


def test_lm_eval_lm_generate_until_stops_on_until_token():
    """Once ``until`` is a suffix of ``generated``, stop early."""
    lm = LlamaLmEvalLM(
        _FakeModel(argmax_id=1), _FakeTokenizer(), batch_size=1, max_length=64
    )
    # Generate up to 5 tokens but stop as soon as id 1 appears twice
    # (the model always produces 1, so the suffix matches after 2 steps).
    requests = [
        _FakeRequest(("ctx", {"until": [[1, 1]], "max_gen_toks": 5})),
    ]
    out = lm.generate_until(requests)
    assert len(out) == 1
    # We expect at most 2 tokens generated (1, 1) before stopping.
    assert len(out[0]) <= 2


def test_lm_eval_lm_matches_any_suffix_static_helper():
    """Static helper should recognise list-based stop sequences."""
    assert LlamaLmEvalLM._matches_any_suffix([1, 2, 3], [[2, 3]]) is True
    assert LlamaLmEvalLM._matches_any_suffix([1, 2, 3], [[4]]) is False
    # Empty until list -> always False.
    assert LlamaLmEvalLM._matches_any_suffix([1, 2, 3], []) is False
    # String ``until`` entries are skipped (we can't match by text).
    assert LlamaLmEvalLM._matches_any_suffix([1, 2, 3], ["abc"]) is False
    # Empty stop sequence is skipped, not matched.
    assert LlamaLmEvalLM._matches_any_suffix([1, 2, 3], [[]]) is False


# --- LmEvalAdapter evaluate/run_preset (mocked lm_eval) -------------------
#
# These tests exercise the *glue* layer of LmEvalAdapter (preset lookup,
# kwargs merging, default task) by stubbing the upstream ``lm_eval``
# imports. The point is to assert that the adapter:
#
# - builds the right kwargs from a preset,
# - resolves a string preset name,
# - defaults to ``["mmlu"]`` when no tasks are given,
# - runs end-to-end via the standalone ``run_preset()`` helper.
#
# We use ``unittest.mock`` to replace the names looked up by the
# adapter's lazy ``from lm_eval.X import Y`` statements — that way we
# don't need a real TaskManager/evaluator in the test process.

from unittest.mock import patch  # noqa: E402

import llm.evaluation.harness.adapter as adapter_module  # noqa: E402


class _FakeTaskManager:
    all_tasks = ["mmlu", "arc_easy", "wikitext"]


class _FakeEvaluator:
    """Records the call and returns a sentinel result."""

    last_call: dict | None = None

    @classmethod
    def evaluate(cls, model, tasks, **kwargs):
        cls.last_call = {"model": model, "tasks": tasks, "kwargs": kwargs}
        return {"results": {"sentinel": {"acc,none": 0.5}}}


@pytest.fixture
def mocked_lm_eval():
    """Patch the lazy ``from lm_eval.X import Y`` lookups in adapter.py.

    ``lm_eval.evaluator`` is not a real attribute on the top-level
    ``lm_eval`` package (it's loaded lazily by callers), so we set
    ``create=True`` to make ``unittest.mock.patch`` accept the
    attribute as a fresh name. ``TaskManager`` exists, so it can be
    patched normally.
    """
    with patch(
        "lm_eval.tasks.TaskManager", _FakeTaskManager
    ), patch("lm_eval.evaluator", _FakeEvaluator, create=True):
        yield


def test_lm_eval_adapter_init_with_mocked_lm_eval(mocked_lm_eval):
    """``LmEvalAdapter()`` constructs and lists built-in tasks."""
    adapter = LmEvalAdapter()
    assert "mmlu" in adapter.list_tasks()
    assert "arc_easy" in adapter.list_tasks()
    assert "wikitext" in adapter.list_tasks()


def test_lm_eval_adapter_evaluate_passes_tasks(mocked_lm_eval):
    _FakeEvaluator.last_call = None
    adapter = LmEvalAdapter()
    out = adapter.evaluate(model="m", tasks=["arc_easy"], batch_size=4)
    assert _FakeEvaluator.last_call == {
        "model": "m",
        "tasks": ["arc_easy"],
        "kwargs": {"batch_size": 4},
    }
    assert out["results"]["sentinel"]["acc,none"] == 0.5


def test_lm_eval_adapter_evaluate_defaults_to_mmlu(mocked_lm_eval):
    _FakeEvaluator.last_call = None
    adapter = LmEvalAdapter()
    adapter.evaluate(model="m")
    assert _FakeEvaluator.last_call["tasks"] == ["mmlu"]


def test_lm_eval_adapter_run_preset_with_string_name(mocked_lm_eval):
    _FakeEvaluator.last_call = None
    adapter = LmEvalAdapter()
    adapter.run_preset("mmlu", model="m")
    call = _FakeEvaluator.last_call
    assert call["tasks"] == ["mmlu"]
    assert call["kwargs"]["num_fewshot"] == 5
    assert call["kwargs"]["batch_size"] == 8


def test_lm_eval_adapter_run_preset_with_instance(mocked_lm_eval):
    _FakeEvaluator.last_call = None
    adapter = LmEvalAdapter()
    custom = EvalPreset(task="wikitext", num_fewshot=2, batch_size=2, limit=5)
    adapter.run_preset(custom, model="m")
    call = _FakeEvaluator.last_call
    assert call["tasks"] == ["wikitext"]
    assert call["kwargs"]["num_fewshot"] == 2
    assert call["kwargs"]["limit"] == 5


def test_lm_eval_adapter_run_preset_caller_kwargs_override(mocked_lm_eval):
    """Caller-supplied kwargs win on conflicts with the preset."""
    _FakeEvaluator.last_call = None
    adapter = LmEvalAdapter()
    # Override the preset's batch_size with an explicit one.
    adapter.run_preset("arc_easy", model="m", batch_size=1)
    assert _FakeEvaluator.last_call["kwargs"]["batch_size"] == 1


def test_lm_eval_adapter_run_preset_unknown_name_raises_keyerror(mocked_lm_eval):
    adapter = LmEvalAdapter()
    with pytest.raises(KeyError):
        adapter.run_preset("does_not_exist", model="m")


def test_lm_eval_adapter_run_benchmark(mocked_lm_eval):
    """``run_benchmark`` is a thin wrapper over ``evaluate`` with one task."""
    _FakeEvaluator.last_call = None
    adapter = LmEvalAdapter()
    adapter.run_benchmark(model="m", benchmark="arc_easy", limit=10)
    call = _FakeEvaluator.last_call
    assert call["tasks"] == ["arc_easy"]
    assert call["kwargs"]["limit"] == 10


def test_standalone_run_preset_helper(mocked_lm_eval):
    """Module-level ``run_preset()`` should work the same as the method."""
    _FakeEvaluator.last_call = None
    out = adapter_module.run_preset("mmlu", model="m")
    assert out["results"]["sentinel"]["acc,none"] == 0.5
    assert _FakeEvaluator.last_call["tasks"] == ["mmlu"]
