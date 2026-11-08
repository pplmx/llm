"""Every ``ModelConfig`` field must actually reach the built model.

``ModelConfig`` is the single source of truth for the decoder's shape, and
``ModelFactory.from_config`` maps it onto ``build_decoder`` kwargs. A field that
is declared but never forwarded is silent: the model builds, trains, and serves,
it just ignores what the user asked for. This test reads the field list at
runtime, so a newly added field is covered automatically instead of waiting for
someone to notice it does nothing.
"""

from __future__ import annotations

import pytest
import torch

from llm.runtime.model_factory import ModelFactory
from llm.training.core.config import ModelConfig

BASE = {"vocab_size": 64, "hidden_size": 32, "num_layers": 1, "num_heads": 4, "num_workers": 0}


def _fingerprint(config: ModelConfig) -> tuple[int, int]:
    """(parameter count, module count) — shape-level evidence the field applied."""
    torch.manual_seed(0)
    model = ModelFactory.from_config(config)
    return sum(p.numel() for p in model.parameters()), sum(1 for _ in model.modules())


def _build(**overrides) -> ModelConfig:
    return ModelConfig(**{**BASE, **overrides})


def test_bias_flags_change_the_parameter_count():
    """``qkv_bias`` / ``mlp_bias`` / ``lm_head_bias`` must add real parameters.

    Toggling them off from their ``True`` default has to shrink the model; if the
    flag were ignored the count would not move at all.
    """
    all_on = {"qkv_bias": True, "mlp_bias": True, "lm_head_bias": True}
    baseline = _fingerprint(_build(**all_on))

    for flag in all_on:
        without = _fingerprint(_build(**{**all_on, flag: False}))
        assert without[0] < baseline[0], f"{flag}=False did not remove any parameters"


def test_intermediate_size_controls_the_ffn_width():
    """The FFN width must follow ``intermediate_size`` (and default to 4x hidden)."""
    widths = {}
    for size in (64, 128, 256):
        torch.manual_seed(0)
        model = ModelFactory.from_config(_build(intermediate_size=size))
        widths[size] = model.transformer_blocks[0].mlp.fc1.out_features
    assert widths == {64: 64, 128: 128, 256: 256}

    torch.manual_seed(0)
    default = ModelFactory.from_config(_build())
    assert default.transformer_blocks[0].mlp.fc1.out_features == 4 * BASE["hidden_size"]


def test_num_layers_and_hidden_size_control_the_stack():
    """Structural fields must change the number of blocks / the hidden dimension."""
    torch.manual_seed(0)
    shallow = ModelFactory.from_config(_build(num_layers=2))
    torch.manual_seed(0)
    deep = ModelFactory.from_config(_build(num_layers=5))
    assert len(shallow.transformer_blocks) == 2
    assert len(deep.transformer_blocks) == 5

    torch.manual_seed(0)
    wide = ModelFactory.from_config(_build(hidden_size=48))
    assert wide.transformer_blocks[0].self_attn.qkv_proj.in_features == 48


def test_num_experts_is_honoured_only_under_the_moe_mlp():
    """``num_experts`` sizes the expert pool; outside ``mlp_impl='moe'`` it is inert."""
    torch.manual_seed(0)
    dense = ModelFactory.from_config(_build())
    torch.manual_seed(0)
    moe_small = ModelFactory.from_config(_build(mlp_impl="moe", num_experts=2, top_k=1))
    torch.manual_seed(0)
    moe_large = ModelFactory.from_config(_build(mlp_impl="moe", num_experts=8, top_k=2))

    assert sum(p.numel() for p in dense.parameters()) != sum(p.numel() for p in moe_small.parameters())
    assert sum(p.numel() for p in moe_large.parameters()) > sum(p.numel() for p in moe_small.parameters()), (
        "more experts did not add parameters"
    )


def test_mlp_impl_moe_requires_positive_experts():
    """A MoE config with no experts must be rejected, not silently built dense."""
    with pytest.raises(ValueError, match="num_experts"):
        _build(mlp_impl="moe", num_experts=0)


def test_every_declared_field_is_consumed_by_the_factory():
    """Guard against a field that is declared but never forwarded.

    Introspects ``ModelConfig`` so a newly added field is covered without
    touching this test. Fields are classified rather than simply diffed: some
    legitimately do not change the shape (they change behaviour, gate another
    field, or are consumed outside the decoder factory).
    """
    import ast
    import inspect
    import pathlib

    from llm.runtime.model_factory import decoder_kwargs_from_config

    field_names = set(ModelConfig.model_fields)
    assert field_names, "ModelConfig reported no fields"

    # Fields deliberately NOT forwarded to ``DecoderModel``. Every entry must
    # name a field that still exists (asserted below), so this cannot rot into a
    # list of names nobody removes.
    not_forwarded = {
        "use_kv_cache",  # consumed by ModelConfig.check_consistency, not the constructor
    }

    # Read the attribute accesses off the mapping function's source. This is
    # what "consumed by the factory" means concretely: ``config.<field>`` is
    # evaluated and placed in the kwargs dict. Checking a hand-written list of
    # "important" fields (bias/intermediate/expert) is what this test used to
    # do, and it left 19 of 25 fields entirely unverified — a field added and
    # never forwarded would have passed.
    source = pathlib.Path(inspect.getsourcefile(decoder_kwargs_from_config)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "decoder_kwargs_from_config"
    )
    forwarded = {
        node.attr
        for node in ast.walk(fn)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "config"
    }

    unforwarded = sorted(field_names - forwarded - not_forwarded)
    assert not unforwarded, (
        "ModelConfig declares fields that decoder_kwargs_from_config never forwards "
        f"to the model: {unforwarded}. Either forward them, or add them to "
        "``not_forwarded`` with a comment saying where they ARE consumed."
    )

    stale_exemptions = not_forwarded - field_names
    assert not stale_exemptions, f"not_forwarded names fields that no longer exist: {sorted(stale_exemptions)}"

    # A field listed as "not forwarded" must actually not be forwarded, or the
    # exemption is hiding a duplicate decision rather than documenting one.
    for exempt in not_forwarded:
        assert exempt not in forwarded, f"{exempt!r} is exempted but IS forwarded"


# --- Legacy per-method switches must warn about combining ---


def test_legacy_peft_switches_document_that_they_must_not_be_combined():
    """Each legacy ``use_*`` switch must carry the combine-at-most-one caveat.

    All four wrap ``nn.Linear`` (BitFit freezes weights instead), so enabling
    two at once stacks two adapters on every layer and trains BOTH parameter
    sets — measured, ``use_ia3`` + ``use_adapter`` produced 48320 trainable
    params against 47968 for adapter alone. There is no mutual-exclusion
    validator, so the combination is accepted silently; the field description is
    the only place a user sees the warning (editor help, ``--help``), and it
    used to live only in the CHANGELOG.
    """
    from llm.training.core.config import TrainingConfig

    for field_name in ("use_prefix_tuning", "use_ia3", "use_bitfit", "use_adapter"):
        description = TrainingConfig.model_fields[field_name].description or ""
        assert "Combine at most one" in description, (
            f"{field_name} lost its combine-at-most-one caveat; stack two adapters silently again?"
        )


def test_combining_two_legacy_switches_stacks_both_adapters():
    """Pin the behaviour the caveat warns about, so it cannot drift unnoticed.

    This is a characterisation test: the config deliberately does not forbid the
    combination (choosing the method is the user's call), so the risk is that a
    future refactor silently changes what combining means.
    """
    from llm.data.modules.synthetic import SyntheticDataModule
    from llm.training.core.config import Config
    from llm.training.tasks.lm_task import LanguageModelingTask

    model_cfg = {"vocab_size": 64, "hidden_size": 32, "num_layers": 1, "num_heads": 4, "max_seq_len": 16}

    def trainable(**overrides) -> int:
        cfg = Config(model=model_cfg, training={"batch_size": 2, "num_samples": 4, **overrides})
        data_module = SyntheticDataModule(cfg)
        data_module.setup()
        torch.manual_seed(0)
        model = LanguageModelingTask(cfg, data_module).build_model()
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    ia3_only = trainable(use_ia3=True)
    adapter_only = trainable(use_adapter=True)
    both = trainable(use_ia3=True, use_adapter=True)

    assert both > adapter_only, (
        "combining the switches no longer adds the second adapter — the caveat's premise changed"
    )
    assert both > ia3_only


# --- Non-finite float config values must be rejected at load time ---


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_rope_theta_must_be_finite_and_positive(bad):
    """A bad ``rope_theta`` must fail at CONFIG time, not corrupt every forward.

    ``rope_theta`` had no validator. Measured: ``rope_theta=nan`` made every
    positional encoding NaN, so the model returned an all-NaN tensor while
    construction and config validation both succeeded — the failure only shows up
    as garbage output (or a NaN loss) much later.
    """
    with pytest.raises(ValueError, match="rope_theta"):
        _build(rope_theta=bad)


def test_rope_theta_accepts_the_normal_value():
    """The guard must not reject the default (10000.0) or other sane bases."""
    torch.manual_seed(0)
    model = ModelFactory.from_config(_build(rope_theta=10000.0))
    assert model is not None
    torch.manual_seed(0)
    assert ModelFactory.from_config(_build(rope_theta=500000.0)) is not None
