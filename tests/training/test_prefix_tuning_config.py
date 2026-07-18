"""Tests for the Prefix Tuning fields added to ``TrainingConfig``.

Mirrors the AdaLoRA trainer-integration config tests (T3 #42): defaults
preserve current behavior (so existing configs are unaffected), field
constraints catch obvious misuse, and explicit kwargs flow through.
"""

from __future__ import annotations

import pytest

from llm.training.core.config import TrainingConfig


class TestTrainingConfigPrefixTuningFields:
    """Defaults preserve current behavior; opt-in via ``use_prefix_tuning=True``."""

    def test_defaults_match_paper_recommendation(self):
        """Off by default; paper-recommended prefix length (10) when on."""
        config = TrainingConfig()
        assert config.use_prefix_tuning is False
        assert config.prefix_tuning_len == 10
        # None → reparam defaults to kv_dim at construction time.
        assert config.prefix_reparam_hidden is None
        # None → every MultiHeadAttention is wrapped.
        assert config.prefix_target_modules is None

    def test_existing_fields_unchanged(self):
        """Adding prefix tuning fields must not break any pre-existing TrainingConfig field."""
        config = TrainingConfig()
        assert config.batch_size == 128
        assert config.epochs == 10
        assert config.lr == pytest.approx(1e-3)
        assert config.scheduler_type == "cosine"
        # AdaLoRA fields untouched.
        assert config.use_adalora is False
        assert config.adalora_init_rank == 12

    def test_prefix_tuning_len_must_be_positive(self):
        with pytest.raises(ValueError, match="prefix_tuning_len"):
            TrainingConfig(prefix_tuning_len=0)
        with pytest.raises(ValueError, match="prefix_tuning_len"):
            TrainingConfig(prefix_tuning_len=-1)

    def test_prefix_reparam_hidden_must_be_positive_when_set(self):
        with pytest.raises(ValueError, match="prefix_reparam_hidden"):
            TrainingConfig(prefix_reparam_hidden=0)
        with pytest.raises(ValueError, match="prefix_reparam_hidden"):
            TrainingConfig(prefix_reparam_hidden=-4)

    def test_prefix_reparam_hidden_none_is_allowed(self):
        """``None`` is the documented "default to kv_dim" sentinel."""
        # Should not raise — None is the explicit "use layer default" value.
        config = TrainingConfig(prefix_reparam_hidden=None)
        assert config.prefix_reparam_hidden is None

    def test_adapt_to_explicit_kwargs(self):
        """Users can opt in with explicit kwargs without touching defaults."""
        config = TrainingConfig(
            use_prefix_tuning=True,
            prefix_tuning_len=20,
            prefix_reparam_hidden=64,
            prefix_target_modules=["attn"],
        )
        assert config.use_prefix_tuning is True
        assert config.prefix_tuning_len == 20
        assert config.prefix_reparam_hidden == 64
        assert config.prefix_target_modules == ["attn"]

    def test_prefix_tuning_does_not_require_adalora_to_be_off(self):
        """Prefix Tuning and AdaLoRA can co-exist; both have independent switches.

        In practice you wouldn't normally combine two PEFT methods on the
        same Linear — they wrap different module classes (AdaLoRA wraps
        ``nn.Linear``, Prefix Tuning wraps ``MultiHeadAttention``) so
        there's no structural conflict — but the config must not
        implicitly forbid the combination.
        """
        config = TrainingConfig(
            use_prefix_tuning=True,
            use_adalora=True,
        )
        assert config.use_prefix_tuning is True
        assert config.use_adalora is True
