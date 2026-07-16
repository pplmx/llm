"""Tests for the AdaLoRA fields added to ``TrainingConfig`` (T3 #42)."""

from __future__ import annotations

import pytest

from llm.training.core.config import TrainingConfig


class TestTrainingConfigAdaLoRAFields:
    """Defaults preserve current behavior; opt-in via ``use_adalora=True``."""

    def test_defaults_match_ticket_spec(self):
        config = TrainingConfig()
        assert config.use_adalora is False
        assert config.adalora_init_rank == 12
        assert config.adalora_target_rank == 6
        assert config.adalora_alpha == 32.0
        assert config.adalora_orth_reg_weight == 0.5
        assert config.adalora_ema_alpha == 0.95
        assert config.adalora_tinit == 0
        assert config.adalora_tfinal is None
        assert config.adalora_prune_every == 50
        assert config.adalora_target_modules is None

    def test_existing_fields_unchanged(self):
        """The schema must not break any pre-existing TrainingConfig field."""
        config = TrainingConfig()
        assert config.batch_size == 128
        assert config.epochs == 10
        assert config.lr == pytest.approx(1e-3)
        assert config.scheduler_type == "cosine"

    def test_adalora_ema_alpha_must_be_in_open_unit_interval(self):
        with pytest.raises(ValueError, match="adalora_ema_alpha"):
            TrainingConfig(adalora_ema_alpha=0.0)
        with pytest.raises(ValueError, match="adalora_ema_alpha"):
            TrainingConfig(adalora_ema_alpha=1.0)
        with pytest.raises(ValueError, match="adalora_ema_alpha"):
            TrainingConfig(adalora_ema_alpha=1.5)

    def test_adalora_prune_every_must_be_positive(self):
        with pytest.raises(ValueError, match="adalora_prune_every"):
            TrainingConfig(adalora_prune_every=0)
        with pytest.raises(ValueError, match="adalora_prune_every"):
            TrainingConfig(adalora_prune_every=-1)

    def test_adalora_tinit_must_be_non_negative(self):
        with pytest.raises(ValueError, match="adalora_tinit"):
            TrainingConfig(adalora_tinit=-1)

    def test_adalora_target_rank_must_not_exceed_init_rank(self):
        with pytest.raises(ValueError, match="adalora_target_rank"):
            TrainingConfig(adalora_init_rank=4, adalora_target_rank=8)

    def test_adalora_tfinal_must_exceed_tinit_when_set(self):
        with pytest.raises(ValueError, match="adalora_tfinal"):
            TrainingConfig(adalora_tinit=10, adalora_tfinal=10)
        with pytest.raises(ValueError, match="adalora_tfinal"):
            TrainingConfig(adalora_tinit=10, adalora_tfinal=5)

    def test_adapt_to_explicit_kwargs(self):
        """Users can opt in with explicit kwargs without touching defaults."""
        config = TrainingConfig(
            use_adalora=True,
            adalora_init_rank=8,
            adalora_target_rank=4,
            adalora_alpha=16.0,
            adalora_ema_alpha=0.9,
            adalora_tinit=10,
            adalora_tfinal=200,
            adalora_prune_every=25,
            adalora_target_modules=["q_proj", "v_proj"],
        )
        assert config.use_adalora is True
        assert config.adalora_init_rank == 8
        assert config.adalora_target_rank == 4
        assert config.adalora_alpha == 16.0
        assert config.adalora_ema_alpha == 0.9
        assert config.adalora_tinit == 10
        assert config.adalora_tfinal == 200
        assert config.adalora_prune_every == 25
        assert config.adalora_target_modules == ["q_proj", "v_proj"]
