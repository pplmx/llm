"""Tests for StreamDataModule — streaming data config validation."""

from dataclasses import dataclass

import pytest
from torch.utils.data import DataLoader, DistributedSampler

from llm.data.base import StreamDataModule


@dataclass
class _MockDataConfig:
    steps_per_epoch: int | None = None


@dataclass
class _MockConfig:
    data: _MockDataConfig


class _ConcreteStreamModule(StreamDataModule):
    """Minimal concrete subclass for testing StreamDataModule methods."""

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
        pass

    def train_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader, DistributedSampler | None]:
        raise NotImplementedError

    def val_dataloader(self, rank: int, world_size: int) -> tuple[DataLoader | None, DistributedSampler | None]:
        raise NotImplementedError


def _make_config(steps: int | None = 100):
    return _MockConfig(data=_MockDataConfig(steps_per_epoch=steps))


def test_stream_module_validate_passes():
    """validate_streaming_config passes when steps_per_epoch > 0."""
    module = _ConcreteStreamModule(_make_config(100))
    module.validate_streaming_config()


def test_stream_module_validate_fails_none():
    """validate_streaming_config raises when steps_per_epoch is None."""
    module = _ConcreteStreamModule(_make_config(None))
    with pytest.raises(ValueError, match="steps_per_epoch > 0"):
        module.validate_streaming_config()


def test_stream_module_validate_fails_zero():
    """validate_streaming_config raises when steps_per_epoch is 0."""
    module = _ConcreteStreamModule(_make_config(0))
    with pytest.raises(ValueError, match="steps_per_epoch > 0"):
        module.validate_streaming_config()


def test_stream_module_validate_fails_negative():
    """validate_streaming_config raises when steps_per_epoch is negative."""
    module = _ConcreteStreamModule(_make_config(-1))
    with pytest.raises(ValueError, match="steps_per_epoch > 0"):
        module.validate_streaming_config()


def test_stream_module_is_streaming_true():
    """is_streaming is True by default."""
    module = _ConcreteStreamModule(_make_config(10))
    assert module.is_streaming is True


def test_stream_module_get_checkpoint_state():
    """get_checkpoint_state returns None by default."""
    module = _ConcreteStreamModule(_make_config(10))
    assert module.get_checkpoint_state() is None


def test_stream_module_load_checkpoint_state():
    """load_checkpoint_state is a no-op."""
    module = _ConcreteStreamModule(_make_config(10))
    module.load_checkpoint_state({"some": "state"})
