"""Tests for PPO task callback integration (Finding F).

Custom-loop tasks (PPO/RLHF) must fire the standard callback hooks so
observability tooling (TensorBoardLogger, MetricsLogger, EarlyStopping)
works for RLHF the same way it does for SFT.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm.data.modules.prompt import PromptDataModule
from llm.training.core.config import Config
from llm.training.core.engine import TrainingEngine
from llm.training.rlhf.ppo_trainer import PPOTrainer
from llm.training.tasks.ppo_task import PPOTask
from tests.support.tokenizers import CharBoundTokenizer


def _write_prompts(path, prompts):
    with path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps({"prompt": prompt}) + "\n")


@pytest.mark.quick
def test_ppo_fires_on_train_step_end(tmp_path, tiny_model, monkeypatch):
    """PPO must emit ``on_train_step_end`` after each PPO step so
    TensorBoardLogger and LRSchedulerCallback observe the loss."""
    prompt_file = tmp_path / "prompts.jsonl"
    _write_prompts(prompt_file, ["Hello", "Hi there", "Test prompt"])
    tokenizer = CharBoundTokenizer()

    config = Config()
    config.data.dataset_path = str(prompt_file)
    config.training.batch_size = 2
    config.training.epochs = 1
    config.optimization.num_workers = 0
    config.optimization.use_compile = False
    config.ppo.ppo_epochs = 1
    config.ppo.response_max_len = 2
    config.ppo.mini_batch_size = 1
    data_module = PromptDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    task = PPOTask(config, data_module)

    from copy import deepcopy

    reward_base = deepcopy(tiny_model)

    def fake_build_model(self):
        if not hasattr(self, "_policy_built"):
            self._policy_built = True
            return tiny_model
        return reward_base

    monkeypatch.setattr(PPOTask, "build_model", fake_build_model)
    monkeypatch.setattr(PPOTask, "_load_tokenizer", lambda self: tokenizer)

    captured: list[dict[str, Any]] = []

    class _Capture(PPOTask if False else object):  # avoid metaclass conflict; real Callback below
        pass

    from llm.training.core.callbacks import Callback

    class CaptureCallback(Callback):
        def on_train_step_end(self, epoch, batch_idx, loss, metrics, logs=None):
            captured.append(
                {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "loss": float(loss.item()),
                    "metrics": dict(metrics),
                }
            )

    engine = TrainingEngine(
        config=config,
        task=task,
        rank=0,
        world_size=1,
        data_module=data_module,
        callbacks=[CaptureCallback()],
    )

    engine.run()

    assert captured, "PPO never fired on_train_step_end"
    for entry in captured:
        assert "loss" in entry
        assert entry["epoch"] == 0
        assert "metrics" in entry
    # At least one PPO step must have fired the step hook. We don't pin
    # the exact count because PromptDataModule may filter prompts whose
    # tokenized length exceeds ``max_seq_len`` — that's an implementation
    # detail of the data module, not the callback contract under test.
    assert len(captured) >= 1


@pytest.mark.quick
def test_ppo_fires_on_epoch_end(tmp_path, tiny_model, monkeypatch):
    """``on_epoch_end`` must fire after each PPO epoch."""
    prompt_file = tmp_path / "prompts.jsonl"
    _write_prompts(prompt_file, ["Hello", "Hi there", "Test prompt", "Another one"])
    tokenizer = CharBoundTokenizer()

    config = Config()
    config.data.dataset_path = str(prompt_file)
    config.training.batch_size = 2
    config.training.epochs = 2
    config.optimization.num_workers = 0
    config.optimization.use_compile = False
    config.ppo.ppo_epochs = 1
    config.ppo.response_max_len = 2
    config.ppo.mini_batch_size = 1
    data_module = PromptDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    task = PPOTask(config, data_module)

    from copy import deepcopy

    reward_base = deepcopy(tiny_model)

    def fake_build_model(self):
        if not hasattr(self, "_policy_built"):
            self._policy_built = True
            return tiny_model
        return reward_base

    monkeypatch.setattr(PPOTask, "build_model", fake_build_model)
    monkeypatch.setattr(PPOTask, "_load_tokenizer", lambda self: tokenizer)

    from llm.training.core.callbacks import Callback

    epoch_calls: list[int] = []

    class EpochCapture(Callback):
        def on_epoch_start(self, epoch, logs=None):
            epoch_calls.append(epoch)

        def on_epoch_end(self, epoch, logs=None):
            epoch_calls.append(100 + epoch)  # mark end with offset

    engine = TrainingEngine(
        config=config,
        task=task,
        rank=0,
        world_size=1,
        data_module=data_module,
        callbacks=[EpochCapture()],
    )

    engine.run()

    # on_epoch_start(0), on_epoch_end(0), on_epoch_start(1), on_epoch_end(1)
    # PromptDataModule may drop prompts that exceed max length, but the
    # callback contract must fire for every configured epoch.
    assert epoch_calls == [0, 100, 1, 101]


@pytest.mark.quick
def test_ppo_honors_should_stop_training(tmp_path, tiny_model, monkeypatch):
    """A callback that sets ``engine.should_stop_training`` must halt the
    RLHF loop after the current epoch."""
    prompt_file = tmp_path / "prompts.jsonl"
    _write_prompts(prompt_file, ["Hello", "Hi there", "Test prompt", "Another one"])
    tokenizer = CharBoundTokenizer()

    config = Config()
    config.data.dataset_path = str(prompt_file)
    config.training.batch_size = 2
    config.training.epochs = 5  # would normally run 5 epochs
    config.optimization.num_workers = 0
    config.optimization.use_compile = False
    config.ppo.ppo_epochs = 1
    config.ppo.response_max_len = 2
    config.ppo.mini_batch_size = 1
    data_module = PromptDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    task = PPOTask(config, data_module)

    from copy import deepcopy

    reward_base = deepcopy(tiny_model)

    def fake_build_model(self):
        if not hasattr(self, "_policy_built"):
            self._policy_built = True
            return tiny_model
        return reward_base

    monkeypatch.setattr(PPOTask, "build_model", fake_build_model)
    monkeypatch.setattr(PPOTask, "_load_tokenizer", lambda self: tokenizer)

    epoch_count = {"n": 0}
    trainer_call_count = {"n": 0}

    from llm.training.core.callbacks import Callback

    class Stopper(Callback):
        def on_epoch_end(self, epoch, logs=None):
            epoch_count["n"] += 1
            if epoch_count["n"] >= 2:
                self.engine.should_stop_training = True

    # Wrap the PPOTrainer.train_step to count invocations; it should stop
    # increasing after the loop halts.
    original_train_step = PPOTrainer.train_step

    def counting_train_step(self, prompts):
        trainer_call_count["n"] += 1
        return original_train_step(self, prompts)

    monkeypatch.setattr(PPOTrainer, "train_step", counting_train_step)

    engine = TrainingEngine(
        config=config,
        task=task,
        rank=0,
        world_size=1,
        data_module=data_module,
        callbacks=[Stopper()],
    )

    engine.run()

    # Epoch 0 completes, epoch 1 completes and triggers stop,
    # epochs 2-4 are skipped.
    assert epoch_count["n"] == 2
    # 2 epochs x 2 batches/epoch = 4 PPO steps total (3 prompts / 2 = 2 batches).
    # Stop happens AFTER epoch 1 finishes, so we expect exactly 4 calls.
    assert trainer_call_count["n"] == 4
