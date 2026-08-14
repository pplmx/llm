"""Tests for PPO task integration."""

import copy
import json

import pytest

from llm.data.modules.prompt import PromptDataModule
from llm.training.core.callbacks import Callback
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
def test_ppo_task_uses_custom_loop(tmp_path, tiny_model, monkeypatch):
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
    assert task.uses_standard_training_loop() is False

    reward_base = copy.deepcopy(tiny_model)

    def fake_build_model(self):
        if not hasattr(self, "_policy_built"):
            self._policy_built = True
            return tiny_model
        return reward_base

    monkeypatch.setattr(PPOTask, "build_model", fake_build_model)
    monkeypatch.setattr(PPOTask, "_load_tokenizer", lambda self: tokenizer)

    engine = TrainingEngine(
        config=config,
        task=task,
        rank=0,
        world_size=1,
        data_module=data_module,
        callbacks=[],
    )

    assert engine.use_standard_loop is False
    assert type(task.ppo_trainer) is PPOTrainer
    engine.run()


class _StepCountingCallback(Callback):
    """Counts optimizer steps seen through the on_train_step_end hook."""

    def __init__(self):
        super().__init__()
        self.steps = 0

    def on_train_step_end(self, epoch, batch_idx, loss, metrics, logs=None):
        self.steps += 1


@pytest.mark.quick
def test_ppo_task_honors_max_steps(tmp_path, tiny_model, monkeypatch):
    """Training.max_steps is a hard cap on total PPO optimizer steps.

    Regression: the custom (non-standard) loop iterated every batch of every
    epoch without consulting ``engine.global_step`` against max_steps, so a
    smoke config meant to stop at N steps ran through the whole dataloader
    (and every remaining epoch), and never advanced ``engine.global_step``
    (leaving TensorBoard at step 0). The standard-loop cap must apply here too.
    """
    prompt_file = tmp_path / "prompts.jsonl"
    _write_prompts(prompt_file, ["Hello", "Hi there", "Test prompt", "More"])
    tokenizer = CharBoundTokenizer()

    config = Config()
    config.data.dataset_path = str(prompt_file)
    config.training.batch_size = 2
    config.training.epochs = 3  # 4 prompts / bs2 => 2 steps/epoch, 6 total
    config.training.max_steps = 3  # stop after 3 optimizer steps total
    config.optimization.num_workers = 0
    config.optimization.use_compile = False
    config.ppo.ppo_epochs = 1
    config.ppo.response_max_len = 2
    config.ppo.mini_batch_size = 1
    data_module = PromptDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    task = PPOTask(config, data_module)

    reward_base = copy.deepcopy(tiny_model)

    def fake_build_model(self):
        if not hasattr(self, "_policy_built"):
            self._policy_built = True
            return tiny_model
        return reward_base

    monkeypatch.setattr(PPOTask, "build_model", fake_build_model)
    monkeypatch.setattr(PPOTask, "_load_tokenizer", lambda self: tokenizer)

    counter = _StepCountingCallback()
    engine = TrainingEngine(
        config=config,
        task=task,
        rank=0,
        world_size=1,
        data_module=data_module,
        callbacks=[counter],
    )

    engine.run()

    # Exactly max_steps PPO updates ran — not all 6 (and not all 3 epochs).
    assert counter.steps == 3, f"expected 3 PPO steps, got {counter.steps}"
    assert engine.global_step == 3


@pytest.mark.quick
def test_ppo_policy_keeps_ddp_wrapper(monkeypatch):
    """Regression (RIL round-47): ``prepare_training`` must hand the trainer
    the DDP-wrapped ``engine.model`` — not its unwrapped ``.module``.

    With the OLD code the trainer's every forward ran on the bare module, so
    on multi-GPU the DDP gradient all-reduce was never triggered and each
    rank silently trained a divergent model (only rank 0's policy was ever
    checkpointed). The policy must keep the wrapper.
    """
    from unittest.mock import MagicMock

    import llm.training.tasks.ppo_task as ppo_module
    from llm.training.core.config import Config

    captured: dict = {}
    recorder = MagicMock()

    def fake_trainer_cls(policy_model, reward_model, tokenizer, config, device):
        captured["policy"] = policy_model
        captured["reward"] = reward_model
        captured["tokenizer"] = tokenizer
        captured["config"] = config
        captured["device"] = device
        return recorder

    class _FakeDDP:
        """Stand-in for DistributedDataParallel: training must receive THIS
        object (a multi-GPU PPO run has one), never the bare ``.module``."""

        def __init__(self):
            self.module = object()

    monkeypatch.setattr(ppo_module, "PPOTrainer", fake_trainer_cls)
    monkeypatch.setattr(ppo_module, "DistributedDataParallel", _FakeDDP)
    monkeypatch.setattr(ppo_module.PPOTask, "_load_tokenizer", lambda self: object())
    monkeypatch.setattr(ppo_module.PPOTask, "_build_reward_model", lambda self: MagicMock())

    ddp_model = _FakeDDP()
    fake_engine = MagicMock()
    fake_engine.device = "cpu"
    fake_engine.model = ddp_model

    task = PPOTask(Config(), MagicMock())
    task.prepare_training(fake_engine)

    assert captured["policy"] is ddp_model, "policy must keep the DDP wrapper (gradient all-reduce)"
    assert captured["policy"] is not ddp_model.module, "policy must not be the unwrapped .module"
    assert captured["reward"] is not None
    assert captured["device"] == "cpu"
