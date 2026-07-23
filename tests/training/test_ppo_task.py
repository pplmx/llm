"""Tests for PPO task integration."""

import copy
import json

import pytest
import torch

from llm.data.modules.prompt import PromptDataModule
from llm.training.core.config import Config
from llm.training.core.engine import TrainingEngine
from llm.training.rlhf.ppo_trainer import PPOTrainer
from llm.training.tasks.ppo_task import PPOTask
from tests.support.tokenizers import CharBoundTokenizer


@pytest.fixture
def device():
    """Force CPU — CUDA OOMs on constrained boxes."""
    return torch.device("cpu")


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
