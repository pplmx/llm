"""PPO training task integrated with the TrainingEngine."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LRScheduler

from llm.runtime.tokenizer_factory import TokenizerFactory
from llm.tokenization.tokenizer import BaseTokenizer
from llm.training.rlhf.config import PPOConfig
from llm.training.rlhf.ppo_trainer import PPOTrainer
from llm.training.tasks.base_task import TrainingTask
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.reward_task import RewardModel

if TYPE_CHECKING:
    from llm.training.core.engine import TrainingEngine


class PPOTask(TrainingTask):
    """RLHF task that delegates rollout + PPO updates to PPOTrainer."""

    uses_standard_loop = False

    def __init__(self, config: Any, data_module: Any):
        super().__init__(config, data_module)
        self.ppo_trainer: PPOTrainer | None = None
        self.tokenizer: BaseTokenizer | None = None
        self._lm_helper = LanguageModelingTask(config, data_module)

    def build_model(self) -> nn.Module:
        return self._lm_helper.build_model()

    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return self._lm_helper.build_optimizer(model)

    def build_scheduler(self, optimizer: optim.Optimizer) -> LRScheduler | None:
        return self._lm_helper.build_scheduler(optimizer)

    def build_criterion(self) -> nn.Module:
        return nn.Identity()

    def train_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        raise RuntimeError("PPOTask uses run_training() instead of train_step().")

    def validation_step(self, batch, model: nn.Module, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
        raise RuntimeError("PPOTask uses run_training() instead of validation_step().")

    def _load_tokenizer(self) -> BaseTokenizer:
        return TokenizerFactory.from_data_config(self.config.data)

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, DDP) else model

    def _build_reward_model(self) -> RewardModel:
        policy = self.build_model()
        reward_model = RewardModel(policy)

        reward_path = self.config.rlhf.reward_model_path
        if reward_path:
            state = torch.load(reward_path, map_location="cpu", weights_only=True)
            reward_model.load_state_dict(state)

        return reward_model

    def _to_ppo_config(self) -> PPOConfig:
        ppo = self.config.ppo
        return PPOConfig(
            clip_epsilon=ppo.clip_epsilon,
            kl_coef=ppo.kl_coef,
            value_coef=ppo.value_coef,
            entropy_coef=ppo.entropy_coef,
            gae_lambda=ppo.gae_lambda,
            gamma=ppo.gamma,
            ppo_epochs=ppo.ppo_epochs,
            mini_batch_size=ppo.mini_batch_size,
            max_grad_norm=ppo.max_grad_norm,
            target_kl=ppo.target_kl,
            rollout_batch_size=ppo.rollout_batch_size,
            response_max_len=ppo.response_max_len,
            temperature=ppo.temperature,
            top_k=ppo.top_k,
            top_p=ppo.top_p,
            normalize_advantages=ppo.normalize_advantages,
            normalize_rewards=ppo.normalize_rewards,
            policy_lr=ppo.policy_lr,
            value_lr=ppo.value_lr,
            use_ref_model=ppo.use_ref_model,
            ref_model_update_freq=ppo.ref_model_update_freq,
        )

    def prepare_training(self, engine: TrainingEngine) -> None:
        self.tokenizer = self._load_tokenizer()
        policy = self._unwrap_model(engine.model)
        reward_model = self._build_reward_model().to(engine.device)

        self.ppo_trainer = PPOTrainer(
            policy_model=policy,
            reward_model=reward_model,
            tokenizer=self.tokenizer,
            config=self._to_ppo_config(),
            device=engine.device,
        )

    def run_training(self, engine: TrainingEngine) -> None:
        if self.ppo_trainer is None:
            raise RuntimeError("prepare_training() must be called before run_training().")

        dataloader = engine.dataloader
        training_start = time.time()

        for epoch in range(engine.start_epoch, engine.config.training.epochs):
            if engine.sampler is not None:
                engine.sampler.set_epoch(epoch)

            epoch_metrics: list[dict[str, float]] = []
            num_batches = len(dataloader)

            for batch_idx, batch in enumerate(dataloader):
                prompts = batch["prompts"]
                metrics = self.ppo_trainer.train_step(prompts)
                epoch_metrics.append(metrics)

                if (batch_idx + 1) % engine.config.logging.log_interval == 0 and engine.rank == 0:
                    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                    engine.logger.info(f"Epoch {epoch + 1:2d} | Batch {batch_idx + 1:4d}/{num_batches} | {metrics_str}")

            avg_loss = 0.0
            if epoch_metrics:
                avg_loss = sum(m.get("loss", 0.0) for m in epoch_metrics) / len(epoch_metrics)

            if engine.rank == 0:
                engine.logger.info("-" * 80)
                engine.logger.info(
                    f"Epoch {epoch + 1:2d}/{engine.config.training.epochs} SUMMARY | PPO Loss: {avg_loss:.4f}"
                )
                engine.logger.info("-" * 80)
                engine.checkpoint_manager.save_checkpoint(
                    epoch,
                    engine.model,
                    self.ppo_trainer.optimizer,
                    None,
                    None,
                    avg_loss,
                )

        if engine.rank == 0:
            total_time = time.time() - training_start
            engine.logger.info(f"PPO training completed in {total_time / 3600:.2f} hours.")
