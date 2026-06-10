"""Register all built-in training tasks."""

from llm.data.modules.dpo import DPODataModule
from llm.data.modules.prompt import PromptDataModule
from llm.data.modules.reward import RewardDataModule
from llm.data.modules.sft import SFTDataModule
from llm.data.modules.streaming import StreamingTextDataModule
from llm.data.modules.synthetic import SyntheticDataModule
from llm.data.modules.text import TextDataModule
from llm.training.task_registry import TASK_REGISTRY
from llm.training.tasks.dpo_task import DPOTask
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.ppo_task import PPOTask
from llm.training.tasks.regression_task import RegressionTask
from llm.training.tasks.reward_task import RewardTask
from llm.training.tasks.sft_task import SFTTask

TASK_REGISTRY.register("regression", RegressionTask, SyntheticDataModule, description="Synthetic regression demo")
TASK_REGISTRY.register("lm", LanguageModelingTask, TextDataModule, description="Map-style language modeling")
TASK_REGISTRY.register(
    "stream_lm",
    LanguageModelingTask,
    StreamingTextDataModule,
    description="Streaming language modeling for large corpora",
)
TASK_REGISTRY.register("sft", SFTTask, SFTDataModule, description="Supervised fine-tuning")
TASK_REGISTRY.register("dpo", DPOTask, DPODataModule, description="Direct preference optimization")
TASK_REGISTRY.register("reward", RewardTask, RewardDataModule, description="Reward model training")
TASK_REGISTRY.register("ppo", PPOTask, PromptDataModule, description="PPO RLHF alignment")
