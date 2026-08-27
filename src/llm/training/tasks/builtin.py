"""Register all built-in training tasks."""

from llm.data.modules.dpo import DPODataModule
from llm.data.modules.grpo import GRPODataModule
from llm.data.modules.prompt import PromptDataModule
from llm.data.modules.reward import RewardDataModule
from llm.data.modules.sft import SFTDataModule
from llm.data.modules.streaming import StreamingTextDataModule
from llm.data.modules.synthetic import SyntheticDataModule
from llm.data.modules.text import TextDataModule
from llm.multimodal.data import MultimodalDataModule
from llm.multimodal.task import MultimodalTask
from llm.training.task_registry import TASK_REGISTRY
from llm.training.tasks.distill_task import DistillationTask
from llm.training.tasks.dpo_task import DPOTask
from llm.training.tasks.grpo_task import GRPOTask
from llm.training.tasks.lm_task import LanguageModelingTask
from llm.training.tasks.ppo_task import PPOTask
from llm.training.tasks.regression_task import RegressionTask
from llm.training.tasks.reward_task import RewardTask
from llm.training.tasks.sft_task import SFTTask
from llm.training.tasks.simpo_task import SimPOTask

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
TASK_REGISTRY.register("simpo", SimPOTask, DPODataModule, description="SimPO (reference-free preference optimization)")
TASK_REGISTRY.register(
    "distill", DistillationTask, TextDataModule, description="Knowledge distillation (student vs frozen teacher)"
)
TASK_REGISTRY.register(
    "grpo", GRPOTask, GRPODataModule, description="GRPO (group-relative advantage policy optimization)"
)
TASK_REGISTRY.register(
    "multimodal", MultimodalTask, MultimodalDataModule, description="Multimodal LM (modal-fusion prefix conditioning)"
)
TASK_REGISTRY.register("reward", RewardTask, RewardDataModule, description="Reward model training")
TASK_REGISTRY.register("ppo", PPOTask, PromptDataModule, description="PPO RLHF alignment")
