"""DataModules that wire datasets into the training engine."""

from llm.data.modules.aifeedback import AIFeedbackDataModule
from llm.data.modules.dpo import DPODataModule
from llm.data.modules.grpo import GRPODataModule
from llm.data.modules.map_base import TokenizedMapDataModule
from llm.data.modules.prompt import PromptDataModule
from llm.data.modules.rejection_sample import RejectionSampleDataModule
from llm.data.modules.reward import RewardDataModule
from llm.data.modules.sft import SFTDataModule
from llm.data.modules.streaming import StreamingTextDataModule
from llm.data.modules.synthetic import SyntheticDataModule
from llm.data.modules.text import TextDataModule

__all__ = [
    "AIFeedbackDataModule",
    "DPODataModule",
    "GRPODataModule",
    "PromptDataModule",
    "RejectionSampleDataModule",
    "RewardDataModule",
    "SFTDataModule",
    "StreamingTextDataModule",
    "SyntheticDataModule",
    "TextDataModule",
    "TokenizedMapDataModule",
]
