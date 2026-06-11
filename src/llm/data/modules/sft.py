from llm.data.datasets.sft import SFTDataset
from llm.data.modules.map_base import TokenizedMapDataModule


class SFTDataModule(TokenizedMapDataModule):
    """DataModule for Supervised Fine-Tuning (SFT) using SFTDataset."""

    def setup(self, stage: str | None = None):
        self.setup_tokenized_file_dataset(SFTDataset, stage)
