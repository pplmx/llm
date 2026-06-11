from llm.data.datasets.dpo import DPODataset
from llm.data.modules.map_base import TokenizedMapDataModule


class DPODataModule(TokenizedMapDataModule):
    """DataModule for Direct Preference Optimization (DPO)."""

    def setup(self, stage: str | None = None):
        self.setup_tokenized_file_dataset(DPODataset, stage)
