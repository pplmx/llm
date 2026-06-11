from llm.data.datasets.text import TextDataset
from llm.data.modules.map_base import TokenizedMapDataModule


class TextDataModule(TokenizedMapDataModule):
    """DataModule for Language Modeling using TextDataset."""

    def setup(self, stage: str | None = None):
        self.setup_tokenized_file_dataset(TextDataset, stage)
