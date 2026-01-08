import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from llm.tokenization.tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning (SFT) / Instruction Tuning.

    Processing flow:
    1. Read JSONL data.
    2. Format into prompt/response using a template.
    3. Tokenize.
    4. Create labels where prompt tokens are masked (set to -100).
    5. Pad to max_seq_len.
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: BaseTokenizer,
        max_seq_len: int = 1024,
        template_fn: Callable[[dict[str, Any]], tuple[str, str]] | None = None,
        padding_value: int = 0,
        ignore_index: int = -100,
    ):
        """
        Args:
            file_path: Path to jsonl file.
            tokenizer: Tokenizer instance.
            max_seq_len: Max sequence length.
            template_fn: Function to convert data item to (prompt, response) tuple.
                         If None, defaults to Alpaca style.
            padding_value: Token ID for padding input_ids.
            ignore_index: Label value for masked tokens (padding/prompt).
        """
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding_value = padding_value
        self.ignore_index = ignore_index

        self.template_fn = template_fn or self.alpaca_template

        self.data = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        data = []
        try:
            with self.file_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except Exception as e:
            raise OSError(f"Error reading SFT file {self.file_path}: {e}")

        logger.info(f"Loaded {len(data)} examples from {self.file_path}")
        return data

    def alpaca_template(self, item: dict[str, Any]) -> tuple[str, str]:
        """Default Alpaca-style template."""
        # Check standard alpaca keys
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        if input_text:
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                "### Response:\n"
            )
        else:
            prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                "### Response:\n"
            )

        return prompt, output_text

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt, response = self.template_fn(item)

        # Tokenize (assuming simpler tokenization where simple concat works roughly ok for now)
        # Ideally, we should check if tokenizer has specific chat formatting methods.
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)

        # Add EOS if tokenizer doesn't (SimpleTokenizer might not, we assume we might need to add one)
        # We append a special EOS token if the tokenizer supports it.
        # For now, let's assume we proceed without explicit EOS unless formatted in response.
        # Actually, best practice is to append EOS to the response.

        # Combine
        input_ids = prompt_ids + response_ids

        # Create labels: mask prompt, keep response
        labels = [self.ignore_index] * len(prompt_ids) + response_ids

        # Truncate if too long
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        # Pad if too short
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.padding_value] * pad_len
            labels += [self.ignore_index] * pad_len

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
            "attention_mask": torch.LongTensor([1] * (len(input_ids) - pad_len) + [0] * pad_len),
        }


def create_sft_dataloader(
    dataset: SFTDataset, batch_size: int, shuffle: bool = True, num_workers: int = 0
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
