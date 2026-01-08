import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from llm.tokenization.tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


class DPODataset(Dataset):
    """
    Dataset for Direct Preference Optimization (DPO).

    Expects JSONL data with keys: 'prompt', 'chosen', 'rejected'.
    Or generic keys mapped via `template_fn`.

    Produces a dict with:
    - chosen_input_ids, chosen_labels, chosen_attention_mask
    - rejected_input_ids, rejected_labels, rejected_attention_mask
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: BaseTokenizer,
        max_seq_len: int = 1024,
        padding_value: int = 0,
        ignore_index: int = -100,
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding_value = padding_value
        self.ignore_index = ignore_index

        self.data = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        data = []
        try:
            with self.file_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # Minimal validation
                        if not all(k in item for k in ("prompt", "chosen", "rejected")):
                            # Allow flexible keys if template handles them?
                            # For now enforce standard DPO keys or we need a map
                            pass
                        data.append(item)
        except Exception as e:
            raise OSError(f"Error reading DPO file {self.file_path}: {e}")

        logger.info(f"Loaded {len(data)} preference pairs from {self.file_path}")
        return data

    def _process_sequence(self, prompt: str, completion: str) -> dict[str, torch.Tensor]:
        """Tokenize and mask a single sequence (prompt + completion)."""
        prompt_ids = self.tokenizer.encode(prompt)
        completion_ids = self.tokenizer.encode(completion)

        input_ids = prompt_ids + completion_ids
        labels = [self.ignore_index] * len(prompt_ids) + completion_ids

        # Truncate
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        # Pad
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.padding_value] * pad_len
            labels += [self.ignore_index] * pad_len

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
            "attention_mask": torch.LongTensor([1] * (len(input_ids) - pad_len) + [0] * pad_len),
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        # We might need to format prompt if it's not pre-formatted.
        # Assuming data is pre-processed or simple text for now.

        chosen_data = self._process_sequence(prompt, chosen)
        rejected_data = self._process_sequence(prompt, rejected)

        return {
            "chosen_input_ids": chosen_data["input_ids"],
            "chosen_labels": chosen_data["labels"],
            "chosen_attention_mask": chosen_data["attention_mask"],
            "rejected_input_ids": rejected_data["input_ids"],
            "rejected_labels": rejected_data["labels"],
            "rejected_attention_mask": rejected_data["attention_mask"],
        }
