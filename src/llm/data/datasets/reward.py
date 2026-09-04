"""
Reward Model Dataset for RLHF.

Handles preference pairs for training a reward model that scores responses.
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from llm.tokenization.tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)


class RewardDataset(Dataset):
    """
    Dataset for Reward Model training.

    Expects JSONL data with keys: 'prompt', 'chosen', 'rejected'.
    Produces pairs of tokenized sequences for comparison.

    Output keys per sample:
    - chosen_input_ids, chosen_attention_mask
    - rejected_input_ids, rejected_attention_mask
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: BaseTokenizer,
        max_seq_len: int = 1024,
        padding_value: int | None = None,
    ):
        if max_seq_len <= 0:
            # RIL ISS-199: mirrors the SFTDataset guard — a non-positive
            # ``max_seq_len`` truncates ids and grows attention_mask past
            # input_ids, crashing downstream with an opaque shape error.
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # ``None`` → tokenizer.pad_token_id (fallback 0), matching the text
        # datasets — hardcoding 0 pads with an arbitrary id for tokenizers
        # whose real pad id differs (RIL ISS-337).
        if padding_value is None:
            tokenizer_pad = getattr(self.tokenizer, "pad_token_id", None)
            self.padding_value = tokenizer_pad if tokenizer_pad is not None else 0
        else:
            self.padding_value = padding_value

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
                        if all(k in item for k in ("prompt", "chosen", "rejected")):
                            if not item["chosen"] or not item["rejected"]:
                                # An empty completion makes the reward model score
                                # the prompt itself (or a fully-masked row), not
                                # the response end it is supposed to score —
                                # silently wrong training signal (RIL ISS-336).
                                logger.warning(
                                    "Skipping Reward item with an empty chosen/rejected completion: nothing to score."
                                )
                                continue
                            # RIL ISS-345: a completion that ALONE reaches
                            # max_seq_len means front-truncation drops the
                            # ENTIRE prompt — the reward model would then score
                            # an un-conditioned mid-response token, and as soon
                            # as only ONE side of the pair overflows the chosen
                            # vs rejected scores come from different context.
                            # Drop the WHOLE pair so kept data stays symmetric.
                            chosen_ids = self.tokenizer.encode(item["chosen"])
                            rejected_ids = self.tokenizer.encode(item["rejected"])
                            if len(chosen_ids) >= self.max_seq_len or len(rejected_ids) >= self.max_seq_len:
                                logger.warning(
                                    "Skipping Reward item whose chosen/rejected completion alone reaches "
                                    "max_seq_len=%d: no prompt context could survive truncation, so the "
                                    "pair would be scored unconditioned/asymmetrically (RIL ISS-345).",
                                    self.max_seq_len,
                                )
                                continue
                            data.append(item)
        except json.JSONDecodeError as e:
            # RIL ISS-201: SFT/DPO already wrap a malformed line in an
            # actionable ValueError with the file path; Reward leaked the raw
            # exception with no context. Align the three datasets.
            raise ValueError(f"Invalid JSON in Reward file {self.file_path}: {e}")

        logger.info(f"Loaded {len(data)} preference pairs from {self.file_path}")
        return data

    def _tokenize_sequence(self, prompt: str, response: str) -> dict[str, torch.Tensor]:
        """Tokenize prompt + response as a single sequence."""
        full_text = prompt + response
        input_ids = self.tokenizer.encode(full_text)

        # Truncate — from the FRONT so the response end stays in the window.
        # The reward model scores the last non-pad token, so
        # ``[:max_seq_len]`` made it score an arbitrary mid-response token
        # whenever prompt+response overflowed (RIL ISS-332).
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len :]

        # Create attention mask before padding
        seq_len = len(input_ids)
        attention_mask = [1] * seq_len

        # Pad
        pad_len = self.max_seq_len - seq_len
        if pad_len > 0:
            input_ids = input_ids + [self.padding_value] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.data[index]

        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_data = self._tokenize_sequence(prompt, chosen)
        rejected_data = self._tokenize_sequence(prompt, rejected)

        return {
            "chosen_input_ids": chosen_data["input_ids"],
            "chosen_attention_mask": chosen_data["attention_mask"],
            "rejected_input_ids": rejected_data["input_ids"],
            "rejected_attention_mask": rejected_data["attention_mask"],
        }
