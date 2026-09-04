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
        padding_value: int | None = None,
        ignore_index: int = -100,
    ):
        """
        Args:
            file_path: Path to jsonl file.
            tokenizer: Tokenizer instance.
            max_seq_len: Max sequence length.
            template_fn: Function to convert data item to (prompt, response) tuple.
                         If None, defaults to Alpaca style.
            padding_value: Token ID for padding input_ids. ``None`` (default)
                resolves to ``tokenizer.pad_token_id`` when the tokenizer has
                one, else ``0`` — hardcoding ``0`` pads with an arbitrary id
                for tokenizers whose real pad id differs (RIL ISS-337).
            ignore_index: Label value for masked tokens (padding/prompt).
        """
        if max_seq_len <= 0:
            # RIL ISS-199: a non-positive ``max_seq_len`` truncates the token
            # ids from the end while ``pad_len`` goes negative, making
            # ``attention_mask`` LONGER than ``input_ids`` (verified 146 vs
            # 293 with max_seq_len=-1) — an opaque shape crash deep in
            # training. Fail fast here instead of mid-run.
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if padding_value is None:
            tokenizer_pad = getattr(self.tokenizer, "pad_token_id", None)
            self.padding_value = tokenizer_pad if tokenizer_pad is not None else 0
        else:
            self.padding_value = padding_value
        self.ignore_index = ignore_index

        self.template_fn = template_fn or self.alpaca_template

        self.data = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        data: list[dict[str, Any]] = []
        try:
            with self.file_path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        # A scalar JSON row (bare string/number) would reach
                        # ``alpaca_template``/``item.get`` and die with a raw
                        # AttributeError mid-setup (RIL ISS-336). Skip it with
                        # context instead.
                        logger.warning("Skipping SFT row that is not a JSON object: %r", item)
                        continue
                    # RIL ISS-345: a response that ALONE reaches max_seq_len
                    # means front-truncation drops the ENTIRE prompt (zero -100
                    # context tokens survive), so the row would train as an
                    # unconditioned response continuation. Drop it with a
                    # warning instead of silently corrupting the signal.
                    # (One template+encode per row here is amortized against the
                    # per-epoch re-encode in ``__getitem__``.)
                    _prompt_text, _response_text = self.template_fn(item)
                    if len(self.tokenizer.encode(_response_text)) >= self.max_seq_len:
                        logger.warning(
                            "Skipping SFT row whose response alone reaches max_seq_len=%d: "
                            "no prompt context could survive truncation (RIL ISS-345).",
                            self.max_seq_len,
                        )
                        continue
                    data.append(item)
        except FileNotFoundError:
            raise FileNotFoundError(f"SFT data file not found: {self.file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in SFT file {self.file_path}: {e}")
        except OSError as e:
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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.data[index]
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

        # Truncate if too long — from the FRONT so the supervised response
        # survives. ``input_ids[:max_seq_len]`` kept the full prompt and
        # chopped the completion tail, discarding the only supervised signal
        # (prompt tokens are masked) (RIL ISS-332).
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len :]
            labels = labels[-self.max_seq_len :]

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
