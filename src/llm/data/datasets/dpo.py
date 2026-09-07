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
        padding_value: int | None = None,
        ignore_index: int = -100,
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
                        # A scalar JSON row (bare string/number) would reach
                        # ``k in item`` / ``item["chosen"]`` below and die with
                        # a raw TypeError/AttributeError mid-setup (RIL
                        # ISS-381; SFT fixed the same class at ISS-336). Skip
                        # it with context instead.
                        if not isinstance(item, dict):
                            logger.warning("Skipping DPO row that is not a JSON object: %r", item)
                            continue
                        # Minimal validation: skip entries missing required keys
                        if not all(k in item for k in ("prompt", "chosen", "rejected")):
                            logger.warning("Skipping DPO item missing required keys (prompt, chosen, rejected)")
                            continue
                        # An empty completion yields all-(-100) labels — both
                        # log-probs come out 0 and the pair contributes a
                        # constant log(2) to the DPO loss, silently diluting
                        # every gradient step (RIL ISS-336).
                        if not item["chosen"] or not item["rejected"]:
                            logger.warning(
                                "Skipping DPO item with an empty chosen/rejected completion: "
                                "there is no preference signal to train on."
                            )
                            continue
                        # An over-long prompt (already >= max_seq_len) truncates
                        # the completion ENTIRELY in `_process_sequence`
                        # (truncation cuts from the end), so the labels become
                        # all -100 — an EMPTY preference signal that silently
                        # contributes a constant log(2) to the DPO loss (deep-
                        # dive finding). Drop the row with a warning instead of
                        # training on it.
                        prompt_ids = self.tokenizer.encode(item["prompt"])
                        if len(prompt_ids) >= self.max_seq_len:
                            logger.warning(
                                "Skipping DPO item whose prompt alone reaches max_seq_len=%d: "
                                "the completion is truncated away entirely and the preference "
                                "signal is empty.",
                                self.max_seq_len,
                            )
                            continue
                        # RIL ISS-345: a completion that ALONE reaches
                        # max_seq_len means front-truncation drops the ENTIRE
                        # prompt (zero -100 context tokens survive) — the
                        # response would be scored with no conditioning, and as
                        # soon as only ONE side of the pair overflows, chosen
                        # and rejected log-probs are computed with different
                        # context (a corrupted, imbalanced preference signal).
                        # Drop the WHOLE pair so the kept data stays symmetric.
                        chosen_ids = self.tokenizer.encode(item["chosen"])
                        rejected_ids = self.tokenizer.encode(item["rejected"])
                        if len(chosen_ids) >= self.max_seq_len or len(rejected_ids) >= self.max_seq_len:
                            logger.warning(
                                "Skipping DPO item whose chosen/rejected completion alone "
                                "reaches max_seq_len=%d: no prompt context could survive "
                                "truncation, so the preference pair would be trained "
                                "unconditioned/asymmetrically (RIL ISS-345).",
                                self.max_seq_len,
                            )
                            continue
                        data.append(item)
        except FileNotFoundError:
            raise FileNotFoundError(f"DPO data file not found: {self.file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in DPO file {self.file_path}: {e}")
        except OSError as e:
            raise OSError(f"Error reading DPO file {self.file_path}: {e}")

        logger.info(f"Loaded {len(data)} preference pairs from {self.file_path}")
        return data

    def _process_sequence(
        self, prompt_ids: list[int], completion_ids: list[int], front_trim: int = 0
    ) -> dict[str, torch.Tensor]:
        """Tokenize and mask a single sequence (prompt + completion).

        Args:
            prompt_ids: Pre-tokenized prompt (encoded once per pair in
                ``__getitem__`` so chosen and rejected share the same
                conditioning and the pair-level trim is computed from one
                encode).
            completion_ids: Pre-tokenized completion.
            front_trim: Pair-shared number of leading tokens to drop. Both
                sides of the pair drop the SAME amount, so chosen and
                rejected are always scored under the same prompt suffix —
                independent per-side truncation sent the shorter side to a
                longer prompt context (RIL ISS-382).
        """
        input_ids = prompt_ids + completion_ids
        labels = [self.ignore_index] * len(prompt_ids) + completion_ids

        if front_trim > 0 and len(input_ids) - front_trim > 0:
            input_ids = input_ids[front_trim:]
            labels = labels[front_trim:]

        # Truncate — from the FRONT so the completion (the supervised / scored
        # part) survives. ``input_ids[:max_seq_len]`` kept the prompt and
        # chopped the completion tail, where chosen/rejected usually diverge
        # (RIL ISS-332).
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len :]
            labels = labels[-self.max_seq_len :]

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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.data[index]

        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        # We might need to format prompt if it's not pre-formatted.
        # Assuming data is pre-processed or simple text for now.

        # Encode once per pair and derive a PAIR-level front trim from the
        # LONGER side's overflow. Trimming both sides by the same leading
        # token count keeps their conditioning context identical — the old
        # per-side ``input_ids[-max_seq_len:]`` sent chosen (longer) to a
        # truncated prompt suffix while rejected (shorter) kept the full
        # prompt, scoring the pair under different contexts (RIL ISS-382).
        # The ISS-345 load guard already dropped pairs where a *completion
        # alone* reaches max_seq_len, so ``front_trim`` never exceeds the
        # prompt length here.
        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = self.tokenizer.encode(chosen)
        rejected_ids = self.tokenizer.encode(rejected)
        overflow = max(len(prompt_ids) + len(chosen_ids), len(prompt_ids) + len(rejected_ids)) - self.max_seq_len
        front_trim = max(overflow, 0)

        chosen_data = self._process_sequence(prompt_ids, chosen_ids, front_trim)
        rejected_data = self._process_sequence(prompt_ids, rejected_ids, front_trim)

        return {
            "chosen_input_ids": chosen_data["input_ids"],
            "chosen_labels": chosen_data["labels"],
            "chosen_attention_mask": chosen_data["attention_mask"],
            "rejected_input_ids": rejected_data["input_ids"],
            "rejected_labels": rejected_data["labels"],
            "rejected_attention_mask": rejected_data["attention_mask"],
        }
