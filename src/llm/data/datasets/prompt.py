"""Prompt dataset for RLHF / PPO rollouts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PromptDataset(Dataset):
    """Dataset of prompt strings loaded from JSONL."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.prompts: list[str] = []
        skipped = 0
        try:
            with self.file_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        # A scalar JSON row (bare string/number) would reach
                        # ``item.get`` and die with a raw AttributeError
                        # mid-setup. Skip it with context.
                        logger.warning("Skipping prompt row that is not a JSON object: %r", item)
                        skipped += 1
                        continue
                    prompt = item.get("prompt") or item.get("instruction") or item.get("text")
                    if prompt:
                        self.prompts.append(str(prompt))
                    else:
                        # A row whose prompt field is missing, empty, or falsy
                        # contributes no rollout to PPO — dropping it silently
                        # means the run trains on a fraction of the corpus with
                        # nothing in the logs saying so. The sibling datasets
                        # (SFT / DPO / Reward) all warn on every dropped row.
                        logger.warning("Skipping prompt row with no non-empty prompt/instruction/text: %r", item)
                        skipped += 1
        except json.JSONDecodeError as exc:
            # Match the SFT/DPO/Reward datasets' contract: a
            # malformed row must surface as a ValueError naming the file,
            # not a raw JSONDecodeError with no context.
            raise ValueError(f"Invalid JSON in prompt file {self.file_path}: {exc}") from None

        if not self.prompts:
            raise ValueError(f"No prompts found in {self.file_path}")

        logger.info("Loaded %d prompts from %s (%d row(s) skipped)", len(self.prompts), self.file_path, skipped)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"prompt": self.prompts[index]}
