import torch
from torch.utils.data import TensorDataset

from llm.data.modules.map_base import SamplerMapDataModule


class SyntheticDataModule(SamplerMapDataModule):
    """DataModule for generating synthetic regression data."""

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
        num_samples = int(self.config.training.num_samples)
        if num_samples < 1:
            # A 0-sample config produces a 0-length train set → an epoch with
            # zero batches → the engine's per-epoch averaging divides by zero
            # (ISS-200 class). Fail fast at the boundary instead.
            raise ValueError(f"SyntheticDataModule requires training.num_samples >= 1, got {num_samples}.")

        # Draw all synthetic data from a DEDICATED fixed-seed generator, NOT
        # the global torch RNG. The distributed layer seeds the global RNG
        # per-rank (``torch.manual_seed(42 + rank)``), so a plain
        # ``torch.randn`` produced DIFFERENT train+val data on every rank in
        # DDP — each rank regressed a different target function at the same
        # DistributedSampler index, making the "global batch" incoherent and
        # the aggregate val loss (save_best / EarlyStopping /
        # ReduceLROnPlateau) statistically meaningless (RIL ISS-134). Same
        # class as the map_base split_train_val fix (RIL ISS-087): fixed
        # seed 0 keeps synthetic runs reproducible and rank-identical.
        generator = torch.Generator()
        generator.manual_seed(0)
        train_x = torch.randn(
            num_samples,
            self.config.model.hidden_size,
            generator=generator,
        )
        train_y = train_x + 0.1 * torch.randn_like(train_x, generator=generator)
        self.train_dataset = TensorDataset(train_x, train_y)

        val_num_samples = max(1, num_samples // 10)
        val_x = torch.randn(
            val_num_samples,
            self.config.model.hidden_size,
            generator=generator,
        )
        val_y = val_x + 0.1 * torch.randn_like(val_x, generator=generator)
        self.val_dataset = TensorDataset(val_x, val_y)
