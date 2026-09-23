"""Data access for frozen Sender trajectories collected by Interlat."""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class HiddenStateDataset(Dataset):
    def __init__(self, path: Path, limit: int | None = None):
        records = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(records, list) or not records:
            raise ValueError(f"No hidden-state records in {path}")
        self.records = records[:limit] if limit else records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]
