"""Shared configuration for the meeting benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ASRConfig:
    enabled: bool = False
    model: str = "small"
    device: str = "cuda"
    compute_type: str = "float16"


@dataclass(frozen=True)
class RunPaths:
    input_path: Path
    output_dir: Path
    reference_path: Path | None = None

