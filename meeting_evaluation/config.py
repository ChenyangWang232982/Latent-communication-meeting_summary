"""Shared configuration for the meeting benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkflowConfig:
    model: str = "Qwen/Qwen3-4B"
    source_max_tokens: int = 8192
    prefill_chunk_tokens: int = 1024
    backfill_rounds: int = 0
    agent_max_new_tokens: int = 128
    receiver_max_new_tokens: int = 192
    prefix_tokens_per_agent: int = 64
    device: str | None = None
    num_samples: int = 0


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

