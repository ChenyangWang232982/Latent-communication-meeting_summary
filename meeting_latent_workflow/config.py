from dataclasses import dataclass
from pathlib import Path
from typing import Literal


CommunicationMode = Literal["text", "cipher"]


@dataclass(frozen=True)
class WorkflowConfig:
    """Runtime settings shared by every workflow node."""

    model_name: str = "google/long-t5-tglobal-large"
    communication_mode: CommunicationMode = "text"
    cipher_checkpoint: Path | None = None
    use_compression: bool = False
    compressed_latent_len: int = 8
    temperature: float = 1.0
    device: str = "auto"
    max_input_tokens: int = 4096
    max_new_tokens: int = 256
    chunk_tokens: int = 2800
    chunk_overlap_tokens: int = 256
    reduce_group_size: int = 3
    max_refinement_rounds: int = 1
