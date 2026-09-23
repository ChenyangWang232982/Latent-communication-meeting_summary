from dataclasses import dataclass
from typing import Literal


CommunicationMode = Literal["text", "cipher"]


@dataclass(frozen=True)
class WorkflowConfig:
    """Runtime settings shared by every workflow node."""

    model_name: str
    communication_mode: CommunicationMode = "cipher"
    device: str = "auto"
    max_input_tokens: int = 4096
    max_new_tokens: int = 256
    sender_max_new_tokens: int = 256
    temperature: float = 1.0
    chunk_tokens: int = 2800
    chunk_overlap_tokens: int = 256
    reduce_group_size: int = 3
    max_refinement_rounds: int = 1
