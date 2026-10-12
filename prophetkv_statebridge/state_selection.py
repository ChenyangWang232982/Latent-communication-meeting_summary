"""Deterministic state-selection policies for fixed-bandwidth StateBridge."""

from __future__ import annotations

STATE_SELECTION_STRATEGIES = ("tail", "stratified")


def state_selection_indices(length: int, budget: int, strategy: str) -> list[int]:
    """Return ``budget`` chronological positions from a generated handoff.

    ``stratified`` preserves early framing, uniformly covers the middle, and
    preserves final conclusions. With K=64 it selects 16 head, 32 middle, and
    16 tail positions. Chronological order is retained for receiver prefixes.
    """
    if length < 1:
        raise ValueError("Cannot select states from an empty handoff.")
    if budget < 1:
        raise ValueError("State-selection budget must be positive.")
    if strategy not in STATE_SELECTION_STRATEGIES:
        raise ValueError(f"Unsupported state-selection strategy: {strategy}")
    if budget >= length:
        return list(range(length))
    if strategy == "tail":
        return list(range(length - budget, length))

    head_count = budget // 4
    tail_count = budget // 4
    middle_count = budget - head_count - tail_count
    middle_start = head_count
    middle_end = length - tail_count - 1

    if middle_count == 1:
        middle = [(middle_start + middle_end) // 2]
    else:
        span = middle_end - middle_start
        middle = [middle_start + (span * index) // (middle_count - 1) for index in range(middle_count)]
    indices = list(range(head_count)) + middle + list(range(length - tail_count, length))
    if len(indices) != budget or len(set(indices)) != budget:
        raise RuntimeError("Stratified state selection did not produce a unique fixed-size prefix.")
    return indices


def select_state_tokens(
    states: "torch.Tensor",
    token_ids: "torch.Tensor",
    budget: int,
    strategy: str,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Select aligned state/token pairs without changing their chronology."""
    import torch

    if states.ndim != 2 or token_ids.ndim != 1:
        raise ValueError("Expected states [length, hidden] and token_ids [length].")
    length = min(states.size(0), token_ids.numel())
    indices = state_selection_indices(length, budget, strategy)
    index_tensor = torch.tensor(indices, device=states.device, dtype=torch.long)
    return states[:length].index_select(0, index_tensor), token_ids[:length].index_select(0, index_tensor)
