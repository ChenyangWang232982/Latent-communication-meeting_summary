"""Deterministic state-selection policies for fixed-bandwidth StateBridge."""

from __future__ import annotations

POSITIONAL_STATE_SELECTION_STRATEGIES = ("tail", "stratified")
STATE_SELECTION_STRATEGIES = POSITIONAL_STATE_SELECTION_STRATEGIES + ("dynamic_compact",)


def state_selection_indices(length: int, budget: int, strategy: str) -> list[int]:
    """Return ``budget`` chronological positions from a generated handoff.

    ``stratified`` samples later middle states and preserves a contiguous tail,
    avoiding early states before the handoff's facts and conclusions have been
    established. With K=64 it selects 32 later-middle and 32 tail positions.
    Chronological order is retained for receiver prefixes.
    """
    if length < 1:
        raise ValueError("Cannot select states from an empty handoff.")
    if budget < 1:
        raise ValueError("State-selection budget must be positive.")
    if strategy not in POSITIONAL_STATE_SELECTION_STRATEGIES:
        raise ValueError(f"Unsupported state-selection strategy: {strategy}")
    if budget >= length:
        return list(range(length))
    if strategy == "tail":
        return list(range(length - budget, length))

    tail_count = budget // 2
    middle_count = budget - tail_count
    middle_end = length - tail_count - 1
    # Prefer to discard the first quarter of a long handoff, but contract the
    # exclusion when the handoff is only slightly larger than the fixed budget.
    middle_start = min(length // 4, middle_end - middle_count + 1)

    if middle_count == 1:
        middle = [(middle_start + middle_end) // 2]
    else:
        span = middle_end - middle_start
        middle = [middle_start + (span * index) // (middle_count - 1) for index in range(middle_count)]
    indices = middle + list(range(length - tail_count, length))
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
