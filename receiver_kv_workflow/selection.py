"""Deterministic span selection, independent of torch and model inference."""

import math


def select_spans(scores, budget, *, span_size=32, protected=()):
    """Select chronological source spans; protected prompt framing counts in budget."""
    if span_size < 1 or budget < 1 or not scores:
        raise ValueError("Scores, budget and span size must be positive/nonempty")
    if any(not math.isfinite(float(x)) for x in scores):
        raise ValueError("Non-finite attention scores")
    selected = set(protected)
    if any(i < 0 or i >= len(scores) for i in selected):
        raise ValueError("Protected index outside source")
    budget = min(budget, len(scores))
    if len(selected) > budget:
        raise ValueError("Budget is smaller than protected prompt framing")
    spans = [list(range(i, min(i + span_size, len(scores))))
             for i in range(0, len(scores), span_size)]
    spans.sort(key=lambda ids: (-sum(scores[i] for i in ids) / len(ids), ids[0]))
    for span in spans:
        available = [i for i in span if i not in selected]
        remaining = budget - len(selected)
        if remaining <= 0:
            break
        if len(available) > remaining:
            peak = max(range(len(available)), key=lambda j: (scores[available[j]], -j))
            start = min(max(0, peak - remaining // 2), len(available) - remaining)
            available = available[start:start + remaining]
        selected.update(available)
    return sorted(selected)


def union_positions(messages):
    """Only for messages selected from the SAME immutable source cache."""
    return sorted({position for message in messages for position in message})
