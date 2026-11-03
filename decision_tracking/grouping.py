"""Retrieve candidate evidence; similarity never establishes a decision."""
import re


def checklist_items(text):
    items = []
    for line in text.splitlines():
        match = re.match(r"^\s*(?:[-*?]|\d+[.)])\s+(.+?)\s*$", line)
        if match and match.group(1) not in items:
            items.append(match.group(1))
    return items or ([text.strip()] if text.strip() else [])


def retrieve_positions(queries, vectors, blocks, initial_positions, protected,
                       *, threshold=0.75, neighbors=1):
    if not -1 <= threshold <= 1 or neighbors < 0:
        raise ValueError("Invalid decision retrieval threshold or neighbor count")
    if len(vectors) != len(blocks) or not blocks:
        raise ValueError("Decision retrieval requires aligned, nonempty source blocks")
    positions = set(initial_positions) | set(protected)
    audit = []
    for scores in (queries @ vectors.T).tolist():
        core = [i for i, score in enumerate(scores) if score >= threshold]
        fallback = not core
        if fallback:
            core = [max(range(len(scores)), key=scores.__getitem__)]
        expanded = set()
        for i in core:
            expanded.update(range(max(0, i-neighbors), min(len(blocks), i+neighbors+1)))
        for i in expanded:
            positions.update(blocks[i])
        audit.append({"matched_blocks": [{"block": i, "similarity": scores[i]} for i in core],
                      "block_ids": sorted(expanded), "below_threshold_fallback": fallback})
    return sorted(positions), audit
