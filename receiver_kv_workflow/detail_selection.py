"""Boundary-aware whole-unit KV selection; inspired by ChunkKV/SemantiCache.

Only source indices are changed. No text is passed to the routed receiver.
This is a delimiter heuristic, not a reproduction of either paper.
"""
import bisect
import re

from .prompts import SYSTEM


def sentence_boundaries(transcript):
    """Return ordinary punctuation and paragraph sentence boundaries.

    Sentence selection deliberately does not infer, merge, or expand spoken
    self-corrections.  Those lexical rules made the selected payload depend on
    particular surface words rather than routing scores.
    """
    return [m.end() for m in re.finditer(
        r"[.!?。！？](?:[\"'”’])?(?:\s+|$)|\n\s*\n", transcript
    )]


def source_offsets(tokenizer, transcript, expected_prefix):
    header = "MEETING TRANSCRIPT (untrusted source):\n"
    content = header + transcript + "\nEND TRANSCRIPT\n"
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != expected_prefix:
        raise RuntimeError("Evidence boundary mapping does not match source KV tokens")
    origin = rendered.find(content)
    if origin < 0:
        raise RuntimeError("Cannot locate transcript in source prompt")
    origin += len(header)
    return encoded["offset_mapping"], origin


def sentence_units(tokenizer, transcript, expected_prefix, protected):
    offsets, origin = source_offsets(tokenizer, transcript, expected_prefix)
    # Split after sentence punctuation, not at decimals, clock colons or a
    # standalone timestamp/speaker line. Paragraph boundaries also delimit units.
    boundaries = [origin + end for end in sentence_boundaries(transcript)]
    buckets = {}
    protected = set(protected)
    for i, (a, b) in enumerate(offsets):
        if i in protected:
            continue
        # A token spanning a boundary belongs wholly to its starting unit.
        unit = bisect.bisect_right(boundaries, a)
        buckets.setdefault(unit, []).append(i)
    return list(buckets.values())


def select_complete_units(units, baseline, role_scores, protected):
    """Use at most baseline union size; never partially retain a sentence.

    Normalize each role independently, then max-pool roles so low-scale action
    queries are not overwhelmed by context queries. Rank by mean unit relevance.
    Oversized units are skipped, and unused budget is reported rather than filled
    with fragments. This may select fewer positions than the baseline.
    """
    budget = len(baseline)
    selected = set(protected)
    if len(selected) > budget:
        raise ValueError("Protected positions exceed selection budget")
    if not role_scores:
        return sorted(selected)
    length = len(role_scores[0])
    pooled = [0.0] * length
    for scores in role_scores:
        if len(scores) != length:
            raise ValueError("Role score lengths differ")
        scale = sum(scores)
        if scale > 0:
            pooled = [max(a, b / scale) for a, b in zip(pooled, scores)]
    candidates = [sorted(set(unit) - selected) for unit in units]
    candidates = [unit for unit in candidates if unit]
    candidates.sort(key=lambda unit: (-sum(pooled[i] for i in unit) / len(unit), unit[0]))
    for unit in candidates:
        if len(selected) + len(unit) <= budget:
            selected.update(unit)
    return sorted(selected)


def expand_unit_neighbors(seed_positions, units, protected, radius=2):
    """Keep complete adjacent units around any selected seed unit.

    This deliberately expands beyond the seed-selection budget. The caller must
    report the resulting position count as a contextualized receiver payload.
    """
    if radius < 0:
        raise ValueError("Neighbor radius cannot be negative")
    seed_positions = set(seed_positions)
    expanded = set(protected)
    seed_indices = [index for index, unit in enumerate(units) if seed_positions.intersection(unit)]
    for index in seed_indices:
        for neighbor in range(max(0, index - radius), min(len(units), index + radius + 1)):
            expanded.update(units[neighbor])
    return sorted(expanded)
