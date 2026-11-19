from bisect import bisect_left
from itertools import accumulate


def target_parts(transcript_tokens):
    """Positive round-half-up; Python round() uses ties-to-even instead."""
    return max(1, (transcript_tokens + 512) // 1024)


def selected_windows(blocks, selected, protected, parts, overlap=0.25):
    """Window already-selected complete speaker turns; never restore dropped source rows."""
    selected, protected = set(selected), set(protected)
    units = [(i, sorted(set(block) & (selected-protected))) for i, block in enumerate(blocks)]
    units = [(i, unit) for i, unit in units if unit]
    if not units:
        return
    count = min(parts, len(units))
    weights = [0, *accumulate(len(unit) for _, unit in units)]
    cuts = [0]
    for index in range(1, count):
        low, high = cuts[-1]+1, len(units)-(count-index)
        target = weights[-1]*index/count
        position = bisect_left(weights, target, low, high+1)
        candidates = {max(low, min(high, position)), max(low, min(high, position-1))}
        cuts.append(min(candidates, key=lambda cut: (abs(weights[cut]-target), cut)))
    cuts.append(len(units))
    for index, (start, end) in enumerate(zip(cuts, cuts[1:])):
        # Expand backward into the preceding base part, without adding specialists.
        if index:
            shared = 0
            target = (weights[start]-weights[cuts[index-1]])*overlap
            while start > cuts[index-1] and shared < target:
                start -= 1
                shared += len(units[start][1])
        yield dict(block_ids=[i for i, _ in units[start:end]],
                   positions=sorted(protected | {p for _, unit in units[start:end] for p in unit}))
