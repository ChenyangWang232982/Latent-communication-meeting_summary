import math


def selected_windows(blocks, selected, protected, parts=8, overlap=0.25):
    """Window already-selected complete speaker turns; never restore dropped source rows."""
    selected, protected = set(selected), set(protected)
    units = [(i, sorted(set(block) & (selected-protected))) for i, block in enumerate(blocks)]
    units = [(i, unit) for i, unit in units if unit]
    target = max(1, math.ceil(sum(len(unit) for _, unit in units)/parts))
    start = 0
    while start < len(units):
        end, size = start, 0
        while end < len(units) and (end == start or size+len(units[end][1]) <= target):
            size += len(units[end][1])
            end += 1
        yield dict(block_ids=[i for i, _ in units[start:end]],
                   positions=sorted(protected | {p for _, unit in units[start:end] for p in unit}))
        if end == len(units):
            break
        next_start, shared = end, 0
        while next_start > start+1 and shared < size*overlap:
            next_start -= 1
            shared += len(units[next_start][1])
        start = next_start
