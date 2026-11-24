"""Retrieve only already-selected evidence; never restore dropped source positions."""


def supplement_windows(blocks, selected, windows, vectors, *, top_k=2, neighbor_radius=1):
    selected = set(selected)
    available = {i: set(block) & selected for i, block in enumerate(blocks)}
    available = {i: positions for i, positions in available.items() if positions}
    result = []
    for window in windows:
        primary = list(window['block_ids'])
        candidates = []
        for block_id in available:
            if any(abs(block_id - anchor) <= neighbor_radius for anchor in primary):
                continue
            score, anchor = max(
                (float(vectors[block_id] @ vectors[a]), a) for a in primary)
            candidates.append(dict(block_id=block_id, anchor_block_id=anchor, cosine=score))
        matches = sorted(candidates, key=lambda row: (-row['cosine'], row['block_id']))[:top_k]
        extras = sorted(row['block_id'] for row in matches)
        positions = set(window['positions'])
        for block_id in extras:
            positions.update(available[block_id])
        result.append(dict(window, primary_block_ids=primary, supplement_block_ids=extras,
            block_ids=sorted(set(primary) | set(extras)), positions=sorted(positions),
            matches=matches, added_positions=len(positions - set(window['positions']))))
    return result


def primary_windows(windows):
    return [dict(w, primary_block_ids=list(w['block_ids']), supplement_block_ids=[],
                 matches=[], added_positions=0) for w in windows]
