"""Overlapping speaker windows and an incremental topic catalogue."""
import math

from .structure import parse_object


def discovery_windows(blocks):
    """Target 1/8 of source tokens, with 1/4 overlap at speaker boundaries."""
    target = max(1, math.ceil(sum(len(b['positions']) for b in blocks) / 8))
    start = 0
    while start < len(blocks):
        end, size = start, 0
        while end < len(blocks) and (end == start or size + len(blocks[end]['positions']) <= target):
            size += len(blocks[end]['positions'])
            end += 1
        yield blocks[start:end]
        if end == len(blocks):
            break
        # Whole turns make overlap approximate. Always include at least one new turn.
        next_start, overlap = end, 0
        while next_start > start + 1 and overlap < size / 4:
            next_start -= 1
            overlap += len(blocks[next_start]['positions'])
        start = next_start


def merge_topics(catalogue, raw, window):
    """The model chooses existing topic IDs; the program allocates new IDs."""
    available = {b['block_id'] for b in window}
    for item in parse_object(raw)['topics']:
        found = next((t for t in catalogue if t['id'] == item.get('id')), None)
        if found is None:
            found = dict(id=f'topic_{len(catalogue):03d}', title=item['title'],
                         scope=item.get('scope', item['title']), source_block_ids=[])
            catalogue.append(found)
        found['source_block_ids'] = sorted(set(found['source_block_ids']) |
                                          (set(item['block_ids']) & available))
    catalogue.sort(key=lambda t: min(t['source_block_ids'], default=math.inf))
    return catalogue
