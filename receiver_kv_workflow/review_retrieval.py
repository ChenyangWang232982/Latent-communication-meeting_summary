"""Retrieve speaker blocks for draft items; only original source KV is handed off."""

import re
import time


def draft_items(draft):
    items = []
    for line in draft.splitlines():
        match = re.match(r'^\s*(?:[-*•]\s+|\d+[.)]\s+)(\S.*)', line)
        if match:
            items.append(match.group(1).strip())
    if not items:
        items = [line.strip() for line in draft.splitlines()
                 if line.strip() and not line.strip().startswith('#')]
    if not items:
        raise ValueError('Cannot retrieve evidence from an empty draft')
    return items


def retrieve_review_positions(router, tokenizer, prefix, blocks, protected, draft,
                              *, neighbors=1, matches_per_item=2):
    """Rank all blocks per draft item, union matches, expand neighbors, retain source order."""
    from .speaker_blocks import expand_neighbor_blocks

    if not blocks:
        raise ValueError('Cannot retrieve review evidence without speaker blocks')
    started = time.perf_counter()
    items = draft_items(draft)
    texts = [tokenizer.decode([prefix[i] for i in block], skip_special_tokens=True)
             for block in blocks]
    block_vectors, block_stats = router.encode(texts)
    item_vectors, item_stats = router.encode(items)
    similarities = item_vectors @ block_vectors.T
    count = min(matches_per_item, len(blocks))
    chosen = set()
    for row in similarities:
        chosen.update(int(i) for i in row.topk(count).indices.tolist())
    seed = set(protected)
    for i in chosen:
        seed.update(blocks[i])
    selected = expand_neighbor_blocks(sorted(seed), blocks, protected, radius=neighbors)
    return selected, {
        'item_count': len(items), 'speaker_block_count': len(blocks),
        'matches_per_item': count, 'matched_block_count': len(chosen),
        'neighbor_blocks': neighbors, 'source_kv_positions': len(selected),
        'embedding_input_tokens': block_stats['embedding_input_tokens'] + item_stats['embedding_input_tokens'],
        'embedding_windows': block_stats['embedding_windows'] + item_stats['embedding_windows'],
        'seconds': time.perf_counter() - started,
    }
