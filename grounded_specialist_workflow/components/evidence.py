"""Source block indexing and relevance-first evidence retrieval."""
import bisect
import math

from receiver_kv_workflow.source_mapping import source_offsets
from receiver_kv_workflow.speaker_blocks import speaker_ranges


def source_blocks(tokenizer, transcript, prefix, protected):
    ranges = speaker_ranges(transcript)
    offsets, origin = source_offsets(tokenizer, transcript, prefix)
    starts = [origin + row['start'] for row in ranges]
    units = [[] for _ in ranges]
    protected = set(protected)
    for position, (start, _) in enumerate(offsets):
        if position not in protected:
            units[max(0, bisect.bisect_right(starts, start) - 1)].append(position)
    result = []
    for row, positions in zip(ranges, units):
        if positions:
            result.append(dict(row, block_id=len(result), positions=positions,
                               text=transcript[row['start']:row['end']]))
    for i, row in enumerate(result):
        row['relative_position'] = i / max(1, len(result) - 1)
    if not result:
        raise ValueError('No source speaker blocks could be mapped to KV positions')
    return result


def retrieve(scores, *, candidate_pool, top_k, neighbors, recency_weight):
    """Keep the best semantic match, rerank other semantic candidates by time.

    Recency cannot introduce a block outside the semantic candidate pool.
    Original cosine scores are retained; no relevance/entailment claim is made.
    """
    if not scores or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError('Evidence similarities must be finite and nonempty')
    if top_k < 1 or candidate_pool < top_k or neighbors < 0:
        raise ValueError('Require candidate_pool >= top_k >= 1 and neighbors >= 0')
    if not math.isfinite(recency_weight) or not 0 <= recency_weight <= 1:
        raise ValueError('recency_weight must be in [0, 1]')
    pool = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:candidate_pool]
    combined = {i: float(scores[i]) + recency_weight * i / max(1, len(scores) - 1) for i in pool}
    ranked = sorted(pool, key=lambda i: (-combined[i], i))
    # Always keep the strongest semantic anchor, including when it is early.
    seeds = [pool[0]] + [i for i in ranked if i != pool[0]][:top_k - 1]
    evidence = sorted({j for i in seeds
                       for j in range(max(0, i - neighbors), min(len(scores), i + neighbors + 1))})
    return {'semantic_candidates': [{'block_id': i, 'cosine': float(scores[i]),
                                     'ranking_score': combined[i]} for i in pool],
            'seed_block_ids': sorted(seeds), 'evidence_block_ids': evidence}


def claim_query(item):
    # A short retrieval instruction, not meeting-specific names or decisions.
    details = ' | '.join(str(item.get(k) or '') for k in ('topic', 'content', 'owners', 'deadline'))
    return ('Instruct: Retrieve meeting statements about this specific issue, including '
            'support, objections, revisions and subsequent outcomes.\nQuery: ' + details)
