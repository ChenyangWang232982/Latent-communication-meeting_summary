"""Offline annotation-based rank inspection; never used to select evidence."""
from .retrieval import (distinct_candidates, exclusion_reasons, information_weights,
                        pair_scores, rank_candidates)


def rank_of(rows, block_id):
    return next((i + 1 for i, row in enumerate(rows) if row['block_id'] == block_id), None)


def target_blocks(units, key):
    """Map annotated character spans to the saved source speaker-unit IDs."""
    targets = {}
    for label, definition in key.items():
        if not isinstance(definition, dict) or 'character_range' not in definition:
            continue
        start, end = definition['character_range']
        targets[label] = [unit['unit'] for unit in units if any(
            part['character_range'][0] < end and part['character_range'][1] > start
            for part in unit['source'])]
    return targets


def diagnose(units, key, blocks, texts, windows, vectors, selected, augmented, *, neighbor_radius=1, top_n=5):
    """Inspect both directions of every annotated pair, including competing window anchors."""
    selected = set(selected)
    available = [i for i, block in enumerate(blocks) if selected.intersection(block)]
    signatures, weights = information_weights({i: texts[i] for i in available})
    targets = target_blocks(units, key)
    pairs = []
    # Reuse the exact production ranking, independent of any annotation.
    window_ranks = [rank_candidates(w['block_ids'], available, vectors, signatures, weights, neighbor_radius)
                    for w in windows]
    for anchor_label, anchors in targets.items():
        for candidate_label, candidates in targets.items():
            if candidate_label == anchor_label:
                continue
            for anchor in anchors:
                for candidate in candidates:
                    row = dict(anchor_label=anchor_label, candidate_label=candidate_label,
                        anchor_block_id=anchor, candidate_block_id=candidate,
                        anchor_text=texts[anchor], candidate_text=texts[candidate],
                        anchor_selected=anchor in available, candidate_selected=candidate in available)
                    if anchor not in available or candidate not in available:
                        row['status'] = 'target_evidence_not_selected'
                        pairs.append(row)
                        continue
                    score, cosine = pair_scores(vectors, weights, candidate, anchor)
                    pool = []
                    for other in available:
                        if abs(other - anchor) <= neighbor_radius:
                            continue
                        weighted, raw = pair_scores(vectors, weights, other, anchor)
                        pool.append(dict(block_id=other, cosine=raw, score=weighted))
                    raw_rows = sorted(pool, key=lambda r: (-r['cosine'], r['block_id']))
                    weighted_rows = sorted(pool, key=lambda r: (-r['score'], r['block_id']))
                    def excerpts(rows):
                        return [dict(r, text=texts[r['block_id']]) for r in rows[:top_n]]
                    row.update(status='ranked', cosine=cosine, score=score,
                        anchor_information=weights[anchor], candidate_information=weights[candidate],
                        raw_anchor_rank=rank_of(raw_rows, candidate),
                        weighted_anchor_rank=rank_of(weighted_rows, candidate),
                        anchor_pool_size=len(pool), anchor_neighbor_excluded=abs(candidate-anchor)<=neighbor_radius,
                        raw_top_candidates=excerpts(raw_rows), weighted_top_candidates=excerpts(weighted_rows),
                        windows=[])
                    for index, window in enumerate(windows):
                        if anchor not in window['block_ids']:
                            continue
                        ranked = window_ranks[index]
                        distinct = distinct_candidates(ranked, signatures)
                        match = next((r for r in ranked if r['block_id'] == candidate), None)
                        reasons = exclusion_reasons(candidate, window['block_ids'], signatures, neighbor_radius)
                        if not reasons and match is None:
                            reasons.append('nonpositive_window_score')
                        winner = next((r for r in distinct if r['block_id'] == candidate), None)
                        if match is not None and winner is None:
                            reasons.append('duplicate_candidate_content')
                        row['windows'].append(dict(window=index, exclusion_reasons=reasons,
                            window_candidate_rank=rank_of(ranked, candidate),
                            window_distinct_rank=rank_of(distinct, candidate),
                            window_score=match['score'] if match else None,
                            best_window_anchor=match['anchor_block_id'] if match else None,
                            actually_supplemented=candidate in augmented[index]['supplement_block_ids'],
                            window_top_candidates=excerpts(distinct)))
                    pairs.append(row)
    return dict(target_blocks=targets, directions=pairs, rank_base=1,
        note='Annotations are inspected after retrieval. Raw/weighted anchor ranks use all selected '
             'non-neighbor blocks, before content exclusions. Window ranks apply actual production '
             'exclusions and best-anchor competition. Null rank means not in that candidate pool. '
             'Character-span overlap identifies speaker blocks, not exact annotated-token coverage.')


def render_diagnostics(data):
    lines = [data['note'], '', f"Target blocks: {data['target_blocks']}"]
    for pair in data['directions']:
        lines.extend(['', f"{pair['anchor_label']} B{pair['anchor_block_id']} -> "
                      f"{pair['candidate_label']} B{pair['candidate_block_id']}",
                      f"status={pair['status']}"])
        if pair['status'] != 'ranked':
            continue
        lines.extend([f"ANCHOR: {pair['anchor_text'].strip()}",
                      f"TARGET: {pair['candidate_text'].strip()}"])
        lines.append(f"cosine={pair['cosine']:.6f}; weighted score={pair['score']:.6f}; "
            f"raw rank={pair['raw_anchor_rank']}/{pair['anchor_pool_size']}; "
            f"weighted rank={pair['weighted_anchor_rank']}/{pair['anchor_pool_size']}")
        for label in ('raw_top_candidates', 'weighted_top_candidates'):
            lines.append(label + ':')
            for row in pair[label]:
                lines.append(f"  B{row['block_id']}: cosine={row['cosine']:.4f}, "
                             f"score={row['score']:.4f}; {row['text'].strip()}")
        for w in pair['windows']:
            lines.append(f"window={w['window']}; rank={w['window_candidate_rank']}; "
                f"distinct rank={w['window_distinct_rank']}; selected={w['actually_supplemented']}; "
                f"exclusions={w['exclusion_reasons']}; best anchor={w['best_window_anchor']}")
            for row in w['window_top_candidates']:
                lines.append(f"  WINDOW CANDIDATE B{row['block_id']}: "
                             f"score={row['score']:.4f}; {row['text'].strip()}")
    return '\n'.join(lines) + '\n'
