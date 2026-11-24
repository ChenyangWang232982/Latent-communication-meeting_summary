"""Retrieve only already-selected evidence; never restore dropped source positions."""
import math
import re
from collections import Counter


def content_terms(text):
    """Ignore speaker headers and generic transcript annotations, not word blacklists."""
    text = re.sub(r'(?m)^[^:\n]{1,100}:\s*', '', text)
    text = re.sub(r'\{[^{}]*\}|<[^<>]*>|\[[^\[\]]*\]', ' ', text)
    return tuple(re.findall(r'[^\W_]+', text.casefold()))


def information_weights(texts):
    """Within-meeting IDF mass: repeated short turns have little distinct content."""
    signatures = {i: content_terms(text) for i, text in texts.items()}
    frequencies = Counter(term for terms in signatures.values() for term in set(terms))
    count = len(signatures)
    masses = {i: sum(math.log((count + 1) / (frequencies[t] + 1)) for t in set(terms))
              for i, terms in signatures.items()}
    scale = sum(masses.values()) / max(1, count)
    weights = {i: mass / (mass + scale) if mass > 0 else 0.0 for i, mass in masses.items()}
    return signatures, weights


def pair_scores(vectors, weights, candidate, anchor):
    cosine = max(-1.0, min(1.0, float(vectors[candidate] @ vectors[anchor])))
    score = max(0.0, cosine) * math.sqrt(weights[candidate] * weights[anchor])
    return score, cosine


def exclusion_reasons(block_id, primary, signatures, neighbor_radius):
    reasons = []
    if block_id in primary:
        reasons.append('inside_primary_window')
    elif any(abs(block_id - anchor) <= neighbor_radius for anchor in primary):
        reasons.append('within_neighbor_radius')
    if not signatures[block_id]:
        reasons.append('empty_content')
    elif signatures[block_id] in {signatures[a] for a in primary}:
        reasons.append('same_content_as_primary')
    return reasons


def rank_candidates(primary, available, vectors, signatures, weights, neighbor_radius):
    candidates = []
    for block_id in available:
        if exclusion_reasons(block_id, primary, signatures, neighbor_radius):
            continue
        score, cosine, anchor = max((*pair_scores(vectors, weights, block_id, anchor), anchor)
                                    for anchor in primary)
        if score > 0:
            candidates.append(dict(block_id=block_id, anchor_block_id=anchor, cosine=cosine,
                score=score, candidate_information=weights[block_id], anchor_information=weights[anchor]))
    return sorted(candidates, key=lambda row: (-row['score'], row['block_id']))


def distinct_candidates(candidates, signatures):
    used = set()
    result = []
    for row in candidates:
        signature = signatures[row['block_id']]
        if signature not in used:
            result.append(row)
            used.add(signature)
    return result


def supplement_windows(blocks, selected, windows, vectors, *, texts=None, top_k=2, neighbor_radius=1):
    selected = set(selected)
    available = {i: set(block) & selected for i, block in enumerate(blocks)}
    available = {i: positions for i, positions in available.items() if positions}
    if texts is None:
        raise ValueError('Source block texts are required for information-aware retrieval')
    signatures, weights = information_weights({i: texts[i] for i in available})
    result = []
    for window in windows:
        primary = list(window['block_ids'])
        candidates = rank_candidates(primary, available, vectors, signatures, weights, neighbor_radius)
        matches = distinct_candidates(candidates, signatures)[:top_k]
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
