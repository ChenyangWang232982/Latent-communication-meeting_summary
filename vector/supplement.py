"""Retrieve only already-selected evidence; never restore dropped source positions."""
import math
import re
from collections import Counter

METHOD = 'relative_diverse_v3'


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


def rank_candidates_absolute(primary, available, vectors, signatures, weights, neighbor_radius):
    """V2 reference ranking, retained for offline comparisons only."""
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


def anchor_proposals(primary, available, vectors, signatures, weights, neighbor_radius):
    """Each anchor nominates its best distinct candidate by within-anchor evidence score."""
    eligible = [i for i in available if not exclusion_reasons(i, primary, signatures, neighbor_radius)]
    proposals = {}
    for anchor in primary:
        if weights[anchor] <= 0:
            continue
        rows = []
        for candidate in eligible:
            score, cosine = pair_scores(vectors, weights, candidate, anchor)
            if score > 0:
                rows.append(dict(block_id=candidate, anchor_block_id=anchor, score=score, cosine=cosine,
                    candidate_information=weights[candidate], anchor_information=weights[anchor]))
        rows = distinct_candidates(sorted(rows, key=lambda r: (-r['score'], r['block_id'])), signatures)
        if not rows:
            continue
        best = dict(rows[0])
        second = rows[1]['score'] if len(rows) > 1 else 0.0
        advantage = (best['score'] - second) / best['score']
        priority = advantage * math.sqrt(weights[anchor] * weights[best['block_id']])
        if priority <= 0:
            continue  # A tied match provides no evidence for a distinctive association.
        best.update(runner_up_score=second, relative_advantage=advantage, base_priority=priority)
        old = proposals.get(best['block_id'])
        if old is None or (priority, best['score'], -anchor) > (
                old['base_priority'], old['score'], -old['anchor_block_id']):
            proposals[best['block_id']] = best
    return list(proposals.values())


def rank_candidates(primary, available, vectors, signatures, weights, neighbor_radius):
    """V3: compare within-anchor advantages, then greedily discourage topic redundancy.

    Similarity between both candidate blocks and their anchors acts as a continuous
    redundancy penalty. No topic count, clustering threshold, or answer label is used.
    """
    remaining = anchor_proposals(primary, available, vectors, signatures, weights, neighbor_radius)
    ranked = []
    used = set()
    while remaining:
        rescored = []
        for proposal in remaining:
            if signatures[proposal['block_id']] in used:
                continue
            redundancy = max((max(0.0, min(1.0, float(vectors[proposal['block_id']] @ vectors[r['block_id']])),
                                  min(1.0, float(vectors[proposal['anchor_block_id']] @ vectors[r['anchor_block_id']])))
                              for r in ranked), default=0.0)
            priority = proposal['base_priority'] * (1.0 - redundancy)
            if priority > 0:
                rescored.append(dict(proposal, redundancy=redundancy, selection_score=priority))
        if not rescored:
            break
        best = min(rescored, key=lambda r: (-r['selection_score'], -r['score'], r['block_id']))
        ranked.append(best)
        used.add(signatures[best['block_id']])
        remaining = [r for r in remaining if r['block_id'] != best['block_id']]
    return ranked


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
