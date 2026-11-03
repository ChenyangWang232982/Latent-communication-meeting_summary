"""Build auditable candidate topic groups without treating similarity as a decision."""


def build_groups(vectors, blocks, decision_positions, protected, *, threshold=0.75, neighbors=1):
    """Cluster decision seeds, then associate all source blocks and neighbors.

    A representative is the first decision-seed block in source order. Other
    seeds join the closest representative above threshold; each non-seed block
    joins its closest representative if above threshold. Groups may overlap
    after neighbor expansion. Similarity is only a candidate-generation signal.
    """
    if not -1 <= threshold <= 1:
        raise ValueError('Decision similarity threshold must be in [-1, 1]')
    if neighbors < 0:
        raise ValueError('Decision neighbor count cannot be negative')
    if len(vectors) != len(blocks):
        raise ValueError('Block and embedding counts differ')
    decision_set = set(decision_positions) - set(protected)
    seeds = [i for i, block in enumerate(blocks) if decision_set.intersection(block)]
    if not seeds:
        return []
    similarities = (vectors @ vectors.T).tolist()
    groups = []
    seed_set = set(seeds)
    for i in seeds:
        if groups:
            best = max(range(len(groups)), key=lambda j: similarities[i][groups[j]['representative']])
            if similarities[i][groups[best]['representative']] >= threshold:
                groups[best]['seed_blocks'].append(i)
                continue
        groups.append({'representative': i, 'seed_blocks': [i], 'matched_blocks': []})
    for i in range(len(blocks)):
        if i in seed_set:
            continue
        best = max(range(len(groups)), key=lambda j: similarities[i][groups[j]['representative']])
        similarity = similarities[i][groups[best]['representative']]
        if similarity >= threshold:
            groups[best]['matched_blocks'].append({'block': i, 'similarity': similarity})
    for group in groups:
        matched = [m['block'] for m in group['matched_blocks']]
        core = set(group['seed_blocks'] + matched)
        expanded = set(core)
        for i in core:
            expanded.update(range(max(0, i-neighbors), min(len(blocks), i+neighbors+1)))
        group['block_ids'] = sorted(expanded)
        positions = set(protected)
        for i in group['block_ids']:
            positions.update(blocks[i])
        group['positions'] = sorted(positions)
    return groups
