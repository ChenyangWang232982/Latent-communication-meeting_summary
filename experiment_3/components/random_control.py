"""Length-matched random evidence, independent of embeddings and target answers."""
import hashlib
import random
from itertools import combinations

from .retrieval import exclusion_reasons, information_weights


def random_windows(blocks, texts, selected, primary, semantic, *, record_id,
                   seed=7, neighbor_radius=1, trials=512):
    retained = set(selected)
    available = {i: set(block) & retained for i, block in enumerate(blocks)}
    available = {i: positions for i, positions in available.items() if positions}
    signatures, weights = information_weights({i: texts[i] for i in available})
    result = []
    if len(primary) != len(semantic):
        raise ValueError('Primary and semantic window counts differ')
    for index, (base, target) in enumerate(zip(primary, semantic)):
        anchors = base['primary_block_ids']
        primary_positions = set(base['positions'])
        count = len(target['supplement_block_ids'])
        target_added = len(set(target['positions']) - primary_positions)
        if target.get('primary_block_ids') != anchors or not primary_positions <= set(target['positions']):
            raise ValueError('Saved semantic and baseline windows have different primary evidence')
        eligible = [i for i in available if weights[i] > 0 and
                    not exclusion_reasons(i, anchors, signatures, neighbor_radius)]
        # Avoid duplicate-content padding, as in semantic supplementation.
        groups = {}
        for i in eligible:
            groups.setdefault(signatures[i], []).append(i)
        groups = list(groups.values())
        if count > len(groups):
            raise ValueError('Too few distinct eligible blocks to match semantic supplement count')
        salt = hashlib.sha256(f'{seed}:{record_id}:{index}'.encode()).digest()
        rng = random.Random(int.from_bytes(salt[:8], 'big'))
        best, best_error, ties, evaluated = [], None, 0, 0
        # For V3's usual 0–2 blocks, exhaust length combinations then sample
        # uniformly among nearest-length sets. No semantic ranking is consulted.
        draws = combinations(eligible, count) if count <= 2 else (
            [rng.choice(group) for group in rng.sample(groups, count)] for _ in range(trials))
        lengths = {i:len(available[i]-primary_positions) for i in eligible}
        for chosen in draws:
            if len({signatures[i] for i in chosen}) != count:
                continue
            evaluated += 1
            error = abs(sum(lengths[i] for i in chosen)-target_added)
            if best_error is None or error < best_error:
                best, best_error, ties = list(chosen), error, 1
            elif error == best_error:
                ties += 1
                if rng.randrange(ties) == 0:
                    best = list(chosen)
        best_positions = primary_positions.union(*(available[i] for i in best))
        added = len(best_positions-primary_positions)
        extras = sorted(best)
        result.append(dict(base, block_ids=sorted(set(anchors)|set(extras)),
            primary_block_ids=list(anchors), supplement_block_ids=extras,
            positions=sorted(best_positions), added_positions=added,
            matches=[dict(block_id=i, method='length_matched_random') for i in extras],
            matching=dict(target_blocks=count, actual_blocks=len(extras),
                target_added_positions=target_added, actual_added_positions=added,
                signed_position_difference=added-target_added,
                absolute_position_difference=best_error, eligible_blocks=len(eligible),
                eligible_distinct_contents=len(groups), evaluated_sets=evaluated,
                matching_method='exhaustive_length_uniform_ties' if count <= 2 else 'random_draw_length_match',
                nearest_length_ties=ties,
                same_supplement_set=set(extras)==set(target['supplement_block_ids']),
                shared_supplement_blocks=len(set(extras)&set(target['supplement_block_ids'])))))
    return result
