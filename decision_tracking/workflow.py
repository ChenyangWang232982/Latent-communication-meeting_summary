"""Decision checklist, full-source retrieval, and one joint evolution pass."""
import time
from .grouping import checklist_items, retrieve_positions
from .prompts import CHECKLIST_REQUEST, evolution_request


def run_two_stage(*, generate, router, source_vectors, blocks, initial_positions,
                  protected, threshold, neighbors, progress):
    started = time.perf_counter()
    progress('decision stage 1/2: extracting candidate checklist')
    checklist, first = generate(initial_positions, CHECKLIST_REQUEST)
    items = checklist_items(checklist)
    progress(f'decision retrieval: matching {len(items)} checklist items against all source blocks')
    retrieval_started = time.perf_counter()
    embedding = None
    if items:
        queries, embedding = router.encode(items)
        positions, matches = retrieve_positions(queries, source_vectors, blocks, initial_positions,
                                                protected, threshold=threshold, neighbors=neighbors)
    else:
        positions, matches = sorted(set(initial_positions) | set(protected)), []
    retrieval_seconds = time.perf_counter() - retrieval_started
    progress(f'decision stage 2/2: joint evolution using {len(positions)} source KV positions')
    answer, second = generate(positions, evolution_request(checklist))
    stages = {'checklist': first, 'evolution': second}
    metrics = {
        'mode': 'two_stage', 'generation_calls': 2, 'checklist_items': len(items),
        'prompt_tokens': sum(s['prompt_tokens'] for s in stages.values()),
        'output_tokens': sum(s['output_tokens'] for s in stages.values()),
        'output_cap_hit': any(s['output_cap_hit'] for s in stages.values()),
        'cap_hits': sum(int(s['output_cap_hit']) for s in stages.values()),
        'seconds': time.perf_counter() - started, 'retrieval_seconds': retrieval_seconds,
        'generation_seconds': sum(s['seconds'] for s in stages.values()),
        'source_kv_positions_across_calls': len(initial_positions) + len(positions),
        'checklist_embedding': embedding, 'stages': stages,
        'note': 'Two additional generations; source embeddings reused. Checklist encoder tokens '
                'are separate from generation model tokens. The checklist is a text intermediate.'}
    audit = {'mode': 'two_stage', 'similarity_threshold': threshold, 'neighbors': neighbors,
             'checklist': checklist, 'items': items, 'retrieval': matches,
             'initial_position_count': len(initial_positions), 'final_positions': positions,
             'added_positions': sorted(set(positions) - set(initial_positions))}
    return answer, metrics, audit
