"""Replay retrieval from saved experiment-3 evidence; no summary model or prefill."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from experiment_1.components.storage import atomic_json
from .components.retrieval import primary_windows, supplement_windows


def saved_evidence(case):
    selection = json.loads((case / 'selection.json').read_text(encoding='utf-8'))
    audit = selection['routing_audit']
    units = audit['sentence_coverage']
    # Existing audit contains each speaker unit's exact source-token ranges and text.
    blocks = [None] * len(units)
    texts = {}
    for unit in units:
        block_id = unit['unit']
        blocks[block_id] = [p for part in unit['source'] for p in range(*part['token_range'])]
        texts[block_id] = ''.join(part['text'] for part in unit['source'])
    original = json.loads((case / 'baseline.specialists.json').read_text(encoding='utf-8'))['windows']
    windows = primary_windows([dict(block_ids=w.get('primary_block_ids', w['block_ids']),
                                   positions=w['positions']) for w in original])
    return blocks, texts, windows, selection


def run(args):
    from vector.routing import SemanticRouter
    folder = args.output_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    manifest = json.loads((args.run_dir / 'manifest.json').read_text(encoding='utf-8'))
    router = SemanticRouter(args.embedding_model, args.device)
    completed = 0
    for case in sorted(args.run_dir.glob('[0-9][0-9][0-9][0-9][0-9]')):
        if not (case / 'selection.json').exists():
            continue
        blocks, texts, windows, selection = saved_evidence(case)
        selected = selection['selected_positions']
        retained = set(selected)
        ids = [i for i, block in enumerate(blocks) if retained.intersection(block)]
        encoded, stats = router.encode([texts[i] for i in ids])
        vectors = {i: encoded[j] for j, i in enumerate(ids)}
        augmented = supplement_windows(blocks, selected, windows, vectors, texts=texts,
            top_k=args.supplement_blocks, neighbor_radius=args.exclude_neighbor_blocks)
        # Source excerpts from the saved audit may differ slightly at token boundaries
        # from tokenizer.decode during live inference. Record this provenance explicitly.
        matches = [dict(window=index, **match,
            candidate_text=texts[match['block_id']], anchor_text=texts[match['anchor_block_id']])
            for index, w in enumerate(augmented) for match in w['matches']]
        target = folder / case.name
        target.mkdir()
        atomic_json(target / 'retrieval.json', dict(method='information_weighted_cosine_v2',
            selected_sha256=selection['selected_sha256'], embedding=stats, windows=augmented,
            matches=matches, candidate_count=len(ids)))
        (target / 'matches.txt').write_text('\n\n'.join(
            f"Window {m['window']}: {m['anchor_block_id']} -> {m['block_id']} "
            f"score={m['score']:.4f}, cosine={m['cosine']:.4f}\n"
            f"ANCHOR: {m['anchor_text']}\nCANDIDATE: {m['candidate_text']}" for m in matches), encoding='utf-8')
        completed += 1
        print(f'{case.name}: {len(matches)} supplements; {target / "matches.txt"}', flush=True)
    atomic_json(folder / 'manifest.json', dict(source_run=str(args.run_dir.resolve()),
        original_manifest=manifest, method='information_weighted_cosine_v2', completed=completed,
        embedding_model=args.embedding_model, device=args.device,
        supplement_blocks=args.supplement_blocks, exclude_neighbor_blocks=args.exclude_neighbor_blocks,
        note='Retrieval only, using audit source excerpts; no summary model, source prefill or generation. '
             'No target answers enter retrieval. Review matches before running paired inference.'))
    print('Retrieval audit:', folder)
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Previous experiment_3 timestamp directory')
    parser.add_argument('--output-dir', type=Path, default=Path('test_result/experiment_3_retrieval_v2'))
    parser.add_argument('--embedding-model', default='models/Qwen3-Embedding-0.6B')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--supplement-blocks', type=int, default=2)
    parser.add_argument('--exclude-neighbor-blocks', type=int, default=1)
    args = parser.parse_args()
    if args.supplement_blocks < 0 or args.exclude_neighbor_blocks < 0:
        parser.error('Counts and neighbor radius must be non-negative')
    run(args)
