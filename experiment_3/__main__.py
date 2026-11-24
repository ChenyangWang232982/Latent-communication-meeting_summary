"""Paired baseline vs semantic supplementation, with one frozen selection per sample."""
import argparse
import csv
import difflib
import gc
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from receiver_kv_workflow.cli import add_workflow_arguments, model_config
from meeting_evaluation.records import load_records
from experiment_1.components.storage import atomic_json
from .components.runner import SupplementRunner
from .components.reporting import measurement


def run(args):
    if args.supplement_blocks < 0 or args.exclude_neighbor_blocks < 0 or args.num_samples < 0:
        raise ValueError('Counts and neighbor radius must be non-negative')
    records = load_records('ami' if args.input.suffix == '.txt' else 'jsonl', args.input)
    if args.num_samples:
        records = records[:args.num_samples]
    folder = args.output_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    config = model_config(args)
    atomic_json(folder / 'manifest.json', dict(config=asdict(config), input=str(args.input.resolve()),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        sample_ids=[r.id for r in records], supplement_blocks=args.supplement_blocks,
        exclude_neighbor_blocks=args.exclude_neighbor_blocks,
        note='Same selected positions and final source KV. Baseline also receives provenance instructions. '
             'Semantic condition has more local evidence. Not an equal-evidence-volume intervention.'))
    print('Experiment directory:', folder, flush=True)
    runner = SupplementRunner(config, supplement_blocks=args.supplement_blocks,
        neighbor_radius=args.exclude_neighbor_blocks, progress=lambda s: print(s, flush=True))
    rows = []
    for index, record in enumerate(records):
        case = folder / f'{index:05d}'
        case.mkdir()
        atomic_json(case / 'source.json', dict(id=record.id, transcript=record.context, reference=record.answer))
        def live(event):
            (case / (event['stage'] + '.partial.txt')).write_text(event['text'], encoding='utf-8')
        runner.paired = None
        try:
            result = runner.run(record.id, record.context, variants=('chunk_kv',),
                                routing_audit=True, on_generation=live)
            pair = runner.paired
            atomic_json(case / 'selection.json', dict(source_positions=result['source_positions'],
                selected_positions=result['selected_source_positions'],
                selected_sha256=hashlib.sha256(json.dumps(result['selected_source_positions']).encode()).hexdigest(),
                routing_audit=result['routing_audit']))
            for condition in ('baseline', 'semantic'):
                entry = pair['conditions'][condition]
                (case / f'{condition}.summary.txt').write_text(entry['answer'] + '\n', encoding='utf-8')
                atomic_json(case / f'{condition}.specialists.json', entry['detail'])
                costs = measurement(result, entry['detail'],
                    extra_seconds=pair['retrieval']['seconds'] if condition == 'semantic' else 0,
                    extra_encoder_tokens=pair['retrieval']['embedding_input_tokens'] if condition == 'semantic' else 0)
                atomic_json(case / f'{condition}.metrics.json', costs)
                rows.append(dict(index=index, id=record.id, condition=condition, **costs))
            atomic_json(case / 'pair.json', dict(execution_order=pair['execution_order'], retrieval=pair['retrieval']))
            (case / 'summary.diff.txt').write_text(''.join(difflib.unified_diff(
                pair['conditions']['baseline']['answer'].splitlines(keepends=True),
                pair['conditions']['semantic']['answer'].splitlines(keepends=True),
                fromfile='baseline', tofile='semantic')), encoding='utf-8')
            # Optional endpoint-pilot targets are used ONLY after generation, never in prompts.
            key_path = args.input.parent / 'answer_key.json'
            if key_path.exists():
                key = json.loads(key_path.read_text(encoding='utf-8')).get(record.id)
                if key and key['modified_source_sha256'] == hashlib.sha256(record.context.encode()).hexdigest():
                    from experiment_2.challenge_samples.run import coverage
                    atomic_json(case / 'target_coverage.json', {name: coverage(runner.tokenizer,
                        record.context, result['selected_source_positions'], key, pair['conditions'][name]['detail']['windows'])
                        for name in ('baseline', 'semantic')})
        except Exception as exc:
            atomic_json(case / 'failure.json', dict(error_type=type(exc).__name__, error=str(exc)))
            print(f'{record.id}: {type(exc).__name__}: {exc}', flush=True)
        finally:
            runner.paired = None
            gc.collect()
            if runner.device.type == 'cuda':
                runner.torch.cuda.empty_cache()
        print(f'{index+1}/{len(records)} finished', flush=True)
    with (folder / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = list(rows[0]) if rows else ['index', 'id', 'condition']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    complete = sum((folder / f'{i:05d}' / 'pair.json').exists() and
                   not (folder / f'{i:05d}' / 'failure.json').exists() for i in range(len(records)))
    atomic_json(folder / 'status.json', dict(planned=len(records), completed_pairs=complete,
        failed=len(records)-complete))
    (folder / 'manual_review.csv').write_text('index,condition,target_correct,scope_errors,omissions,notes\n' +
        ''.join(f'{i},{c},,,,\n' for i in range(len(records)) for c in ('baseline', 'semantic')), encoding='utf-8-sig')
    print(f'Results: {folder}; completed pairs: {complete}/{len(records)}', flush=True)
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_workflow_arguments(parser)
    parser.add_argument('--input', type=Path, required=True, help='Speaker-labelled .txt or normalized .jsonl')
    parser.add_argument('--output-dir', type=Path, default=Path('test_result/experiment_3'))
    parser.add_argument('--num-samples', type=int, default=3, help='0 means all records')
    parser.add_argument('--supplement-blocks', type=int, default=2)
    parser.add_argument('--exclude-neighbor-blocks', type=int, default=1)
    run(parser.parse_args())
