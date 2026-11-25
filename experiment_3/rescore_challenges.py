"""Exploratory original-AMI-reference scores for saved challenge summaries only."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

from experiment_1.components.config import sha256
from experiment_1.components.semantic_quality import add_bertscore
from experiment_1.components.storage import atomic_json
from experiment_2.compare import interval
from meeting_evaluation.metrics import reference_metrics
from meeting_evaluation.records import load_records
from .components.evaluation import comparable_settings

CONDITIONS = ('baseline', 'semantic', 'direct_text')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare(run_dir, reference_path):
    workflow = next(run_dir.glob('workflow/*/manifest.json')).parent
    direct = next(run_dir.glob('direct/*/direct_text/manifest.json')).parent
    wm, dm = read(workflow/'manifest.json'), read(direct/'manifest.json')
    comparable_settings(wm, dm)
    if wm['sample_ids'] != dm['sample_ids'] or wm['input_sha256'] != dm['dataset']['meetings_sha256']:
        raise ValueError('The runs must use the same ordered challenge input')
    originals = {r.id: r for r in load_records('jsonl', reference_path)}
    direct_cases = {read(p)['id']: p.parent for p in direct.glob('*/result.json')}
    samples, missing, references, fingerprints = [], [], {}, {}
    for index, sid in enumerate(wm['sample_ids']):
        folder = workflow/f'{index:05d}'
        dc = direct_cases.get(sid)
        paths = [folder/'baseline.summary.txt', folder/'semantic.summary.txt', dc/'summary.txt' if dc else None]
        if (folder/'failure.json').exists() or not dc or (dc/'failure.json').exists() or any(p is None or not p.exists() for p in paths):
            missing.append(sid)
            continue
        if not sid.startswith('ami_') or not sid.endswith('_endpoint'):
            raise ValueError(f'Unrecognized challenge ID: {sid}')
        original_id = sid[len('ami_'):-len('_endpoint')]
        original = originals.get(original_id)
        if original is None or not original.answer.strip():
            raise ValueError(f'{sid}: original AMI reference missing from {reference_path}')
        source = read(folder/'source.json')
        # The challenge builder adds exactly one opening and one closing line.
        _, first_separator, rest = source['transcript'].partition('\n')
        middle, last_separator, _ = rest.rpartition('\n')
        if source['id'] != sid or not first_separator or not last_separator or middle != original.context:
            raise ValueError(f'{sid}: original meeting body differs; do not match references by ID alone')
        direct_result = read(dc/'result.json')
        if direct_result['variants']['direct_text']['output_cap_hit']:
            raise ValueError(f'{sid}: direct summary is incomplete')
        for condition in CONDITIONS[:2]:
            stages = read(folder/f'{condition}.specialists.json')['stages']
            if any(stage['output_cap_hit'] for stage in stages.values()):
                raise ValueError(f'{sid}/{condition}: incomplete stage output')
        references[sid] = dict(original_id=original_id, reference=original.answer,
            original_source_sha256=sha256(original.context.encode('utf-8')),
            reference_sha256=sha256(original.answer.encode('utf-8')))
        for condition, path in zip(CONDITIONS, paths):
            text = path.read_text(encoding='utf-8').strip()
            samples.append(dict(id=f'{sid}/{condition}', record_id=sid, condition=condition,
                reference=original.answer, result=dict(answers=dict(summary=text)),
                quality=dict(summary=reference_metrics(text, original.answer))))
            fingerprints[str(path.relative_to(run_dir))] = sha256(path.read_bytes())
        fingerprints[str((folder/'source.json').relative_to(run_dir))] = sha256((folder/'source.json').read_bytes())
    if not samples:
        raise ValueError('No completed three-way pairs to score')
    return wm, samples, missing, references, fingerprints


def rescore(run_dir, reference_path, *, output_dir=None,
            model='allenai/longformer-base-4096', layers=12, device='cuda', rouge_only=False):
    run_dir, reference_path = Path(run_dir).resolve(), Path(reference_path).resolve()
    wm, samples, missing, references, fingerprints = prepare(run_dir, reference_path)
    n = len(references)
    print(f'Matched {n}/{len(wm["sample_ids"])} meetings; {len(samples)} saved summaries. No generation.', flush=True)
    metadata = None
    if not rouge_only:
        print(f'Computing BERTScore P/R/F1: {model}, layer {layers}, device {device}', flush=True)
        metadata = add_bertscore(samples, model, layers, device, variants=('summary',))
    rows = [dict(id=s['record_id'], condition=s['condition'], **s['quality']['summary']) for s in samples]
    means = {c: {k: mean(r[k] for r in rows if r['condition']==c) for k in rows[0] if k not in ('id', 'condition')}
             for c in CONDITIONS}
    by_id = {(r['id'], r['condition']): r for r in rows}
    comparisons = {}
    for left, right in (('semantic', 'baseline'), ('semantic', 'direct_text'), ('baseline', 'direct_text')):
        metrics = {}
        for metric in ('rouge_1_f1', 'rouge_2_f1', 'rouge_l_f1', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1'):
            if metric not in rows[0]:
                continue
            differences = [by_id[sid, left][metric]-by_id[sid, right][metric] for sid in references]
            metrics[metric] = dict(n=n, difference=mean(differences), bootstrap_95_interval=interval(differences))
        comparisons[f'{left}_minus_{right}'] = metrics
    notes = [
        'EXPLORATORY ONLY: unchanged original AMI summaries, not complete references for modified meetings.',
        'Synthetic endpoint additions are absent from these references. Correctly including them can lower precision or F1.',
        'ROUGE and BERTScore are reference-similarity metrics, not factual accuracy or target-decision correctness.',
        'All means use matching completed three-way pairs. Failed/missing outputs are disclosed, not assigned zero.',
        'Whole saved final answers are scored. No output edits, target extraction, silent truncation, generation or AI judge.',
        'ROUGE: lowercase ASCII alphanumeric tokens, no stemming; whole-summary ROUGE-L rather than ROUGE-Lsum.',
        'Paired intervals: record-level bootstrap, 2000 repetitions, seed 7; this authored challenge set is not a random population sample.',
        'Evaluation compute is separate from workflow generation cost. No new timing or efficiency conclusion is calculated.']
    report = dict(reference_mode='unchanged_original_ami_exploratory', planned=len(wm['sample_ids']), paired=n,
                  missing_or_failed_ids=missing, means=means, comparisons=comparisons,
                  semantic_evaluation=metadata, factual_accuracy=None, notes=notes)
    output = Path(output_dir).resolve() if output_dir else run_dir/'original_reference_evaluation'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'summary.json', report)
    atomic_json(output/'references.json', references)
    atomic_json(output/'evaluation_manifest.json', dict(run_dir=str(run_dir), reference_input=str(reference_path),
        reference_input_sha256=sha256(reference_path.read_bytes()), summary_source_sha256=fingerprints,
        scoring_code_sha256=sha256(Path(__file__).read_bytes()), semantic_evaluation=metadata))
    with (output/'per_sample.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ('rouge_1_f1', 'rouge_2_f1', 'rouge_l_precision', 'rouge_l_recall', 'rouge_l_f1',
               'bertscore_precision', 'bertscore_recall', 'bertscore_f1')
    lines = ['# Challenge summaries: unchanged original-reference similarity (exploratory)', '',
             f'Completed three-way pairs: {n}/{len(wm["sample_ids"])}', '',
             '| condition | R1 F1 | R2 F1 | RL P | RL R | RL F1 | BERT P | BERT R | BERT F1 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        values = [f'{means[c][k]:.4f}' if k in means[c] else 'N/A' for k in metrics]
        lines.append('| '+' | '.join([c, *values])+' |')
    lines.extend(['', *[f'- {note}' for note in notes]])
    for name, values in comparisons.items():
        lines.extend(['', f'## {name}', '', '| metric | mean difference | paired bootstrap 95% interval |', '|---|---:|---|'])
        for metric, value in values.items():
            ci = value['bootstrap_95_interval']
            lines.append(f'| {metric} | {value["difference"]:.6f} | {ci} |')
    text = '\n'.join(lines)+'\n'
    (output/'summary.md').write_text(text, encoding='utf-8')
    (output/'summary.txt').write_text(text, encoding='utf-8')
    print('\n'.join(lines[:9]), flush=True)
    print('Reports:', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Challenge batch root containing workflow/ and direct/')
    parser.add_argument('--references', type=Path, required=True, help='Original normalized AMI JSONL with transcript and summary')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--bertscore-model', default='allenai/longformer-base-4096')
    parser.add_argument('--bertscore-layers', type=int, default=12)
    parser.add_argument('--device', default='cuda', help='Scoring encoder only; no Qwen is loaded')
    parser.add_argument('--rouge-only', action='store_true')
    args = parser.parse_args()
    rescore(args.run_dir, args.references, output_dir=args.output_dir, model=args.bertscore_model,
            layers=args.bertscore_layers, device=args.device, rouge_only=args.rouge_only)
