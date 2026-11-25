"""Export endpoint coverage and a blank target review sheet; no AI judging."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def export(dataset, workflow_run, direct_run, output_dir):
    definition = read(dataset / 'manifest.json')
    source = dataset / 'meetings.jsonl'
    if hashlib.sha256(source.read_bytes()).hexdigest() != definition['meetings_sha256']:
        raise ValueError('Challenge dataset changed after preparation')
    key_path = dataset / 'answer_key.json'
    if hashlib.sha256(key_path.read_bytes()).hexdigest() != definition['answer_key_sha256']:
        raise ValueError('Challenge answer key changed after preparation')
    keys = read(key_path)
    manifest = read(workflow_run / 'manifest.json')
    if manifest['input_sha256'] != definition['meetings_sha256']:
        raise ValueError('Workflow used a different challenge input')
    if read(direct_run / 'manifest.json')['dataset']['meetings_sha256'] != definition['meetings_sha256']:
        raise ValueError('Direct run used a different challenge input')
    workflow_cases = {read(p)['id']: p.parent for p in workflow_run.glob('*/source.json')}
    direct_cases = {read(p)['id']: p.parent for p in direct_run.glob('*/result.json')}
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, key in keys.items():
        for condition in (*manifest['conditions'], 'direct_text'):
            folder = (direct_cases if condition == 'direct_text' else workflow_cases).get(sid)
            summary = folder / ('summary.txt' if condition == 'direct_text' else f'{condition}.summary.txt') if folder else None
            completed = bool(summary and summary.exists() and not (folder / 'failure.json').exists())
            coverage = None
            if completed and condition != 'direct_text' and (folder / 'target_coverage.json').exists():
                coverage = read(folder / 'target_coverage.json').get(condition)
            rows.append(dict(sample_id=sid, category=key['category'], condition=condition,
                status='completed' if completed else 'failed_or_missing',
                opening_retained=coverage['opening']['fully_retained'] if coverage else '',
                closing_retained=coverage['closing']['fully_retained'] if coverage else '',
                co_location=coverage['both_in_same_local_window'] if coverage else '',
                expected=key['expected'], incorrect=key['incorrect'],
                summary_path=str(summary.resolve()) if completed else '',
                verdict='', notes=''))
    # Keep any previously filled review sheet intact when re-exporting.
    with (output_dir / 'target_review.csv').open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['Endpoint challenge: source coverage only; final target verdicts are blank.',
             f'Planned target items per condition: {len(keys)}',
             'condition | completed | evaluable local coverage | co-location hits | rate']
    for condition in (*manifest['conditions'], 'direct_text'):
        group = [r for r in rows if r['condition'] == condition]
        evaluable = [r for r in group if isinstance(r['co_location'], bool)]
        hits = sum(r['co_location'] for r in evaluable)
        rate = f'{hits/len(evaluable):.1%}' if evaluable else 'N/A'
        lines.append(f'{condition} | {sum(r["status"] == "completed" for r in group)} | '
                     f'{len(evaluable)} | {hits if evaluable else "N/A"} | {rate}')
    lines.extend(['', 'Category counts: ' + str(dict(Counter(k['category'] for k in keys.values()))),
        'Direct sees the full text; specialist co-location is not applicable to it.',
        'Missing/failed outputs are disclosed, not assigned a zero factuality score.',
        'Fill verdict with correct, partial, incorrect, or omitted. Blank means not reviewed.',
        'Correct requires final status and all target-specific owners, quantities, conditions and scope.',
        'An accurate late proposal is not a revised decision. Recency alone does not determine truth.',
        'These are authored endpoint diagnostics, not reference-summary ROUGE/BERTScore tests.'])
    (output_dir / 'coverage.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--workflow-run', type=Path, required=True)
    parser.add_argument('--direct-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print('Report:', export(args.dataset, args.workflow_run, args.direct_run, args.output_dir))
