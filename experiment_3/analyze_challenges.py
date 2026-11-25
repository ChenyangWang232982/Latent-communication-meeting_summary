"""Audit saved endpoint challenge results, without generation or factuality judging."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

from experiment_1.components.repetition import repetition_metrics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def wilson(hits, n):
    z = 1.959963984540054
    p = hits / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0, center-half), min(1, center+half)]


def analyze(root, output):
    workflow = next(root.glob('workflow/*/manifest.json')).parent
    direct = next(root.glob('direct/*/direct_text/manifest.json')).parent
    wm, dm = read(workflow/'manifest.json'), read(direct/'manifest.json')
    if wm['sample_ids'] != dm['sample_ids'] or wm['input_sha256'] != dm['dataset']['meetings_sha256']:
        raise ValueError('Run manifests do not describe the same ordered challenge input')
    with (root/'report/target_review.csv').open(encoding='utf-8-sig') as handle:
        reviews = list(csv.DictReader(handle))
    keys = {row['sample_id']: row for row in reviews}
    conditions = ['baseline', 'semantic', 'direct_text']
    rows, fingerprints = [], {}
    same_selected, cap_hits, reconstructed = [], Counter(), []
    categories = Counter()
    for index, sid in enumerate(wm['sample_ids']):
        case = workflow/f'{index:05d}'
        src = read(case/'source.json')
        if src['id'] != sid:
            raise ValueError('Source ID mismatch')
        reconstructed.append(dict(id=sid, transcript=src['transcript']))
        categories[keys[sid]['category']] += 1
        selection = read(case/'selection.json')['selected_positions']
        coverage = read(case/'target_coverage.json')
        dc = next(direct.glob(f'*_{sid}'))
        result = read(dc/'result.json')
        if result['id'] != sid:
            raise ValueError('Direct result ID mismatch')
        for condition in conditions:
            summary = dc/'summary.txt' if condition == 'direct_text' else case/f'{condition}.summary.txt'
            text = summary.read_text(encoding='utf-8')
            rep = repetition_metrics(text)
            if condition == 'direct_text':
                values = result['variants']['direct_text']
                accounts = values['token_accounting']
                metrics = dict(logical_input_tokens=sum(v for k, v in accounts.items() if not k.endswith('output_tokens') and k != 'specialist_handoff_tokens'),
                    logical_output_tokens=accounts['receiver_output_tokens'],
                    logical_total_tokens=sum(accounts.values()),
                    standalone_pipeline_seconds=values['pipeline_seconds_excluding_model_load'],
                    final_output_tokens=values['generated_tokens'])
                cap_hits[condition] += int(values['output_cap_hit'])
                detail = None
            else:
                metrics = read(case/f'{condition}.metrics.json')
                detail = read(case/f'{condition}.specialists.json')
                cap_hits[condition] += sum(int(stage['output_cap_hit']) for stage in detail['stages'].values())
                same_selected.append(detail['synthesis_selected_positions'] == selection)
            cov = coverage.get(condition)
            rows.append(dict(index=index, sample_id=sid, category=keys[sid]['category'], condition=condition,
                opening_retained=cov['opening']['fully_retained'] if cov else None,
                closing_retained=cov['closing']['fully_retained'] if cov else None,
                co_location=cov['both_in_same_local_window'] if cov else None,
                common_local_windows=sorted(set(cov['opening']['local_windows']) & set(cov['closing']['local_windows'])) if cov else [],
                specialist_count=detail['actual_specialist_windows'] if detail else 0,
                **metrics, **rep, expected=keys[sid]['expected'],
                summary_path=str(summary.resolve())))
        for path in [case/'source.json', case/'selection.json', case/'target_coverage.json',
                     *(case/f'{c}.summary.txt' for c in conditions[:2]),
                     *(case/f'{c}.metrics.json' for c in conditions[:2]),
                     *(case/f'{c}.specialists.json' for c in conditions[:2]),
                     dc/'summary.txt', dc/'result.json']:
            fingerprints[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    # The preparer serializes exactly these two fields, preserving input order.
    text = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in reconstructed)
    input_hash_verified = hashlib.sha256(text.encode('utf-8')).hexdigest() == wm['input_sha256']
    if not input_hash_verified or not all(same_selected):
        raise ValueError('Input reconstruction or final selected-position audit failed')
    for path in (workflow/'manifest.json', direct/'manifest.json', root/'report/target_review.csv'):
        fingerprints[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    groups = {}
    for condition in conditions:
        group = [r for r in rows if r['condition'] == condition]
        covered = [r for r in group if r['co_location'] is not None]
        hits = sum(r['co_location'] for r in covered)
        means = {k: mean(r[k] for r in group) for k in (
            'logical_input_tokens', 'logical_output_tokens', 'logical_total_tokens',
            'standalone_pipeline_seconds', 'final_output_tokens', 'specialist_count')}
        for k in ('local_kv_position_events', 'final_kv_positions', 'total_embedding_input_tokens'):
            means[k] = mean(r[k] for r in group) if condition != 'direct_text' else None
        groups[condition] = dict(n=len(group), coverage_hits=hits if covered else None,
            coverage_n=len(covered), coverage_rate=hits/len(covered) if covered else None,
            coverage_wilson_95=wilson(hits, len(covered)) if covered else None,
            opening_retained=sum(r['opening_retained'] for r in covered) if covered else None,
            closing_retained=sum(r['closing_retained'] for r in covered) if covered else None,
            means=means, median_seconds=median(r['standalone_pipeline_seconds'] for r in group),
            exact_duplicate_samples=sum(r['duplicate_bullet_excess'] > 0 for r in group),
            duplicate_bullet_excess=sum(r['duplicate_bullet_excess'] for r in group),
            cap_hit_stages=cap_hits[condition],
            # Exploratory subset: do not label this a controlled runtime comparison.
            seconds_excluding_first_exploratory=mean(r['standalone_pipeline_seconds'] for r in group if r['index'] != 0))
    by_category = {category: {c: dict(n=sum(r['category']==category and r['condition']==c for r in rows),
        hits=sum(r['co_location'] for r in rows if r['category']==category and r['condition']==c))
        for c in conditions[:2]} for category in categories}
    data = dict(run=str(root.resolve()), workflow=str(workflow.resolve()), direct=str(direct.resolve()),
        git_commit=wm['git_commit'], config=wm['config'], runtime=read(workflow/'runtime.json'),
        workflow_status=read(workflow/'status.json'), direct_status=read(direct/'status.json'),
        reconstructed_input_hash_verified=input_hash_verified, final_positions_equal=all(same_selected),
        final_decoding_settings_equal=all(wm['config'][k]==dm['config'][k] for k in (
            'model', 'seed', 'summary_seed', 'summary_temperature', 'summary_top_p',
            'summary_thinking', 'top_k', 'repetition_penalty', 'fp32_lm_head')),
        category_counts=dict(categories), groups=groups, by_category=by_category,
        filled_original_verdicts=sum(bool(r['verdict'].strip()) for r in reviews),
        factual_accuracy_computed=False, timing_valid_for_formal_comparison=False,
        timing_note='Unauthorized GPU workload interfered with early execution; no clean timing boundary is recorded.',
        fingerprints=fingerprints)
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    # Use a stable union of keys because the direct condition has fewer measurements.
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output/'per_case_metrics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({c: {k:v for k,v in g.items() if k!='means'} | {'means':g['means']}
                      for c,g in groups.items()}, ensure_ascii=True, indent=2))
    print('Saved:', output)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_dir, args.output_dir)
