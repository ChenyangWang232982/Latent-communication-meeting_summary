"""Score existing V3 and full-text direct summaries; never rerun generation."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

from experiment_1.components.config import sha256
from experiment_1.components.semantic_quality import add_bertscore
from experiment_1.components.storage import atomic_json
from experiment_1.components.length_groups import GROUPS, length_group
from experiment_2.compare import paired_summary
from experiment_2.components.reporting import fmt
from experiment_2.rescore import load_samples
from .components.evaluation import (CONDITIONS, SCORE_VARIANT, comparable_settings, direct_entry,
                                    runtime_metadata, score_item, workflow_samples)


def evaluate(run_dir, direct_run, *, output_dir=None, bertscore_model=None, layers=12, device='cpu'):
    run_dir, direct_run = Path(run_dir).resolve(), Path(direct_run).resolve()
    wm, workflow, wf_failed, wf_missing, wf_fingerprints = workflow_samples(run_dir)
    dm, direct_completed, direct_failed, direct_fingerprints = load_samples(direct_run)
    comparable_settings(wm, dm)
    conditions = tuple(wm.get('conditions', CONDITIONS))
    direct = {s['id']: s for s in direct_completed}
    paired_ids = [sid for sid in wm['sample_ids'] if sid in workflow and sid in direct]
    if not paired_ids:
        raise ValueError('No matching completed records between experiment_3 and experiment_2')
    samples = []
    for sid in paired_ids:
        item, baseline = workflow[sid], direct[sid]
        if item['source'] != baseline['source'] or item['reference'] != baseline['reference']:
            raise ValueError(f'Source/reference mismatch for ID {sid}')
        for condition in conditions:
            samples.append(score_item(sid, condition, item['source'], item['reference'],
                item['conditions'][condition], item['transcript_tokens']))
        samples.append(score_item(sid, 'direct_text', item['source'], item['reference'],
            direct_entry(baseline), item['transcript_tokens']))
    semantic_metadata = add_bertscore(samples, bertscore_model, layers, device,
        variants=(SCORE_VARIANT,)) if bertscore_model else None
    rows = [dict(id=s['record_id'], condition=s['condition'], transcript_tokens=s['transcript_tokens'],
        length_group=length_group(s['transcript_tokens']), **s['quality'][SCORE_VARIANT], **s['costs'],
        duplicate_bullet_excess=s['duplicate_bullet_excess']) for s in samples]
    by_id = {(r['id'], r['condition']): r for r in rows}
    comparisons = {}
    for condition in conditions:
        paired_rows = []
        for sid in paired_ids:
            pair = dict(id=sid, transcript_tokens=workflow[sid]['transcript_tokens'])
            for prefix, label in (('workflow', condition), ('direct', 'direct_text')):
                pair.update({prefix + '_' + k: v for k, v in by_id[sid, label].items()
                             if k not in ('id', 'condition', 'length_group')})
            paired_rows.append(pair)
        comparisons[condition] = dict(overall=paired_summary(paired_rows),
            length_groups={label: paired_summary([r for r in paired_rows
                if length_group(r['transcript_tokens']) == label]) for label in GROUPS})
    means = {}
    for condition in (*conditions, 'direct_text'):
        subset = [r for r in rows if r['condition'] == condition]
        metrics = [k for k in subset[0] if k not in ('id', 'condition', 'length_group')]
        means[condition] = {k: mean(r[k] for r in subset) if all(r[k] is not None for r in subset) else None
                            for k in metrics}
    runtimes = {name: runtime_metadata(path) for name, path in (('workflow', run_dir), ('direct', direct_run))}
    notes = [
        'All means use the same matching completed records. Failed or missing records are not assigned zero quality.',
        'ROUGE uses lowercase ASCII alphanumeric tokens, no stemming; whole-summary ROUGE-L, not ROUGE-Lsum.',
        'ROUGE/BERTScore measure reference similarity; source-support is a lexical overlap proxy, not factual accuracy.',
        'This is an overall workflow comparison. Workflow synthesis includes local reports and additional synthesis '
        'instructions; direct_text receives the full transcript and the shared summary request.',
        'Logical decoder token events include thinking and terminal tokens, not attention FLOPs, network traffic or billing.',
        'Embedding input events are separate; embedding/ranking time is included in the workflow pipeline time.',
        'Workflow standalone times are reconstructed from shared prefill/selection and per-condition stages; '
        'the paired run executes shared work only once. Direct times are measured independently.',
        'Workflow CUDA peaks were not recorded by experiment_3; no paired memory comparison is available.',
        'Historical versus new execution is not randomized; matching runtime metadata does not establish equal timing conditions.',
        'Paired intervals reuse experiment_2: record bootstrap, 2000 repetitions, seed 7; tiny pilots are descriptive.',
        'Evidence co-location coverage requires separate source-evidence annotations and is not inferred from these scores.']
    report = dict(planned_workflow_samples=len(wm['sample_ids']), paired_samples=len(paired_ids), paired_ids=paired_ids,
        workflow_failed_ids=wf_failed, workflow_missing_ids=wf_missing,
        direct_failed_matching_ids=[s['id'] for s in direct_failed if s['id'] in wm['sample_ids']],
        workflow_completed_without_direct=[sid for sid in workflow if sid not in direct],
        direct_planned_samples=len(dm['sample_ids']), means=means, comparisons=comparisons,
        runtime_metadata_equal=runtimes['workflow'] is not None and runtimes['workflow'] == runtimes['direct'],
        runtime_metadata=runtimes, semantic_evaluation=semantic_metadata, notes=notes)
    folder = Path(output_dir).resolve() if output_dir else run_dir / 'evaluation' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if any(folder == p or p.is_relative_to(folder) for p in (run_dir, direct_run)):
        raise ValueError('Choose a separate evaluation output directory')
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / 'summary.json', report)
    atomic_json(folder / 'evaluation_manifest.json', dict(workflow_run=str(run_dir), direct_run=str(direct_run),
        workflow_manifest_sha256=sha256((run_dir / 'manifest.json').read_bytes()),
        direct_manifest_sha256=sha256((direct_run / 'manifest.json').read_bytes()),
        workflow_files_sha256=wf_fingerprints, direct_samples_sha256=direct_fingerprints,
        evaluation_code_sha256=sha256(Path(__file__).read_bytes()), semantic_evaluation=semantic_metadata))
    with (folder / 'per_sample.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['Experiment 3: V3 workflow vs full-transcript single-agent baseline',
        f'Paired records: {len(paired_ids)}/{len(wm["sample_ids"])}',
        f'Runtime metadata equal: {report["runtime_metadata_equal"]}',
        'condition | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | logical tokens | pipeline seconds']
    for condition, values in means.items():
        lines.append(' | '.join([condition, *[fmt(values.get(k)) for k in (
            'rouge_1_f1', 'rouge_2_f1', 'rouge_l_f1', 'bertscore_f1', 'total_tokens', 'pipeline_seconds')]]))
    for condition, comparison in comparisons.items():
        lines.extend(['', f'{condition} minus direct_text',
                      'metric | paired n | workflow | direct | difference | bootstrap 95% interval'])
        for metric, values in comparison['overall']['metrics'].items():
            lines.append(' | '.join([metric, str(values['n']), fmt(values['workflow']), fmt(values['direct']),
                fmt(values['difference_workflow_minus_direct']), str(values['bootstrap_95_percent_interval'])]))
    lines.extend(['', *notes])
    (folder / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:4 + len(means)]))
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='experiment_3 generation timestamp directory')
    parser.add_argument('--direct-run', type=Path, required=True, help='experiment_2 generation timestamp directory')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--bertscore-model')
    parser.add_argument('--bertscore-layers', type=int, default=12)
    parser.add_argument('--device', default='cpu', help='BERTScore encoder device only')
    args = parser.parse_args()
    print('Reports:', evaluate(args.run_dir, args.direct_run, output_dir=args.output_dir,
        bertscore_model=args.bertscore_model, layers=args.bertscore_layers, device=args.device))
