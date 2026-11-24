import csv
import math
from statistics import mean, median

from experiment_1.components.storage import atomic_json
from experiment_1.components.token_accounting import token_totals
from experiment_1.components.repetition import repetition_metrics
from experiment_1.components.length_groups import length_group, GROUPS
from .config import VARIANT


def fmt(value, digits=4):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def summarize(samples):
    if not samples:
        return dict(n=0, quality={}, resources={}, tokens={})
    measurements = [s['result']['variants'][VARIANT] for s in samples]
    common = set.intersection(*(set(s['quality'][VARIANT]) for s in samples))
    quality = {k: mean(s['quality'][VARIANT][k] for s in samples)
               for k in sorted(common) if all(s['quality'][VARIANT][k] is not None for s in samples)}
    accounting = [token_totals(m) for m in measurements]
    seconds = sorted(m['pipeline_seconds_excluding_model_load'] for m in measurements)
    peaks = [m.get('cuda_peak_allocated_bytes') for m in measurements if m.get('cuda_peak_allocated_bytes') is not None]
    return dict(n=len(samples), quality=quality,
        tokens={k:dict(mean=mean(r[k] for r in accounting), sum=sum(r[k] for r in accounting)) for k in accounting[0]},
        resources=dict(mean_pipeline_seconds=mean(seconds), median_pipeline_seconds=median(seconds),
            p95_pipeline_seconds=seconds[math.ceil(.95*len(seconds))-1],
            mean_cuda_peak_mib=mean(peaks)/2**20 if peaks else None,
            cap_hits=sum(bool(m['output_cap_hit']) for m in measurements),
            exact_duplicate_samples=sum(repetition_metrics(s['result']['answers'][VARIANT])['duplicate_bullet_excess']>0 for s in samples)))


def write_reports(folder, completed, failures, planned):
    groups = {}
    # Reuse the canonical experiment-1 length bins without imposing a length filter.
    for label in GROUPS:
        subset = [s for s in completed if length_group(s['transcript_tokens'])==label]
        failed = [s for s in failures if length_group(s.get('transcript_tokens'))==label]
        groups[label] = dict(planned=len(subset)+len(failed), failed=len(failed), **summarize(subset))
    result = dict(workflow='full_transcript_single_agent', variant=VARIANT,
                  planned_samples=planned, completed_samples=len(completed), failed_samples=len(failures),
                  failure_ids=[s['id'] for s in failures], overall=summarize(completed), length_groups=groups,
                  notes=[
                    'Same shared SYSTEM and SUMMARY_REQUEST; no local reports, selection, embedding, or independent review.',
                    'The complete chat prompt is counted once in receiver_prompt_tokens; source_prefill_tokens is zero to prevent double counting.',
                    'Reasoning, delimiter and EOS tokens are included in output events; only the final answer is quality-scored.',
                    'CUDA peaks include the generation model and this single prompt cache; no embedding model or separate immutable source cache is resident.',
                    'Model loading and evaluation costs excluded. No truncation and no arbitrary output-token cap; model-context exhaustion is a failure.',
                    'This is an overall workflow comparison, not an identical-evidence or representation-only intervention.'])
    atomic_json(folder/'summary.json', result)
    overall = result['overall']
    lines = ['Experiment 2: full-transcript single-agent baseline',
             f'Planned: {planned}; completed: {len(completed)}; failed: {len(failures)}',
             'group | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | logical tokens | pipeline sec | CUDA peak MiB']
    for label, values in [('overall', overall), *groups.items()]:
        q,r,t = values['quality'], values['resources'], values['tokens']
        lines.append(' | '.join([label,str(values['n']),*[fmt(q.get(k)) for k in ('rouge_1_f1','rouge_2_f1','rouge_l_f1','bertscore_f1')],
            fmt(t.get('total_tokens',{}).get('mean'),1),fmt(r.get('mean_pipeline_seconds'),2),fmt(r.get('mean_cuda_peak_mib'),1)]))
    if completed:
        lines.extend(['',f'Exact duplicate bullet samples: {overall["resources"]["exact_duplicate_samples"]}/{len(completed)}'])
        if 'bertscore_f1' in overall['quality']:
            q=overall['quality']
            lines.append('BERTScore P/R/F1: '+' / '.join(fmt(q[k]) for k in ('bertscore_precision','bertscore_recall','bertscore_f1')))
    lines.extend(['',*result['notes']])
    (folder/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    rows=[]
    for sample in completed:
        m=sample['result']['variants'][VARIANT]
        rows.append(dict(id=sample['id'], variant=VARIANT, transcript_tokens=sample['transcript_tokens'],
            length_group=length_group(sample['transcript_tokens']), **sample['quality'][VARIANT], **token_totals(m),
            pipeline_seconds=m['pipeline_seconds_excluding_model_load'],
            cuda_peak_allocated_bytes=m.get('cuda_peak_allocated_bytes'),
            exact_duplicate_bullet_excess=repetition_metrics(sample['result']['answers'][VARIANT])['duplicate_bullet_excess']))
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['id','variant']
    with (folder/'per_sample.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return result
