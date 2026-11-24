"""Pair existing experiment-1 workflow checkpoints with direct-summary results."""
import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from statistics import mean

from experiment_1.components.config import sha256
from experiment_1.components.storage import atomic_json
from experiment_1.components.token_accounting import token_totals
from experiment_1.components.repetition import repetition_metrics
from experiment_1.components.length_groups import GROUPS, length_group
from .components.config import VARIANT
from .components.reporting import fmt
from .rescore import load_samples


def interval(differences):
    if len(differences)<2:
        return None
    rng=random.Random(7)
    estimates=sorted(mean(rng.choices(differences,k=len(differences))) for _ in range(2000))
    return [estimates[50],estimates[1949]]


def semantic_scores(directory, run_dir, variant):
    if directory is None:
        return {}, None
    metadata=json.loads((directory/'evaluation_manifest.json').read_text(encoding='utf-8'))
    if metadata['source_manifest_sha256']!=sha256((run_dir/'manifest.json').read_bytes()):
        raise ValueError('Evaluation report belongs to a different source run')
    rows=list(csv.DictReader((directory/'per_sample.csv').open(encoding='utf-8-sig')))
    return {r['id']:{k:float(v) for k,v in r.items() if k.startswith('bertscore_') and v}
            for r in rows if r['variant']==variant}, metadata.get('semantic_evaluation')


def paired_summary(rows):
    metrics={}
    for metric in ('rouge_1_f1','rouge_2_f1','rouge_l_f1','bertscore_precision','bertscore_recall','bertscore_f1',
                   'supported_claim_rate','total_tokens','pipeline_seconds','cuda_peak_mib','duplicate_bullet_excess'):
        pairs=[(r.get('workflow_'+metric),r.get('direct_'+metric)) for r in rows]
        pairs=[(a,b) for a,b in pairs if a is not None and b is not None]
        if not pairs:
            continue
        differences=[a-b for a,b in pairs]
        metrics[metric]=dict(n=len(pairs),workflow=mean(a for a,b in pairs),direct=mean(b for a,b in pairs),
            difference_workflow_minus_direct=mean(differences),
            bootstrap_95_percent_interval=interval(differences))
        if metric in ('total_tokens','pipeline_seconds'):
            baseline=sum(b for a,b in pairs)
            metrics[metric]['workflow_reduction_vs_direct_percent']=100*(1-sum(a for a,b in pairs)/baseline) if baseline else None
    return dict(n=len(rows),metrics=metrics)


def compare(run_dir, workflow_run, *, workflow_variant='chunk_kv', direct_evaluation=None,
            workflow_evaluation=None, output_dir=None):
    run_dir,workflow_run=Path(run_dir).resolve(),Path(workflow_run).resolve()
    dm,direct,failed,_=load_samples(run_dir)
    wm=json.loads((workflow_run/'manifest.json').read_text(encoding='utf-8'))
    final_keys=('model','seed','summary_seed','summary_temperature','summary_top_p','summary_thinking',
                'top_k','repetition_penalty','fp32_lm_head')
    mismatches={k:[dm['config'][k],wm['config'].get(k)] for k in final_keys if dm['config'][k]!=wm['config'].get(k)}
    if mismatches:
        raise ValueError(f'Model/final-generation settings differ: {mismatches}')
    workflow={}
    for index,sid in enumerate(wm['sample_ids']):
        path=workflow_run/'samples'/f'{index:05d}.json'
        if path.exists():
            s=json.loads(path.read_text(encoding='utf-8'))
            if s['id']!=sid:
                raise ValueError(f'Workflow sample ID mismatch: {path}')
            workflow[sid]=s
    ds,dmeta=semantic_scores(direct_evaluation,run_dir,VARIANT)
    ws,wmeta=semantic_scores(workflow_evaluation,workflow_run,workflow_variant)
    if ds and ws:
        keys=('model','model_revision','layers','hash','idf','rescale_with_baseline','bert_score_version')
        if any(dmeta.get(k)!=wmeta.get(k) for k in keys):
            raise ValueError('BERTScore configurations differ between evaluation reports')
    rows=[];unmatched=[]
    for sample in direct:
        old=workflow.get(sample['id'])
        if old is None or old['status']!='completed' or workflow_variant not in old['result']['answers']:
            unmatched.append(sample['id'])
            continue
        if sample['source']!=old['source'] or sample['reference']!=old['reference']:
            raise ValueError(f'Source/reference mismatch for ID {sample["id"]}; do not pair different inputs')
        row=dict(id=sample['id'],transcript_tokens=sample['transcript_tokens'],length_group=length_group(sample['transcript_tokens']))
        for prefix,s,v,semantic in [('direct',sample,VARIANT,ds),('workflow',old,workflow_variant,ws)]:
            m=s['result']['variants'][v]
            q={**s['quality'][v],**semantic.get(s['id'],{})}
            row.update({prefix+'_'+k:value for k,value in q.items()})
            row[prefix+'_total_tokens']=token_totals(m)['total_tokens']
            row[prefix+'_pipeline_seconds']=m['pipeline_seconds_excluding_model_load']
            peak=m.get('pipeline_peak_allocated_bytes',m.get('cuda_peak_allocated_bytes'))
            row[prefix+'_cuda_peak_mib']=peak/2**20 if peak is not None else None
            row[prefix+'_duplicate_bullet_excess']=repetition_metrics(s['result']['answers'][v])['duplicate_bullet_excess']
        rows.append(row)
    if not rows:
        raise ValueError('No matching completed pairs')
    runtimes={}
    for label,folder in [('direct',run_dir),('workflow',workflow_run)]:
        path=folder/'runtime.json'
        runtimes[label]=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    destination=Path(output_dir).resolve() if output_dir else run_dir/'comparison'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    # mkdir is exclusive so an existing report or manual annotations cannot be overwritten.
    destination.mkdir(parents=True,exist_ok=False)
    report=dict(direct_run=str(run_dir),workflow_run=str(workflow_run),workflow_variant=workflow_variant,
        direct_planned=len(dm['sample_ids']),direct_failed=len(failed),unmatched_completed_ids=unmatched,
        overall=paired_summary(rows),length_groups={g:paired_summary([r for r in rows if r['length_group']==g]) for g in GROUPS},
        runtimes=runtimes,runtime_equal=runtimes['direct'] is not None and runtimes['direct']==runtimes['workflow'],
        notes=['Means use only matching completed records; failures are not assigned zero quality.',
               'Differences are workflow minus direct; intervals are record-level paired bootstrap, 2000 repetitions, seed 7.',
               'Whole workflow versus direct baseline: source visibility, stage count and prompts necessarily differ.',
               'Peak allocation includes all observed stages when pipeline_peak is available. Resident models/caches differ by design.',
               'Historical versus new execution is not randomized; verify hardware/environment before interpreting timing.',
               'ROUGE/BERTScore indicate reference similarity, not factual accuracy.'])
    atomic_json(destination/'comparison.json',report)
    with (destination/'paired_samples.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)
    lines=[f'Experiment 2: {workflow_variant} workflow vs full-transcript direct baseline',
           f'Completed pairs: {len(rows)}; direct failures: {len(failed)}; unmatched direct completions: {len(unmatched)}',
           f'Runtime metadata equal: {report["runtime_equal"]}',
           'metric | paired n | workflow | direct | workflow-direct | bootstrap 95% interval']
    for metric,values in report['overall']['metrics'].items():
        lines.append(' | '.join([metric,str(values['n']),fmt(values['workflow']),fmt(values['direct']),
                    fmt(values['difference_workflow_minus_direct']),str(values['bootstrap_95_percent_interval'])]))
    lines.extend(['',*report['notes']])
    (destination/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--workflow-run',type=Path,required=True)
    parser.add_argument('--workflow-variant',choices=('chunk_kv','chunk_text'),default='chunk_kv')
    parser.add_argument('--direct-evaluation',type=Path)
    parser.add_argument('--workflow-evaluation',type=Path)
    parser.add_argument('--output-dir',type=Path)
    print('Comparison:',compare(**vars(parser.parse_args())))
