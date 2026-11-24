"""Adapters for saved experiment-3 pairs and experiment-2 direct checkpoints."""
import json
from pathlib import Path

from experiment_1.components.config import sha256
from experiment_1.components.repetition import repetition_metrics
from experiment_1.components.token_accounting import token_totals
from meeting_evaluation.metrics import reference_metrics, source_overlap_support

CONDITIONS = ('baseline', 'semantic')
SCORE_VARIANT = 'summary'


def workflow_samples(run_dir):
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    completed, failed, missing, fingerprints = {}, [], [], {}
    if len(set(manifest['sample_ids'])) != len(manifest['sample_ids']):
        raise ValueError('Workflow manifest has duplicate sample IDs')
    conditions = tuple(manifest.get('conditions', CONDITIONS))
    for index, sample_id in enumerate(manifest['sample_ids']):
        case = run_dir / f'{index:05d}'
        if (case / 'failure.json').exists():
            failed.append(sample_id)
            continue
        names = ['source.json', *[f'{c}.{suffix}' for c in conditions
                                 for suffix in ('summary.txt', 'metrics.json', 'specialists.json')]]
        if any(not (case / name).exists() for name in names):
            missing.append(sample_id)
            continue
        source = json.loads((case / 'source.json').read_text(encoding='utf-8'))
        if source['id'] != sample_id:
            raise ValueError(f'Workflow ID mismatch: {case}')
        if not source['reference'].strip():
            raise ValueError(f'{sample_id}: no reference summary. Use original meetings with references; '
                             'do not score inserted challenge decisions against unchanged AMI summaries.')
        item = dict(source=source['transcript'], reference=source['reference'], conditions={})
        for condition in conditions:
            answer = (case / f'{condition}.summary.txt').read_text(encoding='utf-8').strip()
            costs = json.loads((case / f'{condition}.metrics.json').read_text(encoding='utf-8'))
            detail = json.loads((case / f'{condition}.specialists.json').read_text(encoding='utf-8'))
            if any(stage['output_cap_hit'] for stage in detail['stages'].values()):
                raise ValueError(f'{sample_id}/{condition}: incomplete generated output must not be quality-scored')
            item['transcript_tokens'] = detail['transcript_tokens']
            item['conditions'][condition] = dict(answer=answer,
                costs=dict(total_tokens=costs['logical_total_tokens'],
                           pipeline_seconds=costs['standalone_pipeline_seconds'], cuda_peak_mib=None),
                details=dict(logical_input_tokens=costs['logical_input_tokens'],
                    logical_output_tokens=costs['logical_output_tokens'],
                    embedding_input_tokens=costs['total_embedding_input_tokens'],
                    local_kv_position_events=costs['local_kv_position_events'],
                    final_kv_positions=costs['final_kv_positions']))
        for name in names:
            fingerprints[(case / name).relative_to(run_dir).as_posix()] = sha256((case / name).read_bytes())
        completed[sample_id] = item
    return manifest, completed, failed, missing, fingerprints


def direct_entry(sample):
    from experiment_2.components.config import VARIANT
    answer = sample['result']['answers'][VARIANT]
    m = sample['result']['variants'][VARIANT]
    if m['output_cap_hit']:
        raise ValueError(f'{sample["id"]}: direct output incomplete')
    counts = token_totals(m)
    peak = m.get('pipeline_peak_allocated_bytes', m.get('cuda_peak_allocated_bytes'))
    return dict(answer=answer, costs=dict(total_tokens=counts['total_tokens'],
        pipeline_seconds=m['pipeline_seconds_excluding_model_load'],
        cuda_peak_mib=peak / 2**20 if peak is not None else None),
        details=dict(logical_input_tokens=counts['total_input_tokens'],
                     logical_output_tokens=counts['total_output_tokens'], embedding_input_tokens=0,
                     local_kv_position_events=None, final_kv_positions=None))


def score_item(sample_id, condition, source, reference, entry, transcript_tokens):
    answer = entry['answer']
    quality = {**reference_metrics(answer, reference), **source_overlap_support(answer, source)}
    return dict(id=f'{sample_id}/{condition}', record_id=sample_id, condition=condition,
        reference=reference, transcript_tokens=transcript_tokens,
        result=dict(answers={SCORE_VARIANT: answer}), quality={SCORE_VARIANT: quality},
        costs={**entry['costs'], **entry['details']},
        duplicate_bullet_excess=repetition_metrics(answer)['duplicate_bullet_excess'])


def comparable_settings(workflow, direct):
    keys = ('model', 'seed', 'summary_seed', 'summary_temperature', 'summary_top_p',
            'summary_thinking', 'top_k', 'repetition_penalty', 'fp32_lm_head')
    different = {key: [workflow['config'].get(key), direct['config'].get(key)] for key in keys
                 if workflow['config'].get(key) != direct['config'].get(key)}
    if different:
        raise ValueError(f'Model/final-generation settings differ: {different}')


def runtime_metadata(directory):
    path = Path(directory) / 'runtime.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
