"""Length-stratified reports from saved observations; never rerun generation."""
import csv
from types import SimpleNamespace

from .config import VARIANTS
from .storage import atomic_json


GROUPS = ('0-4000', '4000-8000', '8000-12000', '12000-16000', '16000+')


def length_group(tokens):
    if tokens is None:
        return 'unknown'
    for index, upper in enumerate((4000, 8000, 12000, 16000)):
        if tokens < upper:
            return GROUPS[index]
    return GROUPS[-1]


def sample_tokens(sample):
    tokens = sample.get('transcript_tokens')
    return tokens if tokens is not None else sample.get('result', {}).get('transcript_tokens')


def write_length_reports(store, completed, failures, planned, overall, report_writer):
    groups = {}
    labels = list(GROUPS)
    if any(sample_tokens(sample) is None for sample in completed + failures):
        labels.append('unknown')
    for label in labels:
        done = [s for s in completed if length_group(sample_tokens(s)) == label]
        failed = [s for s in failures if length_group(sample_tokens(s)) == label]
        directory = store.directory / 'length_groups' / label
        directory.mkdir(parents=True, exist_ok=True)
        groups[label] = report_writer(SimpleNamespace(directory=directory), done, failed,
                                     len(done)+len(failed), group_by_length=False)
    definitions = 'Raw transcript tokens, including labels/timestamps, excluding chat templates; [0,4000), [4000,8000), [8000,12000), [12000,16000), [16000,+inf). Unknown legacy lengths are not estimated.'
    atomic_json(store.directory / 'overview.json', dict(
        length_definition=definitions, overall=overall, length_groups=groups,
        unprocessed_samples=planned-len(completed)-len(failures)))
    rows, lines = [], ['Experiment 1: overall and transcript-length groups', definitions,
        'group | planned | completed | failed | KV ROUGE-L | text ROUGE-L | KV sec | text sec | KV tokens | text tokens | token savings %']
    def format_value(value):
        return 'N/A' if value is None else f'{value:.4f}'
    for label, stats in [('overall', overall), *groups.items()]:
        pair = []
        for variant in VARIANTS:
            quality = stats.get('metrics', {}).get('quality', {}).get(variant, {})
            efficiency = stats.get('metrics', {}).get('efficiency', {}).get('variants', {}).get(variant, {})
            tokens = stats.get('tokens', {}).get(variant, {})
            resources = stats.get('additional_resources', {}).get(variant, {})
            peak = resources.get('mean_cuda_peak_allocated_bytes')
            row = dict(length_group=label, variant=variant, planned=stats['planned_samples'],
                completed=stats['completed_samples'], failed=stats['failed_samples'],
                rouge_1_f1=quality.get('rouge_1_f1'), rouge_2_f1=quality.get('rouge_2_f1'),
                rouge_l_f1=quality.get('rouge_l_f1'), bertscore_f1=quality.get('bertscore_f1'),
                source_support_proxy=quality.get('supported_claim_rate'),
                mean_pipeline_seconds=efficiency.get('mean_pipeline_seconds_excluding_model_load'),
                mean_input_tokens=tokens.get('total_input_tokens', {}).get('mean'),
                mean_output_tokens=tokens.get('total_output_tokens', {}).get('mean'),
                mean_total_tokens=tokens.get('total_tokens', {}).get('mean'),
                mean_cuda_peak_mib=peak/2**20 if peak is not None else None,
                handoff_cap_hits=resources.get('handoff_cap_hits'), receiver_cap_hits=resources.get('receiver_cap_hits'),
                duplicate_receiver_samples=stats.get('repetition', {}).get(variant, {}).get('receiver_samples_with_duplicate_bullets'),
                token_savings_percent=stats.get('token_savings_kv_vs_text_percent'))
            rows.append(row)
            pair.append(row)
        cells = [pair[0]['rouge_l_f1'], pair[1]['rouge_l_f1'], pair[0]['mean_pipeline_seconds'],
                 pair[1]['mean_pipeline_seconds'], pair[0]['mean_total_tokens'], pair[1]['mean_total_tokens'],
                 stats.get('token_savings_kv_vs_text_percent')]
        lines.append(f"{label} | {stats['planned_samples']} | {stats['completed_samples']} | {stats['failed_samples']} | " +
                     ' | '.join(format_value(value) for value in cells))
    (store.directory / 'overview.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with (store.directory / 'overview.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
