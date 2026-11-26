"""Replay saved selections and generate only a random-evidence control condition."""
import argparse
import csv
import gc
import json
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from statistics import mean

from experiment_1.components.config import sha256
from experiment_1.components.storage import atomic_json
from .components.random_control import random_windows
from .retrieve import saved_evidence


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def replay_model_snapshot(model, revision):
    """Resolve the recorded Hub commit before loading weights or the tokenizer."""
    if not revision:
        return model, dict(requested_model=model, pinned_revision=None, snapshot=None)
    if Path(model).is_dir():
        raise ValueError('Saved controls name a local model with a Hub revision; use its original Hub repo ID to pin the snapshot.')
    from huggingface_hub import snapshot_download
    print(f'Reusing control model revision: {model}@{revision}', flush=True)
    snapshot = snapshot_download(repo_id=model, revision=revision, allow_patterns=[
        'config.json', 'generation_config.json', 'tokenizer*', 'chat_template*',
        'vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json',
        '*.safetensors', '*.safetensors.index.json'])
    # A full saved commit hash must resolve to that exact cached commit directory.
    if len(revision) == 40 and Path(snapshot).name != revision:
        raise ValueError(f'Hub resolved a different snapshot: expected {revision}, got {snapshot}')
    return snapshot, dict(requested_model=model, pinned_revision=revision, snapshot=str(snapshot))


def run(args):
    source_run = args.source_run.resolve()
    manifest = read(source_run/'manifest.json')
    ids = manifest['sample_ids'][:args.num_samples] if args.num_samples else manifest['sample_ids']
    # Same generation implementation and framing are needed to reuse historical controls.
    project = Path(__file__).resolve().parents[1]
    files = ('chunk_specialist_workflow/components/generation.py', 'receiver_kv_workflow/cache.py',
             'receiver_kv_workflow/prompts.py', 'receiver_kv_workflow/speaker_blocks.py',
             'receiver_kv_workflow/source_mapping.py')
    def code_matches(name):
        raw = (project/name).read_bytes()
        lf = raw.replace(b'\r\n', b'\n')
        return manifest.get('code_sha256', {}).get(name) in {sha256(raw), sha256(lf), sha256(lf.replace(b'\n', b'\r\n'))}
    mismatches = [name for name in files if not code_matches(name)]
    if mismatches and not args.plan_only:
        raise ValueError(f'Generation code differs from saved controls: {mismatches}. Use matching code for a valid replay.')
    output = args.output_dir/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    plans, rows = [], []
    for index, sid in enumerate(ids):
        case = source_run/f'{index:05d}'
        source = read(case/'source.json')
        if source['id'] != sid or (case/'failure.json').exists():
            raise ValueError(f'Incomplete source control: {sid}')
        blocks, texts, primary, selection = saved_evidence(case)
        semantic = read(case/'semantic.specialists.json')
        windows = random_windows(blocks, texts, selection['selected_positions'], primary,
            semantic['windows'], record_id=sid, seed=args.random_seed,
            neighbor_radius=manifest['exclude_neighbor_blocks'], trials=args.match_trials)
        target = output/f'{index:05d}'
        target.mkdir()
        atomic_json(target/'source.json', source)
        atomic_json(target/'random.windows.json', dict(windows=windows,
            selected_sha256=selection['selected_sha256']))
        plans.append((case, target, source, blocks, selection, semantic, windows))
        for wi, window in enumerate(windows):
            rows.append(dict(index=index, id=sid, window=wi, **window['matching']))
    with (output/'window_matching.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    total_target = sum(r['target_added_positions'] for r in rows)
    total_actual = sum(r['actual_added_positions'] for r in rows)
    atomic_json(output/'manifest.json', dict(source_run=str(source_run), sample_ids=ids,
        source_manifest_sha256=sha256((source_run/'manifest.json').read_bytes()),
        config=manifest['config'], random_seed=args.random_seed, match_trials=args.match_trials,
        conditions=['random'], plan_only=args.plan_only,
        saved_generation_code_matches=not mismatches,
        replay_code_sha256={name:sha256((project/name).read_bytes()) for name in
            (*files, 'experiment_3/random_control.py', 'experiment_3/components/random_control.py')},
        source_files_sha256={str(p.relative_to(source_run)):sha256(p.read_bytes())
            for case, *_ in plans for p in (case/'source.json', case/'selection.json',
                                          case/'baseline.specialists.json', case/'semantic.specialists.json')},
        note='Replay exact saved global positions and primary windows; no selectors or embedding rerun. '
             'Random selection uses audit texts for broad content filters, not cosine scores or target annotations. '
             'Same supplementary block count; for at most two blocks, enumerate nearest-length sets '
             'and sample uniformly among ties with a fixed seed. Larger sets use seeded random draws. '
             'Whole-block constraints may leave mismatches, reported per window. '
             'Historical controls are reused; new runtime and memory are not a controlled full-workflow comparison.'))
    print(f'Plans: {len(ids)} meetings, {len(rows)} windows; supplementary positions '
          f'random={total_actual}, semantic={total_target}', flush=True)
    print('Output:', output, flush=True)
    if args.plan_only:
        return output
    from receiver_kv_workflow.engine import ReceiverKVConfig, ReceiverKVRunner
    from receiver_kv_workflow.prompts import prompt_parts
    from receiver_kv_workflow.speaker_blocks import speaker_units
    from receiver_kv_workflow.cache import forward_tokens
    from chunk_specialist_workflow.components.generation import run_specialists
    # Model weights and generation settings are inherited. Embedding is unnecessary in replay.
    config = replace(ReceiverKVConfig(**manifest['config']), device=args.device, vector_routing=False)
    old_runtime = read(source_run/'runtime.json') if (source_run/'runtime.json').exists() else {}
    expected_revision = old_runtime.get('model_revision')
    model_path, model_identity = replay_model_snapshot(config.model, expected_revision)
    config = replace(config, model=model_path)
    runner = ReceiverKVRunner(config, progress=lambda s: print(s, flush=True))
    reported_revision = getattr(runner.model.config, '_commit_hash', None)
    # Local snapshot loading may report _commit_hash=None. The explicitly resolved
    # immutable Hub snapshot above, rather than incidental config metadata, pins identity.
    atomic_json(output/'runtime.json', dict(device=str(runner.device),
        model_revision=expected_revision or reported_revision,
        config_reported_revision=reported_revision, model_identity=model_identity,
        effective_config=asdict(config), embedding_loaded=False))
    completed, coverage_rows = 0, []
    for index, (case, target, src, blocks, selection, old_detail, windows) in enumerate(plans):
        try:
            transcript = src['transcript']
            prefix, _, protected = prompt_parts(runner.tokenizer, transcript)
            selected = selection['selected_positions']
            if len(prefix) != selection['source_positions'] or speaker_units(runner.tokenizer, transcript, prefix, protected) != blocks:
                raise ValueError('Tokenizer source mapping differs from the saved selection')
            if set(selected) != set(old_detail['synthesis_selected_positions']):
                raise ValueError('Saved semantic synthesis positions differ from the frozen selection')
            runner.progress(f'{src["id"]}: replay full-source prefill; random control only')
            def live(event):
                (target/(event['stage']+'.partial.txt')).write_text(event['text'], encoding='utf-8')
            with runner.torch.inference_mode():
                runner._sync(); tick = time.perf_counter()
                pref = forward_tokens(runner.model, runner._tensor(prefix), None, 0, logits_to_keep=1)
                cache = pref.past_key_values
                del pref
                runner._sync(); prefill_seconds = time.perf_counter()-tick
                try:
                    answer, detail = run_specialists(runner, transcript, prefix, protected, selected, cache, blocks,
                        windows=windows, semantic_context=True, communication='kv',
                        transcript_tokens=old_detail['transcript_tokens'], on_generation=live)
                finally:
                    del cache
            (target/'random.summary.txt').write_text(answer+'\n', encoding='utf-8')
            atomic_json(target/'random.specialists.json', detail)
            stages = list(detail['stages'].values())
            old_base = read(case/'baseline.specialists.json')
            old_metrics = read(case/'baseline.metrics.json')
            inherited_shared_inputs = old_metrics['logical_input_tokens']-sum(s['prompt_tokens'] for s in old_base['stages'].values())
            inputs = inherited_shared_inputs+sum(s['prompt_tokens'] for s in stages)
            outputs = sum(s['output_tokens'] for s in stages)
            atomic_json(target/'random.metrics.json', dict(logical_input_tokens=inputs,
                logical_output_tokens=outputs, logical_total_tokens=inputs+outputs,
                inherited_shared_logical_input_tokens=inherited_shared_inputs,
                local_kv_position_events=sum(s['source_kv_positions'] for s in stages[:-1]),
                final_kv_positions=len(selected), replay_prefill_seconds=prefill_seconds,
                replay_generation_seconds=detail['pipeline_seconds'],
                timing_note='Replay excludes selection/embedding; not full standalone workflow latency. '
                            'Logical shared-input accounting is inherited for comparability, not reexecuted probes.'))
            # Target annotations are read only AFTER selection and generation.
            if args.answer_key:
                key = read(args.answer_key)[src['id']]
                if key['modified_source_sha256'] != sha256(transcript.encode()):
                    raise ValueError('Answer annotation does not match source')
                from experiment_2.challenge_samples.run import coverage
                random_cov = coverage(runner.tokenizer, transcript, selected, key, windows)
                atomic_json(target/'target_coverage.json', dict(random=random_cov))
                old_cov = read(case/'target_coverage.json')
                coverage_rows.append(dict(id=src['id'],
                    baseline=old_cov['baseline']['both_in_same_local_window'],
                    semantic=old_cov['semantic']['both_in_same_local_window'],
                    random=random_cov['both_in_same_local_window']))
            completed += 1
        except Exception as exc:
            atomic_json(target/'failure.json', dict(error_type=type(exc).__name__, error=str(exc)))
            print(f'{src["id"]}: {type(exc).__name__}: {exc}', flush=True)
        finally:
            gc.collect()
            if runner.device.type=='cuda': runner.torch.cuda.empty_cache()
        atomic_json(output/'status.json', dict(planned=len(ids), completed=completed, failed=index+1-completed,
                                              processed=index+1))
        print(f'{index+1}/{len(ids)} random control processed', flush=True)
    with (output/'coverage_comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'baseline', 'semantic', 'random'])
        writer.writeheader(); writer.writerows(coverage_rows)
    lines = ['Random evidence control: historical baseline/semantic versus new random',
        f'Completed random generations: {completed}/{len(ids)}',
        f'Coverage-evaluable paired cases: {len(coverage_rows)}',
        f'Exact token-count-matched windows: {sum(r["absolute_position_difference"]==0 for r in rows)}/{len(rows)}',
        f'Total supplementary positions: semantic={total_target}; random={total_actual}',
        f'Mean absolute per-window position mismatch: {mean(r["absolute_position_difference"] for r in rows):.3f}',
        f'Same supplement set as semantic (including empty sets): {sum(r["same_supplement_set"] for r in rows)}/{len(rows)}',
        'Coverage is evidence organization, not final factual accuracy. No target answers enter selection.',
        'Replay timing excludes selectors and embedding; do not compare it to saved standalone latency.']
    if coverage_rows:
        for condition in ('baseline', 'semantic', 'random'):
            hits = sum(r[condition] for r in coverage_rows)
            lines.append(f'{condition}: {hits}/{len(coverage_rows)} = {hits/len(coverage_rows):.1%}')
    (output/'comparison.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return output


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True, help='Saved workflow timestamp directory, not batch root')
    parser.add_argument('--output-dir', type=Path, default=Path('test_result/experiment_3_random_control'))
    parser.add_argument('--num-samples', type=int, default=3, help='0 means all saved cases')
    parser.add_argument('--random-seed', type=int, default=7)
    parser.add_argument('--match-trials', type=int, default=512)
    parser.add_argument('--answer-key', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--plan-only', action='store_true', help='Audit budgets without model loading or generation')
    args = parser.parse_args()
    if args.num_samples < 0 or args.match_trials < 1: parser.error('Invalid sample/trial count')
    run(args)
