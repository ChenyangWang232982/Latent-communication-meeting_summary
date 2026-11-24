import gc
import json

from meeting_evaluation.records import load_records
from meeting_evaluation.metrics import reference_metrics, source_overlap_support
from experiment_1.components.storage import RunStore, atomic_json
from experiment_1.components.resources import ResourceRecorder
from experiment_1.components.progress import ExperimentProgress
from experiment_1.components.length_groups import length_group
from .config import VARIANT, model_config, manifest_for
from .generation import DirectSummaryRunner
from .reporting import write_reports


def run(args, runner_factory=DirectSummaryRunner):
    records = load_records('jsonl', args.input)
    available = len(records)
    records = records[:args.num_samples] if args.num_samples else records
    if not records or len({r.id for r in records}) != len(records):
        raise ValueError('Select nonempty records with unique IDs')
    if any(not r.answer.strip() for r in records):
        raise ValueError('Reference summaries are required for this experiment')
    config = model_config(args)
    manifest = manifest_for(args, records, config)
    manifest['sampling'] = dict(available_records=available, requested_limit=args.num_samples or None,
                                selected_records=len(records))
    store = RunStore(args.output_dir, args.resume, manifest)
    display = ExperimentProgress(label='experiment_2')
    completed, failed = [], []
    runner = None
    display.log(f'Experiment 2 directory: {store.directory}; selected {len(records)} records')
    try:
        for index, record in enumerate(records):
            display.sample(index, len(records), record.id)
            previous = store.load(index, record.id)
            if previous and previous['status'] == 'completed':
                completed.append(previous)
                display.finish_sample('resumed')
                continue
            if runner is None:
                runner = runner_factory(config, progress=display.log)
                runtime = dict(device=str(runner.device), model_revision=getattr(runner.model.config, '_commit_hash', None))
                if runner.device.type == 'cuda':
                    props = runner.torch.cuda.get_device_properties(runner.device)
                    runtime.update(gpu_name=props.name, gpu_total_memory_bytes=props.total_memory,
                                   cuda_version=runner.torch.version.cuda)
                runtime_path = store.directory/'runtime.json'
                if runtime_path.exists() and json.loads(runtime_path.read_text(encoding='utf-8')) != runtime:
                    raise ValueError('Cannot resume resource measurements on different hardware/model revision')
                atomic_json(runtime_path, runtime)
            live = store.directory/'live'/f'{index:05d}'
            live.mkdir(parents=True, exist_ok=True)

            def snapshot(event):
                (live/'direct_summary.partial.txt').write_text(event['text'], encoding='utf-8')
                atomic_json(live/'direct_summary.status.json', {k:v for k,v in event.items() if k != 'text'})

            try:
                resources = ResourceRecorder(runner)
                resources.begin()
                resources(VARIANT, 'start')
                result = runner.run_direct(record, on_progress=display.state, on_generation=snapshot)
                resources(VARIANT, 'end')
                result['variants'][VARIANT].update(resources.values[VARIANT])
                answer = result['answers'][VARIANT]
                sample = dict(id=record.id, status='completed', source=record.context, reference=record.answer,
                    transcript_tokens=result['transcript_tokens'], length_group=length_group(result['transcript_tokens']),
                    quality={VARIANT:{**reference_metrics(answer, record.answer), **source_overlap_support(answer, record.context)}},
                    result=result)
                completed.append(sample)
                if result['thinking']:
                    folder = store.directory/'summaries'/f'{index:05d}'
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder/'direct_text.thinking.md').write_text(result['thinking']+'\n', encoding='utf-8')
            except Exception as exc:
                sample = dict(id=record.id, status='failed', source=record.context, reference=record.answer,
                    transcript_tokens=len(runner.tokenizer.encode(record.context, add_special_tokens=False)),
                    error_type=type(exc).__name__, error=str(exc))
                failed.append(sample)
                gc.collect()
                if runner.device.type == 'cuda':
                    runner.torch.cuda.empty_cache()
            store.save(index, sample)
            atomic_json(store.directory/'progress.json', dict(planned=len(records), completed=len(completed), failed=len(failed)))
            display.log(f'{index+1}/{len(records)}: {record.id}: {sample["status"]}')
            display.finish_sample(sample['status'])
        write_reports(store.directory, completed, failed, len(records))
        atomic_json(store.directory/'progress.json', dict(planned=len(records), completed=len(completed), failed=len(failed), finished=True))
    finally:
        display.close()
    return store.directory, len(failed)
