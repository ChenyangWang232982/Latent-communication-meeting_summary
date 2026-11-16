"""Coordinate paired inference, atomic checkpoints and aggregate reporting."""
import gc
import json

from meeting_evaluation.records import load_records
from receiver_kv_workflow.engine import ReceiverKVRunner
from .config import VARIANTS, arguments, model_config, reproducibility
from .reporting import write_reports
from .resources import ResourceRecorder
from .scoring import score_sample
from .storage import RunStore, atomic_json


def run(args, runner_factory=ReceiverKVRunner):
    records = load_records("jsonl", args.input)
    records = records[:args.num_samples] if args.num_samples else records
    if not records:
        raise ValueError("No samples selected")
    if len({r.id for r in records}) != len(records):
        raise ValueError("Experiment inputs must have unique sample IDs")
    if any(not r.answer.strip() for r in records):
        raise ValueError("Every quality-evaluation sample must have a nonempty reference summary")
    config = model_config(args)
    store = RunStore(args.output_dir, args.resume, reproducibility(args, records, config))
    print(f"Experiment directory: {store.directory}; selected {len(records)} samples", flush=True)
    completed, failures = [], []
    runner = None
    for index, record in enumerate(records):
        previous = store.load(index, record.id)
        if previous and previous["status"] == "completed":
            completed.append(previous)
            print(f"Resumed {index + 1}/{len(records)}: {record.id}", flush=True)
            continue
        if runner is None:
            runner = runner_factory(config, progress=lambda message: print(f"[experiment_1] {message}", flush=True))
            runtime = {"device": str(runner.device), "model_revision": getattr(getattr(getattr(runner, "model", None), "config", None), "_commit_hash", None)}
            if runner.device.type == "cuda":
                props = runner.torch.cuda.get_device_properties(runner.device)
                runtime.update(gpu_name=props.name, gpu_total_memory_bytes=props.total_memory,
                               cuda_version=runner.torch.version.cuda)
            runtime_path = store.directory / "runtime.json"
            if runtime_path.exists() and json.loads(runtime_path.read_text(encoding="utf-8")) != runtime:
                raise ValueError("Cannot resume timing measurements on different hardware or model revision")
            atomic_json(runtime_path, runtime)
        resources = ResourceRecorder(runner)
        try:
            resources.begin()
            result = runner.run(record.id, record.context, variants=VARIANTS, review=False, on_variant=resources)
            result["shared_prefill_selection_peak_allocated_bytes"] = resources.shared_peak_allocated_bytes
            result["gold_answer"] = record.answer
            sample = score_sample(result, record, resources.values)
            completed.append(sample)
        except Exception as exc:
            sample = {"id": record.id, "status": "failed", "source": record.context, "reference": record.answer,
                      "error_type": type(exc).__name__, "error": str(exc)}
            failures.append(sample)
            gc.collect()
            if runner.device.type == "cuda":
                runner.torch.cuda.empty_cache()
        store.save(index, sample)
        atomic_json(store.directory / "progress.json", {"planned": len(records), "completed": len(completed), "failed": len(failures),
                                                       "last_sample_id": record.id})
        print(f"{index + 1}/{len(records)}: {record.id}: {sample['status']}", flush=True)
    atomic_json(store.directory / "progress.json", {"planned": len(records), "completed": len(completed),
                                                   "failed": len(failures), "finished": True})
    write_reports(store, completed, failures, len(records))
    return store.directory, len(failures)


def main():
    directory, failures = run(arguments())
    print(f"Reports: {directory}")
    if failures:
        raise SystemExit(f"{failures} sample(s) failed; inspect checkpoints and resume to retry. Paired means use completed samples only.")
