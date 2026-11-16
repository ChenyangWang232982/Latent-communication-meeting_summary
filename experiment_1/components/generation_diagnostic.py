"""A full-source cache control, isolated from formal experiment scores."""
import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from meeting_evaluation.records import load_records
from receiver_kv_workflow.engine import ReceiverKVRunner
from receiver_kv_workflow.prompts import prompt_parts
from .config import model_config, reproducibility
from .generation_trace import GenerationTrace
from .progress import ExperimentProgress
from .storage import atomic_json


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sample-id", help="Exact record ID of the repeating sample")
    group.add_argument("--sample-index", type=int, default=0, help="Zero-based input row index (default: 0)")
    parser.add_argument("--condition", choices=("full", "routed", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("test_result/generation_diagnostic"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--role-budget-ratio", type=float, default=0.25)
    parser.add_argument("--speaker-neighbor-blocks", type=int, default=1)
    parser.add_argument("--span-tokens", type=int, default=32)
    parser.add_argument("--query-batch-tokens", type=int, default=8)
    from vector.cli import add_arguments
    add_arguments(parser)
    return parser.parse_args()


def fingerprint(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def run(args):
    records = load_records("jsonl", args.input)
    if args.sample_id is not None:
        matches = [record for record in records if record.id == args.sample_id]
        if len(matches) != 1:
            raise ValueError("--sample-id must match exactly one record")
        record = matches[0]
    else:
        if not 0 <= args.sample_index < len(records):
            raise ValueError("--sample-index is outside the input records")
        record = records[args.sample_index]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    directory = args.output_dir / stamp
    directory.mkdir(parents=True, exist_ok=False)
    config = model_config(args)
    # Same code/data/environment fingerprints as experiment 1, but separate
    # manifests and outputs: diagnostic controls are never aggregated as scores.
    args.benchmark = "diagnostic"
    manifest = reproducibility(args, [record], config)
    manifest.update(condition=args.condition, diagnostic=True, config=asdict(config))
    atomic_json(directory / "manifest.json", manifest)
    display = ExperimentProgress()
    try:
        display.log(f"Diagnostic directory: {directory}; sample: {record.id}")
        runner = ReceiverKVRunner(config, progress=display.log)
        runner.progress = lambda _: None
        prefix, suffix, _ = prompt_parts(runner.tokenizer, record.context)
        atomic_json(directory / "prompt.json", {
            "source_positions": len(prefix), "suffix_tokens": len(suffix),
            "source_token_sha256": fingerprint(prefix), "suffix_token_sha256": fingerprint(suffix),
            "model_eos_token_id": runner.model.generation_config.eos_token_id,
            "tokenizer_eos_token_id": runner.tokenizer.eos_token_id,
            "enable_thinking": False, "temperature": 0.0,
        })
        conditions = ("full", "routed") if args.condition == "both" else (args.condition,)
        for index, condition in enumerate(conditions):
            display.sample(index, len(conditions), f"{record.id}: {condition}")
            folder = directory / condition
            atomic_json(folder / "status.json", {"status": "running", "condition": condition})
            trace = GenerationTrace(folder, 0, record.id, display.log)
            try:
                result = runner.run(record.id, record.context, variants=("routed_kv",), review=False,
                                    diagnostic_full_source=condition == "full",
                                    on_state=display.state, on_generation=trace)
            except BaseException as error:
                atomic_json(folder / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                       "error": str(error), "condition": condition})
                raise
            text = result["answers"]["routed_kv"]
            (folder / "summary.txt").write_text(text, encoding="utf-8")
            atomic_json(folder / "status.json", {
                "status": "completed", "condition": condition,
                "diagnostic_full_source": result["diagnostic_full_source"],
                "source_positions": result["source_positions"],
                "retained_positions": len(result["selected_source_positions"]),
                "selected_position_sha256": fingerprint(result["selected_source_positions"]),
                "metrics": result["variants"]["routed_kv"],
            })
            display.finish_sample(f"{condition} completed")
    finally:
        display.close()
    return directory


def main():
    print(f"Diagnostic saved: {run(arguments())}")
