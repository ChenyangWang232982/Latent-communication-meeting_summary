"""Full, routed and refill cache controls, isolated from formal experiment scores."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from meeting_evaluation.records import load_records
from receiver_kv_workflow.engine import ReceiverKVRunner
from .config import add_workflow_arguments, model_config, reproducibility
from .generation_trace import GenerationTrace, fingerprint
from .progress import ExperimentProgress
from .storage import atomic_json


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sample-id", help="Exact record ID of the repeating sample")
    group.add_argument("--sample-index", type=int, default=0, help="Zero-based input row index (default: 0)")
    parser.add_argument("--condition", choices=("full", "routed", "refill", "both", "all"), default="both",
                        help="refill recomputes the selected token IDs at original positions; all runs full, refill, routed")
    parser.add_argument("--output-dir", type=Path, default=Path("test_result/generation_diagnostic"))
    add_workflow_arguments(parser)
    return parser.parse_args()


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
    manifest.update(condition=args.condition, diagnostic=True, variants=["routed_kv"])
    atomic_json(directory / "manifest.json", manifest)
    display = ExperimentProgress()
    try:
        display.log(f"Diagnostic directory: {directory}; sample: {record.id}")
        runner = ReceiverKVRunner(config, progress=display.log)
        runner.progress = lambda _: None
        conditions = (("full", "refill", "routed") if args.condition == "all" else
                      ("full", "routed") if args.condition == "both" else (args.condition,))
        for index, condition in enumerate(conditions):
            display.sample(index, len(conditions), f"{record.id}: {condition}")
            folder = directory / condition
            atomic_json(folder / "status.json", {"status": "running", "condition": condition})
            trace = GenerationTrace(folder, 0, record.id, display.log)
            try:
                result = runner.run(record.id, record.context, variants=("routed_kv",), review=False,
                                    diagnostic_full_source=condition == "full",
                                    diagnostic_refill_selected=condition == "refill",
                                    on_state=display.state, on_generation=trace, on_token=trace.token)
            except BaseException as error:
                atomic_json(folder / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                       "error": str(error), "condition": condition})
                raise
            finally:
                trace.flush()
            text = result["answers"]["routed_kv"]
            (folder / "summary.txt").write_text(text, encoding="utf-8")
            atomic_json(folder / "status.json", {
                "status": "completed", "condition": condition,
                "diagnostic_full_source": result["diagnostic_full_source"],
                "diagnostic_refill_selected": result.get("diagnostic_refill_selected", False),
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
