"""Run receiver-conditioned KV evaluation on normalized MeetingBank records."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .config import RunPaths
from .records import load_records
from .report import write_receiver_kv_communication_reports, write_receiver_kv_reports
from .workflow import run_receiver_kv
from receiver_kv_workflow.engine import ReceiverKVConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    from vector.cli import add_arguments
    add_arguments(parser)
    parser.add_argument("--benchmark", choices=["ami", "icsi", "qmsum_committee", "meetingbank"], default="meetingbank")
    parser.add_argument("--input", type=Path, required=True, help="Normalized MeetingBank JSONL input.")
    parser.add_argument("--output-dir", type=Path, default=Path("test_result"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--num-samples", type=int, default=0, help="0 evaluates every available record.")
    parser.add_argument("--receiver-kv-role-budget-ratio", type=float, default=0.15)
    parser.add_argument("--receiver-kv-span-tokens", type=int, default=32)
    parser.add_argument("--speaker-neighbor-blocks", type=int, default=1)
    parser.add_argument("--receiver-kv-query-batch-tokens", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for model inference (default: 7)")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking for the final summary receiver")
    parser.add_argument("--variants", nargs="+", choices=["routed_kv", "specialist_text"],
                        default=["routed_kv", "specialist_text"])
    parser.add_argument("--communication-metrics-only", action="store_true",
                        help="Discard summaries and write only token, cache and execution measurements.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_variants = {"routed_kv", "specialist_text"}
    if args.communication_metrics_only:
        if set(args.variants) != all_variants:
            raise ValueError(
                "--communication-metrics-only requires routed_kv and specialist_text."
            )
    elif set(args.variants) != all_variants:
        raise ValueError("A quality evaluation requires all receiver-KV variants.")
    paths = RunPaths(args.input, args.output_dir)
    # Every prepared benchmark uses the same normalized JSONL schema. The
    # benchmark name identifies the reported domain, not a different loader.
    records = load_records("jsonl", paths.input_path, None)
    source_by_id = {record.id: record.context for record in records[: args.num_samples] if args.num_samples}
    if not source_by_id:
        source_by_id = {record.id: record.context for record in records}
    from vector.cli import config_kwargs
    receiver_config = ReceiverKVConfig(
        model=args.model, device=args.device or "cuda", enable_thinking=args.enable_thinking, seed=args.seed,
        role_budget_ratio=args.receiver_kv_role_budget_ratio,
        span_tokens=args.receiver_kv_span_tokens,
        query_batch_tokens=args.receiver_kv_query_batch_tokens,
        speaker_neighbor_blocks=args.speaker_neighbor_blocks,
        **config_kwargs(args),
    )
    results = run_receiver_kv(records, receiver_config, num_samples=args.num_samples, variants=tuple(args.variants))
    from .metrics import score_receiver_kv_results
    metrics = score_receiver_kv_results(results, source_by_id, include_quality=not args.communication_metrics_only)
    configuration = {"workflow": "receiver_kv", "num_samples": args.num_samples,
                     "communication_metrics_only": args.communication_metrics_only, **asdict(receiver_config)}
    configuration["variants"] = args.variants
    writer = write_receiver_kv_communication_reports if args.communication_metrics_only else write_receiver_kv_reports
    text_path, json_path = writer(paths.output_dir, args.benchmark, metrics, configuration)
    print(f"Metric report: {text_path}")
    print(f"Machine-readable metrics: {json_path}")


if __name__ == "__main__":
    main()
