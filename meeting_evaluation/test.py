"""Single entry point for QMSum, MeetingBank, AMI, or optional audio-ASR evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .asr import transcribe
from .config import ASRConfig, RunPaths, WorkflowConfig
from .metrics import score_results
from .records import DEFAULT_MEETING_QUESTION, MeetingRecord, load_records
from .report import write_receiver_kv_reports, write_reports
from .workflow import run_receiver_kv, run_statebridge
from receiver_kv_workflow.engine import ReceiverKVConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["qmsum", "meetingbank", "ami", "jsonl"], required=True)
    parser.add_argument("--workflow", choices=["statebridge", "receiver_kv"], default="statebridge")
    parser.add_argument("--input", type=Path, required=True, help="Dataset/transcript input, or audio when --enable-asr is set.")
    parser.add_argument("--reference", type=Path, help="Optional AMI reference summary.")
    parser.add_argument("--output-dir", type=Path, default=Path("test_result"))
    parser.add_argument("--enable-asr", action="store_true", help="Transcribe --input audio before evaluation.")
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--num-samples", type=int, default=0, help="0 evaluates every available record.")
    parser.add_argument("--source-max-tokens", type=int, default=8192)
    parser.add_argument("--prefix-tokens-per-agent", type=int, default=64)
    parser.add_argument("--evidence-policy", choices=["strict", "retrieval", "off"], default="strict")
    parser.add_argument("--alignment-scope", choices=["joint", "per-message"], default="joint")
    parser.add_argument("--receiver-kv-role-budget-ratio", type=float, default=0.15)
    parser.add_argument("--receiver-kv-span-tokens", type=int, default=32)
    parser.add_argument("--receiver-kv-query-batch-tokens", type=int, default=8)
    parser.add_argument("--receiver-kv-max-new-tokens", type=int, default=1024)
    parser.add_argument("--receiver-kv-specialist-text-max-new-tokens", type=int, default=384)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = RunPaths(args.input, args.output_dir, args.reference)
    if args.workflow == "receiver_kv" and args.benchmark != "meetingbank":
        raise ValueError("receiver_kv is currently registered for MeetingBank only; it always generates a meeting summary rather than QMSum query answers.")
    if args.enable_asr:
        transcript = transcribe(paths.input_path, ASRConfig(enabled=True, model=args.asr_model))
        records = [MeetingRecord(paths.input_path.stem, transcript, DEFAULT_MEETING_QUESTION)]
    else:
        records = load_records(args.benchmark, paths.input_path, paths.reference_path)
    config = WorkflowConfig(
        model=args.model,
        source_max_tokens=args.source_max_tokens,
        prefix_tokens_per_agent=args.prefix_tokens_per_agent,
        evidence_policy=args.evidence_policy,
        alignment_scope=args.alignment_scope,
        device=args.device,
        num_samples=args.num_samples,
    )
    source_by_id = {record.id: record.context for record in records[: config.num_samples] if config.num_samples}
    if not source_by_id:
        source_by_id = {record.id: record.context for record in records}
    if args.workflow == "statebridge":
        # QMSum is query-based meeting QA. MeetingBank and AMI require the
        # meeting-summary receiver.
        results = run_statebridge(records, config, meeting_mode=args.benchmark in {"meetingbank", "ami"})
        metrics = score_results(results, source_by_id)
        text_path, json_path = write_reports(paths.output_dir, args.benchmark, metrics, asdict(config))
    else:
        receiver_config = ReceiverKVConfig(
            model=args.model, device=args.device or "cuda",
            role_budget_ratio=args.receiver_kv_role_budget_ratio,
            span_tokens=args.receiver_kv_span_tokens,
            query_batch_tokens=args.receiver_kv_query_batch_tokens,
            max_new_tokens=args.receiver_kv_max_new_tokens,
            specialist_text_max_new_tokens=args.receiver_kv_specialist_text_max_new_tokens,
        )
        results = run_receiver_kv(records, receiver_config, num_samples=args.num_samples)
        from .metrics import score_receiver_kv_results
        metrics = score_receiver_kv_results(results, source_by_id)
        configuration = {"workflow": args.workflow, "num_samples": args.num_samples, **asdict(receiver_config)}
        text_path, json_path = write_receiver_kv_reports(paths.output_dir, args.benchmark, metrics, configuration)
    print(f"Metric report: {text_path}")
    print(f"Machine-readable metrics: {json_path}")


if __name__ == "__main__":
    main()
