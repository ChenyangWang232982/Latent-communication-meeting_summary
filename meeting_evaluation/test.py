"""Single entry point for QMSum, MeetingBank, AMI, or optional audio-ASR evaluation."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import asdict
from pathlib import Path

from .asr import transcribe
from .config import ASRConfig, RunPaths, WorkflowConfig
from .metrics import score_results
from .records import DEFAULT_MEETING_QUESTION, MeetingRecord, load_records
from .report import write_reports
from .workflow import run_statebridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["qmsum", "meetingbank", "ami", "jsonl"], required=True)
    parser.add_argument("--input", type=Path, required=True, help="Dataset/transcript input, or audio when --enable-asr is set.")
    parser.add_argument("--reference", type=Path, help="Optional AMI reference summary.")
    parser.add_argument("--output-dir", type=Path, default=Path("meeting_evaluation/output"))
    parser.add_argument("--enable-asr", action="store_true", help="Transcribe --input audio before evaluation.")
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--num-samples", type=int, default=0, help="0 evaluates every available record.")
    parser.add_argument("--source-max-tokens", type=int, default=8192)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=1024)
    parser.add_argument("--backfill-rounds", type=int, default=0)
    parser.add_argument("--agent-max-new-tokens", type=int, default=128)
    parser.add_argument("--receiver-max-new-tokens", type=int, default=192)
    parser.add_argument("--prefix-tokens-per-agent", type=int, default=64)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = RunPaths(args.input, args.output_dir, args.reference)
    if args.enable_asr:
        transcript = transcribe(paths.input, ASRConfig(enabled=True, model=args.asr_model))
        records = [MeetingRecord(paths.input.stem, transcript, DEFAULT_MEETING_QUESTION)]
    else:
        records = load_records(args.benchmark, paths.input, paths.reference)
    config = WorkflowConfig(
        model=args.model,
        source_max_tokens=args.source_max_tokens,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
        backfill_rounds=args.backfill_rounds,
        agent_max_new_tokens=args.agent_max_new_tokens,
        receiver_max_new_tokens=args.receiver_max_new_tokens,
        prefix_tokens_per_agent=args.prefix_tokens_per_agent,
        device=args.device,
        num_samples=args.num_samples,
    )
    results = run_statebridge(records, config)
    source_by_id = {record.id: record.context for record in records[: config.num_samples] if config.num_samples}
    if not source_by_id:
        source_by_id = {record.id: record.context for record in records}
    metrics = score_results(results, source_by_id)
    text_path, json_path = write_reports(paths.output_dir, args.benchmark, metrics, asdict(config))
    print(f"Metric report: {text_path}")
    print(f"Machine-readable metrics: {json_path}")


if __name__ == "__main__":
    main()

