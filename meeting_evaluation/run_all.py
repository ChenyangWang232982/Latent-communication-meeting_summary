"""Run the configured MeetingBank receiver-KV evaluation in one process."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_MEETINGBANK = Path("input/meetingbank_test.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meetingbank", type=Path, default=DEFAULT_MEETINGBANK)
    parser.add_argument("--output-dir", type=Path, default=Path("test_result"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--receiver-kv-role-budget-ratio", type=float, default=0.15)
    parser.add_argument("--receiver-kv-span-tokens", type=int, default=32)
    parser.add_argument("--receiver-kv-query-batch-tokens", type=int, default=8)
    parser.add_argument("--receiver-kv-max-new-tokens", type=int, default=1024)
    parser.add_argument("--receiver-kv-specialist-text-max-new-tokens", type=int, default=384)
    parser.add_argument("--device")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip a benchmark whose input is unavailable instead of failing before the run.",
    )
    return parser.parse_args()


def test_command(input_path: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, "-m", "meeting_evaluation.test",
        "--benchmark", "meetingbank",
        "--input", str(input_path),
        "--output-dir", str(args.output_dir),
        "--num-samples", str(args.num_samples),
        "--model", args.model,
    ]
    if args.device:
        command.extend(["--device", args.device])
    command.extend([
        "--receiver-kv-role-budget-ratio", str(args.receiver_kv_role_budget_ratio),
        "--receiver-kv-span-tokens", str(args.receiver_kv_span_tokens),
        "--receiver-kv-query-batch-tokens", str(args.receiver_kv_query_batch_tokens),
        "--receiver-kv-max-new-tokens", str(args.receiver_kv_max_new_tokens),
        "--receiver-kv-specialist-text-max-new-tokens", str(args.receiver_kv_specialist_text_max_new_tokens),
    ])
    return command


def main() -> None:
    args = parse_args()
    # One invocation gets one minute-level directory even if benchmark runs
    # themselves extend past the boundary into the next minute.
    run_environment = os.environ.copy()
    run_environment["MEETING_EVAL_RUN_ID"] = datetime.now().strftime("%Y%m%d_%H%M")
    jobs = [("meetingbank", args.meetingbank)]
    missing = [f"{name}: {path}" for name, path in jobs if not path.is_file()]
    if missing and not args.skip_missing:
        raise FileNotFoundError(
            "Missing benchmark input(s):\n- " + "\n- ".join(missing) + "\n"
            "Download/copy them, or use --skip-missing to run only available benchmarks."
        )

    completed = []
    for benchmark, input_path in jobs:
        if not input_path.is_file():
            print(f"Skipping {benchmark}: input not found: {input_path}", flush=True)
            continue
        print(f"\n=== Running {benchmark} ===", flush=True)
        subprocess.run(test_command(input_path, args), check=True, env=run_environment)
        completed.append(benchmark)
    if not completed:
        raise RuntimeError("No benchmark was run.")
    print(f"\nCompleted: {', '.join(completed)}")
    print(f"Reports: {args.output_dir}/{run_environment['MEETING_EVAL_RUN_ID']}/<tool>.txt")


if __name__ == "__main__":
    main()
