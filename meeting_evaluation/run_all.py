"""Run all available meeting-evaluation benchmarks with one command.

Each benchmark is launched in its own process so that GPU memory from one
Qwen run is released before the next begins. MESA-style source-overlap review
and communication metrics are included by ``meeting_evaluation.test``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_QMSUM = Path("meeting_evaluation/input/qmsum_test.jsonl")
DEFAULT_MEETINGBANK = Path("meeting_evaluation/input/meetingbank_test.jsonl")
DEFAULT_AMI = Path("prophetkv_statebridge/input/ami_ES2002a.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qmsum", type=Path, default=DEFAULT_QMSUM)
    parser.add_argument("--meetingbank", type=Path, default=DEFAULT_MEETINGBANK)
    parser.add_argument("--ami", type=Path, default=DEFAULT_AMI)
    parser.add_argument("--ami-reference", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("meeting_evaluation/output"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--source-max-tokens", type=int, default=8192)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=1024)
    parser.add_argument("--backfill-rounds", type=int, default=0)
    parser.add_argument("--agent-max-new-tokens", type=int, default=128)
    parser.add_argument("--receiver-max-new-tokens", type=int, default=192)
    parser.add_argument("--prefix-tokens-per-agent", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip a benchmark whose input is unavailable instead of failing before the run.",
    )
    return parser.parse_args()


def test_command(benchmark: str, input_path: Path, args: argparse.Namespace, reference: Path | None = None) -> list[str]:
    command = [
        sys.executable, "-m", "meeting_evaluation.test",
        "--benchmark", benchmark,
        "--input", str(input_path),
        "--output-dir", str(args.output_dir),
        "--num-samples", str(args.num_samples),
        "--model", args.model,
        "--source-max-tokens", str(args.source_max_tokens),
        "--prefill-chunk-tokens", str(args.prefill_chunk_tokens),
        "--backfill-rounds", str(args.backfill_rounds),
        "--agent-max-new-tokens", str(args.agent_max_new_tokens),
        "--receiver-max-new-tokens", str(args.receiver_max_new_tokens),
        "--prefix-tokens-per-agent", str(args.prefix_tokens_per_agent),
    ]
    if reference:
        command.extend(["--reference", str(reference)])
    if args.device:
        command.extend(["--device", args.device])
    return command


def main() -> None:
    args = parse_args()
    # One invocation gets one minute-level directory even if benchmark runs
    # themselves extend past the boundary into the next minute.
    run_environment = os.environ.copy()
    run_environment["MEETING_EVAL_RUN_ID"] = datetime.now().strftime("%Y%m%d_%H%M")
    jobs = [
        ("qmsum", args.qmsum, None),
        ("meetingbank", args.meetingbank, None),
        ("ami", args.ami, args.ami_reference),
    ]
    missing = [f"{name}: {path}" for name, path, _ in jobs if not path.is_file()]
    if missing and not args.skip_missing:
        raise FileNotFoundError(
            "Missing benchmark input(s):\n- " + "\n- ".join(missing) + "\n"
            "Download/copy them, or use --skip-missing to run only available benchmarks."
        )

    completed = []
    for benchmark, input_path, reference in jobs:
        if not input_path.is_file():
            print(f"Skipping {benchmark}: input not found: {input_path}", flush=True)
            continue
        print(f"\n=== Running {benchmark} ===", flush=True)
        subprocess.run(test_command(benchmark, input_path, args, reference), check=True, env=run_environment)
        completed.append(benchmark)
    if not completed:
        raise RuntimeError("No benchmark was run.")
    print(f"\nCompleted: {', '.join(completed)}")
    print(f"Reports: {args.output_dir}/{run_environment['MEETING_EVAL_RUN_ID']}/<tool>.txt")


if __name__ == "__main__":
    main()
