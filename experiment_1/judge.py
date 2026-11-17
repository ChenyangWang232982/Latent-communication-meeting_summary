"""Evaluate saved paired summaries with an independent chat-completions judge."""
import json
import argparse
import os
from pathlib import Path
from datetime import datetime, timezone
from .components.judge_client import JudgeClient
from .components.judge_runner import run_judge
from .components.judge_env import load_judge_env, env_value


def main():
    load_judge_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Reuse this directory to resume identical evaluation; failed calls remain recorded")
    parser.add_argument("--base-url", default=env_value("JUDGE_BASE_URL"), help="Chat-completions API base, including /v1 where required")
    parser.add_argument("--model", default=env_value("JUDGE_MODEL"), help="Independent evaluation model ID")
    parser.add_argument("--seed", type=int, default=12, help="Case ordering and random audit selection only")
    parser.add_argument("--audit-pairs", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=0, help="First N completed pairs in input order; 0 means all")
    args = parser.parse_args()
    if not args.model or not args.base_url or args.audit_pairs < 0 or args.num_samples < 0:
        parser.error("Fill JUDGE_BASE_URL and JUDGE_MODEL in the project root .env, or supply --base-url/--model; audit-pairs/num-samples must be non-negative")
    client = JudgeClient(args.base_url, args.model, os.getenv("JUDGE_API_KEY", ""))
    settings = {"model": args.model, "base_url": args.base_url, "temperature": 0.0,
                "seed": args.seed, "audit_pairs": args.audit_pairs, "timeout": 300,
                "num_samples": args.num_samples}
    destination = args.output_dir or args.run_dir / "judge" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = run_judge(args.run_dir, destination, settings, client)
    print(output)
    report = json.loads((output / "judge_summary.json").read_text(encoding="utf-8"))
    if report["completed_pairs"] < report["planned_pairs"]:
        raise SystemExit("Some judge calls failed or returned invalid/incomplete judgments; inspect judgments/*.json. No scores were invented.")


if __name__ == "__main__":
    main()
