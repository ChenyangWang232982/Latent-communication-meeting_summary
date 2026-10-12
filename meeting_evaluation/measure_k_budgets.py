"""Measure how specialist handoff lengths fit fixed StateBridge K budgets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median

from .records import load_meetingbank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="MeetingBank JSON or JSONL input.")
    parser.add_argument("--output-dir", type=Path, default=Path("meeting_evaluation/output"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--budgets", nargs="+", type=int, default=[64, 96, 128])
    parser.add_argument("--coverage-target", type=float, default=0.90)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--source-max-tokens", type=int, default=8192)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=1024)
    parser.add_argument("--backfill-rounds", type=int, default=0)
    parser.add_argument("--agent-max-new-tokens", type=int, default=256)
    parser.add_argument("--device")
    return parser.parse_args()


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def suggested_budget(lengths: list[int], budgets: list[int], coverage_target: float) -> str:
    for budget in budgets:
        if sum(length <= budget for length in lengths) / len(lengths) >= coverage_target:
            return str(budget)
    return f">{budgets[-1]}"


def suggested_budget_for_length(length: int, budgets: list[int]) -> str:
    for budget in budgets:
        if length <= budget:
            return str(budget)
    return f">{budgets[-1]}"


def run_directory(output_dir: Path) -> Path:
    run_id = os.environ.get("MEETING_EVAL_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M")
    directory = output_dir / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def main() -> None:
    args = parse_args()
    budgets = sorted(set(args.budgets))
    if not budgets or budgets[0] < 2:
        raise ValueError("--budgets must contain positive K values of at least 2.")
    if not 0 < args.coverage_target <= 1:
        raise ValueError("--coverage-target must be in (0, 1].")
    if args.agent_max_new_tokens <= budgets[-1]:
        raise ValueError(
            "--agent-max-new-tokens must exceed the largest K, otherwise this probe cannot observe longer handoffs."
        )

    records = load_meetingbank(args.input)
    if args.num_samples and len(records) < args.num_samples:
        raise ValueError(f"Requested {args.num_samples} records but input contains only {len(records)}.")
    selected = records[: args.num_samples] if args.num_samples else records
    if not selected:
        raise ValueError("No MeetingBank records selected.")

    with tempfile.TemporaryDirectory(prefix="meeting_k_budget_") as temp_dir:
        temp = Path(temp_dir)
        normalized_input = temp / "records.jsonl"
        raw_output = temp / "runner.txt"
        with normalized_input.open("w", encoding="utf-8") as handle:
            for record in selected:
                handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

        command = [
            sys.executable,
            "-m",
            "prophetkv_statebridge.run_prefix_cache",
            "--meeting-data",
            str(normalized_input),
            "--output",
            str(raw_output),
            "--model",
            args.model,
            "--num-samples",
            "0",
            "--source-max-tokens",
            str(args.source_max_tokens),
            "--prefill-chunk-tokens",
            str(args.prefill_chunk_tokens),
            "--backfill-rounds",
            str(args.backfill_rounds),
            "--agent-max-new-tokens",
            str(args.agent_max_new_tokens),
            "--prefix-tokens-per-agent",
            str(budgets[-1]),
            "--communication-metrics-only",
        ]
        if args.device:
            command.extend(["--device", args.device])
        subprocess.run(command, check=True)
        results = [
            json.loads(line)
            for line in raw_output.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    rows = []
    by_role: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        for specialist in result["specialist_token_metrics"]:
            handoff_tokens = specialist["handoff_tokens"]
            row = {
                "sample_id": result["id"],
                "role": specialist["role"],
                "handoff_tokens": handoff_tokens,
                "generation_tokens_total": specialist["generation_tokens_total"],
                "at_generation_cap": handoff_tokens >= args.agent_max_new_tokens,
                "suggested_k": suggested_budget_for_length(handoff_tokens, budgets),
                "retained_states": {str(budget): min(handoff_tokens, budget) for budget in budgets},
                "truncated": {str(budget): handoff_tokens > budget for budget in budgets},
            }
            rows.append(row)
            by_role[row["role"]].append(row)

    role_summary = {}
    for role, role_rows in sorted(by_role.items()):
        lengths = [row["handoff_tokens"] for row in role_rows]
        role_summary[role] = {
            "experts": len(role_rows),
            "mean_handoff_tokens": mean(lengths),
            "median_handoff_tokens": median(lengths),
            "p90_handoff_tokens": percentile(lengths, 0.90),
            "generation_cap_rate": sum(row["at_generation_cap"] for row in role_rows) / len(role_rows),
            "suggested_k_at_coverage_target": suggested_budget(lengths, budgets, args.coverage_target),
            "budgets": {
                str(budget): {
                    "mean_retained_states": mean(row["retained_states"][str(budget)] for row in role_rows),
                    "truncation_rate": sum(row["truncated"][str(budget)] for row in role_rows) / len(role_rows),
                }
                for budget in budgets
            },
        }

    overall_lengths = [row["handoff_tokens"] for row in rows]
    payload = {
        "benchmark": "meetingbank_specialist_k_budget",
        "samples": len(results),
        "configuration": {
            "model": args.model,
            "source_max_tokens": args.source_max_tokens,
            "prefill_chunk_tokens": args.prefill_chunk_tokens,
            "backfill_rounds": args.backfill_rounds,
            "agent_max_new_tokens": args.agent_max_new_tokens,
            "budgets": budgets,
            "coverage_target": args.coverage_target,
        },
        "overall": {
            "experts": len(rows),
            "mean_handoff_tokens": mean(overall_lengths),
            "median_handoff_tokens": median(overall_lengths),
            "p90_handoff_tokens": percentile(overall_lengths, 0.90),
            "generation_cap_rate": sum(row["at_generation_cap"] for row in rows) / len(rows),
        },
        "by_role": role_summary,
        "per_expert": rows,
        "note": (
            "This is a handoff-length probe, not a quality evaluation. A larger K is only justified by a later "
            "quality comparison; truncation rate alone does not prove that additional states improve summaries."
        ),
    }

    directory = run_directory(args.output_dir)
    text_path = directory / "meetingbank_specialist_k_budget.txt"
    json_path = directory / "meetingbank_specialist_k_budget.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    headers = ["role", "experts", "mean", "median", "p90", "cap-hit", "suggested K"]
    headers.extend(f"K={budget} retained/truncated" for budget in budgets)
    lines = [
        "MeetingBank Specialist Handoff Length vs K",
        f"Samples: {len(results)}",
        f"Agent generation cap: {args.agent_max_new_tokens} tokens",
        f"Coverage target for suggested K: {args.coverage_target:.0%}",
        "",
        " | ".join(headers),
    ]
    for role, summary in role_summary.items():
        values = [
            role,
            str(summary["experts"]),
            f"{summary['mean_handoff_tokens']:.1f}",
            f"{summary['median_handoff_tokens']:.1f}",
            f"{summary['p90_handoff_tokens']:.0f}",
            f"{summary['generation_cap_rate']:.1%}",
            summary["suggested_k_at_coverage_target"],
        ]
        for budget in budgets:
            budget_metrics = summary["budgets"][str(budget)]
            values.append(f"{budget_metrics['mean_retained_states']:.1f}/{budget_metrics['truncation_rate']:.1%}")
        lines.append(" | ".join(values))

    lines += [
        "",
        "Per-expert token records",
        "sample_id | role | handoff tokens | total generated | cap-hit | suggested K | "
        + " | ".join(f"K={budget}" for budget in budgets),
    ]
    for row in rows:
        retained = [
            f"{row['retained_states'][str(budget)]}{'*' if row['truncated'][str(budget)] else ''}"
            for budget in budgets
        ]
        lines.append(
            f"{row['sample_id']} | {row['role']} | {row['handoff_tokens']} | "
            f"{row['generation_tokens_total']} | {str(row['at_generation_cap']).lower()} | {row['suggested_k']} | "
            + " | ".join(retained)
        )
    lines += [
        "",
        "Legend: K column is retained states; * means the handoff was longer than that K and was truncated.",
        payload["note"],
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"K-budget report: {text_path}")
    print(f"Machine-readable K-budget data: {json_path}")


if __name__ == "__main__":
    main()
