"""Fixed-bandwidth MeetingBank ablation for StateBridge handoff strategies."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from .config import WorkflowConfig
from .metrics import score_results
from .records import load_meetingbank
from .workflow import run_statebridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="MeetingBank JSON or JSONL input.")
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
    return parser.parse_args()


def run_directory(output_dir: Path) -> Path:
    run_id = os.environ.get("MEETING_EVAL_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M")
    directory = output_dir / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def value(metrics: dict, strategy: str, key: str) -> float | None:
    return metrics[strategy]["quality"]["statebridge"][key]


def write_comparison(output_dir: Path, configurations: dict[str, WorkflowConfig], metrics: dict[str, dict]) -> tuple[Path, Path]:
    directory = run_directory(output_dir)
    text_path = directory / "meetingbank_state_selection.txt"
    json_path = directory / "meetingbank_state_selection.json"
    tail = metrics["tail"]
    tail_rouge = value(metrics, "tail", "rouge_l_f1")
    stratified_rouge = value(metrics, "stratified", "rouge_l_f1")
    compact_rouge = value(metrics, "dynamic_compact", "rouge_l_f1")
    rouge_delta = None if tail_rouge is None or stratified_rouge is None else stratified_rouge - tail_rouge
    compact_delta = None if tail_rouge is None or compact_rouge is None else compact_rouge - tail_rouge
    payload = {
        "benchmark": "meetingbank_state_selection",
        "configurations": {name: asdict(config) for name, config in configurations.items()},
        "metrics": metrics,
        "stratified_minus_tail_statebridge_rouge_l_f1": rouge_delta,
        "dynamic_compact_minus_tail_statebridge_rouge_l_f1": compact_delta,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def metric_row(strategy: str) -> str:
        result = metrics[strategy]
        quality = result["quality"]
        efficiency = result["efficiency"]
        rouge = quality["statebridge"]["rouge_l_f1"]
        text_rouge = quality["text"]["rouge_l_f1"]
        support = quality["statebridge"]["supported_claim_rate"]
        marker_success = result.get("state_selection", {}).get("dynamic_marker_success_rate")
        return (
            f"{strategy:<17} "
            f"{('N/A' if rouge is None else f'{rouge:.4f}'):<13} "
            f"{('N/A' if text_rouge is None else f'{text_rouge:.4f}'):<11} "
            f"{support:.4f}     "
            f"{efficiency['latent_state_tokens']:<13} "
            f"{efficiency['latent_minus_text_percent']:+.1f}%       "
            f"{'N/A' if marker_success is None else f'{marker_success:.1%}'}"
        )

    lines = [
        "MeetingBank State-Selection Ablation",
        f"Samples: {tail['samples']}",
        f"Fixed state budget per specialist: {configurations['tail'].prefix_tokens_per_agent}",
        "",
        "Results",
        "strategy          latent ROUGE  text ROUGE  source-support  latent states  latent/text  marker success",
        metric_row("tail"),
        metric_row("stratified"),
        metric_row("dynamic_compact"),
        "",
        "Comparison",
        "stratified - tail ROUGE-L: " + ("N/A" if rouge_delta is None else f"{rouge_delta:+.4f}"),
        "dynamic_compact - tail ROUGE-L: " + ("N/A" if compact_delta is None else f"{compact_delta:+.4f}"),
        "",
        "Controls",
        "Each strategy is run with the same MeetingBank records and generates text/no_comm controls independently.",
        "Source-support is a lexical proxy, not an LLM or human MESA score.",
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path, json_path


def main() -> None:
    args = parse_args()
    records = load_meetingbank(args.input)
    if args.num_samples and len(records) < args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} MeetingBank samples, but {args.input} contains only {len(records)}. "
            "Export a larger input file before running this ablation."
        )
    selected = records[: args.num_samples] if args.num_samples else records
    if not selected:
        raise ValueError("No MeetingBank records selected.")
    source_by_id = {record.id: record.context for record in selected}
    base = WorkflowConfig(
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
    configurations = {
        strategy: replace(base, state_selection_strategy=strategy)
        for strategy in ("tail", "stratified", "dynamic_compact")
    }
    metrics = {}
    for strategy, config in configurations.items():
        print(f"\n=== MeetingBank state selection: {strategy} ===", flush=True)
        results = run_statebridge(selected, config, meeting_mode=True)
        metrics[strategy] = score_results(results, source_by_id)
    text_path, json_path = write_comparison(args.output_dir, configurations, metrics)
    print(f"Comparison report: {text_path}")
    print(f"Machine-readable comparison: {json_path}")


if __name__ == "__main__":
    main()
