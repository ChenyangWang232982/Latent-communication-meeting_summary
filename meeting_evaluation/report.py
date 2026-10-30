"""Write concise timestamped metric reports without transcript or handoff content."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def write_receiver_kv_reports(output_dir: Path, benchmark: str, metrics: dict[str, Any], configuration: dict[str, Any]) -> tuple[Path, Path]:
    """Write receiver-KV metrics without exposing generated meeting summaries."""
    run_id = os.environ.get("MEETING_EVAL_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M")
    directory = output_dir / run_id
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{benchmark}_receiver_kv.json"
    text_path = directory / f"{benchmark}_receiver_kv.txt"
    json_path.write_text(json.dumps({"benchmark": benchmark, "workflow": "receiver_kv", "configuration": configuration, "metrics": metrics}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "Meeting Receiver-Conditioned KV Evaluation", f"Benchmark: {benchmark}",
        f"Samples: {metrics['samples']}", "", "Quality (mean)",
        "variant         ROUGE-L F1    source-support proxy",
    ]
    for variant in ("routed_kv", "specialist_text"):
        result = metrics["quality"][variant]
        rouge = result["rouge_l_f1"]
        lines.append(f"{variant:<15} {'N/A' if rouge is None else f'{rouge:.4f}':<13} {result['supported_claim_rate']:.4f}")
    efficiency = metrics["efficiency"]
    lines += ["", "Execution and cache (mean)",
              f"full source prefill: {efficiency['mean_source_prefill_seconds']:.3f}s",
              f"KV selection probes: {efficiency['mean_selection_seconds']:.3f}s",
              f"full source KV cache: {efficiency['mean_full_source_cache_mib']:.1f} MiB",
              "", "variant | receiver positions | receiver KV MiB | generated handoff tokens | logical payload bytes | specialist sec | receiver sec | pipeline sec | output tokens | cap hits"]
    for variant in ("routed_kv", "specialist_text"):
        result = efficiency["variants"][variant]
        lines.append(
            f"{variant} | {result['mean_incoming_positions']:.1f} | {result['mean_incoming_cache_mib']:.1f} | "
            f"{result['mean_specialist_handoff_tokens']:.1f} | {result['mean_logical_payload_bytes']:.1f} | "
            f"{result['mean_specialist_seconds']:.3f} | {result['mean_receiver_seconds']:.3f} | "
            f"{result['mean_pipeline_seconds_excluding_model_load']:.3f} | {result['mean_generated_tokens']:.1f} | {result['output_cap_hits']}"
        )
    lines += ["", "Logical model-token events per sample (mean)",
              "variant | source prefill | selection probes | specialist prompts | specialist handoffs | receiver prompt | receiver output | total input | total output | total"]
    for variant in ("routed_kv", "specialist_text"):
        result = efficiency["variants"][variant]
        lines.append(
            f"{variant} | {result['mean_source_prefill_tokens']:.1f} | {result['mean_selection_probe_tokens']:.1f} | "
            f"{result['mean_specialist_prompt_tokens']:.1f} | {result['mean_specialist_handoff_tokens']:.1f} | "
            f"{result['mean_receiver_prompt_tokens']:.1f} | {result['mean_receiver_output_tokens']:.1f} | "
            f"{result['mean_end_to_end_input_tokens']:.1f} | {result['mean_end_to_end_output_tokens']:.1f} | "
            f"{result['mean_end_to_end_total_tokens']:.1f}"
        )
    lines += ["", metrics["efficiency_note"], metrics["factuality_note"]]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path, json_path


def write_receiver_kv_communication_reports(
    output_dir: Path, benchmark: str, metrics: dict[str, Any], configuration: dict[str, Any],
) -> tuple[Path, Path]:
    """Write token/cost-only measurements without saving or scoring summaries."""
    run_id = os.environ.get("MEETING_EVAL_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M")
    directory = output_dir / run_id
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{benchmark}_communication.json"
    text_path = directory / f"{benchmark}_communication.txt"
    payload = {"benchmark": benchmark, "workflow": "receiver_kv", "mode": "communication_metrics_only",
               "configuration": configuration, "metrics": metrics}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    efficiency = metrics["efficiency"]
    variants = ("routed_kv", "specialist_text")
    lines = [
        "Meeting Communication Token Evaluation", f"Benchmark: {benchmark}",
        f"Samples: {metrics['samples']}",
        "Summary text is neither saved nor quality-scored in this mode.", "",
        "Execution and cache (mean)",
        f"full source prefill: {efficiency['mean_source_prefill_seconds']:.3f}s",
        f"KV selection probes: {efficiency['mean_selection_seconds']:.3f}s",
        f"full source KV cache: {efficiency['mean_full_source_cache_mib']:.1f} MiB", "",
        "variant | receiver positions | receiver KV MiB | generated handoff tokens | logical payload bytes | pipeline sec | output tokens | cap hits",
    ]
    for variant in variants:
        result = efficiency["variants"][variant]
        lines.append(
            f"{variant} | {result['mean_incoming_positions']:.1f} | {result['mean_incoming_cache_mib']:.1f} | "
            f"{result['mean_specialist_handoff_tokens']:.1f} | {result['mean_logical_payload_bytes']:.1f} | "
            f"{result['mean_pipeline_seconds_excluding_model_load']:.3f} | {result['mean_generated_tokens']:.1f} | {result['output_cap_hits']}"
        )
    lines += ["", "Logical model-token events per sample (mean)",
              "variant | source prefill | selection probes | sender prompt | text handoff | receiver prompt | receiver output | total input | total output | total"]
    for variant in variants:
        result = efficiency["variants"][variant]
        lines.append(
            f"{variant} | {result['mean_source_prefill_tokens']:.1f} | {result['mean_selection_probe_tokens']:.1f} | "
            f"{result['mean_specialist_prompt_tokens']:.1f} | {result['mean_specialist_handoff_tokens']:.1f} | "
            f"{result['mean_receiver_prompt_tokens']:.1f} | {result['mean_receiver_output_tokens']:.1f} | "
            f"{result['mean_end_to_end_input_tokens']:.1f} | {result['mean_end_to_end_output_tokens']:.1f} | "
            f"{result['mean_end_to_end_total_tokens']:.1f}"
        )
    lines += ["", efficiency_note := metrics["efficiency_note"],
              "Receiver output is generated only to count actual output-token events, then discarded. "
              "No summary content, ROUGE, or factuality metric is written in this mode."]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path, json_path
