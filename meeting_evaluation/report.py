"""Write concise timestamped metric reports without transcript or handoff content."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def write_reports(output_dir: Path, benchmark: str, metrics: dict[str, Any], configuration: dict[str, Any]) -> tuple[Path, Path]:
    # Keep results comparable and easy to browse: one minute-precise run
    # folder with stable benchmark names, e.g. output/20261009_1437/qmsum.txt.
    run_id = os.environ.get("MEETING_EVAL_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M")
    dated_output_dir = output_dir / run_id
    dated_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = dated_output_dir / f"{benchmark}.json"
    text_path = dated_output_dir / f"{benchmark}.txt"
    payload = {"benchmark": benchmark, "configuration": configuration, "metrics": metrics}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    quality = metrics["quality"]
    efficiency = metrics["efficiency"]
    lines = [
        "Meeting Latent-Communication Evaluation",
        f"Benchmark: {benchmark}",
        f"Samples: {metrics['samples']}",
        "",
        "Quality (mean)",
        "variant       ROUGE-L F1    source-support proxy",
    ]
    for variant in ("statebridge", "text", "no_comm"):
        rouge = quality[variant]["rouge_l_f1"]
        support = quality[variant]["supported_claim_rate"]
        rouge_text = "N/A" if rouge is None else f"{rouge:.4f}"
        lines.append(f"{variant:<13} {rouge_text:<13} {support:.4f}")
    lines += [
        "",
        "Efficiency",
        f"text handoff tokens: {efficiency['text_handoff_tokens']}",
        f"latent states:       {efficiency['latent_state_tokens']}",
        f"latent - text:       {efficiency['latent_minus_text_tokens']:+d}",
        "latent - text (%):   " + ("N/A" if efficiency["latent_minus_text_percent"] is None else f"{efficiency['latent_minus_text_percent']:+.1f}%"),
        f"mean KV prefill:     {efficiency['mean_prefix_prefill_seconds']:.3f}s",
        f"mean KV cache:       {efficiency['mean_prefix_cache_mib']:.1f} MiB",
        "",
        "Interpretation: a negative latent - text value means fewer token-equivalent communication units than the text handoff.",
        metrics["factuality_note"],
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path, json_path


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
    for variant in ("routed_kv", "full_kv", "selected_text"):
        result = metrics["quality"][variant]
        rouge = result["rouge_l_f1"]
        lines.append(f"{variant:<15} {'N/A' if rouge is None else f'{rouge:.4f}':<13} {result['supported_claim_rate']:.4f}")
    efficiency = metrics["efficiency"]
    lines += ["", "Execution and cache (mean)",
              f"full source prefill: {efficiency['mean_source_prefill_seconds']:.3f}s",
              f"KV selection probes: {efficiency['mean_selection_seconds']:.3f}s",
              f"full source KV cache: {efficiency['mean_full_source_cache_mib']:.1f} MiB",
              "", "variant | receiver positions | receiver KV MiB | logical payload bytes | receiver sec | pipeline sec | output tokens | cap hits"]
    for variant in ("routed_kv", "full_kv", "selected_text"):
        result = efficiency["variants"][variant]
        lines.append(
            f"{variant} | {result['mean_incoming_positions']:.1f} | {result['mean_incoming_cache_mib']:.1f} | "
            f"{result['mean_logical_payload_bytes']:.1f} | {result['mean_receiver_seconds']:.3f} | "
            f"{result['mean_pipeline_seconds_excluding_model_load']:.3f} | {result['mean_generated_tokens']:.1f} | {result['output_cap_hits']}"
        )
    lines += ["", metrics["efficiency_note"], metrics["factuality_note"]]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path, json_path
