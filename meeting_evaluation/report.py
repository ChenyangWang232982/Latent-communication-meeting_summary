"""Write concise timestamped metric reports without transcript or handoff content."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_reports(output_dir: Path, benchmark: str, metrics: dict[str, Any], configuration: dict[str, Any]) -> tuple[Path, Path]:
    # Keep results comparable and easy to browse: one dated run folder with
    # stable benchmark names, e.g. output/20261009/qmsum.txt.
    dated_output_dir = output_dir / datetime.now().strftime("%Y%m%d")
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
