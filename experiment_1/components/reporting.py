"""Aggregate paired observations, export CSV and prepare human review sheets."""
import csv
import math
import random
from statistics import mean, median

from meeting_evaluation.metrics import score_receiver_kv_results
from .config import VARIANTS
from .storage import atomic_json


def paired_interval(samples, repetitions=2000):
    differences = [s["quality"]["routed_kv"]["rouge_l_f1"] - s["quality"]["specialist_text"]["rouge_l_f1"]
                   for s in samples if all(s["quality"][v]["rouge_l_f1"] is not None for v in VARIANTS)]
    if not differences:
        return {"n": 0, "mean": None, "bootstrap_95_percent_interval": None}
    if len(differences) < 2:
        return {"n": 1, "mean": differences[0], "bootstrap_95_percent_interval": None}
    rng = random.Random(7)
    estimates = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(repetitions))
    return {"n": len(differences), "mean": mean(differences),
            "bootstrap_95_percent_interval": [estimates[int(.025 * repetitions)], estimates[int(.975 * repetitions) - 1]],
            "resampling_unit": "input record; not independent meetings if several records share a meeting"}


def _mean(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def write_reports(store, completed, failures, planned):
    result = {"planned_samples": planned, "completed_samples": len(completed), "failed_samples": len(failures),
              "failure_ids": [s["id"] for s in failures], "review": False,
              "comparison_note": "Same selected source positions; text adds a generated handoff. This is a workflow comparison, not a representation-only intervention.",
              "memory_note": "CUDA peaks are PyTorch allocated bytes during each variant, including model, embedding model and full resident source KV. They exclude shared prefill/selection peaks, reserved allocator memory and non-PyTorch allocations.",
              "payload_note": "KV tensor bytes and UTF-8 text handoff bytes are separately defined logical payloads, not measured network traffic.",
              "quality_note": "ROUGE-L and lexical source overlap do not measure factual accuracy; use the human review sheet with source evidence."}
    if completed:
        results = [s["result"] for s in completed]
        source_by_id = {s["id"]: s["source"] for s in completed}
        result["metrics"] = score_receiver_kv_results(results, source_by_id)
        result["paired_rouge_l_difference_kv_minus_text"] = paired_interval(completed)
        extra = {}
        for name in VARIANTS:
            measurements = [s["result"]["variants"][name] for s in completed]
            seconds = sorted(m["pipeline_seconds_excluding_model_load"] for m in measurements)
            extra[name] = {
                "median_pipeline_seconds": median(seconds),
                "p95_pipeline_seconds": seconds[max(0, math.ceil(.95 * len(seconds)) - 1)],
                "mean_cuda_peak_allocated_bytes": _mean([m.get("cuda_peak_allocated_bytes") for m in measurements]),
                "mean_cuda_peak_extra_allocated_bytes": _mean([m.get("cuda_peak_extra_allocated_bytes") for m in measurements]),
                "mean_pipeline_peak_allocated_bytes": _mean([m.get("pipeline_peak_allocated_bytes") for m in measurements]),
                "handoff_cap_hits": sum(bool(m.get("specialist_handoff_cap_hit")) for m in measurements),
                "receiver_cap_hits": sum(bool(m["output_cap_hit"]) for m in measurements),
            }
        result["additional_resources"] = extra
    atomic_json(store.directory / "summary.json", result)
    lines = ["Experiment 1: paired draft-summary comparison (review disabled)",
             f"Planned: {planned}; completed: {len(completed)}; failed: {len(failures)}",
             "variant | ROUGE-L | source-support proxy | mean pipeline sec | mean CUDA peak MiB | handoff/receiver cap hits"]
    for name in VARIANTS:
        if not completed:
            continue
        metric = result["metrics"]
        q = metric["quality"][name]
        eff = metric["efficiency"]["variants"][name]
        extra = result["additional_resources"][name]
        peak = extra["mean_cuda_peak_allocated_bytes"]
        rouge = "N/A" if q["rouge_l_f1"] is None else f"{q['rouge_l_f1']:.4f}"
        peak_text = "N/A" if peak is None else f"{peak / 2**20:.1f}"
        lines.append(f"{name} | {rouge} | {q['supported_claim_rate']:.4f} | "
                     f"{eff['mean_pipeline_seconds_excluding_model_load']:.3f} | {peak_text} | "
                     f"{extra['handoff_cap_hits']}/{extra['receiver_cap_hits']}")
    lines.extend(["", result["comparison_note"], result["memory_note"], result["payload_note"], result["quality_note"]])
    (store.directory / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fields = ["id", "variant", "rouge_l_f1", "supported_claim_rate", "source_positions", "selected_positions",
              "actual_retained_fraction", "pipeline_seconds", "generated_tokens", "handoff_tokens",
              "handoff_cap_hit", "receiver_cap_hit", "cuda_peak_allocated_bytes", "logical_payload_bytes"]
    with (store.directory / "per_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in completed:
            r = sample["result"]
            for name in VARIANTS:
                m = r["variants"][name]
                writer.writerow({"id": sample["id"], "variant": name, **{k: sample["quality"][name][k] for k in ("rouge_l_f1", "supported_claim_rate")},
                    "source_positions": r["source_positions"], "selected_positions": len(r["selected_source_positions"]),
                    "actual_retained_fraction": len(r["selected_source_positions"]) / r["source_positions"],
                    "pipeline_seconds": m["pipeline_seconds_excluding_model_load"], "generated_tokens": m["generated_tokens"],
                    "handoff_tokens": m.get("specialist_handoff_tokens", 0), "handoff_cap_hit": m.get("specialist_handoff_cap_hit", False),
                    "receiver_cap_hit": m["output_cap_hit"], "cuda_peak_allocated_bytes": m.get("cuda_peak_allocated_bytes"),
                    "logical_payload_bytes": m["logical_payload_bytes"]})
    # Preserve filled annotations and append rows for newly completed samples.
    annotations = store.directory / "human_review.csv"
    fields = ["id", "variant", "decision_precision", "decision_recall", "action_precision", "action_recall",
              "owner_accuracy", "date_accuracy", "open_issue_coverage", "redundancy_notes", "evidence_notes"]
    existing = set()
    if annotations.exists():
        with annotations.open(encoding="utf-8-sig", newline="") as handle:
            existing = {(row["id"], row["variant"]) for row in csv.DictReader(handle)}
    new_file = not annotations.exists()
    with annotations.open("a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for sample in completed:
            for name in VARIANTS:
                if (sample["id"], name) not in existing:
                    writer.writerow({"id": sample["id"], "variant": name})
    return result
