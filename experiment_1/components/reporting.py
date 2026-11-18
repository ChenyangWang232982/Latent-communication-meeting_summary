"""Aggregate paired observations, export CSV and prepare human review sheets."""
import csv
import math
import random
from statistics import mean, median

from meeting_evaluation.metrics import score_receiver_kv_results
from .config import VARIANTS
from .storage import atomic_json
from .repetition import repetition_metrics
from .token_accounting import token_totals, FIELDS
from meeting_evaluation.metrics import reference_metrics, source_overlap_support


def paired_interval(samples, repetitions=2000, metric="rouge_l_f1"):
    differences = [s["quality"]["routed_kv"][metric] - s["quality"]["specialist_text"][metric]
                   for s in samples if all(s["quality"][v].get(metric) is not None for v in VARIANTS)]
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


def _format(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def write_reports(store, completed, failures, planned):
    chunk_mode = any(s['result'].get('chunk_specialists_by_variant') for s in completed)
    result = {"planned_samples": planned, "completed_samples": len(completed), "failed_samples": len(failures),
              "failure_ids": [s["id"] for s in failures], "review": False,
              "comparison_note": "Same selected source positions; text adds a generated handoff. This is a workflow comparison, not a representation-only intervention.",
              "memory_note": "CUDA peaks are PyTorch allocated bytes during each variant, including model, embedding model and full resident source KV. They exclude shared prefill/selection peaks, reserved allocator memory and non-PyTorch allocations.",
              "payload_note": "KV tensor bytes and UTF-8 text handoff bytes are separately defined logical payloads, not measured network traffic.",
              "quality_note": "ROUGE-L and lexical source overlap do not measure factual accuracy; use the human review sheet with source evidence."}
    result['workflow'] = 'chunk_specialists' if chunk_mode else 'direct_summary'
    if chunk_mode:
        result['comparison_note'] = ('Same globally selected source positions, full-transcript chronological windows, stage instructions and decoding settings. '
            'Every specialist receives global selected context as KV or verbatim selected text, plus the same current-part original text and location. '
            'Final synthesis receives global selected context and ordered local reports. '
            'Both paths generate local natural-language reports. Full-source contextualized KV and re-prefilled text differ '
            'in positional history and available prior context; this is a workflow comparison, not a pure representation intervention.')
        result['payload_note'] = ('Logical payload sums all stage KV tensor bytes or selected evidence UTF-8 bytes, current-part UTF-8 bytes and report UTF-8 bytes. '
            'These quantities are separately defined; they are not measured network traffic.')
    if completed:
        for sample in completed:
            for name in VARIANTS:
                sample["quality"][name].update(reference_metrics(sample["result"]["answers"][name], sample["reference"]))
                sample["quality"][name].update(source_overlap_support(sample["result"]["answers"][name], sample["source"]))
        results = [s["result"] for s in completed]
        source_by_id = {s["id"]: s["source"] for s in completed}
        result["metrics"] = score_receiver_kv_results(results, source_by_id)
        result["paired_rouge_l_difference_kv_minus_text"] = paired_interval(completed)
        keys = sorted(set.intersection(*(set(s["quality"][v]) for s in completed for v in VARIANTS)))
        result["metrics"]["quality"] = {v: {key: _mean([s["quality"][v][key] for s in completed]) for key in keys} for v in VARIANTS}
        result["paired_quality_differences"] = {key: paired_interval(completed, metric=key)
            for key in keys if key.endswith("_f1")}
        token_rows = [{"id": s["id"], "variant": v, **token_totals(s["result"]["variants"][v])}
                      for s in completed for v in VARIANTS]
        result["tokens"] = {v: {key: {"mean": mean(row[key] for row in token_rows if row["variant"] == v),
                                    "sum": sum(row[key] for row in token_rows if row["variant"] == v)}
                                  for key in (*FIELDS, "total_input_tokens", "total_output_tokens", "total_tokens")}
                            for v in VARIANTS}
        baseline = result["tokens"]["specialist_text"]["total_tokens"]["sum"]
        result["token_savings_kv_vs_text_percent"] = 100 * (1 - result["tokens"]["routed_kv"]["total_tokens"]["sum"] / baseline) if baseline else None
        atomic_json(store.directory / "tokens.json", {"variants": result["tokens"],
                    "savings_percent": result["token_savings_kv_vs_text_percent"], "per_sample": token_rows})
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
        repetition = [{"id": sample["id"], "variant": name,
                       "receiver": repetition_metrics(sample["result"].get("answers", {}).get(name, "")),
                       "sender": repetition_metrics(sample["result"].get("handoffs", {}).get(name, ""))}
                      for sample in completed for name in VARIANTS]
        atomic_json(store.directory / "repetition.json", repetition)
        result["repetition"] = {name: {
            "receiver_samples_with_duplicate_bullets": sum(row["receiver"]["duplicate_bullet_excess"] > 0 for row in repetition if row["variant"] == name),
            "sender_samples_with_duplicate_bullets": sum(row["sender"]["duplicate_bullet_excess"] > 0 for row in repetition if row["variant"] == name)} for name in VARIANTS}
    atomic_json(store.directory / "summary.json", result)
    lines = ["Experiment 1: paired chunk-specialist comparison (review disabled)" if chunk_mode else "Experiment 1: paired draft-summary comparison (review disabled)",
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
    if completed:
        lines.extend(["", "Reference overlap (unstemmed ASCII tokens; macro means):",
                      "variant | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L precision | ROUGE-L recall | words/reference ratio"])
        for name in VARIANTS:
            q = result["metrics"]["quality"][name]
            lines.append(name + " | " + " | ".join(_format(q[key]) for key in ("rouge_1_f1", "rouge_2_f1", "rouge_l_precision", "rouge_l_recall", "word_length_ratio")))
            if "bertscore_f1" in q:
                lines.append(f"  BERTScore P/R/F1: {q['bertscore_precision']:.4f} / {q['bertscore_recall']:.4f} / {q['bertscore_f1']:.4f}")
        lines.extend(["", "Logical decoder token events per sample (mean):",
                      "variant | source prefill | probes | sender prompt | sender output | receiver prompt | receiver output | total input | total output | total"])
        for name in VARIANTS:
            row = result["tokens"][name]
            lines.append(name + " | " + " | ".join(f"{row[key]['mean']:.1f}" for key in (*FIELDS, "total_input_tokens", "total_output_tokens", "total_tokens")))
        saving = result["token_savings_kv_vs_text_percent"]
        lines.append(f"Logical decoder token savings KV vs text: {saving:.2f}%" if saving is not None else "Token savings: N/A")
        lines.append(result["metrics"]["efficiency_note"])
        lines.append("Shared prefill/probes are counted once per standalone workflow; paired-run sums are not actual total GPU work. Embedding encoder tokens are separate.")
        lines.append("\nExact duplicate bullet diagnostics (not semantic or factual evaluation):")
        for name, counts in result["repetition"].items():
            lines.append(f"{name}: receiver samples={counts['receiver_samples_with_duplicate_bullets']}; sender samples={counts['sender_samples_with_duplicate_bullets']}")
    lines.extend(["", result["comparison_note"], result["memory_note"], result["payload_note"], result["quality_note"]])
    (store.directory / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fields = [*FIELDS, "total_input_tokens", "total_output_tokens", "total_tokens",
              "rouge_1_f1", "rouge_2_f1", "rouge_l_precision", "rouge_l_recall", "prediction_words", "reference_words", "word_length_ratio",
              "bertscore_precision", "bertscore_recall", "bertscore_f1",
              "id", "variant", "rouge_l_f1", "supported_claim_rate", "source_positions", "selected_positions",
              "actual_retained_fraction", "pipeline_seconds", "generated_tokens", "handoff_tokens",
              "handoff_cap_hit", "receiver_cap_hit", "cuda_peak_allocated_bytes", "logical_payload_bytes"]
    with (store.directory / "per_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in completed:
            r = sample["result"]
            for name in VARIANTS:
                m = r["variants"][name]
                writer.writerow({**token_totals(m),
                    **{key: sample["quality"][name][key] for key in sample["quality"][name] if key in fields},
                    "id": sample["id"], "variant": name, **{k: sample["quality"][name][k] for k in ("rouge_l_f1", "supported_claim_rate")},
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
