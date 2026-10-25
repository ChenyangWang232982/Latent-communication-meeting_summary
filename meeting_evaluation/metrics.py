"""Reference, factuality-proxy, and efficiency metrics for one evaluation run."""

from __future__ import annotations

import re
from statistics import mean
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def rouge_l_f1(prediction: str, reference: str) -> float | None:
    """Dependency-free ROUGE-L F1; returns None when no reference exists."""
    if not reference.strip():
        return None
    left, right = _tokens(prediction), _tokens(reference)
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, target in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if token == target else max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(left), lcs / len(right)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def source_overlap_support(summary: str, source: str, minimum_overlap: float = 0.45) -> dict[str, float]:
    """A transparent proxy only, not a reproduction of MESA's LLM evaluator."""
    source_terms = set(_tokens(source))
    claims = [sentence.strip() for sentence in SENTENCE_RE.split(summary) if len(_tokens(sentence)) >= 4]
    if not claims:
        return {"claim_count": 0, "supported_claim_rate": 0.0}
    supported = 0
    for claim in claims:
        terms = set(_tokens(claim))
        if terms and len(terms & source_terms) / len(terms) >= minimum_overlap:
            supported += 1
    return {"claim_count": len(claims), "supported_claim_rate": supported / len(claims)}


def _average(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return mean(valid) if valid else None


def score_receiver_kv_results(results: list[dict[str, Any]], source_by_id: dict[str, str]) -> dict[str, Any]:
    """Score the receiver-KV controls without pretending KV bytes are tokens."""
    variants = ("routed_kv", "full_kv", "selected_text", "specialist_text")
    quality = {name: {"rouge_l_f1": [], "supported_claim_rate": []} for name in variants}
    source_prefill, selection, source_cache = [], [], []
    efficiency = {
        name: {"incoming_positions": [], "incoming_cache_mib": [], "logical_payload_bytes": [],
               "specialist_handoff_tokens": [], "specialist_seconds": [], "receiver_seconds": [],
               "pipeline_seconds": [], "generated_tokens": [], "output_cap_hits": 0,
               "source_prefill_tokens": [], "selection_probe_tokens": [], "specialist_prompt_tokens": [],
               "receiver_prompt_tokens": [], "receiver_output_tokens": []}
        for name in variants
    }
    for result in results:
        source = source_by_id[result["id"]]
        source_prefill.append(result["prefill_seconds"])
        selection.append(result["selection_seconds"])
        source_cache.append(result["source_cache_bytes"] / 2**20)
        references = result.get("gold_answers") or [result.get("gold_answer", "")]
        for name in variants:
            if name not in result["answers"]:
                continue
            answer = result["answers"][name]
            scores = [rouge_l_f1(answer, str(reference)) for reference in references]
            valid = [score for score in scores if score is not None]
            quality[name]["rouge_l_f1"].append(max(valid) if valid else None)
            quality[name]["supported_claim_rate"].append(
                source_overlap_support(answer, source)["supported_claim_rate"]
            )
            measurement = result["variants"][name]
            data = efficiency[name]
            data["incoming_positions"].append(measurement["incoming_positions"])
            data["incoming_cache_mib"].append(measurement["incoming_cache_bytes"] / 2**20)
            data["logical_payload_bytes"].append(measurement["logical_payload_bytes"])
            data["specialist_handoff_tokens"].append(measurement.get("specialist_handoff_tokens", 0))
            data["specialist_seconds"].append(measurement.get("specialist_seconds", 0.0))
            data["receiver_seconds"].append(measurement["receiver_seconds"])
            data["pipeline_seconds"].append(measurement["pipeline_seconds_excluding_model_load"])
            data["generated_tokens"].append(measurement["generated_tokens"])
            data["output_cap_hits"] += int(measurement["output_cap_hit"])
            accounting = measurement.get("token_accounting", {})
            for field in ("source_prefill_tokens", "selection_probe_tokens", "specialist_prompt_tokens",
                          "receiver_prompt_tokens", "receiver_output_tokens"):
                data[field].append(accounting.get(field, 0))
    return {
        "samples": len(results),
        "quality": {name: {metric: _average(values) for metric, values in metrics.items()}
                    for name, metrics in quality.items()},
        "factuality_note": "supported_claim_rate is a lexical source-overlap proxy inspired by MESA-style source checking; it is not an LLM or human MESA score.",
        "efficiency": {
            "mean_source_prefill_seconds": _average(source_prefill),
            "mean_selection_seconds": _average(selection),
            "mean_full_source_cache_mib": _average(source_cache),
            "variants": {
                name: {
                    "mean_incoming_positions": _average(values["incoming_positions"]),
                    "mean_incoming_cache_mib": _average(values["incoming_cache_mib"]),
                    "mean_logical_payload_bytes": _average(values["logical_payload_bytes"]),
                    "mean_specialist_handoff_tokens": _average(values["specialist_handoff_tokens"]),
                    "mean_specialist_seconds": _average(values["specialist_seconds"]),
                    "mean_receiver_seconds": _average(values["receiver_seconds"]),
                    "mean_pipeline_seconds_excluding_model_load": _average(values["pipeline_seconds"]),
                    "mean_generated_tokens": _average(values["generated_tokens"]),
                    "output_cap_hits": values["output_cap_hits"],
                    "mean_source_prefill_tokens": _average(values["source_prefill_tokens"]),
                    "mean_selection_probe_tokens": _average(values["selection_probe_tokens"]),
                    "mean_specialist_prompt_tokens": _average(values["specialist_prompt_tokens"]),
                    "mean_receiver_prompt_tokens": _average(values["receiver_prompt_tokens"]),
                    "mean_receiver_output_tokens": _average(values["receiver_output_tokens"]),
                    "mean_end_to_end_input_tokens": _average([
                        source + probe + specialist + receiver
                        for source, probe, specialist, receiver in zip(
                            values["source_prefill_tokens"], values["selection_probe_tokens"],
                            values["specialist_prompt_tokens"], values["receiver_prompt_tokens"],
                        )
                    ]),
                    "mean_end_to_end_output_tokens": _average([
                        handoff + receiver for handoff, receiver in zip(
                            values["specialist_handoff_tokens"], values["receiver_output_tokens"]
                        )
                    ]),
                    "mean_end_to_end_total_tokens": _average([
                        source + probe + specialist + handoff + receiver_prompt + receiver_output
                        for source, probe, specialist, handoff, receiver_prompt, receiver_output in zip(
                            values["source_prefill_tokens"], values["selection_probe_tokens"],
                            values["specialist_prompt_tokens"], values["specialist_handoff_tokens"],
                            values["receiver_prompt_tokens"], values["receiver_output_tokens"],
                        )
                    ]),
                }
                for name, values in efficiency.items()
            },
        },
        "efficiency_note": (
            "KV positions, KV bytes and text-token payloads are reported separately. "
            "They are not interchangeable measures of token or cost savings. End-to-end logical model tokens "
            "count input/output token events, not attention FLOPs, wall-clock cost, or provider billing."
        ),
    }

