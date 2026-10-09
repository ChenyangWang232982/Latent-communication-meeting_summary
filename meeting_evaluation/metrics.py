"""Reference, factuality-proxy, and efficiency metrics for one evaluation run."""

from __future__ import annotations

import re
from collections import Counter
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


def score_results(results: list[dict[str, Any]], source_by_id: dict[str, str]) -> dict[str, Any]:
    variants = ("statebridge", "text", "no_comm")
    per_variant: dict[str, dict[str, list[float | None]]] = {
        name: {"rouge_l_f1": [], "supported_claim_rate": []} for name in variants
    }
    communication = {"text_handoff_tokens": 0, "latent_state_tokens": 0, "prefix_prefill_seconds": [], "prefix_cache_bytes": []}
    for result in results:
        communication["text_handoff_tokens"] += result["communication"]["text_handoff_tokens"]
        communication["latent_state_tokens"] += result["communication"]["latent_state_tokens"]
        communication["prefix_prefill_seconds"].append(result["prefix_prefill_seconds"])
        communication["prefix_cache_bytes"].append(result["prefix_cache_bytes"])
        for variant in variants:
            answer = result.get("answers", {}).get(variant, "")
            per_variant[variant]["rouge_l_f1"].append(rouge_l_f1(answer, result.get("gold_answer", "")))
            per_variant[variant]["supported_claim_rate"].append(
                source_overlap_support(answer, source_by_id[result["id"]])["supported_claim_rate"]
            )
    text_tokens = communication["text_handoff_tokens"]
    latent_states = communication["latent_state_tokens"]
    delta = latent_states - text_tokens
    return {
        "samples": len(results),
        "quality": {
            variant: {metric: _average(values) for metric, values in metrics.items()}
            for variant, metrics in per_variant.items()
        },
        "factuality_note": "supported_claim_rate is a lexical source-overlap proxy inspired by MESA-style source checking; it is not an LLM or human MESA score.",
        "efficiency": {
            "text_handoff_tokens": text_tokens,
            "latent_state_tokens": latent_states,
            "latent_minus_text_tokens": delta,
            "latent_minus_text_percent": None if not text_tokens else delta * 100.0 / text_tokens,
            "mean_prefix_prefill_seconds": _average(communication["prefix_prefill_seconds"]),
            "mean_prefix_cache_mib": _average([value / 2**20 for value in communication["prefix_cache_bytes"]]),
        },
    }

