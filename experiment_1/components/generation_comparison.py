"""Streaming comparison: logits are comparable only under a shared token prefix."""
import argparse
import json
from itertools import zip_longest
from pathlib import Path

from .storage import atomic_json


def identity_path(trace):
    if not trace.name.endswith(".tokens.jsonl"):
        raise ValueError("Expected a *.tokens.jsonl trace")
    return trace.with_name(trace.name[:-len(".tokens.jsonl")] + ".input.json")


def entries(path):
    with path.open(encoding="utf-8") as handle:
        for step, line in enumerate(handle):
            value = json.loads(line)
            if value["step"] != step:
                raise ValueError(f"Non-contiguous token steps in {path}")
            yield value


def compare(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    left = json.loads(identity_path(reference).read_text(encoding="utf-8"))
    right = json.loads(identity_path(candidate).read_text(encoding="utf-8"))
    keys = ("sample_id", "source_token_sha256", "suffix_token_sha256", "config",
            "model_revision", "model_dtype", "attention_backend", "temperature", "eos_token_ids",
            "lm_head_weight_dtype", "lm_head_output_dtype", "top_p", "top_k", "seed", "repetition_penalty", "repetition_history_scope")
    differences = [key for key in keys if left.get(key) != right.get(key)]
    result = {
        "reference": str(reference), "candidate": str(candidate),
        "reference_condition": left["cache_condition"], "candidate_condition": right["cache_condition"],
        "identity_mismatches": differences,
        "same_selected_positions": left["selected_position_sha256"] == right["selected_position_sha256"],
        "reference_retained_positions": left["retained_positions"],
        "candidate_retained_positions": right["retained_positions"],
        "logits_scope": "Top 8 raw logits and EOS scores, not the full vocabulary. "
                        "Scores at the first divergence share the same preceding generated token IDs.",
    }
    if differences:
        return {**result, "status": "incompatible_inputs", "first_divergence": None}
    matched = 0
    for a, b in zip_longest(entries(reference), entries(candidate)):
        if a is None or b is None:
            return {**result, "status": "matching_prefix_only", "matched_tokens": matched,
                    "first_divergence": None,
                    "note": "One trace ends here; no shared step with different token IDs is recorded. "
                            "Check generation status before treating either trace as complete."}
        if a["token_id"] != b["token_id"]:
            return {**result, "status": "diverged", "matched_tokens": matched,
                    "first_divergence": {"zero_based_step": matched, "reference": a, "candidate": b}}
        matched += 1
    return {**result, "status": "identical_recorded_tokens" if matched else "no_recorded_tokens", "matched_tokens": matched,
            "first_divergence": None,
            "note": "Recorded token IDs match; verify both status files for normal completion."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Normal full/refill/routed *.tokens.jsonl")
    parser.add_argument("--candidate", type=Path, required=True, help="Formal routed *.tokens.jsonl, including an interrupted trace")
    parser.add_argument("--output", type=Path, help="Default: first_divergence.json beside the candidate trace")
    args = parser.parse_args()
    output = args.output or args.candidate.parent / "first_divergence.json"
    result = compare(args.reference, args.candidate)
    atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Comparison saved: {output}")
