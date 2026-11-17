"""Logical decoder token events; KV positions and encoder tokens stay separate."""
FIELDS = ("source_prefill_tokens", "selection_probe_tokens", "specialist_prompt_tokens",
          "specialist_handoff_tokens", "receiver_prompt_tokens", "receiver_output_tokens")


def token_totals(measurement):
    saved = measurement.get("token_accounting", {})
    if any(key not in saved for key in FIELDS):
        raise ValueError("Incomplete saved token_accounting; cannot report missing events as zero")
    counts = {key: saved[key] for key in FIELDS}
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
        raise ValueError("Token counts must be non-negative integers")
    incoming = sum(counts[key] for key in (FIELDS[0], FIELDS[1], FIELDS[2], FIELDS[4]))
    outgoing = counts[FIELDS[3]] + counts[FIELDS[5]]
    return {**counts, "total_input_tokens": incoming, "total_output_tokens": outgoing,
            "total_tokens": incoming + outgoing}
