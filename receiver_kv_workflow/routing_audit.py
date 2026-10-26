"""Offline source-coordinate audit. Its text is never included in model input."""
from .detail_selection import source_offsets, sentence_units


def contiguous_ranges(positions):
    ranges = []
    for position in sorted(set(positions)):
        if ranges and position == ranges[-1][1]:
            ranges[-1][1] += 1
        else:
            ranges.append([position, position + 1])
    return ranges


def build_audit(tokenizer, transcript, prefix, protected, selected, baseline, role_positions, strategy,
                *, seed_selection=None, neighbor_blocks=0):
    offsets, origin = source_offsets(tokenizer, transcript, prefix)
    protected = set(protected)
    selected, baseline = set(selected), set(baseline)
    seed_selection = set(seed_selection if seed_selection is not None else selected)

    def excerpts(positions):
        result = []
        for start, end in contiguous_ranges(set(positions) - protected):
            a = max(0, offsets[start][0] - origin)
            b = min(len(transcript), offsets[end - 1][1] - origin)
            result.append({"token_range": [start, end], "character_range": [a, b],
                           "text": transcript[a:b]})
        return result

    sentences = []
    if strategy == "speaker":
        from .speaker_blocks import speaker_units
        units = speaker_units(tokenizer, transcript, prefix, protected)
    else:
        units = sentence_units(tokenizer, transcript, prefix, protected)
    for index, unit in enumerate(units):
        kept = selected.intersection(unit)
        sentences.append({"unit": index, "source_tokens": len(unit), "retained_tokens": len(kept),
                          "status": "full" if len(kept) == len(unit) else "partial" if kept else "dropped",
                          "source": excerpts(unit), "retained_ranges": contiguous_ranges(kept)})
    return {"strategy": strategy, "coordinate_convention": "zero-based half-open; token indices include source chat framing; characters index transcript",
            "note": "Offline audit only. Unselected surrounding text is not sent to routed_kv. A retained passage does not guarantee summary correctness.",
            "selected_positions": len(selected), "baseline_positions": len(baseline),
            "seed_selected_positions": len(seed_selection),
            "context_added_positions": len(selected - seed_selection),
            "speaker_neighbor_blocks": neighbor_blocks if strategy == "speaker" else 0,
            "protected_positions": len(protected), "selected_excerpts": excerpts(selected),
            "added_vs_fixed": excerpts(selected - baseline), "removed_vs_fixed": excerpts(baseline - selected),
            "role_fixed_ranges": {role: contiguous_ranges(set(ids) - protected) for role, ids in role_positions.items()},
            "coverage_unit": "speaker_block" if strategy == "speaker" else "sentence",
            "sentence_coverage": sentences}


def render_audit(audit):
    lines = ["KV routing audit (not model input)", f"strategy: {audit['strategy']}",
             audit["coordinate_convention"], audit["note"],
             f"retained={audit['selected_positions']}; seed={audit.get('seed_selected_positions', audit['selected_positions'])}; fixed baseline={audit['baseline_positions']}",
             "", "EXACT SELECTED SOURCE EXCERPTS"]
    for item in audit["selected_excerpts"]:
        lines.extend([f"tokens {item['token_range']}; chars {item['character_range']}", item["text"], ""])
    if audit["strategy"] == "speaker":
        lines.append(f"SPEAKER CONTEXT: +/-{audit['speaker_neighbor_blocks']} blocks; added positions={audit['context_added_positions']}")
    lines.append(f"{audit.get('coverage_unit', 'sentence').upper()} COVERAGE (includes dropped source for diagnosis)")
    for item in audit["sentence_coverage"]:
        lines.append(f"unit {item['unit']}: {item['status']} {item['retained_tokens']}/{item['source_tokens']} tokens; retained ranges={item['retained_ranges']}")
        lines.extend(part["text"] for part in item["source"])
        lines.append("")
    return "\n".join(lines)
