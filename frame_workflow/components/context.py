"""Stable source anchors and exact fact-to-KV references."""
import bisect
import json

from receiver_kv_workflow.speaker_blocks import speaker_ranges
from receiver_kv_workflow.source_mapping import source_offsets


def anchor_transcript(transcript):
    parts = []
    for index, row in enumerate(speaker_ranges(transcript)):
        text = transcript[row['start']:row['end']]
        header, _, rest = text.partition('\n')
        parts.append(header + f'\n[B{index}]\n' + rest + ('' if text.endswith('\n') else '\n'))
    return ''.join(parts)


def context_blocks(tokenizer, transcript, prefix, protected, selected):
    ranges = speaker_ranges(transcript)
    offsets, origin = source_offsets(tokenizer, transcript, prefix)
    starts = [origin + row['start'] for row in ranges]
    units = [[] for _ in ranges]
    selected = set(selected) - set(protected)
    for position, (start, _) in enumerate(offsets):
        if position in selected:
            units[max(0, bisect.bisect_right(starts, start)-1)].append(position)
    return {f'B{i}': dict(speaker=row['speaker'], positions=unit)
            for i, (row, unit) in enumerate(zip(ranges, units)) if unit}


def fact_context(raw, blocks, protected, selected):
    """Resolve references, retaining all available evidence if references cannot be resolved."""
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    try:
        facts = json.loads(text)['facts']
        refs = {ref for fact in facts for ref in fact['context_refs']}
        # Do not silently discard a statement because its model-generated pointer is bad.
        if not facts or not refs or any(not f['context_refs'] for f in facts) or not refs <= blocks.keys():
            raise ValueError('Missing or unavailable context references')
        positions = sorted(set(protected) | {p for ref in refs for p in blocks[ref]['positions']})
        return facts, positions, None
    except (ValueError, KeyError, TypeError) as exc:
        return [dict(raw_report=raw, context_refs=list(blocks))], list(selected), f'{type(exc).__name__}: {exc}'
