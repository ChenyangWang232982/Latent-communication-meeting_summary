"""Speaker-block anchors and source KV coordinates."""
import bisect

from receiver_kv_workflow.source_mapping import source_offsets
from receiver_kv_workflow.speaker_blocks import speaker_ranges


def annotate_transcript(transcript):
    """Add stable block markers after headers, preserving all original text."""
    result = []
    for i, row in enumerate(speaker_ranges(transcript)):
        text = transcript[row['start']:row['end']]
        header, newline, rest = text.partition('\n')
        result.append(header + '\n' + f'[B{i}]\n' + rest + ('' if text.endswith('\n') else '\n'))
    return ''.join(result)


def source_blocks(tokenizer, transcript, prefix, protected):
    ranges = speaker_ranges(transcript)
    offsets, origin = source_offsets(tokenizer, transcript, prefix)
    starts = [origin + row['start'] for row in ranges]
    units = [[] for _ in ranges]
    protected = set(protected)
    for position, (start, _) in enumerate(offsets):
        if position not in protected:
            units[max(0, bisect.bisect_right(starts, start) - 1)].append(position)
    result = []
    for row, positions in zip(ranges, units):
        if positions:
            result.append(dict(row, block_id=len(result), positions=positions,
                               text=transcript[row['start']:row['end']]))
    for i, row in enumerate(result):
        row['relative_position'] = i / max(1, len(result) - 1)
    if not result:
        raise ValueError('No source speaker blocks could be mapped to KV positions')
    return result
