import math
import bisect

from receiver_kv_workflow.speaker_blocks import speaker_ranges
from receiver_kv_workflow.source_mapping import source_offsets


def transcript_windows(tokenizer, transcript, prefix, protected, parts=8, overlap=0.25):
    """Cover the original transcript, including turns omitted by global KV routing."""
    ranges = speaker_ranges(transcript)
    offsets, origin = source_offsets(tokenizer, transcript, prefix)
    starts = [row['start'] for row in ranges]
    units = [[] for _ in ranges]
    protected = set(protected)
    for position, (start, end) in enumerate(offsets):
        if position not in protected and end > origin and start < origin + len(transcript):
            index = max(0, bisect.bisect_right(starts, start-origin)-1)
            units[index].append(position)
    positions = sorted(protected | {p for unit in units for p in unit})
    for window in selected_windows(units, positions, protected, parts, overlap):
        first, last = window['block_ids'][0], window['block_ids'][-1]
        start, end = ranges[first]['start'], ranges[last]['end']
        yield dict(window, char_start=start, char_end=end, total_source_blocks=len(ranges),
                   transcript_fraction_start=start/len(transcript), transcript_fraction_end=end/len(transcript),
                   text=transcript[start:end])


def selected_windows(blocks, selected, protected, parts=8, overlap=0.25):
    """Window already-selected complete speaker turns; never restore dropped source rows."""
    selected, protected = set(selected), set(protected)
    units = [(i, sorted(set(block) & (selected-protected))) for i, block in enumerate(blocks)]
    units = [(i, unit) for i, unit in units if unit]
    target = max(1, math.ceil(sum(len(unit) for _, unit in units)/parts))
    start = 0
    while start < len(units):
        end, size = start, 0
        while end < len(units) and (end == start or size+len(units[end][1]) <= target):
            size += len(units[end][1])
            end += 1
        yield dict(block_ids=[i for i, _ in units[start:end]],
                   positions=sorted(protected | {p for _, unit in units[start:end] for p in unit}))
        if end == len(units):
            break
        next_start, shared = end, 0
        while next_start > start+1 and shared < size*overlap:
            next_start -= 1
            shared += len(units[next_start][1])
        start = next_start
