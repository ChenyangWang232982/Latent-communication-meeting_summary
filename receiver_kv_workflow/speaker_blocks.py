"""Contiguous speaker-turn blocks, built before relevance selection."""
import bisect
import re

from .source_mapping import source_offsets

_NAME = r"[^\W\d_][\w .,'’()\-]{0,99}?"
_TIME = r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?"
_HEADERS = (
    re.compile(rf"^(?P<speaker>{_NAME})[ \t]+\[?(?:{_TIME})\]?[ \t]*$"),
    re.compile(rf"^\[(?:{_TIME})\][ \t]*(?P<speaker>{_NAME})[：:]"),
    re.compile(rf"^(?P<speaker>{_NAME})[：:](?:[ \t]+|$)"),
)


def speaker_ranges(transcript):
    """Return half-open character ranges; adjacent equal labels are merged.

    Labels must start a line. Names are compared case-insensitively with
    whitespace normalized; no speaker identity is inferred from prose.
    Unlabelled preamble is its own block. Missing labels are an explicit error.
    """
    turns = []
    offset = 0
    for line in transcript.splitlines(keepends=True):
        for pattern in _HEADERS:
            match = pattern.match(line.strip())
            if match:
                speaker = match.group("speaker").strip()
                key = " ".join(speaker.casefold().split())
                if not turns or key != turns[-1][2]:
                    turns.append((offset, speaker, key))
                break
        offset += len(line)
    if not turns:
        raise ValueError("speaker selection requires line-start speaker labels, e.g. 'Alice: ...' or 'Alice  01:23'. Use sentence for unlabelled transcripts.")
    blocks = []
    if turns[0][0] > 0:
        blocks.append({"speaker": None, "start": 0, "end": turns[0][0]})
    for i, (start, speaker, _) in enumerate(turns):
        end = turns[i + 1][0] if i + 1 < len(turns) else len(transcript)
        blocks.append({"speaker": speaker, "start": start, "end": end})
    return blocks


def select_complete_blocks(blocks, baseline, role_scores, protected, block_scores=None):
    """Select complete speaker blocks without exceeding the seed budget."""
    budget = len(baseline)
    selected = set(protected)
    if len(selected) > budget:
        raise ValueError("Protected positions exceed selection budget")
    if not role_scores:
        return sorted(selected)
    length = len(role_scores[0])
    pooled = [0.0] * length
    for scores in role_scores:
        if len(scores) != length:
            raise ValueError("Role score lengths differ")
        scale = sum(scores)
        if scale > 0:
            pooled = [max(a, b / scale) for a, b in zip(pooled, scores)]
    candidates = [(index, sorted(set(block) - selected)) for index, block in enumerate(blocks)]
    candidates = [(index, block) for index, block in candidates if block]
    candidates.sort(key=lambda item: (-(block_scores[item[0]] if block_scores is not None else
                                        sum(pooled[i] for i in item[1]) / len(item[1])), item[1][0]))
    for _, block in candidates:
        if len(selected) + len(block) <= budget:
            selected.update(block)
    return sorted(selected)


def expand_neighbor_blocks(seed_positions, blocks, protected, radius=1):
    """Retain complete adjacent speaker blocks around every selected seed block."""
    if radius < 0:
        raise ValueError("Neighbor radius cannot be negative")
    seeds = set(seed_positions)
    expanded = set(protected)
    for index, block in enumerate(blocks):
        if seeds.intersection(block):
            for neighbor in range(max(0, index - radius), min(len(blocks), index + radius + 1)):
                expanded.update(blocks[neighbor])
    return sorted(expanded)


def speaker_units(tokenizer, transcript, prefix, protected):
    blocks = speaker_ranges(transcript)
    offsets, origin = source_offsets(tokenizer, transcript, prefix)
    starts = [origin + block["start"] for block in blocks]
    units = [[] for _ in blocks]
    protected = set(protected)
    for i, (start, end) in enumerate(offsets):
        if i not in protected:
            index = max(0, bisect.bisect_right(starts, start) - 1)
            units[index].append(i)
    return [unit for unit in units if unit]
