"""Contiguous speaker-turn blocks, built before relevance selection."""
import bisect
import re

from .detail_selection import source_offsets

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
