"""Normalize standalone elapsed-time ranges in generated timeline headings."""

import re

_HEADING_RANGE = re.compile(
    r"^(?P<prefix>\s*(?:#{1,6}\s*)?(?:[-*+]\s+|\d+[.)]\s+)?(?:\*\*)?"
    r"(?:(?:approximate\s+time|time|时间|大概时间|约)\s*[：:]\s*(?:\*\*)?\s*)?)"
    r"(?P<start>\d{1,3}:\d{2}(?::\d{2})?)\s*(?:-|–|—|~|～|至|到)\s*"
    r"(?P<end>\d{1,3}:\d{2}(?::\d{2})?)"
    r"(?P<suffix>(?:\*\*)?\s*)$",
    re.IGNORECASE,
)


def _seconds(value):
    parts = [int(part) for part in value.split(":")]
    if any(part >= 60 for part in parts[1:]):
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def midpoint_timeline_ranges(timeline):
    """Replace standalone elapsed-time heading ranges with their midpoint.

    A two-part timestamp means minutes:seconds; a three-part timestamp means
    hours:minutes:seconds. Prose, meeting schedules, single times, invalid and
    reversed ranges are preserved, and no source timestamp is invented.
    """
    result = []
    for line in timeline.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        ending = line[len(content):]
        match = _HEADING_RANGE.fullmatch(content)
        if not match:
            result.append(line)
            continue
        start, end = _seconds(match.group("start")), _seconds(match.group("end"))
        if start is None or end is None or end < start:
            result.append(line)
            continue
        midpoint = (start + end) // 2
        hours, remainder = divmod(midpoint, 3600)
        minutes, seconds = divmod(remainder, 60)
        value = f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"
        result.append(f"{match.group('prefix')}{value}{match.group('suffix')}{ending}")
    return "".join(result)
