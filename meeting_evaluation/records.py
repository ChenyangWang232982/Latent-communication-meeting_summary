"""Input adapters that normalize meeting benchmarks into one record schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MEETING_QUESTION = "Create an evidence-grounded meeting summary covering the most important facts."


@dataclass(frozen=True)
class MeetingRecord:
    id: str
    context: str
    question: str
    answer: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "context": self.context, "question": self.question, "answer": self.answer}


def _read_json_or_jsonl(path: Path) -> Any:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Input is empty: {path}")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    return json.loads(content)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(_text(item) for item in value if _text(item))
    if isinstance(value, dict):
        return _text(value.get("content") or value.get("text") or value.get("utterance") or "")
    return ""


def _qmsum_context(raw: dict[str, Any]) -> str:
    transcript = raw.get("meeting_transcripts") or raw.get("transcript") or raw.get("meeting") or ""
    if isinstance(transcript, list):
        lines = []
        for turn in transcript:
            if isinstance(turn, dict):
                speaker = str(turn.get("speaker") or turn.get("role") or "").strip()
                utterance = _text(turn)
                lines.append(f"{speaker}: {utterance}" if speaker else utterance)
            else:
                lines.append(_text(turn))
        return "\n".join(line for line in lines if line)
    return _text(transcript)


def _queries(raw: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("general_query_list", "specific_query_list", "query_list", "queries"):
        value = raw.get(key)
        if isinstance(value, list):
            yield from (item for item in value if isinstance(item, dict))


def load_qmsum(path: Path) -> list[MeetingRecord]:
    """Load official QMSum JSON, including both general and specific queries."""
    raw = _read_json_or_jsonl(path)
    meetings = raw if isinstance(raw, list) else raw.get("data", [])
    records: list[MeetingRecord] = []
    for meeting_index, meeting in enumerate(meetings):
        if not isinstance(meeting, dict):
            continue
        context = _qmsum_context(meeting)
        meeting_id = str(meeting.get("meeting_id") or meeting.get("id") or meeting_index)
        for query_index, query in enumerate(_queries(meeting)):
            question = _text(query.get("query") or query.get("question"))
            answer = _text(query.get("answer") or query.get("summary"))
            if context and question:
                query_id = str(query.get("query_id") or query_index)
                records.append(MeetingRecord(f"qmsum_{meeting_id}_{query_id}", context, question, answer))
    if not records:
        raise ValueError("No QMSum meeting/query records found. Check that this is the official QMSum JSON format.")
    return records


def load_meetingbank(path: Path) -> list[MeetingRecord]:
    """Load common MeetingBank JSON/JSONL exports or a normalized JSONL export."""
    raw = _read_json_or_jsonl(path)
    rows = raw if isinstance(raw, list) else raw.get("data", raw.get("meetings", []))
    records: list[MeetingRecord] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        context = _text(row.get("transcript") or row.get("meeting_transcript") or row.get("context"))
        answer = _text(row.get("summary") or row.get("reference_summary") or row.get("answer"))
        question = _text(row.get("question")) or DEFAULT_MEETING_QUESTION
        if context:
            records.append(MeetingRecord(str(row.get("id") or row.get("meeting_id") or f"meetingbank_{index}"), context, question, answer))
    if not records:
        raise ValueError("No MeetingBank records found. Expected transcript/summary fields or normalized JSONL.")
    return records


def load_normalized(path: Path) -> list[MeetingRecord]:
    raw = _read_json_or_jsonl(path)
    rows = raw if isinstance(raw, list) else raw.get("data", [raw])
    records = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        context = _text(row.get("context") or row.get("transcript"))
        if context:
            records.append(MeetingRecord(
                str(row.get("id") or index),
                context,
                _text(row.get("question")) or DEFAULT_MEETING_QUESTION,
                _text(row.get("answer") or row.get("summary")),
            ))
    if not records:
        raise ValueError("No normalized records found. Each row needs context or transcript.")
    return records


def load_ami_transcript(path: Path, reference_path: Path | None = None) -> list[MeetingRecord]:
    answer = reference_path.read_text(encoding="utf-8").strip() if reference_path else ""
    context = path.read_text(encoding="utf-8").strip()
    if not context:
        raise ValueError(f"Transcript is empty: {path}")
    return [MeetingRecord(path.stem, context, DEFAULT_MEETING_QUESTION, answer)]


def load_records(kind: str, path: Path, reference_path: Path | None = None) -> list[MeetingRecord]:
    if kind == "qmsum":
        return load_qmsum(path)
    if kind == "meetingbank":
        return load_meetingbank(path)
    if kind == "ami":
        return load_ami_transcript(path, reference_path)
    if kind == "jsonl":
        return load_normalized(path)
    raise ValueError(f"Unknown input kind: {kind}")

