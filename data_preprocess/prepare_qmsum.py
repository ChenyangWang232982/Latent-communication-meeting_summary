"""Export one QMSum domain as speaker-labelled general-summary records.

The official QMSum files retain ``meeting_transcripts`` as speaker/content
turns.  This adapter writes one general-summary record per meeting so a
speaker-block workflow is evaluated on meetings, rather than on duplicated
query instances from the same meeting.

Example:
    python -m data_preprocess.prepare_qmsum \
      --input external/QMSum/data/Product/jsonl/test.jsonl \
      --output input/ami_test.jsonl --max-records 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable

from .token_filter import add_token_arguments, load_token_filter, write_report


DEFAULT_QUESTION = "Create an evidence-grounded meeting summary covering the most important facts."


def _read_one(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Input is empty: {path}")
    raw = [json.loads(line) for line in content.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(content)
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("meetings", [raw]))
    if not isinstance(raw, list):
        raise ValueError("Expected an official QMSum JSON/JSONL meeting collection.")
    return [row for row in raw if isinstance(row, dict)]


def _read(path: Path) -> list[dict[str, Any]]:
    """Read either a QMSum JSON/JSONL collection or an ``all/`` directory."""
    if path.is_file():
        return _read_one(path)
    if not path.is_dir():
        raise FileNotFoundError(f"QMSum input does not exist: {path}")
    files = sorted(candidate for candidate in path.rglob("*") if candidate.suffix.lower() in {".json", ".jsonl"})
    if not files:
        raise ValueError(f"No JSON/JSONL meeting files found below: {path}")
    rows = []
    for candidate in files:
        rows.extend(_read_one(candidate))
    return rows


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def speaker_transcript(turns: Any) -> str:
    if not isinstance(turns, list):
        return ""
    lines = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = _text(turn.get("speaker") or turn.get("role"))
        content = _text(turn.get("content") or turn.get("text") or turn.get("utterance"))
        # Some source files contain empty annotation artefacts. They carry no
        # evidence and can be skipped. A nonempty unlabelled turn, however,
        # would invalidate a speaker-block experiment, so reject that meeting
        # rather than inventing a speaker ID.
        if not content:
            continue
        if not speaker:
            return ""
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def general_queries(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    queries = row.get("general_query_list")
    if isinstance(queries, list):
        yield from (query for query in queries if isinstance(query, dict))
        return
    # Some processed QMSum files have already flattened a general query.
    query = row.get("query") or row.get("question")
    answer = row.get("answer") or row.get("summary")
    if isinstance(query, str) and isinstance(answer, str):
        yield {"query": query, "answer": answer}


def normalize(rows: Iterable[dict[str, Any]], *, max_records: int, seed: int = 7,
              token_filter=None) -> list[dict[str, Any]]:
    if max_records < 0:
        raise ValueError("--max-records must be 0 or positive.")
    records = []
    seen_meetings = set()
    for index, row in enumerate(rows):
        meeting_id = _text(row.get("meeting_id") or row.get("id")) or str(index)
        if meeting_id in seen_meetings:
            continue
        transcript = speaker_transcript(row.get("meeting_transcripts") or row.get("transcript"))
        if not transcript:
            continue
        query = next(iter(general_queries(row)), None)
        if not query:
            continue
        summary = _text(query.get("answer") or query.get("summary"))
        if not summary:
            continue
        record = {
            "id": meeting_id,
            "transcript": transcript,
            "summary": summary,
            "question": _text(query.get("query")) or DEFAULT_QUESTION,
        }
        seen_meetings.add(meeting_id)
        if token_filter is None or token_filter.retain(record):
            records.append(record)
    if not records:
        raise ValueError("No eligible speaker-labelled QMSum records remain after transcript-length filtering.")
    return random.Random(seed).sample(records, max_records) if max_records and len(records) > max_records else records


def write(records: list[dict[str, str]], output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it deliberately.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=output.parent, suffix=".tmp") as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Official QMSum domain JSON/JSONL file or its all/ directory.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=100,
                        help="Export at most this many eligible meetings; fewer means export all; 0 removes the cap.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic meeting-level sample seed.")
    parser.add_argument("--overwrite", action="store_true")
    add_token_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}. Use --overwrite to replace it deliberately.")
    token_filter = load_token_filter(args)
    records = normalize(_read(args.input), max_records=args.max_records, seed=args.seed, token_filter=token_filter)
    write(records, args.output, overwrite=args.overwrite)
    report_path = write_report(args.output, {
        "input": str(args.input), "output": str(args.output), "seed": args.seed,
        "record_limit": args.max_records or None, "records_written": len(records),
        "token_filter": token_filter.report(records),
    })
    print(f"Wrote {len(records)} speaker-labelled QMSum records: {args.output}")
    print(f"Preparation report: {report_path}")


if __name__ == "__main__":
    main()
