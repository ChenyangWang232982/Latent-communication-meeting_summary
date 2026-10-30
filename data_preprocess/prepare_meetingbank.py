"""Download MeetingBank from Hugging Face and export normalized evaluation JSONL.

Example:
    python -m data_preprocess.prepare_meetingbank
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


DEFAULT_DATASET = "huuuyeah/meetingbank"
DEFAULT_OUTPUT = Path("input/meetingbank_test.jsonl")


@dataclass(frozen=True)
class PreparationReport:
    dataset: str
    split: str
    output: str
    records_written: int
    records_skipped: int
    source_columns: list[str]
    created_at_utc: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test", help="MeetingBank split, normally test, validation, or train.")
    parser.add_argument("--input", type=Path,
                        help="Optional local JSON/JSONL export. Use this for raw diarized MeetingBank turns.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face datasets cache directory.")
    parser.add_argument("--max-records", type=int, default=0, help="0 exports the entire split; a positive value exports that many records.")
    parser.add_argument("--require-speakers", action="store_true",
                        help="Require structured speaker-labelled turns; needed for --selection-strategy speaker.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output JSONL deliberately.")
    return parser.parse_args()


def text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _speaker_transcript(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    lines = []
    for turn in value:
        if not isinstance(turn, Mapping):
            return ""
        speaker = text(turn.get("speaker") or turn.get("role") or turn.get("speaker_id"))
        content = text(turn.get("content") or turn.get("text") or turn.get("utterance"))
        if not speaker or not content:
            return ""
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def normalize_row(row: Mapping[str, Any], index: int, *, require_speakers: bool = False) -> dict[str, str] | None:
    """Map the official MeetingBank schema into meeting_evaluation's schema."""
    raw_transcript = row.get("turns") or row.get("meeting_transcripts") or row.get("segments")
    transcript = _speaker_transcript(raw_transcript)
    if not transcript and not require_speakers:
        transcript = text(row.get("transcript") or row.get("meeting_transcript") or row.get("context"))
    summary = text(row.get("summary") or row.get("reference_summary") or row.get("answer"))
    if not transcript or not summary:
        return None
    record_id = text(row.get("uid")) or (str(row["id"]) if row.get("id") is not None else str(index))
    return {"id": record_id, "transcript": transcript, "summary": summary}


def export_rows(rows: Iterable[dict[str, Any]], output: Path, *, max_records: int, overwrite: bool,
                require_speakers: bool = False) -> tuple[int, int, list[str]]:
    if max_records < 0:
        raise ValueError("--max-records must be 0 or positive")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it deliberately.")
    output.parent.mkdir(parents=True, exist_ok=True)
    written, skipped, columns = 0, 0, set()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=output.parent, suffix=".jsonl.tmp") as handle:
            temporary = Path(handle.name)
            for index, row in enumerate(rows):
                if max_records and written >= max_records:
                    break
                if not isinstance(row, Mapping):
                    skipped += 1
                    continue
                columns.update(str(key) for key in row)
                record = normalize_row(row, index, require_speakers=require_speakers)
                if record is None:
                    skipped += 1
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            if not written:
                raise ValueError("No valid MeetingBank rows contained both transcript and summary.")
            handle.flush()
        temporary.replace(output)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return written, skipped, sorted(columns)


def _read_local(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8").strip()
    raw = [json.loads(line) for line in content.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(content)
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("meetings", [raw]))
    if not isinstance(raw, list):
        raise ValueError("Local input must be a JSON/JSONL list of MeetingBank records.")
    return [row for row in raw if isinstance(row, dict)]


def main() -> None:
    args = parse_args()
    if args.input:
        rows = _read_local(args.input)
        dataset_name = str(args.input)
        print(f"Loaded local MeetingBank export: {args.input} ({len(rows)} rows)", flush=True)
    else:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError("Missing dependency: install datasets (for example: pip install datasets).") from error
        rows = load_dataset(args.dataset, split=args.split, cache_dir=str(args.cache_dir) if args.cache_dir else None)
        dataset_name = args.dataset
        print(f"Loaded {args.dataset} split={args.split}: {len(rows)} rows", flush=True)
    written, skipped, columns = export_rows(
        rows, args.output, max_records=args.max_records, overwrite=args.overwrite,
        require_speakers=args.require_speakers,
    )
    report = PreparationReport(
        dataset=dataset_name, split=args.split, output=str(args.output), records_written=written,
        records_skipped=skipped, source_columns=columns,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {written} normalized records: {args.output}")
    print(f"Preparation report: {report_path}")


if __name__ == "__main__":
    main()
