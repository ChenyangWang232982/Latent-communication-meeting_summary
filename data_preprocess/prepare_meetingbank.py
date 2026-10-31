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
import random
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
                        help="Optional local JSON/JSONL export or official MeetingBank root directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face datasets cache directory.")
    parser.add_argument("--max-records", type=int, default=0, help="0 exports the entire split; a positive value exports that many records.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Deterministic sampling seed when --max-records is positive.")
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
        speaker_value = turn.get("speaker")
        if speaker_value is None:
            speaker_value = turn.get("role")
        if speaker_value is None:
            speaker_value = turn.get("speaker_id")
        speaker = text(str(speaker_value)) if speaker_value is not None else ""
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _official_meetingbank_rows(root: Path, split: str) -> list[dict[str, Any]]:
    """Load official segment summaries with their speaker-diarized item transcript.

    The official split files are JSONL despite their ``.json`` extension.  Their
    IDs are formed as ``<meeting id>_<item id>``, matching ``itemInfo`` entries
    in ``Metadata/MeetingBank.json``.
    """
    metadata_path = root / "Metadata" / "MeetingBank.json"
    split_path = root / "Metadata" / "Splits" / f"{split}.json"
    if not metadata_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(
            "Official MeetingBank root must contain Metadata/MeetingBank.json and "
            f"Metadata/Splits/{split}.json: {root}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    diarized: dict[str, list[dict[str, Any]]] = {}
    for meeting_id, meeting in metadata.items():
        if not isinstance(meeting, Mapping):
            continue
        for item_id, item in (meeting.get("itemInfo") or {}).items():
            if not isinstance(item, Mapping):
                continue
            turns = item.get("transcripts")
            transcript = _speaker_transcript(turns)
            if transcript:
                diarized[f"{meeting_id}_{item_id}"] = [
                    {"speaker": f"Speaker_{turn.get('speaker')}", "content": text(turn.get("text"))}
                    for turn in turns
                    if isinstance(turn, Mapping) and text(turn.get("text")) and turn.get("speaker") is not None
                ]

    rows = []
    for row in _read_jsonl(split_path):
        record_id = text(row.get("id"))
        turns = diarized.get(record_id)
        summary = text(row.get("summary"))
        if turns and summary:
            rows.append({"uid": record_id, "turns": turns, "summary": summary})
    return rows


def _sample(rows: list[dict[str, Any]], max_records: int, seed: int) -> list[dict[str, Any]]:
    if max_records <= 0 or len(rows) <= max_records:
        return rows
    return random.Random(seed).sample(rows, max_records)


def main() -> None:
    args = parse_args()
    if args.input:
        if args.input.is_dir():
            rows = _official_meetingbank_rows(args.input, args.split)
            rows = _sample(rows, args.max_records, args.seed)
            max_records = 0
            print(
                f"Loaded official diarized MeetingBank split={args.split}: {len(rows)} selected rows "
                f"from {args.input}", flush=True,
            )
        else:
            rows = _read_local(args.input)
            max_records = args.max_records
            print(f"Loaded local MeetingBank export: {args.input} ({len(rows)} rows)", flush=True)
        dataset_name = str(args.input)
    else:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError("Missing dependency: install datasets (for example: pip install datasets).") from error
        rows = load_dataset(args.dataset, split=args.split, cache_dir=str(args.cache_dir) if args.cache_dir else None)
        dataset_name = args.dataset
        max_records = args.max_records
        print(f"Loaded {args.dataset} split={args.split}: {len(rows)} rows", flush=True)
    written, skipped, columns = export_rows(
        rows, args.output, max_records=max_records, overwrite=args.overwrite,
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
