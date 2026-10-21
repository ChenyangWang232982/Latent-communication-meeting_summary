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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face datasets cache directory.")
    parser.add_argument("--max-records", type=int, default=0, help="0 exports the entire split; a positive value exports that many records.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output JSONL deliberately.")
    return parser.parse_args()


def text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize_row(row: Mapping[str, Any], index: int) -> dict[str, str] | None:
    """Map the official MeetingBank schema into meeting_evaluation's schema."""
    transcript = text(row.get("transcript") or row.get("meeting_transcript") or row.get("context"))
    summary = text(row.get("summary") or row.get("reference_summary") or row.get("answer"))
    if not transcript or not summary:
        return None
    record_id = text(row.get("uid")) or (str(row["id"]) if row.get("id") is not None else str(index))
    return {"id": record_id, "transcript": transcript, "summary": summary}


def export_rows(rows: Iterable[dict[str, Any]], output: Path, *, max_records: int, overwrite: bool) -> tuple[int, int, list[str]]:
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
                record = normalize_row(row, index)
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


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Missing dependency: install datasets (for example: pip install datasets).") from error
    dataset = load_dataset(args.dataset, split=args.split, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    print(f"Loaded {args.dataset} split={args.split}: {len(dataset)} rows", flush=True)
    written, skipped, columns = export_rows(dataset, args.output, max_records=args.max_records, overwrite=args.overwrite)
    report = PreparationReport(
        dataset=args.dataset, split=args.split, output=str(args.output), records_written=written,
        records_skipped=skipped, source_columns=columns,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {written} normalized records: {args.output}")
    print(f"Preparation report: {report_path}")


if __name__ == "__main__":
    main()
