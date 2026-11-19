"""Shared full-transcript length filtering, before benchmark sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def add_token_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-8B",
                        help="Generation model tokenizer ID or local model directory.")
    parser.add_argument("--min-transcript-tokens", type=int, default=0)
    parser.add_argument("--max-transcript-tokens", type=int, default=None,
                        help="Optional length ceiling; default keeps all lengths.")


class TranscriptTokenFilter:
    def __init__(self, tokenizer, *, model: str, minimum: int, maximum: int | None):
        if minimum < 0 or (maximum is not None and maximum < minimum):
            raise ValueError("Transcript token bounds must satisfy 0 <= min <= max.")
        self.tokenizer = tokenizer
        self.model = model
        self.minimum = minimum
        self.maximum = maximum
        self.scanned = self.too_short = self.too_long = 0
        self.accepted_lengths: list[int] = []

    def retain(self, record: dict) -> bool:
        # Count the exact normalized transcript, including speaker labels and
        # timestamps. Never truncate, add a chat template, or count references.
        count = len(self.tokenizer.encode(record["transcript"], add_special_tokens=False))
        self.scanned += 1
        if count < self.minimum:
            self.too_short += 1
            return False
        if self.maximum is not None and count > self.maximum:
            self.too_long += 1
            return False
        record["transcript_tokens"] = count
        self.accepted_lengths.append(count)
        return True

    def report(self, selected: list[dict]) -> dict:
        lengths = [row["transcript_tokens"] for row in selected]
        return {
            "tokenizer_model": self.model,
            "tokenizer_commit": getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash"),
            "min_transcript_tokens": self.minimum,
            "max_transcript_tokens": self.maximum,
            "length_filter_enabled": self.minimum > 0 or self.maximum is not None,
            "bounds_inclusive": True,
            "counting": "full normalized transcript; add_special_tokens=False; no truncation",
            "valid_records_scanned": self.scanned,
            "excluded_too_short": self.too_short,
            "excluded_too_long": self.too_long,
            "eligible_records": len(self.accepted_lengths),
            "records_written": len(selected),
            "selected_min_tokens": min(lengths) if lengths else None,
            "selected_max_tokens": max(lengths) if lengths else None,
            "selected_mean_tokens": sum(lengths) / len(lengths) if lengths else None,
        }


def load_token_filter(args) -> TranscriptTokenFilter:
    if args.min_transcript_tokens < 0 or (args.max_transcript_tokens is not None and args.max_transcript_tokens < args.min_transcript_tokens):
        raise ValueError("Transcript token bounds must satisfy 0 <= min <= max.")
    from transformers import AutoTokenizer
    print(f"Loading tokenizer only: {args.tokenizer_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    # Individual candidates may exceed the model limit. We only count them;
    # overlength candidates are excluded in full rather than truncated.
    tokenizer.model_max_length = 10**30
    return TranscriptTokenFilter(tokenizer, model=args.tokenizer_model,
                                 minimum=args.min_transcript_tokens, maximum=args.max_transcript_tokens)


def write_report(output: Path, report: dict) -> Path:
    path = output.with_suffix(output.suffix + ".report.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
