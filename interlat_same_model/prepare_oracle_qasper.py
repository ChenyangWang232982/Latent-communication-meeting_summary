"""Build QASPER oracle-evidence JSONL splits for latent-channel experiments.

This is an oracle diagnostic: each Sender receives the human-annotated evidence
for its question. It must not be reported as an end-to-end retrieval result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare QASPER oracle-evidence data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/qasper_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/qasper_oracle_evidence"))
    parser.add_argument("--num-train", type=int, default=None, help="Optional cap after filtering.")
    parser.add_argument("--num-val", type=int, default=None, help="Optional cap after filtering.")
    parser.add_argument(
        "--include-no-evidence",
        action="store_true",
        help="Keep annotations without evidence using the paper abstract as context."
    )
    return parser.parse_args()


def answer_text(answer: dict[str, Any]) -> str:
    if answer.get("unanswerable"):
        return "unanswerable"
    if answer.get("yes_no") is not None:
        return "yes" if answer["yes_no"] else "no"
    spans = [span.strip() for span in answer.get("extractive_spans", []) if span.strip()]
    if spans:
        return " ".join(spans)
    return answer.get("free_form_answer", "").strip()


def unique_nonempty(paragraphs: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for paragraph in paragraphs:
        normalized = paragraph.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def build_split(
    raw_path: Path,
    output_path: Path,
    limit: int | None,
    include_no_evidence: bool,
) -> tuple[int, int]:
    papers = json.loads(raw_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    skipped_no_evidence = 0

    for paper_id, paper in papers.items():
        for question_index, qa in enumerate(paper["qas"]):
            for answer_index, annotation in enumerate(qa["answers"]):
                answer = annotation["answer"]
                target = answer_text(answer)
                evidence = unique_nonempty(answer.get("evidence", []))
                if not target:
                    continue
                if not evidence:
                    skipped_no_evidence += 1
                    if not include_no_evidence:
                        continue
                    evidence = unique_nonempty([paper.get("abstract", "")])

                rows.append(
                    {
                        "id": f"{paper_id}_{question_index}_{answer_index}",
                        "title": paper.get("title", ""),
                        "context": "\n\n".join(evidence),
                        "question": qa["question"].strip(),
                        "answer": target,
                        "evidence": evidence,
                        "context_source": "gold_evidence" if answer.get("evidence") else "abstract_fallback",
                    }
                )
                if limit is not None and len(rows) >= limit:
                    break
            if limit is not None and len(rows) >= limit:
                break
        if limit is not None and len(rows) >= limit:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows), skipped_no_evidence


def main() -> None:
    args = parse_args()
    train_count, train_skipped = build_split(
        args.raw_dir / "qasper-train-v0.3.json",
        args.output_dir / "train.jsonl",
        args.num_train,
        args.include_no_evidence,
    )
    val_count, val_skipped = build_split(
        args.raw_dir / "qasper-dev-v0.3.json",
        args.output_dir / "validation.jsonl",
        args.num_val,
        args.include_no_evidence,
    )
    print(
        f"saved train={train_count}, validation={val_count}; "
        f"skipped_without_evidence train={train_skipped}, validation={val_skipped}; "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
