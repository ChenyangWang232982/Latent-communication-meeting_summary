"""Prepare long-context QASPER question-answering examples for CIPHER training.

Each QASPER paper contains multiple questions.  This script expands it into
JSONL rows compatible with :class:`QALatentDataset`.  Context is a token-bounded
window from the full paper, centred on the first annotated evidence passage
when one is available.
"""

import argparse
import json
import random
import tarfile
from urllib.request import urlretrieve
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "qasper_latent"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "qasper_raw"
MODEL_NAME = "google/long-t5-tglobal-large"
CONTEXT_WINDOW_TOKENS = 3900
TRAIN_DEV_ARCHIVE_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/"
    "qasper-train-dev-v0.3.tgz"
)


def flatten_text(value):
    """Convert the slightly nested QASPER paper representation to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        preferred_keys = ("section_name", "section", "title", "paragraphs", "text")
        pieces = [flatten_text(value[key]) for key in preferred_keys if key in value]
        if pieces:
            return "\n\n".join(piece for piece in pieces if piece)
        return "\n\n".join(flatten_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "\n\n".join(flatten_text(item) for item in value if flatten_text(item))
    return str(value).strip()


def answer_text(answer):
    """Return a generatable answer for QASPER's multiple answer formats."""
    if not isinstance(answer, dict):
        return str(answer).strip()

    # The dataset builder may preserve the source's outer ``answer`` object.
    if isinstance(answer.get("answer"), dict):
        answer = answer["answer"]

    free_form = answer.get("free_form_answer") or answer.get("answer")
    if isinstance(free_form, str) and free_form.strip():
        return free_form.strip()

    spans = answer.get("extractive_spans") or answer.get("spans") or []
    if isinstance(spans, str):
        spans = [spans]
    spans = [span.strip() for span in spans if isinstance(span, str) and span.strip()]
    if spans:
        return " ".join(spans)

    yes_no = answer.get("yes_no")
    if isinstance(yes_no, bool):
        return "yes" if yes_no else "no"
    if isinstance(yes_no, str) and yes_no.lower() in {"yes", "no"}:
        return yes_no.lower()
    return ""


def answer_evidence(answer):
    if not isinstance(answer, dict):
        return []
    if isinstance(answer.get("answer"), dict):
        answer = answer["answer"]
    evidence = answer.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    return [item.strip() for item in evidence if isinstance(item, str) and item.strip()]


def make_context_window(document, evidence, tokenizer, max_tokens):
    """Keep a long-document window around evidence without splitting tokens."""
    token_ids = tokenizer(document, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_tokens:
        return document.strip()

    first_evidence_position = -1
    document_lower = document.lower()
    for passage in evidence:
        first_evidence_position = document_lower.find(passage.lower())
        if first_evidence_position >= 0:
            break

    if first_evidence_position < 0:
        start_token = 0
    else:
        prefix_ids = tokenizer(
            document[:first_evidence_position], add_special_tokens=False
        )["input_ids"]
        start_token = max(0, len(prefix_ids) - max_tokens // 3)

    start_token = min(start_token, max(0, len(token_ids) - max_tokens))
    return tokenizer.decode(
        token_ids[start_token : start_token + max_tokens],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def iter_rows(split, tokenizer, max_tokens):
    if isinstance(split, dict):
        papers = []
        for paper_id, paper in split.items():
            paper = dict(paper)
            paper.setdefault("id", paper_id)
            papers.append(paper)
    else:
        papers = split

    for paper_index, paper in enumerate(papers):
        document = flatten_text(paper.get("full_text"))
        if not document:
            document = "\n\n".join(
                part for part in (paper.get("title", ""), paper.get("abstract", "")) if part
            )
        if not document:
            continue

        paper_id = paper.get("id") or paper.get("paper_id") or str(paper_index)
        qas = paper.get("qas") or paper.get("questions") or []
        if isinstance(qas, dict):
            qas = [qas] if "question" in qas else qas.values()
        for qa_index, qa in enumerate(qas):
            question = (qa.get("question") or "").strip()
            answers = qa.get("answers") or []
            if isinstance(answers, dict):
                # Some dataset versions expose a struct of answer sequences.
                answers = answers.get("answer", [answers])
            if not isinstance(answers, list):
                answers = [answers]

            for answer_index, answer in enumerate(answers):
                target = answer_text(answer)
                # Unanswerable examples do not teach the communication channel
                # to preserve document-grounded information, so omit them.
                if not target:
                    continue
                yield {
                    "id": f"{paper_id}_{qa_index}_{answer_index}",
                    "title": paper.get("title", ""),
                    "context": make_context_window(
                        document, answer_evidence(answer), tokenizer, max_tokens
                    ),
                    "question": question,
                    "answer": target,
                }


def write_split(name, split, tokenizer, max_tokens, limit, seed):
    rows = list(iter_rows(split, tokenizer, max_tokens))
    random.Random(seed).shuffle(rows)
    if limit is not None:
        rows = rows[:limit]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{name}.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {name}: {output_path}, rows={len(rows)}")


def extract_archive(archive_path, destination):
    """Extract the trusted official archive without allowing path traversal."""
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination)


def ensure_official_data(raw_dir):
    train_path = raw_dir / "qasper-train-v0.3.json"
    dev_path = raw_dir / "qasper-dev-v0.3.json"
    if train_path.is_file() and dev_path.is_file():
        return train_path, dev_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "qasper-train-dev-v0.3.tgz"
    if not archive_path.is_file():
        print("downloading official QASPER v0.3 train/dev archive...", flush=True)
        urlretrieve(TRAIN_DEV_ARCHIVE_URL, archive_path)
    print("extracting QASPER archive...", flush=True)
    extract_archive(archive_path, raw_dir)
    if not train_path.is_file() or not dev_path.is_file():
        raise FileNotFoundError("Official QASPER archive did not contain train/dev JSON files.")
    return train_path, dev_path


def load_official_json(path):
    with path.open("r", encoding="utf-8") as data_file:
        return json.load(data_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-val", type=int, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=CONTEXT_WINDOW_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Cache location for the official QASPER v0.3 JSON files.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_path, dev_path = ensure_official_data(args.raw_dir.resolve())
    write_split(
        "train",
        load_official_json(train_path),
        tokenizer,
        args.max_context_tokens,
        args.num_train,
        args.seed,
    )
    write_split(
        "validation",
        load_official_json(dev_path),
        tokenizer,
        args.max_context_tokens,
        args.num_val,
        args.seed + 1,
    )


if __name__ == "__main__":
    main()
