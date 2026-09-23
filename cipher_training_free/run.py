"""Run the training-free CIPHER baseline on prepared QA JSONL data."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from cipher_training_free.cipher import DEFAULT_MODEL_NAME, TrainingFreeCipher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "qasper_latent" / "validation.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "cipher_training_free" / "output"


def read_rows(path, limit):
    rows = []
    with path.open("r", encoding="utf-8") as data_file:
        for line in data_file:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--source-max-tokens", type=int, default=4096)
    parser.add_argument("--sender-max-new-tokens", type=int, default=128)
    parser.add_argument("--receiver-max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing JSONL data: {data_path}")
    if args.num_samples < 1:
        raise ValueError("num-samples must be at least 1.")

    rows = read_rows(data_path, args.num_samples)
    cipher = TrainingFreeCipher(model_name=args.model, device=args.device)
    blocks = []
    for index, row in enumerate(rows, start=1):
        result = cipher.answer(
            context=row["context"],
            question=row["question"],
            source_max_tokens=args.source_max_tokens,
            sender_max_new_tokens=args.sender_max_new_tokens,
            receiver_max_new_tokens=args.receiver_max_new_tokens,
            temperature=args.temperature,
        )
        print(f"processed {index}/{len(rows)}", flush=True)
        blocks.append(
            "\n".join(
                (
                    "=" * 80,
                    f"Sample {index}",
                    f"Question id: {row['id']}",
                    "",
                    "Question:",
                    row["question"],
                    "",
                    "Sender internal draft (not transmitted as text):",
                    result.sender_draft,
                    "",
                    f"CIPHER latent token count: {result.latent_token_count}",
                    "",
                    "Receiver answer from CIPHER latent:",
                    result.receiver_answer,
                    "",
                    "Gold answer:",
                    row["answer"],
                )
            )
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.rsplit("/", maxsplit=1)[-1].replace("-", "_")
    output_path = OUTPUT_DIR / f"{timestamp}_training_free_cipher_{model_tag}.txt"
    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"saved output: {output_path}")


if __name__ == "__main__":
    main()
