import argparse
import json
import random
from pathlib import Path

from speech_embedding.paths import DATA_DIR, SPLIT_DIR, project_path


def read_jsonl(path):
    path = project_path(path)
    rows = []
    with path.open("r", encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        for row in rows:
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_counts(total):
    train_count = int(total * 5 / 7)
    val_count = int(total * 1 / 7)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def parse_args():
    parser = argparse.ArgumentParser(description="Split AMI metadata into train/val/test.")
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "metadata.jsonl"),
        help="Input metadata JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SPLIT_DIR),
        help="Directory for train/val/test JSONL files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    random.Random(args.seed).shuffle(rows)

    train_count, val_count, test_count = split_counts(len(rows))
    train_rows = rows[:train_count]
    val_rows = rows[train_count : train_count + val_count]
    test_rows = rows[train_count + val_count :]

    output_dir = project_path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)

    print(f"total: {len(rows)}")
    print(f"train: {len(train_rows)}")
    print(f"val: {len(val_rows)}")
    print(f"test: {len(test_rows)}")


if __name__ == "__main__":
    main()
