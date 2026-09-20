"""Download SQuAD v1.1 and create answer-preserving QA JSONL files."""

import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "squad_v1_latent"
MODEL_NAME = "google/long-t5-tglobal-large"
CONTEXT_WINDOW_TOKENS = 4096


def make_answer_window(context, answer_start, answer_text, tokenizer):
    encoded = tokenizer(
        context,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    answer_end = answer_start + len(answer_text)

    answer_tokens = [
        index
        for index, (start, end) in enumerate(offsets)
        if start < answer_end and end > answer_start
    ]
    if not answer_tokens:
        raise ValueError("Could not locate the answer in its context window.")

    first_answer = answer_tokens[0]
    last_answer = answer_tokens[-1]
    answer_length = last_answer - first_answer + 1
    if answer_length > CONTEXT_WINDOW_TOKENS:
        raise ValueError("Answer is longer than the configured context window.")

    left_budget = (CONTEXT_WINDOW_TOKENS - answer_length) // 2
    start_index = max(0, first_answer - left_budget)
    end_index = min(len(token_ids), start_index + CONTEXT_WINDOW_TOKENS)
    start_index = max(0, end_index - CONTEXT_WINDOW_TOKENS)

    return tokenizer.decode(
        token_ids[start_index:end_index],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def write_split(split_name, split, tokenizer):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{split_name}.jsonl"

    with output_path.open("w", encoding="utf-8") as output_file:
        for example in split:
            answer = example["answers"]["text"][0]
            answer_start = example["answers"]["answer_start"][0]
            row = {
                "id": example["id"],
                "title": example["title"],
                "context": make_answer_window(
                    example["context"],
                    answer_start,
                    answer,
                    tokenizer,
                ),
                "question": example["question"],
                "answer": answer,
            }
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved {split_name}: {output_path}, rows={len(split)}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    squad = load_dataset("rajpurkar/squad")
    write_split("train", squad["train"], tokenizer)
    write_split("validation", squad["validation"], tokenizer)


if __name__ == "__main__":
    main()
