"""Build short targets from the receiver's normal prompt/text behavior."""

import json

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from speech_embedding.paths import DATA_DIR, project_path


SUMMARY_MODEL_NAME = "google/flan-t5-small"
SOURCE_DIR = DATA_DIR / "chunk_blocks_teacher"
OUTPUT_DIR = DATA_DIR / "chunk_blocks_receiver_targets"
# Keep this first pass cheap: it creates targets only for the eight examples
# used by the adapter capacity diagnostic.  Set this to None and add val/test
# after the diagnostic has passed.
SPLIT_NAMES = ["train"]
MAX_SAMPLES_PER_SPLIT = 8

PROMPT_TEXT = "summarize the key factual information from the speech in one sentence:"
MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 48


def read_jsonl(path):
    rows = []
    with project_path(path).open("r", encoding="utf-8") as in_file:
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


@torch.no_grad()
def generate_target(model, tokenizer, transcript, device):
    inputs = tokenizer(
        f"{PROMPT_TEXT} {transcript}",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=4,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        early_stopping=True,
    )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def build_split(split_name, model, tokenizer, device):
    rows = read_jsonl(SOURCE_DIR / f"{split_name}.jsonl")
    if MAX_SAMPLES_PER_SPLIT is not None:
        rows = rows[:MAX_SAMPLES_PER_SPLIT]
    output_rows = []

    for row in rows:
        transcript = row.get("teacher_transcript", "").strip()
        if not transcript:
            raise ValueError(f"Missing teacher_transcript for {row['sample_id']}")
        row = dict(row)
        row["receiver_text_target"] = generate_target(
            model, tokenizer, transcript, device
        )
        output_rows.append(row)

    output_path = OUTPUT_DIR / f"{split_name}.jsonl"
    write_jsonl(output_path, output_rows)
    print(f"saved {split_name}: {output_path}, rows={len(output_rows)}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL_NAME).to(device)
    model.generation_config.max_length = None
    model.eval()

    for split_name in SPLIT_NAMES:
        build_split(split_name, model, tokenizer, device)


if __name__ == "__main__":
    main()
