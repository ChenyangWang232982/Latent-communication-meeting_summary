"""Check whether the frozen receiver can repeat a normal text transcript."""

import json
from datetime import datetime

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from speech_embedding.paths import DATA_DIR, PROJECT_ROOT


SUMMARY_MODEL_NAME = "google/flan-t5-small"
METADATA_PATH = DATA_DIR / "chunk_blocks_teacher" / "train.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "output"

PROMPT_TEXT = "repeat the speech transcript:"
MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 128
MAX_SAMPLES = 8


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
def generate_text(model, tokenizer, transcript, device):
    # This is the normal prompt/text route, deliberately without latent vectors.
    model_input = f"{PROMPT_TEXT} {transcript}"
    inputs = tokenizer(
        model_input,
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
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def format_result(index, row, generated_text):
    return "\n".join(
        [
            "=" * 80,
            f"Sample {index}",
            f"Sample id: {row['sample_id']}",
            "",
            "Generated transcript:",
            generated_text,
            "",
            "Teacher transcript:",
            row["teacher_transcript"],
        ]
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL_NAME).to(device)
    model.generation_config.max_length = None
    model.eval()

    rows = read_jsonl(METADATA_PATH)[:MAX_SAMPLES]
    if not rows:
        raise ValueError(f"No rows found in {METADATA_PATH}")

    blocks = []
    for index, row in enumerate(rows, start=1):
        teacher_transcript = row.get("teacher_transcript", "").strip()
        if not teacher_transcript:
            raise ValueError(f"Missing teacher_transcript for {row['sample_id']}")
        generated_text = generate_text(model, tokenizer, teacher_transcript, device)
        blocks.append(format_result(index, row, generated_text))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{timestamp}_teacher_text_baseline.txt"
    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"saved output: {output_path}")


if __name__ == "__main__":
    main()
