import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import librosa
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from speech_embedding.chunked_data import select_chunk_indices_for_count
from speech_embedding.paths import CHECKPOINT_DIR, DATA_DIR, SPLIT_DIR, project_path


METADATA_PATH = DATA_DIR / "metadata.jsonl"
SPLIT_FILES = [
    SPLIT_DIR / "train.jsonl",
    SPLIT_DIR / "val.jsonl",
    SPLIT_DIR / "test.jsonl",
]

SUMMARY_MODEL_NAME = "google/flan-t5-small"
CHUNK_SECONDS = 30
MAX_CHUNKS = 100
TEACHER_PROMPT_TEXT = "summarize the meeting:"
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 256
NUM_BEAMS = 4
MAKE_BACKUP = True


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


def backup_file(path):
    path = project_path(path)
    if not MAKE_BACKUP or not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".teacher_backup_{timestamp}")
    shutil.copy2(path, backup_path)


def read_words(transcript_path):
    with project_path(transcript_path).open("r", encoding="utf-8") as in_file:
        return json.load(in_file)


def append_token(tokens, token, is_punctuation):
    if not token:
        return
    if not tokens:
        tokens.append(token)
        return
    if is_punctuation or token in {".", ",", "?", "!", ":", ";", "%"}:
        tokens[-1] = tokens[-1] + token
    else:
        tokens.append(token)


def words_to_text(words):
    tokens = []
    for word in words:
        append_token(tokens, word["token"], word.get("is_punctuation", False))
    return " ".join(tokens)


def get_audio_duration(audio_path):
    audio_path = project_path(audio_path)
    return librosa.get_duration(path=str(audio_path))


def selected_chunk_ranges(audio_duration):
    num_audio_chunks = max(1, math.ceil(audio_duration / CHUNK_SECONDS))
    selected_indices = select_chunk_indices_for_count(num_audio_chunks, MAX_CHUNKS)
    ranges = []
    for chunk_idx in selected_indices:
        start = chunk_idx * CHUNK_SECONDS
        end = start + CHUNK_SECONDS
        ranges.append((start, end))
    return ranges


def word_overlaps_range(word, start, end):
    word_start = float(word.get("start", 0.0))
    word_end = float(word.get("end", word_start))
    return word_start < end and word_end >= start


def build_selected_chunk_transcript(row):
    if not row.get("transcript_path"):
        raise ValueError(f"Missing transcript_path for meeting_id={row['meeting_id']}")

    words = read_words(row["transcript_path"])
    ranges = selected_chunk_ranges(get_audio_duration(row["audio_path"]))
    selected_words = []

    for start, end in ranges:
        selected_words.extend(
            word for word in words
            if word_overlaps_range(word, start, end)
        )

    selected_words.sort(key=lambda item: (item["start"], item["end"], item["token"]))
    return words_to_text(selected_words)


@torch.no_grad()
def generate_teacher_summary(model, tokenizer, text, device):
    prompt = f"{TEACHER_PROMPT_TEXT} {text}"
    inputs = tokenizer(
        prompt,
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        do_sample=False,
        length_penalty=1.2,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        early_stopping=True,
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def update_rows(rows, model, tokenizer, device):
    updated = []
    for idx, row in enumerate(rows, start=1):
        row = dict(row)
        print(f"teacher summary {idx}/{len(rows)}: {row['meeting_id']}", flush=True)
        chunk_transcript = build_selected_chunk_transcript(row)
        row["chunk_transcript"] = chunk_transcript
        row["teacher_summary"] = generate_teacher_summary(
            model=model,
            tokenizer=tokenizer,
            text=chunk_transcript,
            device=device,
        )
        updated.append(row)
    return updated


def update_file(path, model, tokenizer, device):
    rows = read_jsonl(path)
    backup_file(path)
    updated_rows = update_rows(rows, model, tokenizer, device)
    write_jsonl(path, updated_rows)
    print(f"updated {path}: rows={len(updated_rows)}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print(
        "teacher config: "
        f"max_chunks={MAX_CHUNKS}, "
        f"chunk_seconds={CHUNK_SECONDS}, "
        f"model={SUMMARY_MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL_NAME).to(device)
    model.eval()

    update_file(METADATA_PATH, model, tokenizer, device)
    for split_path in SPLIT_FILES:
        if split_path.exists():
            update_file(split_path, model, tokenizer, device)

    print("teacher summaries are ready.")


if __name__ == "__main__":
    main()
