import json

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from speech_embedding.paths import DATA_DIR, project_path


SPEECH_MODEL_NAME = "openai/whisper-base"
SOURCE_DIR = DATA_DIR / "chunk_blocks"
OUTPUT_DIR = DATA_DIR / "chunk_blocks_teacher"
SPLIT_NAMES = ["train", "val", "test"]

SAMPLING_RATE = 16000
BATCH_SIZE = 8
MAX_NEW_TOKENS = 128
LANGUAGE = "english"
TASK = "transcribe"


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


def load_audio_chunk(row):
    audio, _ = librosa.load(
        project_path(row["audio_path"]),
        sr=SAMPLING_RATE,
        mono=True,
        offset=float(row["start_time"]),
        duration=float(row["duration"]),
    )
    return audio


def get_forced_decoder_ids(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None
    if not hasattr(tokenizer, "get_decoder_prompt_ids"):
        return None
    return tokenizer.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)


@torch.no_grad()
def transcribe_batch(model, processor, rows, device):
    audios = [load_audio_chunk(row) for row in rows]
    inputs = processor(
        audios,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_features = inputs["input_features"].to(device)

    generate_kwargs = {
        "input_features": input_features,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_beams": 1,
        "do_sample": False,
    }
    forced_decoder_ids = get_forced_decoder_ids(processor)
    if forced_decoder_ids is not None:
        generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

    generated_ids = model.generate(**generate_kwargs)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def transcribe_split(split_name, model, processor, device):
    source_path = SOURCE_DIR / f"{split_name}.jsonl"
    output_path = OUTPUT_DIR / f"{split_name}.jsonl"
    rows = read_jsonl(source_path)
    output_rows = []

    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start : start + BATCH_SIZE]
        teacher_texts = transcribe_batch(model, processor, batch_rows, device)

        for row, teacher_text in zip(batch_rows, teacher_texts):
            row = dict(row)
            row["teacher_transcript"] = " ".join(teacher_text.strip().split())
            output_rows.append(row)

        done = min(start + BATCH_SIZE, len(rows))
        print(
            f"{split_name}: {done}/{len(rows)} teacher transcripts generated",
            flush=True,
        )

    write_jsonl(output_path, output_rows)
    print(f"saved {split_name}: {output_path}, rows={len(output_rows)}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("teacher model:", SPEECH_MODEL_NAME)
    print("source:", SOURCE_DIR)
    print("output:", OUTPUT_DIR)

    processor = WhisperProcessor.from_pretrained(SPEECH_MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(SPEECH_MODEL_NAME).to(device)
    model.eval()

    for split_name in SPLIT_NAMES:
        transcribe_split(split_name, model, processor, device)


if __name__ == "__main__":
    main()
