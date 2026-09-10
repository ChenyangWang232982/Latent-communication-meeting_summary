import json
import math
from datetime import datetime
from pathlib import Path

import librosa
import torch
from transformers import WhisperProcessor

from speech_embedding.chunked_speech_to_summary_model import (
    ChunkedSpeechToSummaryLatentModel,
)
from speech_embedding.paths import CHECKPOINT_DIR, PROJECT_ROOT, SPLIT_DIR, project_path


# MODE = 0: use test.jsonl to generate summaries and print gold summaries.
# MODE = 1: read all audio files under input/ and generate summaries only.
MODE = 0

SPEECH_MODEL_NAME = "openai/whisper-base"
SUMMARY_MODEL_NAME = "google/flan-t5-small"
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint_chunked_embedding_alignment.pt"

TEST_METADATA_PATH = SPLIT_DIR / "test.jsonl"
INPUT_AUDIO_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

CHUNK_SECONDS = 30
# Keep this consistent with train_chunked_embedding.py.
MAX_CHUNKS = 100
CHUNK_LATENT_LEN = 4
MAX_PROMPT_LENGTH = 32
MAX_SUMMARY_NEW_TOKENS = 256
SAMPLING_RATE = 16000
BATCH_SIZE = 1
SUMMARY_PROMPT_TEXT = "summarize the meeting:"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def read_jsonl(path):
    rows = []
    with project_path(path).open("r", encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_input_audio_files(input_dir):
    input_dir = project_path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input audio directory not found: {input_dir}")

    audio_files = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(path)

    if not audio_files:
        raise FileNotFoundError(f"No audio files found under: {input_dir}")

    return sorted(audio_files)


def select_chunk_indices(num_audio_chunks, max_chunks):
    if num_audio_chunks <= max_chunks:
        return list(range(num_audio_chunks))

    indices = torch.linspace(0, num_audio_chunks - 1, steps=max_chunks)
    return indices.round().long().tolist()


def split_audio(audio, chunk_samples, max_chunks):
    num_audio_chunks = max(1, math.ceil(len(audio) / chunk_samples))
    selected_indices = select_chunk_indices(num_audio_chunks, max_chunks)
    total_chunks = len(selected_indices)

    chunks = []
    for chunk_idx in selected_indices:
        start = chunk_idx * chunk_samples
        end = start + chunk_samples
        chunk = audio[start:end]

        if len(chunk) < chunk_samples:
            padded = torch.zeros(chunk_samples, dtype=torch.float32)
            if len(chunk) > 0:
                padded[: len(chunk)] = torch.tensor(chunk, dtype=torch.float32)
            chunk = padded.numpy()

        chunks.append(chunk)

    while len(chunks) < max_chunks:
        chunks.append(torch.zeros(chunk_samples, dtype=torch.float32).numpy())

    chunk_attention_mask = torch.zeros(max_chunks, dtype=torch.long)
    chunk_attention_mask[:total_chunks] = 1
    return chunks, chunk_attention_mask


def encode_audio(audio_path, speech_processor):
    audio_path = project_path(audio_path)
    audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    chunk_samples = SAMPLING_RATE * CHUNK_SECONDS
    chunks, chunk_attention_mask = split_audio(audio, chunk_samples, MAX_CHUNKS)

    features = []
    for chunk in chunks:
        speech_inputs = speech_processor(
            chunk,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
        )
        features.append(speech_inputs["input_features"].squeeze(0))

    input_features = torch.stack(features, dim=0)
    return input_features, chunk_attention_mask


def get_prompt_text():
    return SUMMARY_PROMPT_TEXT


def get_max_new_tokens():
    return MAX_SUMMARY_NEW_TOKENS


def encode_prompt(tokenizer, prompt_text):
    prompt = tokenizer(
        prompt_text,
        max_length=MAX_PROMPT_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return prompt["input_ids"].squeeze(0), prompt["attention_mask"].squeeze(0)


def make_batch(audio_path, speech_processor, tokenizer, device, prompt_text):
    input_features, chunk_attention_mask = encode_audio(audio_path, speech_processor)
    prompt_input_ids, prompt_attention_mask = encode_prompt(tokenizer, prompt_text)

    batch = {
        "input_features": input_features.unsqueeze(0),
        "chunk_attention_mask": chunk_attention_mask.unsqueeze(0),
        "prompt_input_ids": prompt_input_ids.unsqueeze(0),
        "prompt_attention_mask": prompt_attention_mask.unsqueeze(0),
    }
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def generate_text(model, batch):
    latent_embeds, latent_mask = model.encode_all_chunks(
        batch["input_features"],
        batch["chunk_attention_mask"],
    )

    prompt_embeds = model.summary_model.get_input_embeddings()(
        batch["prompt_input_ids"]
    )
    combined_embeds = torch.cat([latent_embeds, prompt_embeds], dim=1)
    combined_mask = torch.cat([latent_mask, batch["prompt_attention_mask"]], dim=1)

    encoder_outputs = model.summary_model.get_encoder()(
        inputs_embeds=combined_embeds,
        attention_mask=combined_mask,
        return_dict=True,
    )

    generated_ids = model.summary_model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=combined_mask,
        max_new_tokens=get_max_new_tokens(),
        num_beams=4,
        length_penalty=1.2,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        early_stopping=True,
    )

    generated_text = model.summary_tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    return generated_text[0]


def load_model(device):
    model = ChunkedSpeechToSummaryLatentModel(
        speech_model_name=SPEECH_MODEL_NAME,
        summary_model_name=SUMMARY_MODEL_NAME,
        chunk_latent_len=CHUNK_LATENT_LEN,
        freeze_speech=True,
        freeze_summary=True,
    ).to(device)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_mode_0_items():
    rows = read_jsonl(TEST_METADATA_PATH)
    return [
        {
            "name": row.get("meeting_id", Path(row["audio_path"]).stem),
            "audio_path": row["audio_path"],
            "gold_text": row.get("summary", ""),
            "gold_label": "Gold summary",
        }
        for row in rows
    ]


def build_mode_1_items():
    return [
        {
            "name": path.stem,
            "audio_path": path,
            "gold_text": "",
            "gold_label": "",
        }
        for path in find_input_audio_files(INPUT_AUDIO_DIR)
    ]


def get_generated_label():
    return "Generated summary"


def format_result(index, item, generated_text):
    lines = [
        "=" * 80,
        f"Sample {index}",
        f"Name: {item['name']}",
        f"Audio: {item['audio_path']}",
        "",
        f"{get_generated_label()}:",
        generated_text,
    ]

    if item["gold_text"]:
        lines.extend(
            [
                "",
                f"{item['gold_label']}:",
                item["gold_text"],
            ]
        )

    return "\n".join(lines)


def main():
    if MODE not in {0, 1}:
        raise ValueError("MODE must be 0 or 1.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("mode:", MODE)
    print("checkpoint:", CHECKPOINT_PATH)

    model = load_model(device)
    speech_processor = WhisperProcessor.from_pretrained(SPEECH_MODEL_NAME)

    if MODE == 0:
        items = build_mode_0_items()
    else:
        items = build_mode_1_items()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{timestamp}_mode{MODE}_embedding_alignment.txt"
    prompt_text = get_prompt_text()
    print("prompt:", prompt_text)

    result_blocks = []
    for index, item in enumerate(items, start=1):
        print(f"generate {index}/{len(items)}: {item['name']}", flush=True)
        batch = make_batch(
            audio_path=item["audio_path"],
            speech_processor=speech_processor,
            tokenizer=model.summary_tokenizer,
            device=device,
            prompt_text=prompt_text,
        )
        generated_text = generate_text(model, batch)
        result_blocks.append(format_result(index, item, generated_text))

    output_text = "\n\n".join(result_blocks) + "\n"
    output_path.write_text(output_text, encoding="utf-8")
    print(f"saved output: {output_path}")


if __name__ == "__main__":
    main()
