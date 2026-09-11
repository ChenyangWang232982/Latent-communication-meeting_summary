import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from speech_embedding.chunk_block_data import ChunkBlockSpeechTranscriptDataset
from speech_embedding.chunked_speech_to_summary_model import (
    ChunkedSpeechToSummaryLatentModel,
)
from speech_embedding.paths import CHECKPOINT_DIR, DATA_DIR, PROJECT_ROOT


SPEECH_MODEL_NAME = "openai/whisper-base"
SUMMARY_MODEL_NAME = "google/flan-t5-small"
COMM_METHOD = "direct_projection"
COMM_TEMPERATURE = 1.0
COMM_TOP_K = None

PROMPT_TEXT = "repeat the speech transcript:"
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint_chunk_block_direct_embedding_t5_encoder.pt"
TEST_METADATA_PATH = DATA_DIR / "chunk_blocks_teacher" / "test.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "output"

CHUNK_LATENT_LEN = 64
MAX_TARGET_LENGTH = 128
MAX_NEW_TOKENS = 128
BATCH_SIZE = 1
MAX_TEST_SAMPLES = 30


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_chunk_latent_len(state_dict):
    query_tokens = state_dict.get("comm.query_tokens")
    if query_tokens is None:
        return CHUNK_LATENT_LEN
    return query_tokens.shape[0]


def load_model(device):
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    chunk_latent_len = infer_chunk_latent_len(state_dict)
    print("checkpoint chunk_latent_len:", chunk_latent_len)

    model = ChunkedSpeechToSummaryLatentModel(
        speech_model_name=SPEECH_MODEL_NAME,
        summary_model_name=SUMMARY_MODEL_NAME,
        chunk_latent_len=chunk_latent_len,
        freeze_speech=True,
        freeze_summary=True,
        comm_method=COMM_METHOD,
        comm_temperature=COMM_TEMPERATURE,
        comm_top_k=COMM_TOP_K,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


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
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=4,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        early_stopping=True,
    )
    return model.summary_tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def format_result(index, row, generated_text):
    lines = [
        "=" * 80,
        f"Sample {index}",
        f"Sample id: {row['sample_id']}",
        f"Meeting id: {row['meeting_id']}",
        f"Chunk index: {row['chunk_index']}",
        f"Start time: {row['start_time']}s",
        "",
        "Generated transcript:",
        generated_text,
        "",
        "Teacher transcript:",
        row.get("teacher_transcript", ""),
    ]
    if row.get("transcript"):
        lines.extend(
            [
                "",
                "AMI transcript:",
                row["transcript"],
            ]
        )
    return "\n".join(lines)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("checkpoint:", CHECKPOINT_PATH)

    model = load_model(device)
    rows = read_jsonl(TEST_METADATA_PATH)
    if MAX_TEST_SAMPLES is not None:
        rows = rows[:MAX_TEST_SAMPLES]

    dataset = ChunkBlockSpeechTranscriptDataset(
        metadata_path=TEST_METADATA_PATH,
        speech_model_name=SPEECH_MODEL_NAME,
        summary_tokenizer=model.summary_tokenizer,
        prompt_text=PROMPT_TEXT,
        max_target_length=MAX_TARGET_LENGTH,
    )
    if MAX_TEST_SAMPLES is not None:
        dataset.samples = dataset.samples[:MAX_TEST_SAMPLES]

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{timestamp}_chunk_block_test.txt"

    blocks = []
    for index, batch in enumerate(loader, start=1):
        print(f"generate {index}/{len(loader)}", flush=True)
        batch = move_batch_to_device(batch, device)
        generated_text = generate_text(model, batch)
        blocks.append(format_result(index, rows[index - 1], generated_text))

    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"saved output: {output_path}")


if __name__ == "__main__":
    main()
