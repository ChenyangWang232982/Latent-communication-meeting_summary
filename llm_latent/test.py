"""Generate meeting summaries through the trained same-model CIPHER channel."""

import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from llm_latent.data import MeetingLatentDataset
from llm_latent.model import SameModelCipherSystem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "toy_meetings" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "output"

MODEL_NAME = "google/flan-t5-small"
# Keep these values identical to train.py for the checkpoint being tested.
USE_COMPRESSION = False
COMPRESSED_LATENT_LEN = 8
SAMPLE_COUNT = 8
EVALUATE_TRAIN_SAMPLES = True

COMMUNICATION_NAME = "compressed" if USE_COMPRESSION else "uncompressed"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "llm_latent"
    / f"cipher_same_model_{COMMUNICATION_NAME}_samples{SAMPLE_COUNT or 'all'}.pt"
)
OVERFIT_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"

TEMPERATURE = 1.0
SENDER_MAX_LENGTH = 256
RECEIVER_MAX_LENGTH = 32
TARGET_MAX_LENGTH = 128
MAX_NEW_TOKENS = 128


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_latent_len(state_dict):
    query_tokens = state_dict.get("comm.query_tokens")
    return (
        query_tokens.size(0)
        if query_tokens is not None
        else COMPRESSED_LATENT_LEN
    )


def load_model(device):
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model = SameModelCipherSystem(
        model_name=MODEL_NAME,
        latent_len=infer_latent_len(state_dict),
        temperature=TEMPERATURE,
        use_compression=USE_COMPRESSION,
        freeze_sender=True,
        freeze_receiver=True,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def generate_summary(model, batch):
    combined_embeds, combined_mask = model.build_receiver_inputs(
        sender_input_ids=batch["sender_input_ids"],
        sender_attention_mask=batch["sender_attention_mask"],
        receiver_input_ids=batch["receiver_input_ids"],
        receiver_attention_mask=batch["receiver_attention_mask"],
    )
    receiver_encoder_outputs = model.receiver.get_encoder()(
        inputs_embeds=combined_embeds,
        attention_mask=combined_mask,
        return_dict=True,
    )
    generated_ids = model.receiver.generate(
        encoder_outputs=receiver_encoder_outputs,
        attention_mask=combined_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=4,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        early_stopping=True,
    )
    return model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def move_batch_to_device(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def format_result(index, row, generated_summary):
    source_text = row.get("teacher_transcript") or row.get("transcript", "")
    return "\n".join(
        [
            "=" * 80,
            f"Sample {index}",
            f"Meeting id: {row.get('meeting_id', '')}",
            "",
            "Generated summary:",
            generated_summary,
            "",
            "Gold summary:",
            row.get("summary", ""),
            "",
            "Source transcript:",
            source_text,
        ]
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata_path = OVERFIT_PATH if EVALUATE_TRAIN_SAMPLES else TEST_PATH
    rows = read_jsonl(metadata_path)
    if SAMPLE_COUNT is not None:
        rows = rows[:SAMPLE_COUNT]

    model = load_model(device)
    dataset = MeetingLatentDataset(
        metadata_path=metadata_path,
        tokenizer=model.tokenizer,
        sender_max_length=SENDER_MAX_LENGTH,
        receiver_max_length=RECEIVER_MAX_LENGTH,
        target_max_length=TARGET_MAX_LENGTH,
    )
    if SAMPLE_COUNT is not None:
        dataset.samples = dataset.samples[:SAMPLE_COUNT]

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    result_blocks = []
    for index, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        generated_summary = generate_summary(model, batch)
        result_blocks.append(format_result(index, rows[index - 1], generated_summary))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "train_samples" if EVALUATE_TRAIN_SAMPLES else "test"
    output_path = OUTPUT_DIR / f"{timestamp}_cipher_same_model_{mode}.txt"
    output_path.write_text("\n\n".join(result_blocks) + "\n", encoding="utf-8")
    print(f"saved output: {output_path}")


if __name__ == "__main__":
    main()
