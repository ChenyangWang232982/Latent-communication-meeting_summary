"""Generate QA answers through a trained same-model CIPHER channel."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from llm_latent.data import QALatentDataset
from llm_latent.model import SameModelCipherSystem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "squad_v1_latent"
OUTPUT_DIR = PROJECT_ROOT / "output"

MODEL_NAME = "google/long-t5-tglobal-large"
# Keep these values identical to train.py for the checkpoint being tested.
USE_COMPRESSION = False
COMPRESSED_LATENT_LEN = 8
SAMPLE_COUNT = 30

COMMUNICATION_NAME = "compressed" if USE_COMPRESSION else "uncompressed"
MODEL_NAME_TAG = MODEL_NAME.rsplit("/", maxsplit=1)[-1].replace("-", "_")
TEMPERATURE = 1.0
SENDER_MAX_LENGTH = 4096
RECEIVER_MAX_LENGTH = 128
TARGET_MAX_LENGTH = 256
MAX_NEW_TOKENS = 256


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


def load_model(device, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)
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
def generate_answer(model, batch):
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


def format_result(index, row, generated_answer):
    return "\n".join(
        [
            "=" * 80,
            f"Sample {index}",
            f"Question id: {row['id']}",
            "",
            "Question:",
            row["question"],
            "",
            "Generated answer:",
            generated_answer,
            "",
            "Gold answer:",
            row["answer"],
            "",
            "Sender-only context:",
            row["context"],
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing train.jsonl and validation.jsonl.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint created by llm_latent.train.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
        help="Use train for the 30-sample overfit check.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=SAMPLE_COUNT,
        help="Number of examples to generate; use 0 for all examples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata_path = args.data_dir.resolve() / f"{args.split}.jsonl"
    checkpoint_path = args.checkpoint.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing prepared data split: {metadata_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    rows = read_jsonl(metadata_path)
    sample_count = None if args.num_samples == 0 else args.num_samples
    if sample_count is not None:
        rows = rows[:sample_count]

    model = load_model(device, checkpoint_path)
    dataset = QALatentDataset(
        metadata_path=metadata_path,
        tokenizer=model.tokenizer,
        sender_max_length=SENDER_MAX_LENGTH,
        receiver_max_length=RECEIVER_MAX_LENGTH,
        target_max_length=TARGET_MAX_LENGTH,
    )
    if sample_count is not None:
        dataset.samples = dataset.samples[:sample_count]

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    result_blocks = []
    for index, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        generated_answer = generate_answer(model, batch)
        result_blocks.append(format_result(index, rows[index - 1], generated_answer))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{timestamp}_{checkpoint_path.stem}_{args.split}.txt"
    output_path.write_text("\n\n".join(result_blocks) + "\n", encoding="utf-8")
    print(f"saved output: {output_path}")


if __name__ == "__main__":
    main()
