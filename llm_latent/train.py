"""Train same-model CIPHER embedding communication on prepared QA data."""

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from llm_latent.data import QALatentDataset
from llm_latent.model import SameModelCipherSystem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "squad_v1_latent"

MODEL_NAME = "google/long-t5-tglobal-large"
# Communication controls.  No compression keeps one latent message for every
# non-padding sender token; compression reduces it to COMPRESSED_LATENT_LEN.
USE_COMPRESSION = False
COMPRESSED_LATENT_LEN = 8

# Set to an integer for a small diagnostic run, or None to use all samples.
SAMPLE_COUNT = 30
VALIDATE_ON_TRAIN_SAMPLES = True

COMMUNICATION_NAME = "compressed" if USE_COMPRESSION else "uncompressed"
MODEL_NAME_TAG = MODEL_NAME.rsplit("/", maxsplit=1)[-1].replace("-", "_")

TEMPERATURE = 1.0
SENDER_MAX_LENGTH = 4096
RECEIVER_MAX_LENGTH = 128
TARGET_MAX_LENGTH = 256

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
USE_BF16 = True
MAX_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
UNFREEZE_RECEIVER_ENCODER_LAYERS = 2
UNFREEZE_DECODER_CROSS_ATTENTION = True

RECEIVER_TUNING_NAME = (
    f"encoder{UNFREEZE_RECEIVER_ENCODER_LAYERS}_decoder_cross_attention"
    if UNFREEZE_DECODER_CROSS_ATTENTION
    else f"encoder{UNFREEZE_RECEIVER_ENCODER_LAYERS}_frozen_decoder"
)


def configure_trainable_parameters(model):
    """Train CIPHER plus receiver layers that must consume latent messages."""
    for parameter in model.sender.parameters():
        parameter.requires_grad = False

    for parameter in model.receiver.parameters():
        parameter.requires_grad = False

    for parameter in model.comm.parameters():
        parameter.requires_grad = True

    total_encoder_layers = model.receiver.config.num_layers
    first_trainable_layer = total_encoder_layers - UNFREEZE_RECEIVER_ENCODER_LAYERS

    for layer_index in range(first_trainable_layer, total_encoder_layers):
        for parameter in model.receiver.encoder.block[layer_index].parameters():
            parameter.requires_grad = True

    for parameter in model.receiver.encoder.final_layer_norm.parameters():
        parameter.requires_grad = True

    if UNFREEZE_DECODER_CROSS_ATTENTION:
        # T5 decoder blocks contain self-attention, encoder-decoder attention,
        # then feed-forward layers.  Only the middle cross-attention module
        # needs to adapt to the non-native CIPHER encoder representation.
        for decoder_block in model.receiver.decoder.block:
            for parameter in decoder_block.layer[1].parameters():
                parameter.requires_grad = True

        for parameter in model.receiver.decoder.final_layer_norm.parameters():
            parameter.requires_grad = True


def move_batch_to_device(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def run_epoch(model, loader, optimizer, device):
    is_training = optimizer is not None
    model.train(is_training)

    # Sender is frozen, so keep its dropout disabled. Receiver stays in train
    # mode only during training because its last encoder layers are trainable.
    model.sender.eval()
    if is_training:
        # The decoder is frozen in this first experiment.  Keeping it in eval
        # mode makes its teacher-forced loss deterministic for a given latent.
        model.receiver.decoder.eval()

    total_loss = 0.0
    if is_training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=USE_BF16 and device.startswith("cuda"),
            ):
                outputs = model(**batch)
                loss = outputs.loss

            if is_training:
                (loss / GRADIENT_ACCUMULATION_STEPS).backward()
                if (
                    step % GRADIENT_ACCUMULATION_STEPS == 0
                    or step == len(loader)
                ):
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        GRAD_CLIP_NORM,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def build_dataset(path, tokenizer):
    return QALatentDataset(
        metadata_path=path,
        tokenizer=tokenizer,
        sender_max_length=SENDER_MAX_LENGTH,
        receiver_max_length=RECEIVER_MAX_LENGTH,
        target_max_length=TARGET_MAX_LENGTH,
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
        "--experiment-name",
        default="squad",
        help="Checkpoint name prefix, for example qasper.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "validation.jsonl"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(
            f"Expected train.jsonl and validation.jsonl under {data_dir}. "
            "Run a preparation script first."
        )

    checkpoint_path = (
        PROJECT_ROOT
        / "llm_latent"
        / (
            f"cipher_{args.experiment_name}_{MODEL_NAME_TAG}_{COMMUNICATION_NAME}_"
            f"{RECEIVER_TUNING_NAME}_samples{SAMPLE_COUNT or 'all'}.pt"
        )
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SameModelCipherSystem(
        model_name=MODEL_NAME,
        latent_len=COMPRESSED_LATENT_LEN,
        temperature=TEMPERATURE,
        use_compression=USE_COMPRESSION,
        freeze_sender=True,
        freeze_receiver=True,
    ).to(device)
    configure_trainable_parameters(model)
    model.receiver.gradient_checkpointing_enable()
    model.receiver.config.use_cache = False

    train_dataset = build_dataset(train_path, model.tokenizer)
    if SAMPLE_COUNT is not None:
        if len(train_dataset) < SAMPLE_COUNT:
            raise ValueError(
                f"Need {SAMPLE_COUNT} train examples, found {len(train_dataset)}"
            )

        train_dataset.samples = train_dataset.samples[:SAMPLE_COUNT]

    if VALIDATE_ON_TRAIN_SAMPLES:
        val_dataset = train_dataset
    else:
        val_dataset = build_dataset(val_path, model.tokenizer)
        if SAMPLE_COUNT is not None:
            val_dataset.samples = val_dataset.samples[:SAMPLE_COUNT]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(f"device: {device}")
    print(
        f"communication: {COMMUNICATION_NAME}, "
        f"receiver_tuning: {RECEIVER_TUNING_NAME}, "
        f"sample_count: {SAMPLE_COUNT or 'all'}, "
        f"train_samples: {len(train_dataset)}, val_samples: {len(val_dataset)}, "
        f"compressed_latent_len: {COMPRESSED_LATENT_LEN}"
    )
    print(
        f"sender_max_length: {SENDER_MAX_LENGTH}, "
        f"batch_size: {BATCH_SIZE}, "
        f"gradient_accumulation: {GRADIENT_ACCUMULATION_STEPS}, "
        f"bf16: {USE_BF16 and device.startswith('cuda')}"
    )
    print(f"parameters: total={total_parameters:,}, trainable={trainable_count:,}")

    best_val_loss = float("inf")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"saved checkpoint: {checkpoint_path}", flush=True)

        print(
            f"Epoch {epoch}/{MAX_EPOCHS}, "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"best_val_loss={best_val_loss:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
