"""Train same-model CIPHER embedding communication on meeting summaries."""

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from llm_latent.data import MeetingLatentDataset
from llm_latent.model import SameModelCipherSystem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "toy_meetings" / "splits"

MODEL_NAME = "google/flan-t5-small"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
# Communication controls.  No compression keeps one latent message for every
# non-padding sender token; compression reduces it to COMPRESSED_LATENT_LEN.
USE_COMPRESSION = False
COMPRESSED_LATENT_LEN = 8

# Set to an integer for a small diagnostic run, or None to use all samples.
SAMPLE_COUNT = 30
VALIDATE_ON_TRAIN_SAMPLES = True

COMMUNICATION_NAME = "compressed" if USE_COMPRESSION else "uncompressed"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "llm_latent"
    / f"cipher_same_model_{COMMUNICATION_NAME}_samples{SAMPLE_COUNT or 'all'}.pt"
)

TEMPERATURE = 1.0
SENDER_MAX_LENGTH = 256
RECEIVER_MAX_LENGTH = 32
TARGET_MAX_LENGTH = 128

BATCH_SIZE = 2
MAX_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
UNFREEZE_RECEIVER_ENCODER_LAYERS = 2


def configure_trainable_parameters(model):
    """Freeze sender; train CIPHER plus the receiver's final encoder layers."""
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
    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(is_training):
            outputs = model(**batch)
            loss = outputs.loss

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    GRAD_CLIP_NORM,
                )
                optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def build_dataset(path, tokenizer):
    return MeetingLatentDataset(
        metadata_path=path,
        tokenizer=tokenizer,
        sender_max_length=SENDER_MAX_LENGTH,
        receiver_max_length=RECEIVER_MAX_LENGTH,
        target_max_length=TARGET_MAX_LENGTH,
    )


def main():
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

    train_dataset = build_dataset(TRAIN_PATH, model.tokenizer)
    if SAMPLE_COUNT is not None:
        if len(train_dataset) < SAMPLE_COUNT:
            raise ValueError(
                f"Need {SAMPLE_COUNT} train examples, found {len(train_dataset)}"
            )

        train_dataset.samples = train_dataset.samples[:SAMPLE_COUNT]

    if VALIDATE_ON_TRAIN_SAMPLES:
        val_dataset = train_dataset
    else:
        val_dataset = build_dataset(VAL_PATH, model.tokenizer)
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
        f"sample_count: {SAMPLE_COUNT or 'all'}, "
        f"train_samples: {len(train_dataset)}, val_samples: {len(val_dataset)}, "
        f"compressed_latent_len: {COMPRESSED_LATENT_LEN}"
    )
    print(f"parameters: total={total_parameters:,}, trainable={trainable_count:,}")

    best_val_loss = float("inf")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)

        print(
            f"Epoch {epoch}/{MAX_EPOCHS}, "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"best_val_loss={best_val_loss:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
