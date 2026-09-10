import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from speech_embedding.chunked_data import ChunkedMeetingSpeechSummaryDataset
from speech_embedding.chunked_speech_to_summary_model import (
    ChunkedSpeechToSummaryLatentModel,
)
from speech_embedding.paths import CHECKPOINT_DIR, SPLIT_DIR


SPEECH_MODEL_NAME = "openai/whisper-base"
SUMMARY_MODEL_NAME = "google/flan-t5-small"

MAX_CHUNKS = 16
CHUNK_LATENT_LEN = 4
MAX_SUMMARY_LENGTH = 256

BATCH_SIZE = 1
MAX_EPOCHS = 60
PATIENCE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MIN_DELTA = 1e-4

FREEZE_SPEECH = True
FREEZE_SUMMARY = False


def move_batch_to_device(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(is_train):
            outputs = model(**batch)
            loss = outputs.loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print(
        "config: "
        f"max_chunks={MAX_CHUNKS}, "
        f"chunk_latent_len={CHUNK_LATENT_LEN}, "
        f"batch_size={BATCH_SIZE}, "
        f"max_epochs={MAX_EPOCHS}, "
        f"patience={PATIENCE}, "
        f"lr={LEARNING_RATE}, "
        f"weight_decay={WEIGHT_DECAY}, "
        f"freeze_speech={FREEZE_SPEECH}, "
        f"freeze_summary={FREEZE_SUMMARY}"
    )

    model = ChunkedSpeechToSummaryLatentModel(
        speech_model_name=SPEECH_MODEL_NAME,
        summary_model_name=SUMMARY_MODEL_NAME,
        chunk_latent_len=CHUNK_LATENT_LEN,
        freeze_speech=FREEZE_SPEECH,
        freeze_summary=FREEZE_SUMMARY,
    ).to(device)

    train_dataset = ChunkedMeetingSpeechSummaryDataset(
        metadata_path=SPLIT_DIR / "train.jsonl",
        speech_model_name=SPEECH_MODEL_NAME,
        summary_tokenizer=model.summary_tokenizer,
        max_chunks=MAX_CHUNKS,
        max_summary_length=MAX_SUMMARY_LENGTH,
    )
    val_dataset = ChunkedMeetingSpeechSummaryDataset(
        metadata_path=SPLIT_DIR / "val.jsonl",
        speech_model_name=SPEECH_MODEL_NAME,
        summary_tokenizer=model.summary_tokenizer,
        max_chunks=MAX_CHUNKS,
        max_summary_length=MAX_SUMMARY_LENGTH,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    trainable_params = [
        param for param in model.parameters()
        if param.requires_grad
    ]
    optimizer = AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    checkpoint_path = CHECKPOINT_DIR / "checkpoint_chunked_embedding_best.pt"
    best_val_loss = float("inf")
    bad_epochs = 0

    for epoch in range(MAX_EPOCHS):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{MAX_EPOCHS}, "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"best_val_loss={best_val_loss:.4f}"
        )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"saved best checkpoint: {checkpoint_path}")
        else:
            bad_epochs += 1
            print(f"no improvement: {bad_epochs}/{PATIENCE}")

        if bad_epochs >= PATIENCE:
            print(
                f"early stopping at epoch {epoch + 1}; "
                f"best_val_loss={best_val_loss:.4f}"
            )
            break


if __name__ == "__main__":
    train()
