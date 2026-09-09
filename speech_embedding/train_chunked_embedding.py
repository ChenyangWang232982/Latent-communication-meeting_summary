import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from speech_embedding.chunked_data import ChunkedMeetingSpeechSummaryDataset
from speech_embedding.chunked_speech_to_summary_model import (
    ChunkedSpeechToSummaryLatentModel,
)
from speech_embedding.paths import CHECKPOINT_DIR, SPLIT_DIR


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

    speech_model_name = "openai/whisper-base"
    summary_model_name = "google/flan-t5-small"
    max_chunks = 48
    chunk_latent_len = 4
    max_summary_length = 256
    epochs = 150

    model = ChunkedSpeechToSummaryLatentModel(
        speech_model_name=speech_model_name,
        summary_model_name=summary_model_name,
        chunk_latent_len=chunk_latent_len,
        freeze_speech=True,
        freeze_summary=False,
    ).to(device)

    train_dataset = ChunkedMeetingSpeechSummaryDataset(
        metadata_path=SPLIT_DIR / "train.jsonl",
        speech_model_name=speech_model_name,
        summary_tokenizer=model.summary_tokenizer,
        max_chunks=max_chunks,
        max_summary_length=max_summary_length,
    )
    val_dataset = ChunkedMeetingSpeechSummaryDataset(
        metadata_path=SPLIT_DIR / "val.jsonl",
        speech_model_name=speech_model_name,
        summary_tokenizer=model.summary_tokenizer,
        max_chunks=max_chunks,
        max_summary_length=max_summary_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    trainable_params = [
        param for param in model.parameters()
        if param.requires_grad
    ]
    optimizer = AdamW(trainable_params, lr=5e-5)

    checkpoint_path = CHECKPOINT_DIR / "checkpoint_chunked_embedding_best.pt"
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"saved best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    train()
