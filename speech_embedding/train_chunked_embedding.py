import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from speech_embedding.chunked_data import ChunkedMeetingSpeechSummaryDataset
from speech_embedding.chunked_speech_to_summary_model import (
    ChunkedSpeechToSummaryLatentModel,
)
from speech_embedding.paths import CHECKPOINT_DIR, SPLIT_DIR


SPEECH_MODEL_NAME = "openai/whisper-base"
SUMMARY_MODEL_NAME = "google/flan-t5-small"

TEACHER_TARGET_FIELD = "teacher_summary"
TEACHER_PROMPT_TEXT = "summarize the meeting:"
TEACHER_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "checkpoint_chunked_embedding_teacher_distilled.pt"
)

# Keep this consistent with build_teacher_summary_metadata.py and test_chunked_embedding.py.
MAX_CHUNKS = 100
CHUNK_LATENT_LEN = 4
MAX_TEACHER_SUMMARY_LENGTH = 256

BATCH_SIZE = 1
MAX_EPOCHS = 60
PATIENCE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MIN_DELTA = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0
PRINT_EVERY = 5

FREEZE_SPEECH = True
FREEZE_SUMMARY = True

TRAIN_TARGET_FIELD = TEACHER_TARGET_FIELD


def get_prompt_text():
    return TEACHER_PROMPT_TEXT


def get_checkpoint_path():
    return TEACHER_CHECKPOINT_PATH


def get_max_target_length():
    return MAX_TEACHER_SUMMARY_LENGTH


def move_batch_to_device(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    phase = "train" if is_train else "val"

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(is_train):
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=LABEL_SMOOTHING,
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP_NORM,
                )
                optimizer.step()

        total_loss += loss.item()

        if step % PRINT_EVERY == 0 or step == len(loader):
            avg_loss = total_loss / step
            print(
                f"{phase} step {step}/{len(loader)}, "
                f"loss={loss.item():.4f}, "
                f"avg_loss={avg_loss:.4f}",
                flush=True,
            )

    return total_loss / max(len(loader), 1)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)
    print(
        "config: "
        f"mode_name=text_communication_distillation, "
        f"target_field={TRAIN_TARGET_FIELD}, "
        f"max_chunks={MAX_CHUNKS}, "
        f"chunk_latent_len={CHUNK_LATENT_LEN}, "
        f"batch_size={BATCH_SIZE}, "
        f"max_epochs={MAX_EPOCHS}, "
        f"patience={PATIENCE}, "
        f"lr={LEARNING_RATE}, "
        f"weight_decay={WEIGHT_DECAY}, "
        f"label_smoothing={LABEL_SMOOTHING}, "
        f"grad_clip_norm={GRAD_CLIP_NORM}, "
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
        prompt_text=get_prompt_text(),
        max_chunks=MAX_CHUNKS,
        max_summary_length=get_max_target_length(),
        target_field=TRAIN_TARGET_FIELD,
    )
    val_dataset = ChunkedMeetingSpeechSummaryDataset(
        metadata_path=SPLIT_DIR / "val.jsonl",
        speech_model_name=SPEECH_MODEL_NAME,
        summary_tokenizer=model.summary_tokenizer,
        prompt_text=get_prompt_text(),
        max_chunks=MAX_CHUNKS,
        max_summary_length=get_max_target_length(),
        target_field=TRAIN_TARGET_FIELD,
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
            torch.save(model.state_dict(), get_checkpoint_path())
            print(f"saved best checkpoint: {get_checkpoint_path()}")
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
