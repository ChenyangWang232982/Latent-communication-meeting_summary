import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from speech_embedding.chunk_block_data import ChunkBlockSpeechTranscriptDataset
from speech_embedding.chunked_speech_to_summary_model import (
    ChunkedSpeechToSummaryLatentModel,
)
from speech_embedding.paths import CHECKPOINT_DIR, DATA_DIR


SPEECH_MODEL_NAME = "openai/whisper-base"
SUMMARY_MODEL_NAME = "google/flan-t5-small"
PROMPT_TEXT = "repeat the speech transcript:"
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint_decoder_latent_adapter_overfit8.pt"

TRAIN_METADATA_PATH = DATA_DIR / "chunk_blocks_teacher" / "train.jsonl"
VAL_METADATA_PATH = DATA_DIR / "chunk_blocks_teacher" / "val.jsonl"
TARGET_FIELD = "teacher_transcript"

CHUNK_LATENT_LEN = 64
MAX_TARGET_LENGTH = 128
MAX_WHISPER_NEW_TOKENS = 128

# This is a capacity diagnostic, not a generalization experiment.  It asks one
# narrow question: can the frozen-T5 adapter memorize eight audio/text pairs?
OVERFIT_MODE = True
OVERFIT_SAMPLE_COUNT = 8

BATCH_SIZE = 8
MAX_EPOCHS = 300
PATIENCE = None
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
MIN_DELTA = 1e-4
GRAD_CLIP_NORM = 1.0

FREEZE_SPEECH = True
FREEZE_SUMMARY = True


def configure_trainable_parameters(model):
    for param in model.speech_model.parameters():
        param.requires_grad = False

    for param in model.summary_model.parameters():
        param.requires_grad = False

    for param in model.comm.parameters():
        param.requires_grad = True



def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    # Frozen sender/receiver must also stay deterministic: ``model.train()``
    # would otherwise activate their dropout layers without updating weights.
    model.speech_model.eval()
    model.summary_model.eval()
    total_loss = 0.0
    phase = "train" if is_train else "val"

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(is_train):
            outputs = model(
                input_features=batch["input_features"],
                chunk_attention_mask=batch["chunk_attention_mask"],
                prompt_input_ids=batch["prompt_input_ids"],
                prompt_attention_mask=batch["prompt_attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print(
        "config: "
        f"mode_name=chunk_block_whisper_decoder_latent_adapter, "
        f"target_field={TARGET_FIELD}, "
        f"chunk_latent_len={CHUNK_LATENT_LEN}, "
        f"max_whisper_new_tokens={MAX_WHISPER_NEW_TOKENS}, "
        f"max_target_length={MAX_TARGET_LENGTH}, "
        f"overfit_mode={OVERFIT_MODE}, "
        f"overfit_sample_count={OVERFIT_SAMPLE_COUNT}, "
        f"batch_size={BATCH_SIZE}, "
        f"max_epochs={MAX_EPOCHS}, "
        f"patience={PATIENCE}, "
        f"lr={LEARNING_RATE}, "
        f"weight_decay={WEIGHT_DECAY}, "
        f"freeze_speech={FREEZE_SPEECH}, "
        f"freeze_summary={FREEZE_SUMMARY}, "
        "trainable=adapter_only"
    )

    model = ChunkedSpeechToSummaryLatentModel(
        speech_model_name=SPEECH_MODEL_NAME,
        summary_model_name=SUMMARY_MODEL_NAME,
        chunk_latent_len=CHUNK_LATENT_LEN,
        freeze_speech=FREEZE_SPEECH,
        freeze_summary=FREEZE_SUMMARY,
        max_whisper_new_tokens=MAX_WHISPER_NEW_TOKENS,
    ).to(device)
    configure_trainable_parameters(model)
    total_params, trainable_params_count = count_parameters(model)
    print(f"total parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params_count:,}")

    train_dataset = ChunkBlockSpeechTranscriptDataset(
        metadata_path=TRAIN_METADATA_PATH,
        speech_model_name=SPEECH_MODEL_NAME,
        summary_tokenizer=model.summary_tokenizer,
        prompt_text=PROMPT_TEXT,
        max_target_length=MAX_TARGET_LENGTH,
        target_field=TARGET_FIELD,
    )
    val_dataset = ChunkBlockSpeechTranscriptDataset(
        metadata_path=VAL_METADATA_PATH,
        speech_model_name=SPEECH_MODEL_NAME,
        summary_tokenizer=model.summary_tokenizer,
        prompt_text=PROMPT_TEXT,
        max_target_length=MAX_TARGET_LENGTH,
        target_field=TARGET_FIELD,
    )

    if OVERFIT_MODE:
        if len(train_dataset) < OVERFIT_SAMPLE_COUNT:
            raise ValueError(
                f"Need at least {OVERFIT_SAMPLE_COUNT} training samples, "
                f"but found {len(train_dataset)}."
            )
        train_dataset.samples = train_dataset.samples[:OVERFIT_SAMPLE_COUNT]
        # Validation deliberately uses the same examples.  A low loss here is
        # evidence of model capacity, not a claim about generalization.
        val_dataset = train_dataset

    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
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
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            bad_epochs += 1
            if PATIENCE is not None:
                print(f"no improvement: {bad_epochs}/{PATIENCE}")

        if PATIENCE is not None and bad_epochs >= PATIENCE:
            print(
                f"early stopping at epoch {epoch + 1}; "
                f"best_val_loss={best_val_loss:.4f}"
            )
            break


if __name__ == "__main__":
    train()
