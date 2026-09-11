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
COMM_METHOD = "direct_projection"
COMM_TEMPERATURE = 1.0
COMM_TOP_K = None

PROMPT_TEXT = "repeat the speech transcript:"
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint_chunk_block_direct_embedding_t5_encoder.pt"

TRAIN_METADATA_PATH = DATA_DIR / "chunk_blocks_teacher" / "train.jsonl"
VAL_METADATA_PATH = DATA_DIR / "chunk_blocks_teacher" / "val.jsonl"
TARGET_FIELD = "teacher_transcript"

CHUNK_LATENT_LEN = 64
MAX_TARGET_LENGTH = 128

BATCH_SIZE = 8
MAX_EPOCHS = 40
PATIENCE = 6
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MIN_DELTA = 1e-4
GRAD_CLIP_NORM = 1.0
PRINT_EVERY = 20

FREEZE_SPEECH = True
FREEZE_SUMMARY = True
TRAIN_T5_ENCODER = True
TRAIN_T5_DECODER = False


def configure_trainable_parameters(model):
    for param in model.speech_model.parameters():
        param.requires_grad = False

    for param in model.summary_model.parameters():
        param.requires_grad = False

    for param in model.comm.parameters():
        param.requires_grad = True

    if TRAIN_T5_ENCODER:
        for name, param in model.summary_model.named_parameters():
            if name.startswith("encoder.block") or name.startswith(
                "encoder.final_layer_norm"
            ):
                param.requires_grad = True

    if TRAIN_T5_DECODER:
        for name, param in model.summary_model.named_parameters():
            if name.startswith("decoder.block") or name.startswith(
                "decoder.final_layer_norm"
            ):
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

        if step % PRINT_EVERY == 0 or step == len(loader):
            print(
                f"{phase} step {step}/{len(loader)}, "
                f"loss={loss.item():.4f}, avg_loss={total_loss / step:.4f}",
                flush=True,
            )

    return total_loss / max(len(loader), 1)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print(
        "config: "
        f"mode_name=chunk_block_direct_embedding_t5_encoder, "
        f"comm_method={COMM_METHOD}, "
        f"target_field={TARGET_FIELD}, "
        f"chunk_latent_len={CHUNK_LATENT_LEN}, "
        f"max_target_length={MAX_TARGET_LENGTH}, "
        f"batch_size={BATCH_SIZE}, "
        f"max_epochs={MAX_EPOCHS}, "
        f"patience={PATIENCE}, "
        f"lr={LEARNING_RATE}, "
        f"weight_decay={WEIGHT_DECAY}, "
        f"comm_temperature={COMM_TEMPERATURE}, "
        f"comm_top_k={COMM_TOP_K}, "
        f"freeze_speech={FREEZE_SPEECH}, "
        f"freeze_summary={FREEZE_SUMMARY}, "
        f"train_t5_encoder={TRAIN_T5_ENCODER}, "
        f"train_t5_decoder={TRAIN_T5_DECODER}"
    )

    model = ChunkedSpeechToSummaryLatentModel(
        speech_model_name=SPEECH_MODEL_NAME,
        summary_model_name=SUMMARY_MODEL_NAME,
        chunk_latent_len=CHUNK_LATENT_LEN,
        freeze_speech=FREEZE_SPEECH,
        freeze_summary=FREEZE_SUMMARY,
        comm_method=COMM_METHOD,
        comm_temperature=COMM_TEMPERATURE,
        comm_top_k=COMM_TOP_K,
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
            print(f"saved best checkpoint: {CHECKPOINT_PATH}")
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
