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
ALIGNMENT_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "checkpoint_chunked_embedding_alignment.pt"
)

# Keep this consistent with build_teacher_summary_metadata.py and test_chunked_embedding.py.
MAX_CHUNKS = 100
CHUNK_LATENT_LEN = 4
MAX_TEXT_LENGTH = 1024

BATCH_SIZE = 1
MAX_EPOCHS = 60
PATIENCE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MIN_DELTA = 1e-4
MSE_WEIGHT = 0.1
COSINE_WEIGHT = 1.0
GRAD_CLIP_NORM = 1.0
PRINT_EVERY = 5

FREEZE_SPEECH = True
FREEZE_SUMMARY = True

TRAIN_TARGET_FIELD = TEACHER_TARGET_FIELD


def get_prompt_text():
    return TEACHER_PROMPT_TEXT


def get_checkpoint_path():
    return ALIGNMENT_CHECKPOINT_PATH


def get_max_target_length():
    return MAX_TEXT_LENGTH


def move_batch_to_device(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def compress_text_embeddings(text_embeds, text_attention_mask, target_len):
    text_latents = []

    for sample_embeds, sample_mask in zip(text_embeds, text_attention_mask):
        valid_embeds = sample_embeds[sample_mask.bool()]

        if valid_embeds.numel() == 0:
            pooled = sample_embeds.new_zeros(target_len, sample_embeds.size(-1))
        elif valid_embeds.size(0) < target_len:
            pad_len = target_len - valid_embeds.size(0)
            padding = valid_embeds[-1:].expand(pad_len, -1)
            pooled = torch.cat([valid_embeds, padding], dim=0)
        else:
            pooled = F.adaptive_avg_pool1d(
                valid_embeds.transpose(0, 1).unsqueeze(0),
                target_len,
            ).squeeze(0).transpose(0, 1)

        text_latents.append(pooled)

    return torch.stack(text_latents, dim=0)


def alignment_loss(speech_latents, text_latents, latent_mask):
    speech_latents = F.layer_norm(speech_latents, speech_latents.shape[-1:])
    text_latents = F.layer_norm(text_latents, text_latents.shape[-1:])
    speech_latents = F.normalize(speech_latents, p=2, dim=-1)
    text_latents = F.normalize(text_latents, p=2, dim=-1)

    mask = latent_mask.to(speech_latents.dtype)
    token_count = mask.sum().clamp_min(1.0)

    mse = ((speech_latents - text_latents) ** 2).sum(dim=-1)
    mse = (mse * mask).sum() / (token_count * speech_latents.size(-1))

    cosine = F.cosine_similarity(speech_latents, text_latents, dim=-1)
    cosine = 1.0 - (cosine * mask).sum() / token_count

    return MSE_WEIGHT * mse + COSINE_WEIGHT * cosine, mse, cosine


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    phase = "train" if is_train else "val"

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(is_train):
            speech_latents, latent_mask = model.encode_all_chunks(
                batch["input_features"],
                batch["chunk_attention_mask"],
            )
            with torch.no_grad():
                text_embeds = model.summary_model.get_input_embeddings()(
                    batch["target_input_ids"]
                )
                text_latents = compress_text_embeddings(
                    text_embeds,
                    batch["target_attention_mask"],
                    target_len=speech_latents.size(1),
                )

            loss, mse, cosine = alignment_loss(
                speech_latents,
                text_latents,
                latent_mask,
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
                f"mse={mse.item():.4f}, "
                f"cosine={cosine.item():.4f}, "
                f"avg_loss={avg_loss:.4f}",
                flush=True,
            )

    return total_loss / max(len(loader), 1)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)
    print(
        "config: "
        f"mode_name=embedding_alignment, "
        f"target_field={TRAIN_TARGET_FIELD}, "
        f"max_chunks={MAX_CHUNKS}, "
        f"chunk_latent_len={CHUNK_LATENT_LEN}, "
        f"batch_size={BATCH_SIZE}, "
        f"max_epochs={MAX_EPOCHS}, "
        f"patience={PATIENCE}, "
        f"lr={LEARNING_RATE}, "
        f"weight_decay={WEIGHT_DECAY}, "
        f"mse_weight={MSE_WEIGHT}, "
        f"cosine_weight={COSINE_WEIGHT}, "
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
