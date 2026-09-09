import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from speech_embedding.data import MeetingSpeechSummaryDataset
from speech_embedding.paths import CHECKPOINT_DIR, DATA_DIR
from speech_embedding.speech_to_summary_model import SpeechToSummaryLatentModel

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    #initialize the model
    model = SpeechToSummaryLatentModel(
        speech_model_name = "openai/whisper-base",
        summary_model_name = "google/flan-t5-small",
        latent_len = 32,
        freeze_speech = True,
        freeze_summary = False,
    ).to(device)

    dataset = MeetingSpeechSummaryDataset(
        metadata_path = DATA_DIR / "metadata.jsonl",
        speech_model_name = "openai/whisper-base",
        summary_tokenizer = model.summary_tokenizer,
        max_summary_length = 256,
    )

    loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = True
    )

    trainable_params = [
        param for param in model.parameters()
        if param.requires_grad
    ]

    optimizer = AdamW(trainable_params, lr=1e-4)

    model.train()
    for epoch in range(80):
        total_loss = 0.0

        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{80}, Loss: {avg_loss:.4f}")

    torch.save(
        model.state_dict(),
        CHECKPOINT_DIR / "checkpoint_embedding_overfit.pt",
    )


if __name__ == "__main__":
    train()
