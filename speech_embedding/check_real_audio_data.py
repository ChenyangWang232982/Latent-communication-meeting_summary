from torch.utils.data import DataLoader

from speech_embedding.data import MeetingSpeechSummaryDataset
from speech_embedding.paths import DATA_DIR
from speech_embedding.speech_to_summary_model import SpeechToSummaryLatentModel


def main():
    model = SpeechToSummaryLatentModel(
        speech_model_name="openai/whisper-base",
        summary_model_name="google/flan-t5-small",
        latent_len=32,
        freeze_speech=True,
        freeze_summary=False,
    )

    dataset = MeetingSpeechSummaryDataset(
        metadata_path=DATA_DIR / "metadata.jsonl",
        speech_model_name="openai/whisper-base",
        summary_tokenizer=model.summary_tokenizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    batch = next(iter(loader))

    print("input_features:", batch["input_features"].shape)
    print("prompt_input_ids:", batch["prompt_input_ids"].shape)
    print("prompt_attention_mask:", batch["prompt_attention_mask"].shape)
    print("labels:", batch["labels"].shape)

    outputs = model(**batch)

    print("loss:", outputs.loss.item())


if __name__ == "__main__":
    main()
